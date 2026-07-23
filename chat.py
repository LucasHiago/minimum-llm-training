"""
chat.py — Interface web estilo ChatGPT para o mini-GPT (com MoE opcional)
=========================================================================

Sobe um servidor web local (só com a biblioteca padrão do Python — nenhuma
dependência nova além do PyTorch) com um layout de chat: você escreve, o
modelo responde com efeito de digitação (streaming, um caractere por vez).

Um modelo só:

    python chat.py                       # abre em http://127.0.0.1:8000
    python chat.py --port 9000 --temperature 0.7

Vários especialistas (Mixture of Experts — veja moe.py e build_vocab.py):

    python chat.py --experts out/vetores out/classes

    Com >1 especialista, a interface ganha um seletor de modo:
      • route  — um "porteiro" escolhe o especialista que melhor combina com o
                 seu prompt (menor perplexidade) e SÓ ele responde.
      • blend  — a cada letra TODOS votam, com pesos ao vivo (o "neurônios em
                 grupo"). Exige vocabulário compartilhado (build_vocab.py).

IMPORTANTE — o que este modelo é (e o que não é):
    Ele é um GPT de NÍVEL DE CARACTERE treinado para COMPLETAR código C++.
    Não foi ajustado para "responder perguntas" como o ChatGPT. Ele continua
    o texto que você digitou no estilo do corpus. Digite um trecho de C++
    (ex.: "int main()") para ver o comportamento mais convincente.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import torch
from torch.nn import functional as F

from moe import Expert


# ---------------------------------------------------------------------------
# Motor: carrega 1+ especialistas e gera em streaming (um caractere por vez)
# ---------------------------------------------------------------------------

class ChatEngine:
    """Carrega os especialistas e gera texto em streaming.

    Com um único especialista, é um chat comum. Com vários, vira um Mixture of
    Experts: `route` (escolhe um) ou `blend` (funde os votos de todos).

    Os métodos de streaming produzem tuplas (tipo, dado):
        ("meta",    {...})  — info de roteamento no início da resposta
        ("weights", [...])  — pesos ao vivo de cada especialista (modo blend)
        ("token",   "c")    — o próximo caractere gerado
    """

    def __init__(self, expert_dirs, device):
        self.device = device
        self.experts = [Expert(d, device) for d in expert_dirs]
        self.names = [e.name for e in self.experts]
        self.multi = len(self.experts) > 1
        # `blend` só é possível se todos falam o mesmo alfabeto (mesmo vocab).
        base = self.experts[0].tok.chars
        self.blend_ok = self.multi and all(e.tok.chars == base for e in self.experts)

    def stream(self, prompt, mode, max_new_tokens, temperature, top_k, router_temp):
        if mode == "blend" and self.blend_ok:
            yield from self._blend(prompt, max_new_tokens, temperature, top_k, router_temp)
        else:
            yield from self._route(prompt, max_new_tokens, temperature, top_k)

    @torch.no_grad()
    def _route(self, prompt, max_new_tokens, temperature, top_k):
        """Escolhe o especialista de menor perplexidade e gera só com ele."""
        if self.multi:
            scored = sorted(((e.nll(prompt), e) for e in self.experts), key=lambda t: t[0])
            winner = scored[0][1]
            yield ("meta", {"mode": "route", "chosen": winner.name,
                            "scores": [{"name": e.name, "nll": round(s, 3)} for s, e in scored]})
        else:
            winner = self.experts[0]

        ids = winner.encode(prompt) or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        block = winner.block_size
        for _ in range(max_new_tokens):
            logits, _ = winner.model(idx[:, -block:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            yield ("token", winner.tok.itos[int(next_id)])

    @torch.no_grad()
    def _blend(self, prompt, max_new_tokens, temperature, top_k, router_temp):
        """A cada caractere, todos votam; combina pesado pela relevância atual."""
        tok = self.experts[0].tok
        yield ("meta", {"mode": "blend", "experts": self.names})

        ids = [tok.stoi[c] for c in prompt if c in tok.stoi] or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        for step in range(max_new_tokens):
            probs_each, ctx_nll = [], []
            for e in self.experts:
                cond = idx[:, -e.block_size:]
                logits, _ = e.model(cond)
                p = F.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1)
                probs_each.append(p.squeeze(0))
                if cond.size(1) >= 2:
                    nll = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                          cond[:, 1:].reshape(-1)).item()
                else:
                    nll = 0.0
                ctx_nll.append(nll)

            w = F.softmax(-torch.tensor(ctx_nll) / max(router_temp, 1e-6), dim=0)
            mixed = sum(wi * pi for wi, pi in zip(w, probs_each))
            if top_k:
                v, _ = torch.topk(mixed, min(top_k, mixed.numel()))
                mixed[mixed < v[-1]] = 0.0
                mixed /= mixed.sum()
            next_id = torch.multinomial(mixed, num_samples=1)
            idx = torch.cat((idx, next_id.view(1, 1)), dim=1)

            if step % 8 == 0:  # atualiza as barras de peso de vez em quando
                yield ("weights", [{"name": n, "w": round(float(wi), 3)}
                                    for n, wi in zip(self.names, w)])
            yield ("token", tok.itos[int(next_id)])


# ---------------------------------------------------------------------------
# Página HTML (layout de chat, tudo embutido — zero arquivos externos)
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mini-GPT · chat</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --panel2: #1c2230; --border: #2a3140;
    --text: #e6edf3; --muted: #8b949e; --accent: #2f81f7; --user: #1f6feb;
    --code: #0a0e14;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg); color: var(--text);
    font: 15px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex; flex-direction: column;
  }
  header {
    padding: 14px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px; background: var(--panel);
  }
  header .dot { width: 9px; height: 9px; border-radius: 50%; background: #3fb950; box-shadow: 0 0 8px #3fb950; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .2px; }
  header .sub { color: var(--muted); font-size: 12px; margin-left: auto; }

  #chat {
    flex: 1; overflow-y: auto; padding: 24px 0;
    display: flex; flex-direction: column; gap: 18px;
  }
  .row { width: 100%; max-width: 780px; margin: 0 auto; padding: 0 20px; display: flex; gap: 12px; }
  .avatar {
    width: 30px; height: 30px; border-radius: 7px; flex: 0 0 30px;
    display: grid; place-items: center; font-size: 13px; font-weight: 700; margin-top: 2px;
  }
  .user .avatar { background: var(--user); color: #fff; }
  .bot  .avatar { background: #238636; color: #fff; }
  .bubble { flex: 1; min-width: 0; }
  .who { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .msg {
    white-space: pre-wrap; word-wrap: break-word;
    font-family: "SFMono-Regular", ui-monospace, "JetBrains Mono", Consolas, monospace;
    font-size: 13.5px; line-height: 1.55;
    background: var(--code); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
  }
  .user .msg { background: var(--panel2); border-color: #30395022; font-family: inherit; font-size: 15px; }
  .cursor::after { content: "▍"; color: var(--accent); animation: blink 1s steps(1) infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  /* Painel de roteamento MoE (badges no route, barras no blend) */
  .route { margin-bottom: 8px; display: none; flex-direction: column; gap: 5px; }
  .route.show { display: flex; }
  .badges { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .badge { font-size: 11px; padding: 3px 9px; border-radius: 20px; border: 1px solid var(--border);
           color: var(--muted); font-family: ui-monospace, monospace; }
  .badge.win { background: #23863622; border-color: #3fb95066; color: #7ee787; }
  .bar { display: grid; grid-template-columns: 96px 1fr 44px; gap: 8px; align-items: center; font-size: 11px; }
  .bar .name { color: var(--muted); font-family: ui-monospace, monospace; text-align: right; overflow: hidden; text-overflow: ellipsis; }
  .bar .track { height: 7px; background: var(--panel2); border-radius: 6px; overflow: hidden; }
  .bar .fill { height: 100%; background: linear-gradient(90deg, var(--accent), #7ee787); transition: width .15s; }
  .bar .pct { color: var(--text); text-align: right; font-variant-numeric: tabular-nums; }

  .empty { margin: auto; text-align: center; color: var(--muted); max-width: 460px; padding: 0 20px; }
  .empty h2 { color: var(--text); font-weight: 600; margin: 0 0 8px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 18px; }
  .chip {
    background: var(--panel2); border: 1px solid var(--border); color: var(--text);
    border-radius: 20px; padding: 7px 13px; font-size: 13px; cursor: pointer;
    font-family: ui-monospace, monospace;
  }
  .chip:hover { border-color: var(--accent); }

  footer { border-top: 1px solid var(--border); background: var(--panel); padding: 14px 0 18px; }
  .composer { max-width: 780px; margin: 0 auto; padding: 0 20px; }
  .inputwrap {
    display: flex; align-items: flex-end; gap: 10px;
    background: var(--panel2); border: 1px solid var(--border); border-radius: 14px; padding: 8px 8px 8px 14px;
  }
  .inputwrap:focus-within { border-color: var(--accent); }
  textarea {
    flex: 1; resize: none; background: transparent; border: 0; color: var(--text);
    font: inherit; outline: none; max-height: 180px; padding: 6px 0;
    font-family: ui-monospace, monospace;
  }
  button.send {
    background: var(--accent); color: #fff; border: 0; border-radius: 9px;
    width: 38px; height: 38px; cursor: pointer; font-size: 17px; flex: 0 0 38px;
    display: grid; place-items: center;
  }
  button.send:disabled { opacity: .4; cursor: not-allowed; }
  .controls { display: flex; gap: 16px; align-items: center; margin-top: 10px; color: var(--muted); font-size: 12px; flex-wrap: wrap; }
  .controls label { display: flex; align-items: center; gap: 7px; }
  .controls input[type=range] { accent-color: var(--accent); width: 110px; }
  .controls .val { color: var(--text); min-width: 30px; font-variant-numeric: tabular-nums; }
  .hint { text-align: center; color: var(--muted); font-size: 11px; margin-top: 10px; }

  /* Seletor de modo route/blend (só aparece com vários especialistas) */
  #modewrap { display: none; }
  #modewrap.show { display: flex; }
  .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .seg button { background: transparent; color: var(--muted); border: 0; padding: 4px 12px; cursor: pointer; font-size: 12px; }
  .seg button.on { background: var(--accent); color: #fff; }
  #rtwrap { display: none; }
  #rtwrap.show { display: flex; }
</style>
</head>
<body>
  <header>
    <span class="dot"></span>
    <h1>mini-GPT</h1>
    <span class="sub">__SUBTITLE__</span>
  </header>

  <div id="chat">
    <div class="empty" id="empty">
      <h2>Complete um trecho de C++</h2>
      <div>Este modelo <b>completa código</b> no estilo do corpus — não é um assistente que responde perguntas. Comece com um trecho:</div>
      <div class="chips">
        <span class="chip">int main()</span>
        <span class="chip">#include &lt;vector&gt;</span>
        <span class="chip">for (int i = 0;</span>
        <span class="chip">class Node {</span>
      </div>
    </div>
  </div>

  <footer>
    <div class="composer">
      <div class="inputwrap">
        <textarea id="input" rows="1" placeholder="Escreva um trecho de C++ e pressione Enter…"></textarea>
        <button class="send" id="send" title="Enviar (Enter)">↑</button>
      </div>
      <div class="controls">
        <label id="modewrap">modo
          <span class="seg" id="seg">
            <button data-mode="route" class="on">route</button>
            <button data-mode="blend">blend</button>
          </span>
        </label>
        <label id="rtwrap">mistura <input type="range" id="rt" min="0.1" max="4" step="0.1" value="__RT__">
          <span class="val" id="rtv">__RT__</span></label>
        <label>temperatura <input type="range" id="temp" min="0.1" max="1.5" step="0.1" value="__TEMP__">
          <span class="val" id="tempv">__TEMP__</span></label>
        <label>tokens <input type="range" id="tokens" min="50" max="600" step="50" value="__MAXTOK__">
          <span class="val" id="tokensv">__MAXTOK__</span></label>
        <label>top-k <input type="range" id="topk" min="0" max="80" step="5" value="__TOPK__">
          <span class="val" id="topkv">__TOPKLABEL__</span></label>
      </div>
      <div class="hint">Enter envia · Shift+Enter quebra linha · Esc interrompe a geração</div>
    </div>
  </footer>

<script>
const EXPERTS = __EXPERTS__;      // nomes dos especialistas
const BLEND_OK = __BLENDOK__;     // fusão disponível (vocab compartilhado)

const chat = document.getElementById('chat');
const input = document.getElementById('input');
const send = document.getElementById('send');
const empty = document.getElementById('empty');
const temp = document.getElementById('temp'), tempv = document.getElementById('tempv');
const tokens = document.getElementById('tokens'), tokensv = document.getElementById('tokensv');
const topk = document.getElementById('topk'), topkv = document.getElementById('topkv');
const rt = document.getElementById('rt'), rtv = document.getElementById('rtv');

temp.oninput = () => tempv.textContent = temp.value;
tokens.oninput = () => tokensv.textContent = tokens.value;
topk.oninput = () => topkv.textContent = topk.value === '0' ? 'off' : topk.value;
rt.oninput = () => rtv.textContent = rt.value;

// Seletor de modo (route/blend), só quando há >1 especialista.
let mode = 'route';
if (EXPERTS.length > 1) {
  document.getElementById('modewrap').classList.add('show');
  const seg = document.getElementById('seg');
  const blendBtn = seg.querySelector('[data-mode=blend]');
  if (!BLEND_OK) { blendBtn.disabled = true; blendBtn.title = 'Precisa de vocabulário compartilhado (build_vocab.py)'; blendBtn.style.opacity = .4; }
  seg.querySelectorAll('button').forEach(b => b.onclick = () => {
    if (b.disabled) return;
    mode = b.dataset.mode;
    seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    document.getElementById('rtwrap').classList.toggle('show', mode === 'blend');
  });
}

let es = null;             // EventSource ativo
let busy = false;

function autosize() { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 180) + 'px'; }
input.addEventListener('input', autosize);

function addRow(role, text) {
  if (empty) empty.remove();
  const row = document.createElement('div');
  row.className = 'row ' + role;
  row.innerHTML = `<div class="avatar">${role === 'user' ? 'Tu' : 'GPT'}</div>
    <div class="bubble"><div class="who">${role === 'user' ? 'Você' : 'mini-GPT'}</div>
    <div class="route"></div><div class="msg"></div></div>`;
  row.querySelector('.msg').textContent = text;
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return { msg: row.querySelector('.msg'), route: row.querySelector('.route') };
}

function renderRoute(panel, m) {
  panel.classList.add('show');
  const badges = m.scores.map(s =>
    `<span class="badge ${s.name === m.chosen ? 'win' : ''}">${s.name} · ${s.nll}</span>`).join('');
  panel.innerHTML = `<div class="badges"><span style="color:var(--muted)">roteado →</span>${badges}</div>`;
}

function renderWeights(panel, list) {
  panel.classList.add('show');
  panel.innerHTML = list.map(x =>
    `<div class="bar"><span class="name">${x.name}</span>
       <span class="track"><span class="fill" style="width:${(x.w*100).toFixed(0)}%"></span></span>
       <span class="pct">${(x.w*100).toFixed(0)}%</span></div>`).join('');
}

function setBusy(b) { busy = b; send.disabled = b; send.textContent = b ? '■' : '↑'; }

function stop() {
  if (es) { es.close(); es = null; }
  document.querySelectorAll('.cursor').forEach(el => el.classList.remove('cursor'));
  setBusy(false);
}

function submit() {
  const prompt = input.value.trim();
  if (!prompt || busy) return;
  addRow('user', prompt);
  input.value = ''; autosize();

  const { msg, route } = addRow('bot', '');
  msg.classList.add('cursor');
  setBusy(true);

  const qs = new URLSearchParams({
    prompt, mode, temperature: temp.value, max_new_tokens: tokens.value,
    top_k: topk.value, router_temp: rt.value,
  });
  es = new EventSource('/stream?' + qs.toString());
  es.addEventListener('meta', e => {
    const m = JSON.parse(e.data);
    if (m.mode === 'route' && m.scores) renderRoute(route, m);
    else if (m.mode === 'blend') renderWeights(route, m.experts.map(n => ({ name: n, w: 1 / m.experts.length })));
  });
  es.addEventListener('weights', e => renderWeights(route, JSON.parse(e.data)));
  es.addEventListener('token', e => {
    msg.textContent += JSON.parse(e.data);
    chat.scrollTop = chat.scrollHeight;
  });
  es.addEventListener('done', () => stop());
  es.onerror = () => stop();
}

send.onclick = () => busy ? stop() : submit();
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});
document.addEventListener('keydown', e => { if (e.key === 'Escape' && busy) stop(); });
document.querySelectorAll('.chip').forEach(c => c.onclick = () => {
  input.value = c.textContent; autosize(); input.focus();
});
input.focus();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Servidor HTTP
# ---------------------------------------------------------------------------

def make_handler(engine, defaults):
    if engine.multi:
        subtitle = f"char-level · {len(engine.experts)} especialistas · {engine.device}"
    else:
        subtitle = f"char-level · C++ · {engine.device}"
    page = (
        PAGE.replace("__SUBTITLE__", subtitle)
        .replace("__EXPERTS__", json.dumps(engine.names))
        .replace("__BLENDOK__", "true" if engine.blend_ok else "false")
        .replace("__RT__", str(defaults["router_temp"]))
        .replace("__TEMP__", str(defaults["temperature"]))
        .replace("__MAXTOK__", str(defaults["max_new_tokens"]))
        .replace("__TOPK__", str(defaults["top_k"] or 0))
        .replace("__TOPKLABEL__", str(defaults["top_k"]) if defaults["top_k"] else "off")
        .encode("utf-8")
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # silencia o log padrão (uma linha por requisição)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send_html(page)
            elif url.path == "/stream":
                self._stream(parse_qs(url.query))
            else:
                self.send_error(404)

        def _send_html(self, body):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _sse(self, event, data):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
            self.wfile.flush()

        def _stream(self, q):
            prompt = q.get("prompt", [""])[0]
            mode = q.get("mode", ["route"])[0]
            temperature = float(q.get("temperature", [defaults["temperature"]])[0])
            max_new = int(float(q.get("max_new_tokens", [defaults["max_new_tokens"]])[0]))
            top_k = int(float(q.get("top_k", [0])[0])) or None
            router_temp = float(q.get("router_temp", [defaults["router_temp"]])[0])
            max_new = max(1, min(max_new, 2000))  # trava de segurança

            # Server-Sent Events: manda cada evento assim que sai do modelo.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                for kind, data in engine.stream(prompt, mode, max_new, temperature,
                                                top_k, router_temp):
                    self._sse(kind, data)  # kind ∈ {meta, weights, token}
                self._sse("done", 1)
            except (BrokenPipeError, ConnectionResetError):
                pass  # o navegador fechou a conexão (usuário parou a geração)

    return Handler


def main():
    p = argparse.ArgumentParser(description="Chat web para o mini-GPT (com MoE opcional).")
    p.add_argument("--out", default="out", help="pasta com ckpt.pt e tokenizer.json")
    p.add_argument("--experts", nargs="+", default=None,
                   help="pastas de especialistas (Mixture of Experts). Sem isto, usa --out")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--router_temp", type=float, default=0.5,
                   help="[blend] alto = mistura mais democrática; baixo = o melhor domina")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = ChatEngine(args.experts or [args.out], device)
    defaults = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "router_temp": args.router_temp,
    }

    handler = make_handler(engine, defaults)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    if engine.multi:
        print(f"{len(engine.experts)} especialistas: {', '.join(engine.names)} "
              f"({'route+blend' if engine.blend_ok else 'só route — vocab difere'}) em {device}.")
    else:
        print(f"Modelo carregado ({engine.experts[0].model.num_params():,} parâmetros) em {device}.")
    print(f"Chat no ar: http://{args.host}:{args.port}  (Ctrl+C para sair)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAté mais!")
        server.shutdown()


if __name__ == "__main__":
    main()
