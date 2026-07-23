"""
chat.py — Interface web estilo ChatGPT para o mini-GPT treinado
================================================================

Sobe um servidor web local (só com a biblioteca padrão do Python — nenhuma
dependência nova além do PyTorch) com um layout de chat: você escreve, o
modelo responde com efeito de digitação (streaming, um caractere por vez).

Uso:

    python chat.py                       # abre em http://127.0.0.1:8000
    python chat.py --port 9000
    python chat.py --temperature 0.7 --top_k 40

IMPORTANTE — o que este modelo é (e o que não é):
    Ele é um GPT de NÍVEL DE CARACTERE treinado para COMPLETAR código C++.
    Não foi ajustado para "responder perguntas" como o ChatGPT. Ele continua
    o texto que você digitou no estilo do corpus. O layout aqui é de chat só
    pela experiência — por baixo é completar texto. Digite um trecho de C++
    (ex.: "int main()") para ver o comportamento mais convincente.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Modelo: carregar e gerar em streaming (um caractere por vez)
# ---------------------------------------------------------------------------

class Engine:
    """Guarda o modelo/tokenizador carregados e gera texto em streaming."""

    def __init__(self, out_dir, device):
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        if not os.path.exists(ckpt_path):
            raise SystemExit(
                f"Não achei {ckpt_path}. Treine primeiro com: python train.py"
            )
        self.device = device
        self.tok = CharTokenizer.load(os.path.join(out_dir, "tokenizer.json"))
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model = GPT(GPTConfig(**ckpt["config"])).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def stream(self, prompt, max_new_tokens, temperature, top_k):
        """Gera texto autoregressivo, devolvendo um caractere por vez.

        É a mesma lógica de `model.generate`, mas com `yield` para que o
        servidor possa mandar cada caractere ao navegador assim que sai.
        """
        model, tok = self.model, self.tok
        block_size = model.config.block_size

        # Converte o prompt em tokens (ignora caracteres fora do vocabulário).
        start_ids = [tok.stoi[c] for c in prompt if c in tok.stoi] or [0]
        idx = torch.tensor([start_ids], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]           # corta pra janela de contexto
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # só a última posição

            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            yield tok.itos[int(next_id)]


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
</style>
</head>
<body>
  <header>
    <span class="dot"></span>
    <h1>mini-GPT</h1>
    <span class="sub">char-level · C++ · __DEVICE__</span>
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
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const send = document.getElementById('send');
const empty = document.getElementById('empty');
const temp = document.getElementById('temp'), tempv = document.getElementById('tempv');
const tokens = document.getElementById('tokens'), tokensv = document.getElementById('tokensv');
const topk = document.getElementById('topk'), topkv = document.getElementById('topkv');

temp.oninput = () => tempv.textContent = temp.value;
tokens.oninput = () => tokensv.textContent = tokens.value;
topk.oninput = () => topkv.textContent = topk.value === '0' ? 'off' : topk.value;

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
    <div class="msg"></div></div>`;
  row.querySelector('.msg').textContent = text;
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return row.querySelector('.msg');
}

function setBusy(b) {
  busy = b;
  send.disabled = b;
  send.textContent = b ? '■' : '↑';
}

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

  const msg = addRow('bot', '');
  msg.classList.add('cursor');
  setBusy(true);

  const qs = new URLSearchParams({
    prompt, temperature: temp.value, max_new_tokens: tokens.value, top_k: topk.value,
  });
  es = new EventSource('/stream?' + qs.toString());
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
    page = (
        PAGE.replace("__DEVICE__", engine.device)
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

        def _stream(self, q):
            prompt = (q.get("prompt", [""])[0])
            temperature = float(q.get("temperature", [defaults["temperature"]])[0])
            max_new = int(float(q.get("max_new_tokens", [defaults["max_new_tokens"]])[0]))
            top_k = int(float(q.get("top_k", [0])[0])) or None
            max_new = max(1, min(max_new, 2000))  # trava de segurança

            # Server-Sent Events: manda cada caractere assim que sai do modelo.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                for ch in engine.stream(prompt, max_new, temperature, top_k):
                    data = json.dumps(ch)  # JSON escapa quebras de linha etc.
                    self.wfile.write(f"event: token\ndata: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"event: done\ndata: 1\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # o navegador fechou a conexão (usuário parou a geração)

    return Handler


def main():
    p = argparse.ArgumentParser(description="Chat web para o mini-GPT.")
    p.add_argument("--out", default="out", help="pasta com ckpt.pt e tokenizer.json")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(args.out, device)
    defaults = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }

    handler = make_handler(engine, defaults)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    n_params = engine.model.num_params()
    print(f"Modelo carregado ({n_params:,} parâmetros) em {device}.")
    print(f"Chat no ar: http://{args.host}:{args.port}  (Ctrl+C para sair)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAté mais!")
        server.shutdown()


if __name__ == "__main__":
    main()
