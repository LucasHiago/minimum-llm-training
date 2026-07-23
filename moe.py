"""
moe.py — Mistura de Especialistas (Mixture of Experts) sobre mini-GPTs
======================================================================

A ideia (a sua!): vários modelos PEQUENOS e especializados que, juntos,
respondem melhor que um sozinho — como neurônios, que só fazem sentido em grupo.
Cada especialista é um checkpoint treinado num corpus/estilo diferente.

Dois modos de eles trabalharem juntos:

  --mode route  (roteador / "hard routing")
      Um "porteiro" mede qual especialista fica MENOS surpreso com o seu prompt
      (menor perplexidade) e deixa SÓ ele responder. É o "acessa o A pelo B"
      via um coordenador que escolhe o melhor.

  --mode blend  (fusão por token / o MoE "de verdade" / "soft routing")
      A cada caractere, TODOS os especialistas votam na próxima letra. Os votos
      são combinados com PESOS proporcionais a quão bem cada um explica o texto
      até aqui. Isso é o "neurônios disparando em grupo" — e o peso pode migrar
      de um especialista para outro no meio da geração.

Uso:

    # roteia o prompt para o melhor especialista e gera com ele
    python moe.py --experts out/vetores out/classes --mode route \
        --prompt "std::vector<int> v;"

    # funde os votos de todos, letra a letra
    python moe.py --experts out/vetores out/classes --mode blend \
        --prompt "std::vector<int> v;"

Pré-requisito do modo `blend`: todos os especialistas precisam do MESMO
vocabulário (veja build_vocab.py). No modo `route` isso é recomendado para as
perplexidades serem comparáveis, mas não é obrigatório.
"""

import argparse
import os

import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


class Expert:
    """Um especialista: um mini-GPT treinado + seu tokenizador + um nome."""

    def __init__(self, path, device):
        self.name = os.path.basename(os.path.normpath(path))
        self.device = device
        ckpt = torch.load(os.path.join(path, "ckpt.pt"), map_location=device)
        self.tok = CharTokenizer.load(os.path.join(path, "tokenizer.json"))
        self.model = GPT(GPTConfig(**ckpt["config"])).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @property
    def block_size(self):
        return self.model.config.block_size

    def encode(self, text):
        """Prompt -> ids, ignorando caracteres fora do vocabulário do especialista."""
        return [self.tok.stoi[c] for c in text if c in self.tok.stoi]

    @torch.no_grad()
    def nll(self, text):
        """Perplexidade (log): quão 'surpreso' este especialista fica com o texto.

        Mede a loss média ao prever cada próximo caractere do prompt. Menor =
        o texto 'combina' mais com o que este especialista aprendeu. É o sinal
        que o roteador usa para escolher."""
        ids = self.encode(text)
        if len(ids) < 2:
            return float("inf")  # curto demais para avaliar
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        idx = idx[:, -self.block_size:]
        _, loss = self.model(idx[:, :-1], idx[:, 1:])
        return loss.item()


# ---------------------------------------------------------------------------
# Modo 1 — Roteador: escolhe UM especialista e gera com ele
# ---------------------------------------------------------------------------

def route(experts, prompt, gen_kwargs):
    print("Perplexidade (log) de cada especialista no prompt:")
    scored = sorted(((e.nll(prompt), e) for e in experts), key=lambda t: t[0])
    for score, e in scored:
        tag = "  <- escolhido" if e is scored[0][1] else ""
        print(f"  {e.name:20s} {score:7.4f}{tag}")

    winner = scored[0][1]
    print(f"\nRoteado para: {winner.name}\n")

    ids = winner.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=winner.device)
    out = winner.model.generate(idx, **gen_kwargs)
    return winner.tok.decode(out[0].tolist())


# ---------------------------------------------------------------------------
# Modo 2 — Fusão por token: todos votam cada letra, pesados pela relevância
# ---------------------------------------------------------------------------

@torch.no_grad()
def blend(experts, prompt, gen_kwargs, router_temp=0.5):
    # A fusão exige um alfabeto comum: os votos são somados índice a índice.
    vocab = experts[0].tok.chars
    for e in experts[1:]:
        if e.tok.chars != vocab:
            raise SystemExit(
                f"O especialista '{e.name}' tem um vocabulário diferente. "
                "Para --mode blend, treine todos com o mesmo --vocab "
                "(veja build_vocab.py)."
            )
    tok = experts[0].tok
    device = experts[0].device
    max_new = gen_kwargs["max_new_tokens"]
    temperature = gen_kwargs.get("temperature", 1.0)
    top_k = gen_kwargs.get("top_k")

    ids = [tok.stoi[c] for c in prompt if c in tok.stoi] or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    contrib = torch.zeros(len(experts))  # quanto cada um "mandou" na média (debug)
    for _ in range(max_new):
        probs_each, ctx_nll = [], []
        for e in experts:
            cond = idx[:, -e.block_size:]
            logits, _ = e.model(cond)                    # (1, T, V)
            # Voto: distribuição do PRÓXIMO caractere (com temperatura).
            p = F.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1)
            probs_each.append(p.squeeze(0))
            # Relevância: quão bem ESTE especialista explica o contexto atual.
            if cond.size(1) >= 2:
                nll = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                      cond[:, 1:].reshape(-1)).item()
            else:
                nll = 0.0
            ctx_nll.append(nll)

        # Pesos: especialista que "explica" melhor o texto pesa mais (softmax
        # sobre -nll). É o roteamento suave — pode migrar durante a geração.
        w = F.softmax(-torch.tensor(ctx_nll) / router_temp, dim=0)
        contrib += w
        mixed = sum(wi * pi for wi, pi in zip(w, probs_each))  # mistura os votos

        if top_k:
            v, _ = torch.topk(mixed, min(top_k, mixed.numel()))
            mixed[mixed < v[-1]] = 0.0
            mixed /= mixed.sum()

        next_id = torch.multinomial(mixed, num_samples=1)
        idx = torch.cat((idx, next_id.view(1, 1)), dim=1)

    contrib /= max_new
    print("Contribuição média de cada especialista na geração:")
    for e, c in sorted(zip(experts, contrib.tolist()), key=lambda t: -t[1]):
        print(f"  {e.name:20s} {c*100:5.1f}%")
    print()
    return tok.decode(idx[0].tolist())


def main():
    p = argparse.ArgumentParser(description="Mistura de especialistas (MoE) de mini-GPTs.")
    p.add_argument("--experts", nargs="+", required=True,
                   help="pastas dos especialistas (cada uma com ckpt.pt e tokenizer.json)")
    p.add_argument("--mode", choices=["route", "blend"], default="route")
    p.add_argument("--prompt", default="int main()")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--router_temp", type=float, default=0.5,
                   help="[blend] temperatura do roteador: alta = mistura mais "
                        "democrática entre especialistas; baixa = o melhor domina")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    experts = [Expert(path, device) for path in args.experts]
    print(f"{len(experts)} especialistas carregados: "
          f"{', '.join(e.name for e in experts)}\n")

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature, top_k=args.top_k)

    if args.mode == "route":
        text = route(experts, args.prompt, gen_kwargs)
    else:
        text = blend(experts, args.prompt, gen_kwargs, router_temp=args.router_temp)
    print("-" * 60)
    print(text)


if __name__ == "__main__":
    main()
