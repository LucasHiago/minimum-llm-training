"""
sample.py — Gera texto a partir de um modelo treinado
=====================================================

Uso:

    python sample.py --prompt "int main()"
    python sample.py --prompt "#include <vector>" --max_new_tokens 500 --temperature 0.8

Parâmetros úteis:
    --temperature : < 1.0 = mais conservador; > 1.0 = mais criativo/aleatório
    --top_k       : amostra só entre os K tokens mais prováveis
"""

import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser(description="Gera texto com o mini-GPT.")
    p.add_argument("--out", default="out", help="pasta com ckpt.pt e tokenizer.json")
    p.add_argument("--prompt", default="\n", help="texto inicial")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"Não achei {ckpt_path}. Treine primeiro com: python train.py"
        )

    # Carrega tokenizador e pesos.
    tok = CharTokenizer.load(os.path.join(args.out, "tokenizer.json"))
    ckpt = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Converte o prompt em tokens (ignora caracteres fora do vocabulário).
    start_ids = [tok.stoi[c] for c in args.prompt if c in tok.stoi] or [0]
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
