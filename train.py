"""
train.py — Loop de treino do mini-GPT
=====================================

Uso básico:

    python train.py                              # usa data/input.txt
    python train.py --data data/meu_texto.txt    # seu próprio texto
    python train.py --iters 3000 --n_layer 6     # ajusta hiperparâmetros

O que acontece aqui:
    1. Lê um arquivo de texto puro.
    2. Constrói um tokenizador de caractere a partir dele.
    3. Divide em treino (90%) e validação (10%).
    4. Treina o modelo a prever o próximo caractere.
    5. Salva pesos + tokenizador em `out/` para você gerar texto depois.
"""

import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def get_batch(data, block_size, batch_size, device):
    """Sorteia `batch_size` trechos aleatórios de `data`.

    x = trecho de tamanho block_size
    y = o mesmo trecho deslocado 1 posição (o "próximo caractere" alvo)
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, eval_iters=50):
    """Mede a loss média em treino e validação (com dropout desligado)."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser(description="Treina o mini-GPT de caractere.")
    p.add_argument("--data", default="data/input.txt", help="arquivo de texto")
    p.add_argument("--out", default="out", help="pasta de saída")
    p.add_argument("--iters", type=int, default=2000, help="passos de treino")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cpu ou cuda (auto se vazio)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # 1) Lê o texto -------------------------------------------------------
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus: {len(text):,} caracteres")

    # 2) Tokenizador ------------------------------------------------------
    tok = CharTokenizer.from_text(text)
    print(f"Vocabulário: {tok.vocab_size} caracteres únicos")
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    # 3) Split treino/validação ------------------------------------------
    n = int(0.9 * len(data))
    splits = {"treino": data[:n], "val": data[n:]}

    # 4) Modelo -----------------------------------------------------------
    config = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"Parâmetros do modelo: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 5) Loop de treino ---------------------------------------------------
    best_val = float("inf")
    os.makedirs(args.out, exist_ok=True)

    for it in range(1, args.iters + 1):
        if it % args.eval_interval == 0 or it == 1:
            losses = estimate_loss(
                model, splits, args.block_size, args.batch_size, device
            )
            print(
                f"passo {it:5d} | treino {losses['treino']:.4f} "
                f"| val {losses['val']:.4f}"
            )
            # Salva o melhor checkpoint (menor loss de validação).
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": config.__dict__},
                    os.path.join(args.out, "ckpt.pt"),
                )
                tok.save(os.path.join(args.out, "tokenizer.json"))

        x, y = get_batch(splits["treino"], args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"\nTreino concluído. Melhor loss de validação: {best_val:.4f}")
    print(f"Checkpoint salvo em: {os.path.join(args.out, 'ckpt.pt')}")
    print(f"Gere texto com:  python sample.py --prompt \"Era uma vez\"")


if __name__ == "__main__":
    main()
