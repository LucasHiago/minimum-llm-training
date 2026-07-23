"""
train.py — Loop de treino do mini-GPT
=====================================

Uso básico:

    python train.py                              # usa data/cpp.txt
    python train.py --data data/cpp.txt          # corpus de C++ do usuário
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
import time

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


# Presets: conjuntos prontos de hiperparâmetros. `tiny`/`fast` são para
# MICRO-TREINOS rápidos (ótimos para experimentar e para treino cumulativo com
# --resume). `balanced` é o padrão histórico. `quality` só compensa com GPU.
#   tiny  ~40s na CPU   |  balanced ~10min na CPU  |  quality: use GPU
PRESETS = {
    "tiny":     dict(iters=1000, batch_size=16, block_size=64,  n_layer=2, n_head=4, n_embd=64,  eval_iters=20),
    "fast":     dict(iters=1500, batch_size=32, block_size=96,  n_layer=3, n_head=4, n_embd=96,  eval_iters=30),
    "balanced": dict(iters=2000, batch_size=32, block_size=128, n_layer=4, n_head=4, n_embd=128, eval_iters=50),
    "quality":  dict(iters=4000, batch_size=64, block_size=192, n_layer=6, n_head=6, n_embd=192, eval_iters=50),
}


def encode_known(tok, text):
    """Codifica `text` ignorando caracteres fora do vocabulário do tokenizador.

    Necessário no treino cumulativo (--resume): o vocabulário é fixado no 1º
    treino, então um pedaço novo pode conter caracteres que o modelo não pode
    representar. Devolve (ids, n_ignorados)."""
    ids = [tok.stoi[c] for c in text if c in tok.stoi]
    return ids, len(text) - len(ids)


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
    p.add_argument("--data", default="data/cpp.txt", help="arquivo de texto")
    p.add_argument("--out", default="out", help="pasta de saída")
    p.add_argument("--preset", choices=list(PRESETS), default="balanced",
                   help="conjunto de hiperparâmetros (tiny/fast/balanced/quality)")
    p.add_argument("--resume", action="store_true",
                   help="continua o treino do checkpoint em --out (treino cumulativo)")
    p.add_argument("--vocab", default=None,
                   help="usa um vocabulário fixo (JSON do build_vocab.py) em vez de "
                        "construir do texto — necessário para especialistas do MoE")
    # Estes ficam como None: se não forem passados, herdam do --preset.
    p.add_argument("--iters", type=int, help="passos de treino DESTE run")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--block_size", type=int)
    p.add_argument("--n_layer", type=int)
    p.add_argument("--n_head", type=int)
    p.add_argument("--n_embd", type=int)
    p.add_argument("--eval_iters", type=int)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cpu ou cuda (auto se vazio)")
    p.add_argument("--amp", action="store_true",
                   help="mixed precision na GPU (mais rápido, usa menos memória)")
    args = p.parse_args()

    # Resolve os hiperparâmetros: valor explícito na linha de comando vence;
    # senão, cai no valor do preset escolhido.
    preset = PRESETS[args.preset]
    for k, v in preset.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # AMP (float16) só faz sentido em GPU CUDA.
    use_amp = args.amp and device == "cuda"
    if args.amp and not use_amp:
        print("Aviso: --amp ignorado (só funciona em GPU CUDA).")
    if use_amp:
        print("Mixed precision (AMP float16) ativado.")

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "ckpt.pt")

    # 1) Lê o texto -------------------------------------------------------
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus: {len(text):,} caracteres")

    # 2) Retomada (opcional) ou início do zero ---------------------------
    start_iter, best_val, ckpt = 0, float("inf"), None
    if args.resume:
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"--resume pediu {ckpt_path}, mas ele não existe. "
                             "Treine uma vez sem --resume primeiro.")
        ckpt = torch.load(ckpt_path, map_location=device)
        # No treino cumulativo o vocabulário e a arquitetura vêm do checkpoint.
        tok = CharTokenizer.load(os.path.join(args.out, "tokenizer.json"))
        config = GPTConfig(**ckpt["config"])
        start_iter = ckpt.get("iter", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Retomando de {ckpt_path} (já treinado por {start_iter:,} passos).")
    else:
        # Vocabulário fixo (para MoE) ou construído a partir do próprio texto.
        tok = CharTokenizer.load(args.vocab) if args.vocab else CharTokenizer.from_text(text)
        config = GPTConfig(
            vocab_size=tok.vocab_size, block_size=args.block_size,
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=args.dropout,
        )
    print(f"Vocabulário: {tok.vocab_size} caracteres únicos")

    ids, ignored = encode_known(tok, text)
    if ignored:
        print(f"Aviso: {ignored:,} caractere(s) fora do vocabulário foram "
              "ignorados (o vocabulário é fixado no 1º treino).")
    data = torch.tensor(ids, dtype=torch.long)

    # 3) Split treino/validação ------------------------------------------
    n = int(0.9 * len(data))
    splits = {"treino": data[:n], "val": data[n:]}

    # 4) Modelo + otimizador ---------------------------------------------
    model = GPT(config).to(device)
    print(f"Parâmetros do modelo: {model.num_params():,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # GradScaler evita underflow dos gradientes em float16 (no-op se AMP off).
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    if ckpt is not None:
        # Restaura pesos + estado do otimizador (o "momentum" do AdamW). Sem
        # isso a retomada daria um solavanco na loss, como se recomeçasse frio.
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if use_amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

    def save(it):
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config.__dict__,
            "iter": it,
            "best_val": best_val,
        }, ckpt_path)
        tok.save(os.path.join(args.out, "tokenizer.json"))

    # 5) Loop de treino (soma `args.iters` passos ao total acumulado) -----
    block_size = config.block_size  # a janela vem do modelo, não do preset
    end_iter = start_iter + args.iters
    t0 = time.time()
    for it in range(start_iter + 1, end_iter + 1):
        if it % args.eval_interval == 0 or it == start_iter + 1:
            losses = estimate_loss(
                model, splits, block_size, args.batch_size, device, args.eval_iters
            )
            best_val = min(best_val, losses["val"])
            done = it - start_iter
            rate = done / max(time.time() - t0, 1e-9)  # passos/s deste run
            eta = (end_iter - it) / max(rate, 1e-9)
            print(
                f"passo {it:5d}/{end_iter} | treino {losses['treino']:.4f} "
                f"| val {losses['val']:.4f} | {rate:4.1f} passos/s "
                f"| falta ~{eta:4.0f}s"
            )
            save(it)  # salva o estado mais recente (permite retomar depois)

        x, y = get_batch(splits["treino"], block_size, args.batch_size, device)
        # autocast roda a passagem em float16 na GPU quando AMP está ligado.
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()  # escala a loss antes do backward
        scaler.step(optimizer)         # aplica o passo (desfaz a escala)
        scaler.update()                # ajusta o fator de escala

    save(end_iter)  # garante o checkpoint final com o total de passos
    elapsed = time.time() - t0
    print(f"\nTreino concluído em {elapsed:.0f}s. Total acumulado: {end_iter:,} passos.")
    print(f"Melhor loss de validação: {best_val:.4f}")
    print(f"Checkpoint salvo em: {ckpt_path}")
    print("Continue treinando com:  python train.py --resume")
    print("Gere código com:  python sample.py --prompt \"#include <iostream>\"")


if __name__ == "__main__":
    main()
