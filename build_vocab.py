"""
build_vocab.py — Monta um vocabulário COMPARTILHADO entre vários corpora
========================================================================

Para os especialistas do MoE (`moe.py`) se combinarem — sobretudo no modo
`blend`, em que eles votam na mesma letra — todos precisam falar o MESMO
"alfabeto": o mesmo conjunto de caracteres, na mesma ordem. Este script lê
vários arquivos de texto e salva a UNIÃO dos caracteres num tokenizador único.

Uso:

    python build_vocab.py --out out/vocab.json data/vetores.txt data/classes.txt

Depois, treine cada especialista com esse vocabulário fixo:

    python train.py --data data/vetores.txt --vocab out/vocab.json --out out/vetores
    python train.py --data data/classes.txt --vocab out/vocab.json --out out/classes
"""

import argparse
import os

from tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser(description="Monta um vocabulário compartilhado.")
    p.add_argument("files", nargs="+", help="arquivos de texto (um por especialista)")
    p.add_argument("--out", default="out/vocab.json", help="onde salvar o vocabulário")
    args = p.parse_args()

    chars = set()
    for path in args.files:
        with open(path, "r", encoding="utf-8") as f:
            chars |= set(f.read())
        print(f"  {path}: vocabulário agora com {len(chars)} caracteres")

    tok = CharTokenizer(sorted(chars))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"\nVocabulário compartilhado ({tok.vocab_size} caracteres) salvo em: {args.out}")
    print("Treine cada especialista com:  python train.py --data <corpus> "
          f"--vocab {args.out} --out out/<nome>")


if __name__ == "__main__":
    main()
