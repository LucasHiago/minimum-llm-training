"""
download_data.py — Baixa um corpus maior para treinos mais interessantes
========================================================================

O texto de exemplo em data/input.txt é pequeno (bom para testar em segundos).
Para o modelo aprender de verdade, use mais dados. Este script baixa o
"tiny shakespeare" (~1 MB de texto em inglês), o dataset clássico para
demonstrações de modelos de caractere.

    python download_data.py                 # salva em data/shakespeare.txt
    python train.py --data data/shakespeare.txt --iters 5000

Prefere português? Qualquer .txt grande serve: junte livros de domínio público
(ex.: Machado de Assis no site do Domínio Público / Projeto Gutenberg) num
único arquivo e aponte --data para ele.
"""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = "data/shakespeare.txt"


def main():
    os.makedirs("data", exist_ok=True)
    print(f"Baixando {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    size = os.path.getsize(DEST)
    print(f"Salvo em {DEST} ({size:,} bytes)")
    print("Treine com:  python train.py --data data/shakespeare.txt --iters 5000")


if __name__ == "__main__":
    main()
