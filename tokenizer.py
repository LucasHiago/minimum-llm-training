"""
tokenizer.py — Tokenizador de caractere (o mais simples possível)
=================================================================

LLMs grandes usam tokenizadores de subpalavra (BPE). Aqui, para ser didático,
cada CARACTERE vira um token. Vantagens: vocabulário minúsculo, zero
dependências, fácil de entender. Desvantagem: sequências mais longas.

    "olá" -> [39, 46, 27]   (codificar / encode)
    [39, 46, 27] -> "olá"   (decodificar / decode)
"""

import json


class CharTokenizer:
    def __init__(self, chars):
        # `chars`: lista ordenada de caracteres únicos que formam o vocabulário.
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # char -> id
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # id -> char

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str):
        """Constrói o vocabulário a partir de todos os caracteres do texto."""
        return cls(sorted(set(text)))

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chars, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))
