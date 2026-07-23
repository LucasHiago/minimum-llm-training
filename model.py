"""
model.py — Um GPT mínimo e didático (nível de caractere)
=========================================================

Este arquivo implementa um Transformer decoder-only (a mesma família do GPT)
do zero, usando só PyTorch. O objetivo é ser LEGÍVEL, não rápido.

Fluxo de uma passagem (forward):

    tokens (inteiros) ->  embeddings  ->  N blocos Transformer  ->  logits

Cada bloco Transformer tem duas partes:
    1. Self-attention causal (cada posição "olha" só para o passado)
    2. MLP (rede feed-forward que processa cada posição)

Com conexões residuais e LayerNorm em volta de cada parte.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Hiperparâmetros do modelo. Valores pequenos = treino leve na CPU."""
    vocab_size: int = 65      # nº de tokens distintos (definido pelos dados)
    block_size: int = 128     # comprimento máximo de contexto (janela)
    n_layer: int = 4          # nº de blocos Transformer empilhados
    n_head: int = 4           # nº de cabeças de atenção
    n_embd: int = 128         # dimensão dos embeddings
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Self-attention multi-cabeça com máscara causal.

    "Causal" = a posição t só pode atender às posições <= t.
    Isso força o modelo a prever o próximo token sem ver o futuro.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Projeção única que gera query, key e value de uma vez (3x n_embd).
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Máscara triangular inferior (1 onde é permitido atender).
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, tempo (tokens), canais (n_embd)

        # Gera q, k, v e separa em cabeças.
        q, k, v = self.attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Scores de atenção: quão relevante cada token é para os outros.
        att = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
        # Aplica a máscara causal: futuro vira -inf -> peso 0 após softmax.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Média ponderada dos values.
        y = att @ v                                  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # junta as cabeças
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Rede feed-forward aplicada em cada posição independentemente."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Um bloco Transformer: atenção + MLP, ambos com residual + LayerNorm."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # x = x + sub-camada(norm(x))  -> conexão residual (pre-norm)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """O modelo completo."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)   # o que é o token
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)   # onde ele está
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding de entrada e camada de saída compartilham pesos.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "sequência maior que block_size"

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # soma "o que" + "onde"
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy entre a previsão de cada posição e o próximo token real.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Gera texto autoregressivamente, um token por vez."""
        self.eval()
        for _ in range(max_new_tokens):
            # Corta o contexto para no máximo block_size tokens.
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # só a última posição

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # amostra
            idx = torch.cat((idx, next_id), dim=1)
        return idx
