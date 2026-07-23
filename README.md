# minigpt — uma LLM mínima e didática

Um GPT (Transformer decoder-only) de **nível de caractere**, escrito do zero em
PyTorch, para você **entender e treinar** um modelo de linguagem no seu próprio
computador — inclusive **só na CPU**.

É a mesma arquitetura do GPT, encolhida ao essencial: ~200 linhas de modelo,
comentadas em português. Inspirado no [nanoGPT](https://github.com/karpathy/nanoGPT)
do Andrej Karpathy, mas ainda mais enxuto e didático.

## O que tem aqui

| Arquivo             | O quê                                                      |
|---------------------|-----------------------------------------------------------|
| `model.py`          | O Transformer: atenção causal, MLP, blocos, GPT           |
| `tokenizer.py`      | Tokenizador de caractere (char → id → char)               |
| `train.py`          | Loop de treino (lê texto, treina, salva checkpoint)       |
| `sample.py`         | Gera texto a partir de um modelo treinado                 |
| `data/input.txt`    | Corpus de exemplo (fábulas em português)                  |
| `download_data.py`  | Baixa um corpus maior (tiny shakespeare)                  |

## Instalação

```bash
# (opcional) ambiente virtual
python -m venv .venv && source .venv/bin/activate

# PyTorch versão CPU (leve). Para GPU, veja pytorch.org.
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Uso em 2 passos

**1. Treinar** (o exemplo padrão roda em ~1–2 min na CPU):

```bash
python train.py
```

Você verá a *loss* caindo:

```
Dispositivo: cpu
Corpus: 4,812 caracteres
Vocabulário: 61 caracteres únicos
Parâmetros do modelo: 812,861
passo     1 | treino 4.2011 | val 4.2039
passo   200 | treino 2.4518 | val 2.4602
passo  2000 | treino 1.4832 | val 1.6201
```

**2. Gerar texto**:

```bash
python sample.py --prompt "A raposa"
```

## Como funciona (visão de 1 minuto)

1. **Tokenizar**: cada caractere do texto vira um número inteiro.
2. **Prever o próximo**: o modelo recebe uma janela de caracteres e aprende a
   prever qual vem em seguida. É só isso — repetido bilhões de vezes nos modelos
   grandes, alguns milhares aqui.
3. **Atenção**: cada posição "olha" para as anteriores para decidir o próximo
   caractere. A máscara *causal* impede olhar para o futuro.
4. **Gerar**: dado um começo (*prompt*), o modelo prevê um caractere, anexa,
   e repete — texto autoregressivo.

## Treinos leves: o que ajustar

Tudo é configurável pela linha de comando. Para deixar **mais rápido/leve**,
diminua os números; para **melhor qualidade**, aumente (e treine mais tempo):

```bash
python train.py \
  --iters 3000 \        # mais passos = aprende mais (e demora mais)
  --n_layer 4 \         # nº de blocos Transformer (profundidade)
  --n_head 4 \          # nº de cabeças de atenção
  --n_embd 128 \        # tamanho dos embeddings (largura)
  --block_size 128 \    # tamanho da janela de contexto
  --batch_size 32 \     # exemplos por passo
  --lr 3e-4             # taxa de aprendizado
```

Dicas:
- **Só quero ver funcionando rápido**: `--iters 500 --n_layer 2 --n_embd 64`.
- **Quero um texto mais convincente**: baixe mais dados
  (`python download_data.py`) e treine com `--iters 5000` ou mais.
- **Tenho GPU NVIDIA**: instale o PyTorch com CUDA; o código usa a GPU
  automaticamente (ou force com `--device cuda`).

## Treinar com o SEU texto

Basta apontar para qualquer arquivo `.txt` em UTF-8:

```bash
python train.py --data data/meu_texto.txt --iters 4000
python sample.py --prompt "Era uma vez"
```

Quanto maior e mais consistente o corpus, melhor o resultado. Junte vários
textos num único arquivo se precisar.

## Controlando a geração

```bash
python sample.py --prompt "O leão" --temperature 0.7 --top_k 40 --max_new_tokens 500
```

- `--temperature`: `< 1.0` mais previsível/repetitivo; `> 1.0` mais criativo/caótico.
- `--top_k`: amostra só entre os K caracteres mais prováveis (reduz besteira).
- `--max_new_tokens`: quantos caracteres gerar.

## Limitações (de propósito)

Isto é material **didático**, não um modelo de produção. É nível de caractere
(não subpalavra/BPE), sem otimizações de performance, sem treino distribuído.
O objetivo é caber na cabeça — depois que entender isto, o
[nanoGPT](https://github.com/karpathy/nanoGPT) e a série "Zero to Hero" do
Karpathy são os próximos passos naturais.

## Licença

MIT — use, modifique e compartilhe à vontade.
