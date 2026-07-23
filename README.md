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
| `data/cpp.txt`      | Corpus de exemplo **em C++** (usado por padrão)           |
| `prepare_cpp.py`    | Monta um corpus de C++ a partir dos seus arquivos         |
| `download_data.py`  | Baixa um corpus maior de exemplo (tiny shakespeare)       |
| `chat.py`           | Interface web estilo GPT (streaming) para o modelo treinado |

> **Foco em C++.** Por padrão o modelo treina em `data/cpp.txt` e aprende a
> **gerar/completar código C++**. Ele completa trechos no estilo do corpus —
> não é um assistente que "responde perguntas" nem garante código correto
> (isso exigiria fine-tuning de um modelo pré-treinado bem maior).

## Atalhos com Make (opcional)

Se você tem `make`, dá para pular os comandos longos:

```bash
make setup                          # cria o venv e instala o PyTorch
make train                          # treina no corpus de C++ (data/cpp.txt)
make sample PROMPT="int main()"     # completa código C++
make train AMP=1                    # treina na GPU com mixed precision
make help                           # lista todos os comandos
```

Tudo é sobrescrevível: `make train ITERS=5000 N_LAYER=6 AMP=1`,
`make sample PROMPT="int main()" TEMPERATURE=0.7 TOP_K=40`. Sem `make`? Use os
comandos `python ...` das seções abaixo — o resultado é o mesmo.

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
Corpus: 6,283 caracteres
Vocabulário: 77 caracteres únicos
Parâmetros do modelo: 812,861
passo     1 | treino 4.2429 | val 4.2118
passo   200 | treino 2.4518 | val 2.4602
passo  2000 | treino 1.4832 | val 1.6201
```

**2. Gerar código**:

```bash
python sample.py --prompt "int main()"
```

## Micro-treinos e treino cumulativo (mais rápido)

O treino padrão (`balanced`) leva ~10 min só na CPU. Para um ciclo de
experimentação rápido, use um **preset** menor — o `tiny` treina em ~40 s:

```bash
python train.py --preset tiny        # ~40s na CPU (modelo minúsculo)
python train.py --preset fast        # meio-termo
make micro                           # (com make) atalho para o preset tiny
```

Presets disponíveis: `tiny`, `fast`, `balanced` (padrão), `quality` (só com GPU).
Qualquer hiperparâmetro segue sobrescrevível: `python train.py --preset tiny --iters 2000`.

**Treino cumulativo (`--resume`).** Você não precisa treinar tudo de uma vez:
treine um pouco, pare, e **continue de onde parou** depois. O treino é
acumulativo — pesos *e* o estado do otimizador (o "momentum" do AdamW) são
salvos e restaurados, então 500 + 500 passos ≈ 1000 passos de uma vez:

```bash
python train.py --preset tiny --iters 500   # 1º pedaço  (total: 500)
python train.py --resume --iters 500        # +500 passos (total: 1000)
python train.py --resume --iters 500        # +500 passos (total: 1500)
make train-more ITERS=500                    # (com make) o mesmo, cumulativo
```

Cada run mostra **passos/s** e **ETA** para você saber quanto falta. Duas
ressalvas do char-level: a arquitetura e o **vocabulário são fixados no 1º
treino** — ao retomar, o modelo reusa o tokenizador salvo e ignora (com aviso)
caracteres novos que não existiam no corpus original.

## Chat web (estilo GPT)

Já tem um modelo treinado? Suba uma interface de chat no navegador — com
respostas em *streaming* (caractere por caractere), como o ChatGPT:

```bash
python chat.py            # abre em http://127.0.0.1:8000
make chat PORT=9000       # (com make) porta customizada
```

Usa **só a biblioteca padrão do Python** (nenhuma dependência além do PyTorch).
Enter envia, Shift+Enter quebra linha, Esc interrompe a geração; dá para ajustar
temperatura, tokens e top-k ao vivo.

> Lembre-se: por baixo o modelo **completa código C++**, não responde perguntas.
> O layout é de chat pela experiência — comece com um trecho como `int main()`.

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
- **Quero código mais convincente**: monte um corpus grande com o seu próprio
  código (`prepare_cpp.py`, abaixo) e treine com `--iters 5000` ou mais.

## Treinando em C++ (o padrão)

O corpus embutido (`data/cpp.txt`) é só uma amostra. Para o modelo gerar C++ de
verdade, ele precisa ver **muito** código. Aponte o `prepare_cpp.py` para os seus
projetos: ele varre a pasta, junta todos os `.cpp/.hpp/.h/...` num único arquivo
de treino (pulando `build/`, `.git/`, etc.):

```bash
python prepare_cpp.py --src ~/meus_projetos --out data/cpp.txt
python train.py --data data/cpp.txt --iters 5000 --amp
python sample.py --prompt "int main()"
```

Quanto maior e mais **consistente** o estilo do código, melhor o resultado — um
modelo pequeno aprende mal com muitos estilos misturados.

## Usando a GPU (NVIDIA)

Instale o PyTorch com CUDA (veja [pytorch.org](https://pytorch.org)); o código
detecta a GPU automaticamente (ou force com `--device cuda`). Para treinar mais
rápido e usando menos memória, ligue **mixed precision**:

```bash
python train.py --data data/cpp.txt --iters 8000 --amp \
  --n_layer 6 --n_embd 256 --block_size 256 --batch_size 64
```

Com GPU você pode subir bastante `n_layer`, `n_embd` e `block_size` — é aí que o
modelo começa a gerar C++ com estrutura de verdade. O `--amp` só tem efeito em
GPU (na CPU é ignorado com um aviso).

## Treinar com QUALQUER texto

Não é só C++: aponte `--data` para qualquer `.txt` em UTF-8 (código de outra
linguagem, prosa, etc.):

```bash
python train.py --data data/meu_corpus.txt --iters 4000
```

## Controlando a geração

```bash
python sample.py --prompt "int main()" --temperature 0.7 --top_k 40 --max_new_tokens 500
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
