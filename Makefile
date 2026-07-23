# Makefile — atalhos para o minimum-llm-training
# Rode `make` (ou `make help`) para ver os comandos disponíveis.

# Interpretador: usa o venv se existir; senão, o python do sistema.
VENV   := .venv
PYTHON := $(shell [ -x $(VENV)/bin/python ] && echo $(VENV)/bin/python || echo python3)

# Variáveis sobrescrevíveis na linha de comando, ex.:
#   make train ITERS=5000 N_LAYER=6
#   make sample PROMPT="int main()" TEMPERATURE=0.7
DATA        ?= data/cpp.txt
ITERS       ?= 2000
N_LAYER     ?= 4
N_HEAD      ?= 4
N_EMBD      ?= 128
BLOCK_SIZE  ?= 128
BATCH_SIZE  ?= 32
LR          ?= 3e-4
AMP         ?= 0
PROMPT      ?= int main()
MAX_TOKENS  ?= 300
TEMPERATURE ?= 0.8
TOP_K       ?=
CPP_SRC     ?= data/cpp_src
HOST        ?= 127.0.0.1
PORT        ?= 8000
PRESET      ?= tiny
EXPERTS     ?= out
MODE        ?= route

# Flags opcionais: só entram na linha de comando quando definidas.
TOPK_FLAG    := $(if $(TOP_K),--top_k $(TOP_K),)
AMP_FLAG     := $(if $(filter 1,$(AMP)),--amp,)
# Passa --experts ao chat só quando EXPERTS difere do padrão "out".
EXPERTS_FLAG := $(if $(filter-out out,$(EXPERTS)),--experts $(EXPERTS),)

.DEFAULT_GOAL := help
.PHONY: help setup install train micro train-more sample chat moe data cpp clean clean-all

help: ## Mostra esta ajuda
	@echo "minimum-llm-training — comandos disponíveis:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Exemplos:"
	@echo "  make setup                            # cria venv e instala o PyTorch"
	@echo "  make cpp CPP_SRC=~/meus_projetos      # monta o corpus com seu codigo"
	@echo "  make train ITERS=5000 AMP=1           # treina na GPU com mixed precision"
	@echo "  make sample PROMPT='int main()'       # completa a partir de um trecho"

setup: ## Cria o venv e instala o PyTorch (versão CPU)
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	@echo "Pronto. Agora rode: make train"

install: setup ## Alias de 'setup'

train: ## Treina o modelo (use ITERS=, N_LAYER=, DATA=, AMP=1 p/ GPU)
	$(PYTHON) train.py \
		--data $(DATA) --iters $(ITERS) \
		--n_layer $(N_LAYER) --n_head $(N_HEAD) --n_embd $(N_EMBD) \
		--block_size $(BLOCK_SIZE) --batch_size $(BATCH_SIZE) --lr $(LR) $(AMP_FLAG)

micro: ## Micro-treino rápido do zero (use PRESET=tiny|fast|balanced|quality)
	$(PYTHON) train.py --data $(DATA) --preset $(PRESET)

train-more: ## Continua o treino do checkpoint, somando +ITERS passos (cumulativo)
	$(PYTHON) train.py --data $(DATA) --resume --iters $(ITERS) $(AMP_FLAG)

sample: ## Gera texto (use PROMPT=, MAX_TOKENS=, TEMPERATURE=, TOP_K=)
	$(PYTHON) sample.py \
		--prompt "$(PROMPT)" --max_new_tokens $(MAX_TOKENS) \
		--temperature $(TEMPERATURE) $(TOPK_FLAG)

chat: ## Abre o chat web estilo GPT (use PORT=, EXPERTS="out/a out/b" p/ MoE)
	$(PYTHON) chat.py --host $(HOST) --port $(PORT) $(EXPERTS_FLAG) \
		--temperature $(TEMPERATURE) --max_new_tokens $(MAX_TOKENS) $(TOPK_FLAG)

moe: ## Junta especialistas (use EXPERTS="out/a out/b", MODE=route|blend, PROMPT=)
	$(PYTHON) moe.py --experts $(EXPERTS) --mode $(MODE) \
		--prompt "$(PROMPT)" --max_new_tokens $(MAX_TOKENS) \
		--temperature $(TEMPERATURE) $(TOPK_FLAG)

cpp: ## Monta data/cpp.txt a partir dos seus arquivos (use CPP_SRC=<pasta>)
	$(PYTHON) prepare_cpp.py --src $(CPP_SRC) --out $(DATA)

data: ## Baixa um corpus maior (tiny shakespeare) para data/
	$(PYTHON) download_data.py

clean: ## Remove checkpoints e caches (out/, __pycache__)
	rm -rf out __pycache__ */__pycache__

clean-all: clean ## Remove também o venv
	rm -rf $(VENV)
