SHELL := /bin/bash
PYTHON := python3
PIP := pip
CSV := 811122_d557e5f46a5e019415b76d52f6d12c72294c38af.csv
PIKACHU_IMAGE := cards/IMG_20260430_103425.jpg
GLOOM_IMAGE := cards/IMG_20260430_104025.jpg
QUALITY ?= M
CARD_LANGUAGE ?= PT
QUANTITY ?= 1
PREPROCESS_INPUT_DIR ?= cards
PREPROCESS_OUTPUT_DIR ?= cards-treated
PREPROCESS_CONTRAST ?= 35
OCR_BACKEND ?= easyocr
WORKERS ?= 1

.PHONY: help install test lint run run-pikachu run-gloom write batch preprocess sync

help:
	@echo "Targets disponiveis:"
	@echo "  make install        - instala dependencias"
	@echo "  make test           - roda todos os testes"
	@echo "  make lint           - roda linter (ruff)"
	@echo "  make run IMAGE=...  - extrai dados de uma imagem"
	@echo "  make run-pikachu    - extrai dados do Pikachu"
	@echo "  make run-gloom      - extrai dados do Gloom"
	@echo "  make write IMAGE=.. - extrai e grava no CSV"
	@echo "  make batch IMAGES_FILE=images.txt - processa lote (WORKERS=4)"
	@echo "  make preprocess     - gera PNGs tratados"
	@echo "  make sync QUERY=... - sincroniza reference_cards.json"

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

run:
	@if [[ -z '$(IMAGE)' ]]; then echo "uso: make run IMAGE=arquivo.jpg"; exit 1; fi
	$(PYTHON) -m src.cli extract --image "$(IMAGE)" --csv "$(CSV)" --quality "$(QUALITY)" --language "$(CARD_LANGUAGE)" --quantity "$(QUANTITY)" --ocr-backend "$(OCR_BACKEND)"

run-pikachu:
	$(PYTHON) -m src.cli extract --image "$(PIKACHU_IMAGE)" --csv "$(CSV)" --quality "$(QUALITY)" --language "$(CARD_LANGUAGE)" --quantity "$(QUANTITY)" --ocr-backend "$(OCR_BACKEND)"

run-gloom:
	$(PYTHON) -m src.cli extract --image "$(GLOOM_IMAGE)" --csv "$(CSV)" --quality "$(QUALITY)" --language "$(CARD_LANGUAGE)" --quantity "$(QUANTITY)" --ocr-backend "$(OCR_BACKEND)"

write:
	@if [[ -z '$(IMAGE)' ]]; then echo "uso: make write IMAGE=arquivo.jpg"; exit 1; fi
	$(PYTHON) -m src.cli extract --image "$(IMAGE)" --csv "$(CSV)" --quality "$(QUALITY)" --language "$(CARD_LANGUAGE)" --quantity "$(QUANTITY)" --write --ocr-backend "$(OCR_BACKEND)"

batch:
	@if [[ -z '$(IMAGES_FILE)' ]]; then echo "uso: make batch IMAGES_FILE=images.txt"; exit 1; fi
	$(PYTHON) -m src.cli batch --images-file "$(IMAGES_FILE)" --csv "$(CSV)" --quality "$(QUALITY)" --language "$(CARD_LANGUAGE)" --quantity "$(QUANTITY)" --write --error-log "batch-errors.jsonl" --ocr-backend "$(OCR_BACKEND)" --workers "$(WORKERS)"

preprocess:
	$(PYTHON) -m src.cli preprocess --input-dir "$(PREPROCESS_INPUT_DIR)" --output-dir "$(PREPROCESS_OUTPUT_DIR)" --contrast "$(PREPROCESS_CONTRAST)"

sync:
	@if [[ -z '$(QUERY)' ]]; then echo "uso: make sync QUERY='set.name:\"Ascended Heroes\"'"; exit 1; fi
	$(PYTHON) -m src.cli sync --source pokemontcg --query "$(QUERY)" --edition-ptbr "$(EDITION_PTBR)" --edition-sigla "$(EDITION_SIGLA)" --out "reference/reference_cards.json" --overrides "reference/reference_overrides.json"
