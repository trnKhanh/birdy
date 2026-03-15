ifeq ($(OS), Windows_NT)
	DETECTED_OS := Windows
	SEP := \\
	SHELL := powershell.exe
	.SHELLFLAGS := -Command
else
	DETECTED_OS := $(shell uname -s)
	SEP := /
endif

DATA_DIR := .$(SEP)local$(SEP)data
ARCHIVE_DIR := .$(SEP)local$(SEP)archive
BIRDCLEF2026_DATA_DIR := $(DATA_DIR)$(SEP)birdclef2026

.PHONY: all
all: info install data-birdclef2026

.PHONY: info
info:
	@echo Detected OS : $(DETECTED_OS)

.PHONY: install
install: dependencies
	pip install -eq .

.PHONY: install-dev
install-dev: dependencies
	pip install -eq .[dev]

.PHONY: dependencies
dependencies:
	pip install kaggle

.PHONY: data-birdclef2026 
data-birdclef2026: dependencies
	-mkdir -p $(ARCHIVE_DIR)
	kaggle competitions download -c birdclef-2026 -p $(ARCHIVE_DIR)
	-mkdir -p $(BIRDCLEF2026_DATA_DIR)
	7z x $(ARCHIVE_DIR)$(SEP)birdclef-2026.zip -y -o$(BIRDCLEF2026_DATA_DIR) \
	|| tar -xf $(ARCHIVE_DIR)$(SEP)birdclef-2026.zip -C $(BIRDCLEF2026_DATA_DIR) 
	rm -f $(ARCHIVE_DIR)$(SEP)birdclef-2026.zip
