.PHONY: all
all: 
	@echo $(shell ls -la)

.PHONY: install
install: dependencies
	pip install -e .

.PHONY: install-dev
install-dev: dependencies
	pip install -e .[dev]

.PHONY: dependencies
dependencies:
	pip install kaggle


DATA_DIR := ./local/data/
ARCHIVE_DIR := ./local/archive/
BIRDCLEF2026_DATA_DIR := $(DATA_DIR)/birdclef2026

.PHONY: data-birdclef2026 
data-birdclef2026: dependencies
	mkdir -p $(ARCHIVE_DIR)
	kaggle competitions download -c birdclef-2026 -p $(ARCHIVE_DIR)
	mkdir -p $(BIRDCLEF2026_DATA_DIR)
	7z x $(ARCHIVE_DIR)/birdclef-2026.zip -y -o$(BIRDCLEF2026_DATA_DIR) \
	|| tar -xf $(ARCHIVE_DIR)/birdclef-2026.zip -C $(BIRDCLEF2026_DATA_DIR) 
	rm -f $(ARCHIVE_DIR)/birdclef-2026.zip

