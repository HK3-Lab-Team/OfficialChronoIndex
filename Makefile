PYTHON ?= python3
CONFIG ?= datasets.toml
DATASET ?= zhao
OUT ?= artifacts/unified_cgm.parquet
UNIFIED_TYPE ?= cgm

.PHONY: help dry-run list load-all load-one save-unified

help:
	@echo "Targets:"
	@echo "  make dry-run                      # validate config and resolved paths"
	@echo "  make list                         # alias of dry-run"
	@echo "  make load-all                     # load every enabled dataset"
	@echo "  make load-one DATASET=zhao        # load one dataset by name"
	@echo "  make save-unified OUT=path.parquet UNIFIED_TYPE=cgm|clinical|food"
	@echo ""
	@echo "Variables:"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  DATASET=$(DATASET)"
	@echo "  OUT=$(OUT)"
	@echo "  UNIFIED_TYPE=$(UNIFIED_TYPE)"

dry-run:
	$(PYTHON) -m chronoindex.cli.load_all_datasets --config $(CONFIG) --dry-run

list: dry-run

load-all:
	$(PYTHON) -m chronoindex.cli.load_all_datasets --config $(CONFIG)

load-one:
	$(PYTHON) -m chronoindex.cli.load_all_datasets --config $(CONFIG) --only $(DATASET)

save-unified:
	$(PYTHON) -m chronoindex.cli.load_all_datasets --config $(CONFIG) --save-unified $(OUT) --unified-type $(UNIFIED_TYPE)
