PY ?= python

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make train DATA=... [DATASET=ntu] [ARGS=\"--extra\"]"
	@echo "  make eval CHECKPOINT=... DATA=... [DATASET=ntu] [ARGS=\"--extra\"]"
	@echo "  make describe        # Print Study 3 summary"

.PHONY: train
train:
	@if [ -z "$(DATA)" ]; then echo "Usage: make train DATA=path/to/paired.pt [DATASET=ntu] [ARGS=...]"; exit 1; fi
	$(PY) tmr.py train --data_path $(DATA) --dataset $(if $(DATASET),$(DATASET),ntu) $(ARGS)

.PHONY: eval
eval:
	@if [ -z "$(CHECKPOINT)" ] || [ -z "$(DATA)" ]; then echo "Usage: make eval CHECKPOINT=ckpt.pth DATA=path/to/paired.pt [DATASET=ntu] [ARGS=...]"; exit 1; fi
	$(PY) tmr.py eval --checkpoint $(CHECKPOINT) --data_path $(DATA) --dataset $(if $(DATASET),$(DATASET),ntu) $(ARGS)

.PHONY: describe
describe:
	$(PY) tmr.py describe
