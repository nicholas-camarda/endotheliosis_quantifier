# Roadmap

## Phase 0: Already Completed
- [x] End-to-end scripts for data prep → segmentation → feature extraction → quantification
- [x] Dataset and feature caching pipeline
- [x] Baseline quantification via Random Forest with validation outputs

## Phase 1: Current Development
- [x] Rename repository to `endotheliosis_quantifier`
- [x] Standardize Conda env as `eq`
- [x] Adopt `src/eq` package layout; add CLI entrypoints (`eq data-load`, `eq train-segmenter`, `eq extract-features`, `eq quantify`, `eq pipeline`)
- [x] Centralize configuration (data paths, cache, model paths) in a config module
- [x] Consolidate scripts: keep a single ROI extractor, feature extractor, and quantifier
- [x] Consolidate `scripts/utils/` and remove/merge redundant utilities
- [x] Update `.gitignore` to stop tracking `notebooks/` and `earlier_models/`
- [x] **COMPLETED**: Migrate to FastAI (PyTorch/MPS/CUDA) for training/inference - TensorFlow completely removed
- [x] **COMPLETED**: Implement comprehensive fastai segmentation system with U-Net support and mode-specific training
- [x] **COMPLETED**: Add comprehensive test coverage for fastai segmentation (46 tests passing)
- [x] **COMPLETED**: Document complete segmentation pipeline workflow (mitochondria pretraining → glomeruli fine-tuning)
- [x] **COMPLETED**: Create configuration system for both pipeline stages with YAML configs
- [x] **COMPLETED**: Implement pipeline orchestrator script for automated execution
- [x] Reproducible segmentation training (configs, metrics, checkpoints)
- [x] **Complete mitochondria workflow documentation with data source attribution**
- [ ] Add a small sample dataset and smoke tests for each pipeline step

## Phase 2: Near-term Enhancements
- [ ] Package release prep (README, examples, versioning)
- [ ] Batch inference CLI for researcher datasets
- [ ] Resolve segmentation_models import issues for full CLI functionality

## Phase 3: Longer-term
- [ ] Optional WSL2 + CUDA (RTX 3080) training flow documented and tested
- [ ] Publish model checkpoints and versioned artifacts
- [ ] Optional demo notebooks with minimal setup
