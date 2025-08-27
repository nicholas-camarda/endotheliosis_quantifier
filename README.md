# Endotheliosis Quantifier (EQ)

A comprehensive pipeline for quantifying endotheliosis in preeclampsia histology images.

## 🍎 Apple Silicon (MPS) Compatibility

Apple Silicon (MPS) is supported out of the box. The CLI auto‑detects hardware and enables MPS fallback where needed.

Key points:
- Automatic device selection and mode suggestions via `eq mode`.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set when appropriate.
- Clear logging for device/mode, with safe fallbacks.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd endotheliosis_quantifier

# Create and activate conda environment
conda env create -f environment.yml
conda activate eq

# Install in development mode (enables CLI `eq`)
pip install -e .[dev]
```

## Quick Start

```bash
# Show hardware mode and suggestions
eq mode --show

# Load and preprocess data (example)
eq data-load --data-dir data/preeclampsia_data/train \
             --test-data-dir data/preeclampsia_data/test \
             --cache-dir data/preeclampsia_data/cache

# Train glomeruli segmenter (QUICK_TEST example)
QUICK_TEST=true eq seg --data-dir data/preeclampsia_data/train

# Train endotheliosis quantification model (requires trained segmentation model)
eq quant-endo --data-dir data/preeclampsia_data/train \
              --segmentation-model path/to/segmentation_model.pkl \
              --epochs 50 --batch-size 8

# Run production inference with pre-trained models
eq production --data-dir data/preeclampsia_data/train \
             --test-data-dir data/preeclampsia_data/test
```

## Configuration

The package uses YAML configuration files for all settings:

- `configs/glomeruli_finetuning_config.yaml` - Glomeruli model configuration
- `configs/mito_pretraining_config.yaml` - Mitochondria model configuration

## Project Structure

```
src/eq/
├── core/           # Canonical loaders, constants (e.g., BINARY_P2C, threshold=127)
├── processing/     # Image conversion and patchification
├── evaluation/     # Metrics and evaluators
├── pipeline/       # Pipeline orchestration and production runners
├── quantification/ # Quantification workflows (in progress)
└── utils/          # Config, logging, hardware detection (MPS/CUDA)
```

## Architecture Overview

- Data flow: raw images → patchification → binary masks (threshold 127) → segmentation → features → quantification.
- Core canon: `eq.core` provides `BINARY_P2C=[0,1]`, mask conversion helpers, and canonical data loaders used across pipelines.
- Training stages: mitochondria pretraining → glomeruli fine‑tuning (see `configs/mito_pretraining_config.yaml` and `configs/glomeruli_finetuning_config.yaml`).
- Key entry points: CLI `eq` dispatches to `data-load`, `seg`, `quant-endo`, `production`, `mode`, and `capabilities`.
- Hardware: mode‑aware execution with MPS/CUDA detection and sensible batch sizes via `eq mode`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python -m pytest -q tests/` and ensure `ruff check .` passes
5. Submit a pull request

## Agent-OS Context

- Decisions Log: `.agent-os/product/decisions.md` (architecture, stack, and repository hygiene decisions)
- Standards: `.agent-os/standards/code-style/python-style.md` and related testing/tech-stack docs
- Product Specs: `.agent-os/specs/completed-specs/` for recent consolidation and migration summaries

## License

MIT (see LICENSE)
