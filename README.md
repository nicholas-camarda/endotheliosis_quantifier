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
git clone https://github.com/nicholas-camarda/endotheliosis_quantifier.git
cd endotheliosis_quantifier

# Create and activate conda environment (use mamba)
mamba env create -f environment.yml
mamba activate eq

# Install in development mode (enables CLI `eq`)
pip install -e .[dev]
```

## Quick Start

```bash
# Show hardware mode and suggestions
eq mode --show

# Check hardware capabilities
eq capabilities

# Process raw data into ready-to-analyze format
eq process-data --input-dir data/raw_images --output-dir derived_data
```

## Production Run: End‑to‑End (Data → Mito Train → Glom Transfer)

The commands below are the canonical, production‑mode workflow using FastAI v2 and our standardized directories. All outputs are PNG, masks are binary, and models/plots are written under `models/segmentation/`.

```bash
# 0) Activate environment
mamba activate eq

# 1) Data processing (raw → derived_data/ with 256×256 patches, binary PNG masks)
#    Replace RAW_DIR with your project root containing images and (optionally) masks
RAW_DIR=raw_data/preeclampsia_project
eq process-data \
  --input-dir "$RAW_DIR" \
  --output-dir derived_data/preeclampsia

# After this step you should have:
# derived_data/preeclampsia/
#   ├─ image_patches/
#   └─ mask_patches/

# 2) Train mitochondria segmentation model (from scratch)
#    Saves parameterized filename and standard plots in models/segmentation/mitochondria
python -m eq.training.train_mitochondria \
  --data-dir derived_data/mito \
  --model-dir models/segmentation/mitochondria \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --image-size 256

# Expected artifacts (examples):
# models/segmentation/mitochondria/
#   mito_dynamic_unet_seg_model- pretrain_e50_b16_lr1e-03_sz256.pkl
#   training_loss.png
#   validation_predictions.png

# 3) Train glomeruli model (transfer learning from mitochondria)
#    If you have a specific mito checkpoint, set BASE below; otherwise glob the latest.
BASE=$(ls models/segmentation/mitochondria/*.pkl | head -n1)
python -m eq.training.train_glomeruli \
  --data-dir derived_data/preeclampsia \
  --model-dir models/segmentation/glomeruli \
  --base-model "$BASE" \
  --epochs 30 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --image-size 256

# Expected artifacts (examples):
# models/segmentation/glomeruli/
#   glomeruli_model- xfer_e30_b8_lr1e-04_sz256.pkl
#   glomeruli_model_validation_predictions.png
#   glomeruli_model_training_loss.png
```

Notes
- Use `--config configs/mito_pretraining_config.yaml` or `configs/glomeruli_finetuning_config.yaml` to drive runs; CLI flags override YAML; both fall back to `eq.core.constants` defaults.
- All experimental or ad‑hoc outputs should go to `test_output/` (not `models/segmentation/`).

### 🚧 Planned (Not Yet Implemented):

#### CLI Commands
- `eq train-segmenter` - Unified training command
- `eq data-load` - Legacy data loading (will be deprecated)
- `eq production` - End-to-end production pipeline
- `eq extract-features` - Feature extraction
- `eq quantify` - Endotheliosis quantification

#### Pipeline Components
- Model evaluation and metrics
- Inference pipeline
- Production deployment
- Feature extraction from trained models
- Endotheliosis scoring and quantification

### 🔄 FastAI v2 Migration Status

#### Current Status:
- ✅ **Data Processing**: Complete - `eq process-data` works with FastAI v2
- ✅ **Data Pipeline**: Complete - DataBlock approach implemented  
- ✅ **Training Modules**: Complete - CLI interfaces just added
- ⏳ **Inference Pipeline**: Pending - Will be implemented after training validation
- ⏳ **Evaluation Pipeline**: Pending - Will be implemented after training validation

## Data Requirements

### Input Data Format
- **Images**: TIF, PNG, or JPEG files
- **Masks**: Optional - PNG files with binary masks (0/255 values)
- **Structure**: Any directory structure - `eq process-data` auto-detects images and masks

### Expected Output from `eq process-data`
```
derived_data/
├── image_patches/          # 256x256 image patches
├── mask_patches/           # 256x256 mask patches (if masks detected)
├── cache/                  # Processed data cache
└── metadata.json          # Processing statistics
```

## Configuration

Configuration is handled through:
- **Constants**: `eq.core.constants` - Default values for patch size, batch size, etc.
- **Environment**: `eq mode` - Hardware-aware configuration
- **CLI Arguments**: All training parameters configurable via command line

## Project Structure

```
src/eq/
├── core/              # Canonical loaders, constants (e.g., BINARY_P2C, threshold=127)
├── data_management/   # Data loading, caching, and organization
├── processing/        # Image conversion and patchification
├── training/          # Training scripts for mitochondria and glomeruli models
├── inference/         # Inference and prediction scripts
├── evaluation/        # Metrics and evaluators
├── pipeline/          # Pipeline orchestration and production runners
├── models/            # Model architecture definitions
├── quantification/    # Quantification workflows (in progress)
└── utils/             # Config, logging, hardware detection (MPS/CUDA)
```

## Architecture Overview

- **Data Flow**: raw images → patchification → binary masks (threshold 127) → segmentation → features → quantification
- **Core Canon**: `eq.core` provides `BINARY_P2C=[0,1]`, mask conversion helpers, and canonical data loaders used across pipelines
- **Training Stages**: 
  - **Stage 1**: Mitochondria pretraining on EM data (Lucchi et al. 2012) for general segmentation features
  - **Stage 2**: Glomeruli fine-tuning using mitochondria model as base for transfer learning
- **Performance Targets**: 
  - Mitochondria: >70% validation accuracy
  - Glomeruli: >70% validation accuracy (transfer learning from mitochondria)
- **Key Entry Points**: CLI `eq` dispatches to `process-data`, `train-segmenter`, `production`, `mode`, and `capabilities`
- **Hardware**: Mode-aware execution with MPS/CUDA detection and sensible batch sizes via `eq mode`

### FastAI v2 Migration Status

- ✅ **Data Processing**: Complete - `eq process-data` with auto mask detection and DataBlock approach
- ✅ **Data Pipeline**: Complete - Unified `patchify_dataset` with proper binary mask handling
- 🔄 **Training Modules**: In Progress - FastAI v2 APIs being updated (DataBlock, unet_learner with n_out)
- ⏳ **Inference Pipeline**: Pending
- ⏳ **Evaluation Pipeline**: Pending

## Troubleshooting

### Common Issues

**Environment Setup**: Always activate the correct environment:
```bash
mamba activate eq
pip install -e .  # Install in development mode
```

**"Not yet implemented" Messages**: Many CLI commands exist but aren't implemented yet:
```bash
# These will show "not yet implemented" messages:
eq train-segmenter    # Use direct module instead
eq data-load         # Use eq process-data instead
eq production        # Not implemented yet
eq extract-features  # Not implemented yet
eq quantify         # Not implemented yet

# Use these working alternatives:
eq process-data --input-dir data/raw_images --output-dir derived_data
python -m eq.training.train_mitochondria --data-dir derived_data/cache --model-dir models/output
```

**Training Data Issues**: Ensure you have the right data structure:
```bash
# Check derived_data structure
ls -la derived_data/
cat derived_data/metadata.json

# For mitochondria training, you need pickle files in cache/
ls -la derived_data/cache/
# Should have: train_images.pickle, train_masks.pickle, val_images.pickle, val_masks.pickle
```

**CUDA Out of Memory**: Reduce batch size:
```bash
python -m eq.training.train_mitochondria --batch-size 4  # Reduce from default 16
```

**Hardware Issues**: Check your hardware configuration:
```bash
eq mode --show
eq capabilities
```

### Performance Optimization

- **Development Mode**: `eq mode --set development` for faster iteration
- **Production Mode**: `eq mode --set production` for maximum performance
- **Quick Testing**: `QUICK_TEST=true` for faster training cycles

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
