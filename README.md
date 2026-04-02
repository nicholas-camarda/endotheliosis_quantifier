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

# Check hardware capabilities
eq capabilities
```

## Step-by-Step Training Pipeline

### 1. Data Preparation

```bash
# Load and preprocess preeclampsia data
eq data-load --data-dir data/preeclampsia_data/train \
             --test-data-dir data/preeclampsia_data/test \
             --cache-dir data/preeclampsia_data/cache

# For mitochondria training, organize Lucchi dataset
python -m eq.data_management.organize_lucchi_dataset \
    --input-dir data/mitochondria_raw \
    --output-dir data/mitochondria_data
```

### 2. Mitochondria Model Training (Stage 1)

```bash
# Train mitochondria segmentation model from scratch
eq train-segmenter \
    --base-model-path models/base_mitochondria_model.pkl \
    --cache-dir data/mitochondria_data/cache \
    --output-dir models/mitochondria_training \
    --epochs 50 \
    --batch-size 16

# Or use the training module directly
python -m eq.training.train_mitochondria \
    --data-dir data/mitochondria_data/cache \
    --model-dir models/mitochondria_training \
    --epochs 50 \
    --batch-size 16
```

### 3. Glomeruli Model Training (Stage 2 - Transfer Learning)

```bash
# Train glomeruli model using mitochondria model as base
eq train-segmenter \
    --base-model-path models/mitochondria_training/mitochondria_model.pkl \
    --cache-dir data/preeclampsia_data/cache \
    --output-dir models/glomeruli_training \
    --epochs 30 \
    --batch-size 8

# Or use the training module directly
python -m eq.training.train_glomeruli \
    --data-dir data/preeclampsia_data/cache \
    --model-dir models/glomeruli_training \
    --base-model models/mitochondria_training/mitochondria_model.pkl \
    --epochs 30 \
    --batch-size 8
```

### 4. Model Evaluation

```bash
# Evaluate mitochondria model performance
python -m eq.evaluation.evaluate_mitochondria_model \
    --model-path models/mitochondria_training/mitochondria_model.pkl \
    --data-dir data/mitochondria_data/cache \
    --output-dir evaluation/mitochondria_results

# Evaluate glomeruli model performance
python -m eq.evaluation.evaluate_glomeruli_model \
    --model-path models/glomeruli_training/glomeruli_model.pkl \
    --data-dir data/preeclampsia_data/cache \
    --output-dir evaluation/glomeruli_results
```

### 5. Production Inference

```bash
# Run complete production pipeline
eq production --data-dir data/preeclampsia_data/train \
             --test-data-dir data/preeclampsia_data/test \
             --segmentation-model models/glomeruli_training/glomeruli_model.pkl

# Or run individual inference steps
python -m eq.inference.run_mitochondria_prediction \
    --model-path models/mitochondria_training/mitochondria_model.pkl \
    --data-path data/mitochondria_data/test \
    --output-dir predictions/mitochondria

python -m eq.inference.run_glomeruli_prediction \
    --model-path models/glomeruli_training/glomeruli_model.pkl \
    --data-path data/preeclampsia_data/test \
    --output-dir predictions/glomeruli
```

### 6. Quick Testing Mode

```bash
# Enable quick testing for faster development cycles
QUICK_TEST=true eq train-segmenter \
    --base-model-path models/base_mitochondria_model.pkl \
    --cache-dir data/mitochondria_data/cache \
    --output-dir models/mitochondria_training \
    --epochs 5 \
    --batch-size 4
```

## Training Data Requirements

### Mitochondria Training Data
- **Source**: [Lucchi et al. 2012 benchmark dataset](http://rhoana.rc.fas.harvard.edu/dataset/lucchi.zip)
- **Format**: TIF files with ground truth masks
- **Organization**: Use `eq.data_management.organize_lucchi_dataset` to structure data
- **Expected Structure**:
  ```
  data/mitochondria_data/
  ├── training/
  │   ├── images/          # Training images (.tif)
  │   └── masks/           # Ground truth masks (.tif)
  └── testing/
      ├── images/          # Test images (.tif)
      └── masks/           # Ground truth masks (.tif)
  ```

### Glomeruli Training Data
- **Source**: Preeclampsia H&E histology images with annotations
- **Format**: TIF images with Label Studio JSON annotations
- **Expected Structure**:
  ```
  data/preeclampsia_data/
  ├── Lauren_PreEclampsia_Raw_TIF_Images/  # Raw TIF images
  ├── annotations.json                      # Label Studio export
  └── cache/                               # Processed data cache
  ```

## Configuration

The package uses YAML configuration files for all settings:

- `configs/mito_pretraining_config.yaml` - Mitochondria model configuration
- `configs/glomeruli_finetuning_config.yaml` - Glomeruli model configuration

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
- **Key Entry Points**: CLI `eq` dispatches to `data-load`, `train-segmenter`, `production`, `mode`, and `capabilities`
- **Hardware**: Mode-aware execution with MPS/CUDA detection and sensible batch sizes via `eq mode`

## Troubleshooting

### Common Issues

**Import Errors**: If you encounter import errors, ensure the `eq` environment is activated:
```bash
mamba activate eq
```

**CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing:
```bash
eq train-segmenter --batch-size 4  # Reduce from default 16
```

**MPS Fallback Issues**: On Apple Silicon, ensure PyTorch MPS is available:
```bash
eq mode --set development  # Uses CPU fallback
```

**Data Loading Errors**: Verify data structure matches expected format:
```bash
# Check data organization
ls -la data/mitochondria_data/training/
ls -la data/preeclampsia_data/Lauren_PreEclampsia_Raw_TIF_Images/
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
