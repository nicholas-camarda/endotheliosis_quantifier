# Endotheliosis Quantifier — Product Overview

## Vision
- Computer vision pipeline to automatically segment glomeruli and quantify glomerular endotheliosis on H&E slides.
- Transfer learning: initialize from mitochondria EM-trained features; fine-tune a U-Net for glomerular line structures.

## Target Users
- Kidney pathology and biomedical imaging researchers needing reproducible, automated endotheliosis quantification.

## Current State
- 4-step pipeline in `scripts/main/` (data prep → segmentation → feature extraction → quantification).
- **COMPLETED**: Full migration to fastai/PyTorch with native Apple Silicon support (MPS) and CUDA support for NVIDIA GPUs.
- **COMPLETED**: Comprehensive fastai segmentation system with U-Net support and mode-specific training strategies.
- **COMPLETED**: Dual-environment architecture with automatic hardware detection and optimization.
- Cross-platform support: macOS M1 for inference and development; Windows/WSL2 + CUDA for production training.

## CLI Usage

The package provides a command-line interface for running the endotheliosis quantification pipeline:

### Quick Start
```bash
# Install in development mode
pip install -e .

# Check hardware capabilities
eq capabilities

# Set training mode (development/production/auto)
eq mode --set development

# Run complete pipeline
eq pipeline --data-dir data/train --test-data-dir data/test --cache-dir data/cache --output-dir output --base-model-path models/base_model.h5

# Or run individual steps
eq data-load --data-dir data/train --test-data-dir data/test --cache-dir data/cache
eq train-segmenter --base-model-path models/base_model.h5 --cache-dir data/cache --output-dir output
eq extract-features --cache-dir data/cache --output-dir output --model-path output/glomerulus_segmentation/glomerulus_segmenter.h5
eq quantify --cache-dir data/cache --output-dir output --model-path output/glomerulus_segmentation/glomerulus_segmenter.h5
```

### Available Commands
- `eq capabilities` - Show hardware capabilities and recommendations
- `eq mode` - Inspect and manage environment mode (development/production/auto)
- `eq data-load` - Load and preprocess data
- `eq train-segmenter` - Train segmentation models (now using fastai)
- `eq extract-features` - Extract features from images using trained models
- `eq quantify` - Run endotheliosis quantification
- `eq pipeline` - Execute complete end-to-end pipeline

For detailed usage information, see the [Technical Documentation](technical/README.md#cli-interface).

## Start Here
- Roadmap: see `./roadmap.md`
- Tech Stack: see `./tech-stack.md`
- Key Decisions: see `./decisions.md`
- Technical Summary: see `./technical/README.md`
- Migration: see root `README.md` for repo rename steps
