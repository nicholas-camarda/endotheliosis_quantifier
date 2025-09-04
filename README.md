# Endotheliosis Quantifier (EQ)

A deep learning pipeline for automated quantification of endotheliosis severity in mouse glomeruli histology images. The system uses a two-stage approach: first training a mitochondria segmentation model on a public electron microscopy data to learn substructure and line features, then transferring this knowledge to segment glomeruli in light microscopy images for endotheliosis quantification.

**Key Features:**
- **Two-stage training**: Mitochondria pretraining → glomeruli transfer learning
- **ROI identification**: Automated segmentation of mouse glomeruli regions
- **Regression modeling**: Predicts endotheliosis severity scores (0-3 scale)
- **Quantitative analysis**: Objective assessment of endotheliosis in preclinical models
- **FastAI v2 optimized**: Binary segmentation with proper augmentation and normalization

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
eq process-data --input-dir data/raw_images/[project] --output-dir data/derived_data/[project]
```

## Production Run: End‑to‑End (Data → Mito Train → Glom Transfer)

The following commands demonstrate the complete pipeline from raw data to trained models:

```bash
# 0) Activate environment
mamba activate eq

# Create standard dirs up front
mkdir -p data/raw_data data/derived_data models/segmentation/{mitochondria,glomeruli} test_output

# Lucchi download and unzip into data/raw_data (matches your extraction path)
cd data/raw_data
wget -O ./lucchi.zip http://rhoana.rc.fas.harvard.edu/dataset/lucchi.zip
unzip -o ./lucchi.zip

# 1) Data processing - convert large images to 256×256 patches, first for mito then for your gloms
RAW_LUCCHI_DIR=data/raw_data/mnt/coxfs01/vcg_connectomics/mitochondria/Lucchi
eq process-data \
  --input-dir "$RAW_LUCCHI_DIR" \
  --output-dir data/derived_data/mito

RAW_DIR=data/raw_data/preeclampsia_project # replace this with your project name
eq process-data \
  --input-dir "$RAW_DIR" \
  --output-dir data/derived_data/preeclampsia

# After this step you should have:
# data/derived_data/preeclampsia/
#   ├─ image_patches/
#   └─ mask_patches/

# 2) Validate derived data integrity (recommended before training)
#    Checks 1:1 image/mask mapping, size consistency, and binary mask values
eq audit-derived --data-dir data/derived_data/mito

# 3) Train mitochondria segmentation model
python -m eq.training.train_mitochondria \
  --data-dir data/derived_data/mito \
  --model-dir models/segmentation/mitochondria \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --image-size 256

# Expected artifacts (examples):
# models/segmentation/mitochondria/
#   └── mitochondria_model/
#       ├── mitochondria_model-epochs_50-batch_16-lr_0.001-size_256.pkl
#       ├── training_loss.png
#       ├── validation_predictions.png
#       └── training_history.json

# 4) Train glomeruli model using transfer learning from mitochondria
BASE=$(ls models/segmentation/mitochondria/*.pkl | head -n1) # set this to your best model trained in step (3)
python -m eq.training.train_glomeruli \
  --data-dir data/derived_data/preeclampsia \
  --model-dir models/segmentation/glomeruli \
  --base-model "$BASE" \
  --epochs 30 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --image-size 256

# Expected artifacts (examples):
# models/segmentation/glomeruli/
#   └── transfer/
#       └── glomeruli_model/
#           ├── glomeruli_model-transfer-epochs_30-batch_8-lr_0.0001-size_256.pkl
#           ├── training_loss.png
#           ├── validation_predictions.png
#           └── training_history.json
```

### Notes
- Use `--config configs/mito_pretraining_config.yaml` or `configs/glomeruli_finetuning_config.yaml` to drive runs; CLI flags override YAML; both fall back to `eq.core.constants` defaults.


### Planned Features
- Feature extraction from segmented regions
- Endotheliosis severity scoring
- Model evaluation and metrics
- Inference pipeline

### 🔄 FastAI v2 Migration Status

#### Current Status:
- ✅ **Data Processing**: Complete - `eq process-data` works with FastAI v2
- ✅ **Data Pipeline**: Complete - DataBlock approach implemented with best practices
- ✅ **Training Modules**: Complete - Optimized binary segmentation with FastAI v2 best practices
- ✅ **Transfer Learning**: Complete - Mitochondria → glomeruli transfer learning working
- ⏳ **Inference Pipeline**: Pending - Will be implemented after training validation
- ⏳ **Evaluation Pipeline**: Pending - Will be implemented after training validation

#### Recent Optimizations (2025-09-04):
- ✅ **Transform Pipeline**: Implemented FastAI v2 best practices with optimal augmentation organization
- ✅ **Normalization**: Added ImageNet normalization for optimal transfer learning performance
- ✅ **Binary Segmentation**: Optimized `n_out=2` approach with proper loss function handling
- ✅ **Lighting Augmentation**: Enabled for improved medical imaging robustness
- ✅ **Directory Structure**: Organized output structure with model-specific subfolders
- ✅ **Error Handling**: Improved data integrity validation and error reporting
- ✅ **Training Infrastructure**: Complete training pipeline with proper file organization

## Data Requirements

### Input Data Format
- **Images**: TIF, PNG, or JPEG files (glomeruli histology images)
- **Masks**: Optional - PNG files with binary masks (0/255 values) for glomeruli segmentation
- **Structure**: Any directory structure - `eq process-data` auto-detects images and masks
- **Data Types**: 
  - **Mitochondria**: Electron microscopy data (Lucchi et al. 2012 dataset)
  - **Glomeruli**: Kidney histology images (H&E stained sections)

### Data Processing

The `eq process-data` command converts large histology images into smaller patches suitable for deep learning:

**Input**: Large histology images (e.g., 2048×2048 pixels)  
**Output**: 256×256 pixel patches

**Process**:
1. Scans directory for image files (TIF, PNG, JPEG)
2. Looks for corresponding mask files (optional)
3. Splits images into 256×256 overlapping patches
4. Creates corresponding patches for masks
5. Validates image-to-mask correspondence
6. Saves processed data for training

**Why patches?**
- Deep learning models require fixed-size inputs
- Large images don't fit in GPU memory
- More patches provide more training samples

### Expected Output from `eq process-data`
```
data/derived_data/
├── mito/                   # Mitochondria data (from Lucchi dataset)
│   ├── image_patches/      # 256x256 image patches
│   ├── mask_patches/       # 256x256 mask patches
│   ├── cache/              # Processed data cache
│   └── metadata.json       # Processing statistics
└── preeclampsia/           # Glomeruli data (your project data)
    ├── image_patches/      # 256x256 image patches
    ├── mask_patches/       # 256x256 mask patches (if masks detected)
    ├── cache/              # Processed data cache
    └── metadata.json       # Processing statistics
```

## Configuration

Configuration is handled through:
- **Constants**: `eq.core.constants` - Default values for patch size, batch size, etc.
- **Environment**: `eq mode` - Hardware-aware configuration
- **CLI Arguments**: All training parameters configurable via command line

## Project Structure

```
src/eq/
├── core/              # Core constants, types, and abstract interfaces
├── data_management/   # Data loading, caching, organization, and model loading
├── processing/        # Image conversion, patchification, and preprocessing
├── training/          # Training scripts for mitochondria and glomeruli models (FastAI v2 optimized)
├── inference/         # Inference and prediction scripts (planned)
├── evaluation/        # Metrics and evaluators (planned)
├── pipeline/          # Pipeline orchestration and production runners (planned)
├── quantification/    # Quantification workflows (planned)
└── utils/             # Config, logging, hardware detection (MPS/CUDA)
```

### Key Files:
- `training/train_mitochondria.py`: Mitochondria model training
- `training/train_glomeruli.py`: Glomeruli transfer learning
- `data_management/datablock_loader.py`: Data loading and preprocessing
- `core/constants.py`: Configuration parameters

## Architecture

**Data Flow**: Raw images → Patchification → Segmentation → Feature extraction → Quantification

**Training Approach**:
1. **Stage 1**: Train mitochondria segmentation model on electron microscopy data
2. **Stage 2**: Transfer learned features to glomeruli segmentation in light microscopy

For detailed technical documentation, see  the [technical documentation](TECHNICAL_LAB_NOTEBOOK.md).

## Troubleshooting

### Common Issues

**Environment Setup**:
```bash
mamba activate eq
pip install -e .
```

**Data Visualization**:
```bash
# Check mask quality
eq visualize --mask data/derived_data/mito/mask_patches/sample_mask.png
eq visualize --image data/derived_data/mito/image_patches/sample.png --mask data/derived_data/mito/mask_patches/sample_mask.png
```

**Training Issues**:
```bash
# Check data structure
ls -la data/derived_data/
cat data/derived_data/metadata.json

# Reduce batch size if out of memory
python -m eq.training.train_mitochondria --batch-size 4
```

**Hardware Configuration**:
```bash
eq mode --show
eq capabilities
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python -m pytest -q tests/` and ensure `ruff check .` passes
5. Submit a pull request
