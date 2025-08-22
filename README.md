# Endotheliosis Quantifier

A comprehensive pipeline for quantifying endotheliosis in histology images using deep learning segmentation and regression models.

## Overview

This project implements a three-stage pipeline for endotheliosis quantification:

1. **Segmentation Training (`seg`)**: Train models to find glomeruli in medical images
2. **Quantification Training (`quant-endo`)**: Use segmenter to extract ROIs, then train regression model for endotheliosis quantification  
3. **Production Inference (`production`)**: End-to-end inference using all pre-trained models

## Environment setup

This project standardizes on a Conda environment named `eq` as defined in `environment.yml`.

Create or update the environment (using mamba):

```bash
mamba env create -f environment.yml  # first time
# or, to update an existing env named eq
mamba env update -f environment.yml --prune
```

Activate the environment:

```bash
mamba activate eq  # if your shell is configured for mamba
# otherwise, standard activation still works:
conda activate eq
```

## Data Organization and Patch System

### Input Data Structure
The system expects data organized with train/test splits:

```
data/
├── preeclampsia_data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
```

### Patch System
Individual histology slides are split into patches for efficient processing:

- **Patch Naming Convention**: `{slide_id}_{patch_number}` or `{slide_id}-{patch_number}`
  - Examples: `T101_1`, `T101_2`, `Sample2-1`, `Sample2-2`
- **Separator Detection**: System automatically detects `_` or `-` separators
- **Slide Grouping**: Patches are automatically grouped by slide ID
- **QUICK_TEST Mode**: Processes only a subset of patches for fast validation
- **Full Production**: Processes all patches from all slides

### Data Traceability
- Maintains traceability from individual patches back to original slides
- Preserves train/test split organization throughout all pipeline stages
- Output directories named based on input data source (e.g., `output/preeclampsia/`)

## Using the Pipeline

### Three-Stage Pipeline Commands

```bash
# Stage 1: Train segmentation model
python -m eq seg --data-dir data/preeclampsia_data

# Stage 2: Train quantification model (uses trained segmenter)
python -m eq quant-endo --data-dir data/preeclampsia_data

# Stage 3: Production inference (uses all pre-trained models)
python -m eq production --data-dir data/preeclampsia_data --test-data-dir data/preeclampsia_data/test
```

### QUICK_TEST Mode
All stages support fast validation mode for development and testing:

```bash
# Quick test mode for any stage
QUICK_TEST=true python -m eq seg --data-dir data/preeclampsia_data
QUICK_TEST=true python -m eq quant-endo --data/preeclampsia_data
QUICK_TEST=true python -m eq production --data-dir data/preeclampsia_data --test-data-dir data/preeclampsia_data/test
```

### Using Pretrained Models for Evaluation

For testing without training (using existing pretrained models):

```bash
# Evaluate mitochondria segmentation model using pretrained weights
USE_PRETRAINED=true QUICK_TEST=true python -m eq.pipeline.segmentation_pipeline --stage mito

# This will:
# - Load the pretrained model from backups/mito_dynamic_unet_seg_model-e50_b16.pkl
# - Skip training entirely
# - Run evaluation on validation data
# - Generate sample predictions and evaluation summary
# - Save all outputs to standard production locations
```

### Hardware Detection and MPS Support
The system automatically detects hardware capabilities and sets appropriate backends:

- **MPS Fallback**: Only enabled on macOS when MPS is available
- **CUDA Support**: Automatically used when available on NVIDIA systems
- **CPU Fallback**: Used when no GPU acceleration is available

## Output Structure

The pipeline generates a clean, organized output structure:

```
output/
└── {data_source}/          # e.g., preeclampsia/
    ├── models/             # Trained models
    ├── plots/              # Training curves, visualizations
    ├── results/            # Inference results, segmentation details
    └── cache/              # Intermediate processing files
```

**Logging**: Console output serves as the log - no separate log files or complex reporting bureaucracy.

## Package Structure

- `eq.segmentation.fastai_segmenter` - FastAI-based segmentation models
- `eq.pipeline.run_production_pipeline` - Main pipeline orchestration
- `eq.utils.hardware_detection` - Hardware capability detection
- `eq.utils.output_manager` - Output directory management
- `eq.utils.config_manager` - Pipeline configuration management

## Legacy Package Functions

The project maintains backward compatibility with existing package functions:

```bash
# Run ROI extraction using the eq package
python -c "from eq.features.preprocess_roi import preprocess_images; preprocess_images('data/preeclampsia_data/train/images', 'data/preeclampsia_data/train/masks', 'output/rois')"

# Run feature extraction
python -c "from eq.features.helpers import extract_features; features = extract_features(images)"

# Run endotheliosis quantification
python scripts/main/quantify_endotheliosis.py
```

### Legacy Package Structure

- `eq.features.helpers` - Core feature extraction and processing functions
- `eq.features.preprocess_roi` - ROI preprocessing with circular extraction
- `eq.pipeline.quantify_endotheliosis` - Endotheliosis quantification pipeline
- `eq.io.convert_files_to_jpg` - File format conversion utilities
- `eq.patches.patchify_images` - Image patching utilities
- `eq.augment.augment_dataset` - Data augmentation functions

## Dependencies

```bash
# Install additional dependencies
brew install geos

# Install PyTorch and related packages
pip install torch torchvision segmentation-models-pytorch ipykernel matplotlib albumentations --force-reinstall --no-cache-dir
pip install ipywidgets jupyter docker -U

# Configure Jupyter
jupyter kernelspec list -Xfrozen_modules=off
```

## Data Sources

You can download the AIDPATH dataset from their website (https://aidpath.org/data/) or access it through their API. You will need to create an account to access the data.

## Patch Size Standardization

**Current Standard: 224x224 pixels**

The pipeline uses 224x224 pixel patches throughout to ensure compatibility with pretrained models. This standardization was implemented after discovering that existing pretrained models (like `backups/mito_dynamic_unet_seg_model-e50_b16.pkl`) were trained on 224x224 input.

### Why 224x224?
- **Model compatibility**: Aligns with pretrained segmentation models
- **Performance**: Slightly faster training than 256x256
- **Consistency**: Uniform patch size across entire pipeline
- **Evaluation accuracy**: Proper metrics without resizing artifacts

### Automatic Compatibility
The pipeline includes automatic resizing for mixed scenarios, so existing 256x256 patches will work during evaluation, but new training will generate 224x224 patches natively.

## Training Details

### Mitochondria Segmentation Training

**Purpose**: Train a U-Net model on mitochondria data to use as a pretrained backbone for glomeruli segmentation (transfer learning).

**Quick Start**:
```python
from eq.segmentation.train_mitochondria_fastai import train_mitochondria_model

# Train mitochondria model (ready for transfer learning)
result = train_mitochondria_model(
    train_images=your_train_images,      # (N, 224, 224, 3) RGB format
    train_masks=your_train_masks,        # (N, 224, 224, 1) binary (0 or 1)
    val_images=your_val_images,          # (N, 224, 224, 3) RGB format  
    val_masks=your_val_masks,            # (N, 224, 224, 1) binary (0 or 1)
    output_dir="models/mitochondria",
    model_name="mito_segmenter_v1",
    batch_size=16,
    epochs=50,
    learning_rate=1e-3,
    image_size=224
)
```

**Data Requirements**:
- **Images**: RGB format, uint8 (0-255), shape (H, W, 3)
- **Masks**: Binary format, uint8 (0 or 1), shape (H, W, 1)
- **Size**: 224x224 pixels (configurable)

**Outputs Generated**:
```
models/mitochondria/mito_segmenter_v1/
├── mito_segmenter_v1.pkl          # Trained model (ready for transfer learning)
├── training_history.pkl            # Training metrics and loss curves
├── lr_finder.png                   # Learning rate finder plot
├── training_curves.png             # Training loss and metrics over time
├── sample_predictions.png          # Sample predictions vs ground truth
└── training_summary.txt            # Text summary of training parameters and results
```

**Monitoring Training**:
```
epoch     train_loss  valid_loss  dice      time
0         0.193577    0.000000    0.000000  00:05
1         0.091904    0.000000    0.000000  00:05
...
```

**Next Steps**: Once training is complete, your mitochondria model is ready for transfer learning on glomeruli data.

### U-Net Model Configuration
- Extended model depth from three to five levels
- Dropout factor: 0.25 (added after each convolution block)
- Training epochs: 10 (configurable)
- Learning rate: 0.01
- Batch size: 3 (configurable)
- Performance monitoring: Dice coefficient on validation set

### Data Augmentation
- Random orientation perturbations (rotation and diagonal flip)
- Random stain color and contrast perturbations
- Applied to both images and corresponding masks

### Testing Process
1. Apply tissue masking algorithm to gray level WSI at 2x magnification
2. Slide U-Net receptive field over tissue image with 50% overlap
3. Apply class majority voting for overlapping areas
4. Stitch output masks and evaluate glomerular object size

## 📋 Complete Workflow Guide

### Data Source
The mitochondria segmentation pipeline uses the **Lucchi et al. 2012 benchmark dataset**:
- **Source**: http://rhoana.rc.fas.harvard.edu/dataset/lucchi.zip
- **Reference**: https://connectomics.readthedocs.io/en/latest/tutorials/mito.html
- **Format**: TIF stacks containing 165 individual electron microscopy images and corresponding ground truth masks

### Dataset Organization
After processing, the data is organized into the following structure:
```
data/mitochondria_data/
├── training/
│   ├── images/          # 165 individual training images (training_0.tif to training_164.tif)
│   └── masks/           # 165 individual training masks (training_groundtruth_0.tif to training_groundtruth_164.tif)
├── testing/
│   ├── images/          # 165 individual test images (testing_0.tif to testing_164.tif)
│   └── masks/           # 165 individual test masks (testing_groundtruth_0.tif to testing_groundtruth_164.tif)
└── cache/               # Dataset-specific cache directory
```

**File Descriptions**:
- **Images**: Individual electron microscopy images extracted from TIF stacks (.tif format)
- **Masks**: Individual ground truth segmentation masks extracted from TIF stacks (.tif format)
- **Training**: 165 images used for model training
- **Testing**: 165 images used for model evaluation

### Step-by-Step Process

#### Step 0: Get the Mito Data and move it to the data directory

```bash
# Make sure you're in the repo root directory
cd /path/to/endotheliosis_quantifier

# Download directly into the repo's data folder
wget https://www.dropbox.com/s/3bjyl5vyqj86h6f/lucchi.zip -O data/lucchi.zip

# Extract in place (creates mnt/ subdirectory)
unzip data/lucchi.zip
```

#### Step 1: Dataset Organization
```bash
python -m eq.utils.organize_lucchi_dataset --input-dir mnt/coxfs01/vcg_connectomics/mitochondria/Lucchi
```
**Creates**:
```
data/mitochondria_data/
├── training/
│   ├── images/          # train_im.tif
│   └── masks/           # train_label.tif
└── testing/
    ├── images/          # test_im.tif
    └── masks/           # test_label.tif
```

#### Step 2: Patch Generation
```bash
python -m eq.utils.create_mitochondria_patches --input-dir data/mitochondria_data
```
**What this does**:
- Creates 224x224 pixel patches from each image with 10% overlap
- Splits patches into training (80%) and validation (20%) sets
- Saves patches as JPG files in organized directories:
```
data/mitochondria_data/
├── training/
│   ├── images/          # Original TIF images
│   ├── masks/           # Original TIF masks
│   ├── image_patches/   # 224x224 JPG patches for U-Net
│   ├── mask_patches/    # Corresponding mask patches
│   ├── image_patch_validation/  # Validation patches (20%)
│   └── mask_patch_validation/   # Validation mask patches
├── testing/
│   ├── images/          # Original test TIFs
│   ├── masks/           # Original test masks
│   ├── image_patches/   # Test patches
│   ├── mask_patches/    # Test mask patches
│   ├── image_patch_validation/  # Test validation
│   └── mask_patch_validation/   # Test validation masks
└── cache/                # Dataset-specific cache
```

#### Step 3: Model Training
```bash
python -m eq.pipeline.segmentation_pipeline --stage mitochondria_pretraining
```
**Outputs**:
- Trained U-Net model: `models/segmentation/mitochondria/mito_dynamic_unet_seg_model-e50_b16.pkl`
- Training history and metrics: `models/segmentation/mitochondria/training_history.pkl`
- Diagnostic plots: `models/segmentation/mitochondria/` (learning curves, sample predictions)
- Model checkpoint ready for transfer learning

#### Step 4: Transfer Learning to Glomeruli
```bash
python -m eq.pipeline.segmentation_pipeline --stage glomeruli_finetuning
```
**Uses**: Mitochondria-pretrained model as starting point
**Trains on**: Preeclampsia glomeruli data with Label Studio annotations
**Outputs**: Fine-tuned glomeruli segmentation model

### Reproducibility Features

- **Data Source**: Public benchmark dataset with documented download instructions
- **Processing Scripts**: All data processing steps are scripted and version-controlled
- **Configuration Files**: YAML configs for both mitochondria pretraining and glomeruli fine-tuning
- **Model Checkpoints**: Trained models are saved with training history and diagnostics
- **Environment**: Exact conda environment (`environment.yml`) ensures reproducibility

### Expected Outputs

After running the complete pipeline, you'll have:

```
models/segmentation/mitochondria/
├── mito_dynamic_unet_seg_model-e50_b16.pkl  # Trained mitochondria model
├── training_history.pkl                      # Training metrics and loss curves
├── lr_finder.png                             # Learning rate finder plot
├── training_curves.png                       # Training loss and metrics over time
└── training_summary.txt                      # Text summary of training parameters and results
```

**Note**: Your existing production models will be moved to the new structure:
- `models/segmentation/mitochondria/` - Mitochondria models
- `models/segmentation/glomeruli_xfer_learn/` - Glomeruli transfer learning models

## 📁 Production Directory Structure

```
├── logs/                             # All pipeline and training logs
├── raw_data/                         # Raw datasets
│   ├── lucchi_dataset/               # Mitochondria EM data (Lucchi et al. 2012)
│   └── preeclampsia_data/            # Glomeruli data for fine-tuning
├── derived_data/                     # Processed data
│   ├── mitochondria_data/            # Mitochondria patches and ROIs
│   └── glomeruli_data/               # Glomeruli patches and ROIs
├── outputs/                          # Training results and plots
│   ├── mitochondria_training/        # Mitochondria training outputs
│   └── glomeruli_training/           # Glomeruli training outputs
└── models/                           # Trained models
    └── segmentation/
        ├── mitochondria/              # Mitochondria segmentation models
        └── glomeruli_xfer_learn/     # Glomeruli transfer learning models
    └── regression/                   # Future regression models
```

## 🔬 Quick Testing Mode

For development and testing, use the `QUICK_TEST=true` environment variable:

```bash
# Quick testing mode (faster training, same data)
QUICK_TEST=true python -m eq.pipeline.segmentation_pipeline --stage mitochondria_pretraining

# Production mode (full training)
python -m eq.pipeline.segmentation_pipeline --stage mitochondria_pretraining
```

**Quick Test Mode Features:**
- **Same outputs**: All outputs go to production folders (no separate test structure)
- **Training**: Reduced epochs (2 instead of 50)
- **Batch size**: Smaller batches (4 instead of 16)
- **Same data**: Uses the same 224x224 patches, just trains faster
- **Easy switching**: Just change environment variable, no code changes needed



