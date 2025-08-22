# Technical Summary

## Pipeline Overview
- Data prep/cache: parse annotations, generate masks, split train/val/test, cache arrays.
- Segmentation: **fastai U-Net fine-tuning** to detect glomeruli with multiple architecture options and mode-specific training.
- ROI extraction: contour-based ROI extraction from masked images; save to disk.
- Feature extraction: ResNet50 (imagenet) features per ROI.
- Quantification: Random Forest regression to estimate endotheliosis score with CI output.

## Segmentation Pipeline Architecture

The segmentation pipeline consists of two main stages designed for transfer learning:

### Stage 1: Mitochondria Pretraining
**Purpose**: Initial U-Net training on mitochondria EM data as a foundation for learning general image segmentation features.

**Data Source**: [Lucchi et al. 2012 benchmark dataset](http://rhoana.rc.fas.harvard.edu/dataset/lucchi.zip) from Harvard's Rhoana server - widely-used EM dataset for mitochondria segmentation research.

**Data Flow**:
1. **Raw Data**: Download `lucchi.zip` from Harvard server, extract individual TIF files
2. **Format Conversion**: TIF → JPG using `src/eq/io/convert_files_to_jpg.py`
3. **Patch Generation**: Images split into 256x256 patches using `src/eq/patches/patchify_images.py`
4. **Model Training**: U-Net with ResNet34 backbone (50 epochs, batch size 16)
5. **Output**: `segmentation_model_dir/mito_dynamic_unet_seg_model-e50_b16.pkl` (228MB)

**Configuration**: `configs/mito_pretraining_config.yaml`
- Programmatic train/test split (80/20, seed=42)
- No stratification needed

### Stage 2: Glomeruli Fine-tuning
**Purpose**: Fine-tune the mitochondria-pretrained model on glomeruli segmentation data to create the final production model.

**Data Flow**:
1. **Raw Data**: `data/preeclampsia_data/Lauren_PreEclampsia_Raw_TIF_Images/`
2. **Annotations**: Label Studio JSON export with RLE-encoded masks and endotheliosis scores
3. **Preprocessing**: TIF → JPG conversion, resize to 256x256, normalization
4. **Fine-tuning**: Transfer learning from mitochondria model with lower learning rate (1e-4 vs 1e-3)
5. **Output**: `segmentation_model_dir/glomerulus_segmentation_model-dynamic_unet-e50_b16_s84.pkl` (250MB)

**Configuration**: `configs/glomeruli_finetuning_config.yaml`
- Programmatic train/val/test split (70/15/15, seed=84)
- Stratified by endotheliosis score

### Label Studio Integration
- **Annotation Workflow**: Manual annotation of glomeruli boundaries + endotheliosis severity scoring (0-4 scale)
- **Data Processing**: `load_annotations_from_json()`, `rle2mask()`, `get_scores_from_annotations()`
- **Export Format**: JSON with RLE-encoded masks and choice-based scoring

### Model Checkpoints
- **Mitochondria Model**: Transfer learning foundation (228MB)
- **Final Glomeruli Model**: Production glomeruli segmentation (250MB)
- **Architecture**: Dynamic U-Net with ResNet34 backbone (fine-tuned)

### Reproducibility
- **Requirements**: Exact conda environment (`environment.yml`), hardware (MPS/CUDA recommended)
- **Seeds**: Mitochondria (42), Glomeruli (84) - different to avoid data leakage
- **Commands**: 
  ```bash
  # Stage 1: Mitochondria pretraining
  python -m eq seg --config configs/mito_pretraining_config.yaml
  
  # Stage 2: Glomeruli fine-tuning  
  python -m eq seg --config configs/glomeruli_finetuning_config.yaml
  ```

### Pipeline Orchestration
- **Script**: `src/eq/pipeline/segmentation_pipeline.py`
- **Usage**: 
  ```bash
  python -m eq.pipeline.segmentation_pipeline --stage mito
  python -m eq.pipeline.segmentation_pipeline --stage glomeruli
  python -m eq.pipeline.segmentation_pipeline --stage all
  ```

## CLI Interface
- Central `eq` command with subcommands for each pipeline step
- `eq data-load` - Load and preprocess data with configurable paths
- `eq train-segmenter` - Train segmentation models with hyperparameter control  
- `eq extract-features` - Extract features from images using trained models
- `eq quantify` - Run endotheliosis quantification pipeline
- `eq pipeline` - Execute complete end-to-end pipeline
- `eq capabilities` - Show hardware capabilities and recommendations
- `eq mode` - Inspect and manage environment mode (development/production/auto)
- All commands support `--help` for usage information

### Enhanced Logging System
The CLI provides comprehensive logging with the following features:

#### Logging Options
- `--verbose, -v`: Enable verbose logging with function names and line numbers
- `--log-file`: Write logs to a specified file in addition to console output

#### Logging Features
- **Colored Output**: Different colors for INFO (green), WARNING (yellow), ERROR (red), etc.
- **Progress Tracking**: Step-by-step progress with percentages and timing
- **Function Timing**: Automatic timing of all major operations
- **Data Statistics**: Counts and shapes of data structures being processed
- **File Operations**: Logging of file reads, writes, and operations
- **Error Details**: Full tracebacks in verbose mode for debugging

#### Example Output
```
2025-08-21 14:10:56,681 - INFO - 🔧 Starting eq command: data-load
2025-08-21 14:10:56,681 - INFO - 🚀 Starting data_load_command...
2025-08-21 14:10:56,681 - INFO - 🔄 Starting data loading and preprocessing pipeline...
2025-08-21 14:10:56,681 - INFO - [1/6] (16.7%) Skipping binary mask generation
2025-08-21 14:10:56,681 - INFO - [2/6] (33.3%) Organizing test data into subdirectories
2025-08-21 14:10:56,683 - INFO - 📊 Found 178 test images
2025-08-21 14:10:56,683 - INFO - ✅ organize_data_into_subdirs completed successfully (took 0.00s)
```

### Installation
After installing the package in development mode:
```bash
pip install -e .
```
The `eq` command will be available in your environment.

### Available Commands

#### 1. Data Loading (`data-load`)
Load and preprocess data for the pipeline:
```bash
eq data-load --data-dir data/train --test-data-dir data/test --cache-dir data/cache
```

**Options:**
- `--data-dir`: Training data directory (required)
- `--test-data-dir`: Test data directory (required)
- `--cache-dir`: Cache directory for processed data (required)
- `--annotation-file`: Annotation JSON file (optional)
- `--image-size`: Image size for processing (default: 256)

#### 2. Train Segmentation Model (`train-segmenter`)
Train a segmentation model for glomeruli detection:
```bash
eq train-segmenter --base-model-path models/base_model.h5 --cache-dir data/cache --output-dir output
```

**Options:**
- `--base-model-path`: Path to base model (required)
- `--cache-dir`: Cache directory (required)
- `--output-dir`: Output directory (required)
- `--model-name`: Model name (default: glomerulus_segmenter)
- `--batch-size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 50)

#### 3. Extract Features (`extract-features`)
Extract features from images using a trained model:
```bash
eq extract-features --cache-dir data/cache --output-dir output --model-path models/trained_model.h5
```

**Options:**
- `--cache-dir`: Cache directory (required)
- `--output-dir`: Output directory (required)
- `--model-path`: Path to trained model (required)

#### 4. Quantify Endotheliosis (`quantify`)
Run the main endotheliosis quantification:
```bash
eq quantify --cache-dir data/cache --output-dir output --model-path models/trained_model.h5
```

**Options:**
- `--cache-dir`: Cache directory (required)
- `--output-dir`: Output directory (required)
- `--model-path`: Path to trained model (required)

#### 5. Complete Pipeline (`pipeline`)
Run the complete pipeline end-to-end:
```bash
eq pipeline --data-dir data/train --test-data-dir data/test --cache-dir data/cache --output-dir output --base-model-path models/base_model.h5
```

**Options:**
- `--data-dir`: Training data directory (required)
- `--test-data-dir`: Test data directory (required)
- `--cache-dir`: Cache directory (required)
- `--output-dir`: Output directory (required)
- `--base-model-path`: Path to base model (required)
- `--annotation-file`: Annotation JSON file (optional)
- `--image-size`: Image size for processing (default: 256)
- `--batch-size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 50)

### Usage Examples

#### Basic Usage
```bash
# Load data
eq data-load --data-dir data/train --test-data-dir data/test --cache-dir data/cache

# Train model
eq train-segmenter --base-model-path models/base_model.h5 --cache-dir data/cache --output-dir output

# Extract features
eq extract-features --cache-dir data/cache --output-dir output --model-path output/glomerulus_segmentation/glomerulus_segmenter.h5

# Quantify
eq quantify --cache-dir data/cache --output-dir output --model-path output/glomerulus_segmentation/glomerulus_segmenter.h5
```

#### Complete Pipeline
```bash
eq pipeline \
  --data-dir data/train \
  --test-data-dir data/test \
  --cache-dir data/cache \
  --output-dir output \
  --base-model-path models/base_model.h5 \
  --annotation-file data/annotations.json
```

#### Verbose Logging
```bash
# Enable verbose logging for detailed output
eq --verbose data-load --data-dir data/train --test-data-dir data/test --cache-dir data/cache

# Save logs to file
eq --log-file pipeline.log pipeline --data-dir data/train --test-data-dir data/test --cache-dir data/cache --output-dir output --base-model-path models/base_model.h5
```

#### Help Commands
Get help for any command:
```bash
eq --help
eq data-load --help
eq train-segmenter --help
eq extract-features --help
eq quantify --help
eq pipeline --help
```

### Notes
- All commands require the `eq` conda environment to be activated
- The cache directory is used to store intermediate processed data
- Output directories will be created automatically if they don't exist
- Some commands may take significant time depending on data size and model complexity
- Use `--verbose` for detailed debugging information
- Use `--log-file` to save logs for later analysis

## Package Structure
- `src/eq/` - Main package with modular organization
- `src/eq/__main__.py` - CLI entry point with argument parsing
- `src/eq/features/` - Data loading and preprocessing
- `src/eq/segmentation/` - Model training and inference
  - `fastai_segmenter.py` - Comprehensive fastai segmentation implementation
  - `train_segmenter_fastai.py` - Fastai training script (replaces TensorFlow version)
- `src/eq/models/` - Feature extraction and model utilities
- `src/eq/pipeline/` - Main quantification logic
  - `segmentation_pipeline.py` - Complete segmentation pipeline orchestrator
- `src/eq/utils/` - Common utilities and helpers

## Fastai Segmentation System

The package now includes a comprehensive fastai-based segmentation system that replaces the previous TensorFlow implementation:

### Key Features
- **Multiple U-Net Architectures**: Support for ResNet18, 34, 50, and 101 backbones
- **Mode-Specific Training**: Production (conservative), Development (aggressive), and Auto (balanced) strategies
- **Hardware Optimization**: Automatic device selection (MPS/CUDA/CPU) with optimal batch sizing
- **Advanced Augmentation**: Comprehensive data augmentation pipeline with mode-specific settings
- **Cache Integration**: Support for existing pickle-based data pipeline
- **Comprehensive Monitoring**: Early stopping, learning rate scheduling, and model checkpointing

### Training Modes
- **Production Mode**: Conservative training with early stopping, learning rate reduction, and best model saving
- **Development Mode**: Aggressive training with one-cycle policy and frequent model saving
- **Auto Mode**: Balanced training with automatic hardware detection and optimization

### Hardware Support
- **Apple Silicon (M1/M2)**: Native MPS acceleration with automatic fallback
- **NVIDIA GPUs**: CUDA support for high-performance training
- **CPU Fallback**: Automatic fallback when GPU acceleration unavailable
- **Memory Optimization**: Automatic batch size optimization based on available memory

### Usage Example
```bash
# Train with fastai (replaces TensorFlow training)
eq train-segmenter --base-model-path models/base_model.h5 --cache-dir data/cache --output-dir output

# Check hardware capabilities
eq capabilities

# Set training mode
eq mode --set development

# Run with specific mode
eq --mode production train-segmenter --base-model-path models/base_model.h5 --cache-dir data/cache --output-dir output
```

## Notable Paths (to be centralized)
- Data root: `data/preeclampsia_data` (train/test/cache subdirs)
- Outputs: `output/` (models, predictions)

## Next Steps (engineering)
- Resolve segmentation_models import issues for full CLI functionality
- Add sample data and smoke tests runnable on M1 and in WSL2
- Package release prep (README, examples, versioning)
- Batch inference CLI for researcher datasets
