# API Specification

This is the API specification for the spec detailed in @.agent-os/specs/2025-08-31-pipeline-development-testing/spec.md

## CLI Interface

### Main Command Structure
The primary CLI entry point is `eq` with subcommands for different pipeline components:

```bash
eq [--mode {auto,development,production}] [--verbose] [--log-file FILE] <command> [<args>]
```

### Core Commands

#### Data Management
- **`eq data-load`** - Load and preprocess training data
- **`eq capabilities`** - Show hardware capabilities and recommendations
- **`eq mode`** - Inspect and manage environment mode

#### Training Pipeline
- **`eq train-segmenter`** - Train segmentation models with configurable parameters
- **`eq seg`** - Train glomeruli segmentation model
- **`eq quant-endo`** - Train endotheliosis quantification model

#### Inference and Evaluation
- **`eq extract-features`** - Extract features from trained models
- **`eq quantify`** - Run endotheliosis quantification
- **`eq production`** - Execute complete production pipeline

#### Pipeline Orchestration
- **`eq orchestrator`** - Interactive pipeline orchestrator with menu selection

## Endpoints

### Data Loading Command

**Purpose:** Load and preprocess training data for model training
**Parameters:** 
- `--data-dir`: Training data directory (required)
- `--test-data-dir`: Test data directory (required)
- `--cache-dir`: Cache directory for processed data (required)
- `--annotation-file`: Annotation JSON file (optional)
- `--image-size`: Image size for processing (default: 256)

**Response:** Success/failure status with processing statistics
**Errors:** Invalid data paths, corrupted files, insufficient memory

### Training Command

**Purpose:** Train segmentation models with transfer learning support
**Parameters:**
- `--base-model-path`: Path to base model for transfer learning (required)
- `--cache-dir`: Cache directory with processed data (required)
- `--output-dir`: Output directory for trained models (required)
- `--model-name`: Model name identifier (default: glomerulus_segmenter)
- `--batch-size`: Training batch size (default: 8)
- `--epochs`: Number of training epochs (default: 50)

**Response:** Training progress, metrics, and final model path
**Errors:** Invalid model path, insufficient data, hardware limitations

### Production Pipeline Command

**Purpose:** Execute complete end-to-end pipeline for production inference
**Parameters:**
- `--data-dir`: Path to training data directory (required)
- `--test-data-dir`: Path to test data directory (required)
- `--segmentation-model`: Segmentation model to use (required)
- `--annotation-file`: Path to annotation file (optional)
- `--image-size`: Image size for processing (default: 256)
- `--batch-size`: Batch size for processing (default: 8)

**Response:** Complete pipeline results with quantification metrics
**Errors:** Model loading failures, data processing errors, insufficient resources

### Mode Management Command

**Purpose:** Inspect and configure environment execution mode
**Parameters:**
- `--set`: Set environment mode (auto/development/production)
- `--show`: Show current mode and configuration summary
- `--validate`: Validate current mode against hardware capabilities

**Response:** Current mode status, hardware recommendations, configuration summary
**Errors:** Invalid mode selection, hardware compatibility issues

## Internal API Structure

### Core Modules Interface

#### Data Management (`eq.data_management`)
- **`DataLoader`** - Abstract base class for data loading operations
- **`ImagePreprocessor`** - Image preprocessing and augmentation
- **`AnnotationParser`** - Annotation file parsing and validation
- **`CacheManager`** - Data caching and retrieval system

#### Training (`eq.training`)
- **`BaseTrainer`** - Abstract base class for training operations
- **`MitochondriaTrainer`** - Mitochondria model training implementation
- **`GlomeruliTrainer`** - Glomeruli model training with transfer learning
- **`TrainingConfig`** - Configuration management for training parameters

#### Inference (`eq.inference`)
- **`ModelLoader`** - Model loading and validation
- **`Predictor`** - Inference execution and result generation
- **`BatchProcessor`** - Batch processing for multiple images
- **`OutputFormatter`** - Standardized output formatting

#### Evaluation (`eq.evaluation`)
- **`MetricsCalculator`** - Performance metrics computation
- **`VisualizationGenerator`** - Charts and plots generation
- **`ReportGenerator`** - Evaluation report creation
- **`StatisticalAnalyzer`** - Statistical analysis and significance testing

#### Pipeline (`eq.pipeline`)
- **`PipelineOrchestrator`** - Main pipeline coordination
- **`ComponentManager`** - Individual component lifecycle management
- **`ErrorHandler`** - Error handling and recovery mechanisms
- **`ProgressTracker`** - Progress monitoring and reporting

### Configuration Management

#### YAML Configuration Files
- **`configs/mito_pretraining_config.yaml`** - Mitochondria training configuration
- **`configs/glomeruli_finetuning_config.yaml`** - Glomeruli fine-tuning configuration
- **`configs/inference_config.yaml`** - Inference pipeline configuration
- **`configs/evaluation_config.yaml`** - Evaluation metrics and reporting configuration

#### Configuration Schema
```yaml
# Example configuration structure
training:
  model:
    architecture: "unet"
    pretrained: true
    freeze_layers: ["encoder.0", "encoder.1"]
  
  data:
    batch_size: 8
    image_size: 256
    augmentation: true
  
  optimization:
    learning_rate: 0.001
    optimizer: "adam"
    scheduler: "cosine"
  
  hardware:
    device: "auto"
    mixed_precision: true
    gradient_accumulation: 1
```

### Error Handling and Recovery

#### Error Categories
- **Data Errors**: Invalid file formats, corrupted data, missing annotations
- **Model Errors**: Architecture mismatches, weight loading failures, compatibility issues
- **Hardware Errors**: Insufficient memory, device unavailability, performance degradation
- **Configuration Errors**: Invalid parameters, missing dependencies, path resolution issues

#### Recovery Mechanisms
- **Automatic Fallbacks**: Hardware fallbacks (MPS → CPU), batch size reduction
- **Graceful Degradation**: Continue processing with available resources
- **Detailed Logging**: Comprehensive error reporting with context and suggestions
- **User Guidance**: Clear error messages with resolution steps and examples
