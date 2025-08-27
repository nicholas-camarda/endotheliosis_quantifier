"""
Data Management Module

This module consolidates all data management functionality including:
- Data loading and organization
- Configuration management
- Output management
- Metadata processing
- Dataset organization utilities

Consolidated from:
- data/loaders.py
- data/config.py
- utils/organize_lucchi_dataset.py
- utils/output_manager.py
- utils/metadata_processor.py
"""

# Data loading functions
from .data_loading import (
    get_glom_mask_file,
    get_glom_y,
    get_mask_path_patterns,
    n_glom_codes,
    setup_global_functions,
)

# Model loading functions
from .model_loading import (
    get_model_info,
    load_model_with_historical_support,
    setup_model_loading_environment,
    validate_model_compatibility,
)

# Unified data loaders
from .loaders import (
    Annotation,
    UnifiedDataLoader,
    load_glomeruli_data,
    load_mitochondria_patches,
    load_annotations_from_json,
    get_scores_from_annotations,
)

# Configuration
from .config import (
    DataConfig,
    AugmentationConfig,
)

# Output management
from .output_manager import (
    OutputManager,
    create_output_directories,
)

# Dataset organization
from .organize_lucchi_dataset import (
    organize_lucchi_dataset,
    extract_tif_stack,
)

# Metadata processing
from .metadata_processor import (
    MetadataProcessor,
    process_metadata_file,
)

# Public API
__all__ = [
    # Data loading
    'get_glom_mask_file',
    'get_glom_y',
    'get_mask_path_patterns',
    'n_glom_codes',
    'setup_global_functions',
    
    # Model loading
    'get_model_info',
    'load_model_with_historical_support',
    'setup_model_loading_environment',
    'validate_model_compatibility',
    
    # Unified data loaders
    'Annotation',
    'UnifiedDataLoader',
    'load_glomeruli_data',
    'load_mitochondria_patches',
    'load_annotations_from_json',
    'get_scores_from_annotations',
    
    # Configuration
    'DataConfig',
    'AugmentationConfig',
    
    # Output management
    'OutputManager',
    'create_output_directories',
    
    # Dataset organization
    'organize_lucchi_dataset',
    'extract_tif_stack',
    
    # Metadata processing
    'MetadataProcessor',
    'process_metadata_file',
]

# Version info
__version__ = "1.0.0"
__author__ = "EQ Development Team"
__description__ = "Consolidated data management functionality for endotheliosis quantifier"
