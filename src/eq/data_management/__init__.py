"""
Data Management Module

This module provides essential data management functionality:
- FastAI v2 DataBlock loading (datablock_loader.py)
- Model loading with historical support (model_loading.py)
- Output directory management (output_manager.py)
- Metadata processing (metadata_processor.py)
- Automatic train/val splitting via FastAI v2 RandomSplitter

CONSOLIDATED ARCHITECTURE:
- Single source of truth for data loading (datablock_loader.py)
- Removed duplicate DataConfig classes
- Removed redundant data loading functions
- Moved dataset-specific utilities to utils/
"""

# Data loading functions (consolidated into datablock_loader)
from .datablock_loader import (
    build_segmentation_datablock,
    build_segmentation_dls,
)
from .standard_getters import get_y as get_glom_y

# Model loading functions (consolidated from model_loader.py and model_loading.py)
from .model_loading import (
    get_model_info,
    load_model_with_historical_support,
    setup_model_loading_environment,
    validate_model_compatibility,
    load_model_safely,
    load_mitochondria_model,
    load_glomeruli_model,
)

# FastAI v2 DataBlock loader (canonical) - already imported above

# Configuration - moved to datablock_loader

# Output management
from .output_manager import (
    OutputManager,
    create_output_directories,
)

# Dataset organization - moved to utils

# Metadata processing
from .metadata_processor import (
    MetadataProcessor,
    process_metadata_file,
)

# Train/val/test splitting - handled automatically by FastAI v2 RandomSplitter

# Public API
__all__ = [
    # Data loading (consolidated)
    'get_glom_y',
    'build_segmentation_datablock',
    'build_segmentation_dls',
    
    # Model loading (consolidated)
    'get_model_info',
    'load_model_with_historical_support',
    'setup_model_loading_environment',
    'validate_model_compatibility',
    'load_model_safely',
    'load_mitochondria_model',
    'load_glomeruli_model',
    
    # Output management
    'OutputManager',
    'create_output_directories',
    
    # Metadata processing
    'MetadataProcessor',
    'process_metadata_file',
    
    # Train/val/test splitting - handled automatically by FastAI v2
]

# Version info
__version__ = "1.0.0"
__author__ = "EQ Development Team"
__description__ = "Consolidated data management functionality for endotheliosis quantifier"
