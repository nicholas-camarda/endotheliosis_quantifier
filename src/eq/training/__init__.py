#!/usr/bin/env python3
"""
Training Module

This module consolidates training functionality that was previously
scattered across multiple folders:
- models/ (training scripts)
- pipeline/ (training scripts mixed with pipeline orchestration)

Provides unified training infrastructure for mitochondria and glomeruli models.
"""

from eq.core.constants import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE

# Create wrapper functions with expected names
def train_mitochondria(data_dir, model_dir, epochs=None, batch_size=None, **kwargs):
    """
    Train mitochondria model from data directory.
    
    Args:
        data_dir: Directory containing training data
        model_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        **kwargs: Additional training parameters
        
    Returns:
        Trained model
    """
    # Lazy import to avoid importing submodules at package import time
    from eq.training.train_mitochondria import train_mitochondria_with_datablock
    
    # Use constants defaults if not provided
    epochs = epochs or DEFAULT_EPOCHS
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    
    # For now, assume data_dir contains cached pickle files
    return train_mitochondria_with_datablock(
        data_dir=data_dir,
        output_dir=model_dir,
        model_name="mitochondria_model",
        epochs=epochs,
        batch_size=batch_size,
        **kwargs
    )


def train_glomeruli(data_dir, model_dir, base_model=None, epochs=None, batch_size=None, **kwargs):
    """
    Train glomeruli model using transfer learning from mitochondria model.
    
    Args:
        data_dir: Directory containing training data
        model_dir: Directory to save the trained model
        base_model: Path to pretrained mitochondria model
        epochs: Number of training epochs
        batch_size: Training batch size
        **kwargs: Additional training parameters
        
    Returns:
        Trained model
    """
    # Lazy import to avoid importing submodules at package import time
    from eq.training.train_glomeruli import train_glomeruli_with_transfer_learning
    
    # Use constants defaults if not provided
    epochs = epochs or DEFAULT_EPOCHS
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    
    # Forward parameters to the concrete implementation
    return train_glomeruli_with_transfer_learning(
        data_dir=data_dir,
        output_dir=model_dir,
        model_name="glomeruli_model",
        base_model_path=base_model,
        batch_size=batch_size,
        epochs=epochs,
        **kwargs,
    )


# Training utilities (placeholder functions for now)
def setup_training_environment(output_dir):
    """Setup training environment."""
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return True


def validate_training_data(data_dir):
    """Validate training data."""
    from pathlib import Path
    data_path = Path(data_dir)
    return data_path.exists() and data_path.is_dir()


def save_training_results(results, output_dir):
    """Save training results."""
    from pathlib import Path
    import pickle
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "training_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    return True


# Public API
__all__ = [
    # Public wrappers (lazy import inside functions)
    'train_mitochondria',
    'train_glomeruli',
    # Training utilities
    'setup_training_environment',
    'validate_training_data',
    'save_training_results',
]

# Version info
__version__ = "1.0.0"
__description__ = "Unified training infrastructure for mitochondria and glomeruli models"
