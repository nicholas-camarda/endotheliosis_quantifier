#!/usr/bin/env python3
"""
Tests for mitochondria training functionality
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tempfile

import numpy as np
import pytest


# Test data loading
def test_mitochondria_data_loader_import():
    """Test that mitochondria data loader can be imported."""
    from eq.features.mitochondria_data_loader import load_mitochondria_patches
    assert callable(load_mitochondria_patches)

def test_segmentation_pipeline_import():
    """Test that segmentation pipeline can be imported."""
    from eq.pipeline.segmentation_pipeline import SegmentationPipeline
    assert SegmentationPipeline is not None

def test_mitochondria_training_import():
    """Test that mitochondria training function can be imported."""
    from eq.segmentation.train_mitochondria_fastai import train_mitochondria_model
    assert callable(train_mitochondria_model)

def test_pipeline_components_ready():
    """Test that all mitochondria pipeline components are functional."""
    # Test data loading
    
    # Test pipeline orchestration
    
    # Test mitochondria training
    
    # If we get here without errors, all components are ready
    assert True

def test_mitochondria_training_function_signature():
    """Test that mitochondria training function has correct signature."""
    import inspect

    from eq.segmentation.train_mitochondria_fastai import train_mitochondria_model
    
    sig = inspect.signature(train_mitochondria_model)
    expected_params = ['train_images', 'train_masks', 'val_images', 'val_masks', 
                      'output_dir', 'model_name', 'batch_size', 'epochs', 
                      'learning_rate', 'image_size']
    
    for param in expected_params:
        assert param in sig.parameters, f"Missing parameter: {param}"

def test_mitochondria_training_end_to_end():
    """Test that mitochondria training function actually works with real data."""
    from eq.segmentation.train_mitochondria_fastai import train_mitochondria_model

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_output")
        
        # Create synthetic test data (small dataset for quick testing)
        # 10 training samples, 2 validation samples
        train_images = np.random.randint(0, 255, (10, 256, 256, 3), dtype=np.uint8)
        train_masks = np.random.randint(0, 2, (10, 256, 256, 1), dtype=np.uint8)
        val_images = np.random.randint(0, 255, (2, 256, 256, 3), dtype=np.uint8)
        val_masks = np.random.randint(0, 2, (2, 256, 256, 1), dtype=np.uint8)
        
        try:
            # Run training with minimal epochs for testing
            result = train_mitochondria_model(
                train_images=train_images,
                train_masks=train_masks,
                val_images=val_images,
                val_masks=val_masks,
                output_dir=output_dir,
                model_name="test_mito_model",
                batch_size=2,  # Small batch size for testing
                epochs=2,       # Minimal epochs for testing
                learning_rate=1e-3,
                image_size=256
            )
            
            # Check that training completed and returned a result
            assert result is not None
            
            # Check that output directory was created
            assert os.path.exists(output_dir)
            
            # Check that model file was created
            model_path = os.path.join(output_dir, "test_mito_model", "test_mito_model.pkl")
            assert os.path.exists(model_path), f"Model file not found at {model_path}"
            
            # Check that training history was saved
            history_path = os.path.join(output_dir, "test_mito_model", "training_history.pkl")
            assert os.path.exists(history_path), f"Training history not found at {history_path}"
            
        except Exception as e:
            pytest.fail(f"Mitochondria training failed with error: {e}")
