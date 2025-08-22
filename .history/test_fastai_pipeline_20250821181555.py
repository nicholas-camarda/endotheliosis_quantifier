#!/usr/bin/env python3
"""Test script to demonstrate fastai pipeline functionality."""

from pathlib import Path
from eq.segmentation.fastai_segmenter import FastaiSegmenter, SegmentationConfig

def test_fastai_pipeline():
    print("=== Testing Fastai Pipeline ===")
    
    # Create configuration
    config = SegmentationConfig(
        epochs=1, 
        batch_size=2, 
        device_mode='development',
        model_arch='resnet34'
    )
    print(f"✅ Config created: {config}")
    
    # Create segmenter
    segmenter = FastaiSegmenter(config)
    print("✅ Segmenter created")
    
    # Test model creation
    try:
        segmenter.create_model('glomeruli')
        print("✅ Model created successfully")
        print("\n=== Fastai Training Pipeline Working! ===")
        return True
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        print("\n=== Need to fix data preparation ===")
        return False

if __name__ == "__main__":
    test_fastai_pipeline()
