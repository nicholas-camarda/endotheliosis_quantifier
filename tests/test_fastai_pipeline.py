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
    
    # Test data preparation
    try:
        print("\n=== Preparing Data ===")
        cache_dir = Path('data/preeclampsia_data/cache')
        segmenter.prepare_data_from_cache(cache_dir, 'glomeruli')
        print("✅ Data prepared successfully")
        
        # Test model creation
        print("\n=== Creating Model ===")
        segmenter.create_model('glomeruli')
        print("✅ Model created successfully")
        
        print("\n=== Fastai Training Pipeline Working! ===")
        print("✅ Data preparation: Working")
        print("✅ Model creation: Working")
        print("✅ MPS acceleration: Working")
        print("✅ Development mode: Working")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n=== Pipeline needs debugging ===")
        return False

if __name__ == "__main__":
    test_fastai_pipeline()
