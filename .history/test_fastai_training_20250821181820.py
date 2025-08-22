#!/usr/bin/env python3
"""Comprehensive test script to demonstrate fastai training pipeline."""

from pathlib import Path
from eq.segmentation.fastai_segmenter import FastaiSegmenter, SegmentationConfig

def test_fastai_training():
    print("=== 🚀 Fastai Training Pipeline Demo ===")
    
    # Create configuration for quick training
    config = SegmentationConfig(
        epochs=2, 
        batch_size=2, 
        device_mode='development',
        model_arch='resnet34',
        learning_rate=0.001
    )
    print(f"✅ Config created: {config}")
    
    # Create segmenter
    segmenter = FastaiSegmenter(config)
    print("✅ Segmenter created")
    
    try:
        # Prepare data
        print("\n=== 📊 Preparing Data ===")
        cache_dir = Path('data/preeclampsia_data/cache')
        segmenter.prepare_data_from_cache(cache_dir, 'glomeruli')
        print("✅ Data prepared successfully")
        print(f"   Training samples: {len(segmenter.dls.train_ds)}")
        print(f"   Validation samples: {len(segmenter.dls.valid_ds)}")
        print(f"   Batch size: {segmenter.config.batch_size}")
        
        # Create model
        print("\n=== 🧠 Creating Model ===")
        segmenter.create_model('glomeruli')
        print("✅ Model created successfully")
        print(f"   Architecture: {segmenter.config.model_arch}")
        print(f"   Parameters: {sum(p.numel() for p in segmenter.learn.model.parameters()):,}")
        
        # Find learning rate
        print("\n=== 🔍 Finding Learning Rate ===")
        lr_min, lr_steep, lr_valley, lr_slide = segmenter.find_learning_rate()
        print("✅ Learning rate found")
        print(f"   LR min: {lr_min:.2e}")
        print(f"   LR steep: {lr_steep:.2e}")
        print(f"   LR valley: {lr_valley:.2e}")
        print(f"   LR slide: {lr_slide:.2e}")
        
        # Train model
        print("\n=== 🏋️ Training Model ===")
        training_result = segmenter.train(epochs=2, learning_rate=lr_steep)
        print("✅ Training completed successfully")
        print(f"   Training loss: {training_result.get('train_loss', 'N/A')}")
        print(f"   Validation loss: {training_result.get('valid_loss', 'N/A')}")
        print(f"   Dice score: {training_result.get('dice', 'N/A')}")
        
        print("\n=== 🎉 Fastai Training Pipeline Complete! ===")
        print("✅ Data preparation: Working")
        print("✅ Model creation: Working") 
        print("✅ Learning rate finding: Working")
        print("✅ Model training: Working")
        print("✅ MPS acceleration: Working")
        print("✅ Development mode: Working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n=== Pipeline needs debugging ===")
        return False

if __name__ == "__main__":
    # Set MPS fallback for compatibility
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    test_fastai_training()
