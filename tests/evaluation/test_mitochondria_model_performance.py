#!/usr/bin/env python3
"""
Test mitochondria model inference performance to check for FastAI v1 vs v2 incompatibility
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, 'src')

def test_mitochondria_model_performance():
    """Test if mitochondria model has the same 0% detection issue as glomeruli model"""
    
    print("🔍 Testing mitochondria model inference performance...")
    
    try:
        # Set up namespace for model loading
        from fastai.vision.all import PILImage, load_learner

        import __main__
        from eq.core.data_loading import get_glom_mask_file, get_glom_y, n_glom_codes
        __main__.__dict__['get_y'] = get_glom_y
        __main__.__dict__['get_glom_mask_file'] = get_glom_mask_file
        __main__.__dict__['n_glom_codes'] = n_glom_codes
        
        # Load mitochondria model
        mito_model_path = Path("backups/mito_dynamic_unet_seg_model-e50_b16.pkl")
        if not mito_model_path.exists():
            print(f"❌ Mitochondria model not found: {mito_model_path}")
            return False
        
        print(f"📁 Loading mitochondria model from: {mito_model_path}")
        mito_model = load_learner(mito_model_path)
        print("✅ Mitochondria model loaded successfully")
        
        # Get test data
        test_image_path = Path("derived_data/mitochondria_data/testing/image_patches")
        if not test_image_path.exists():
            print(f"❌ Test data not found: {test_image_path}")
            return False
        
        test_images = list(test_image_path.glob("*.jpg"))[:5]  # Test first 5 images
        print(f"📊 Testing on {len(test_images)} images")
        
        # Test predictions
        total_pixels = 0
        positive_pixels = 0
        
        for i, img_path in enumerate(test_images):
            print(f"  Testing image {i+1}/{len(test_images)}: {img_path.name}")
            
            # Load and predict
            img = PILImage.create(img_path)
            pred = mito_model.predict(img)
            
            # Extract prediction array
            if hasattr(pred[0], 'numpy'):
                pred_array = pred[0].numpy()
            else:
                pred_array = np.array(pred[0])
            
            # Count positive pixels
            positive_pixels += np.sum(pred_array > 0.5)  # Threshold at 0.5
            total_pixels += pred_array.size
            
            print(f"    Prediction shape: {pred_array.shape}")
            print(f"    Prediction range: [{pred_array.min():.4f}, {pred_array.max():.4f}]")
            print(f"    Positive pixels: {np.sum(pred_array > 0.5)} / {pred_array.size}")
        
        # Calculate detection rate
        detection_rate = positive_pixels / total_pixels if total_pixels > 0 else 0
        print("\n📊 RESULTS:")
        print(f"  Total pixels tested: {total_pixels}")
        print(f"  Positive pixels: {positive_pixels}")
        print(f"  Detection rate: {detection_rate:.4f} ({detection_rate*100:.2f}%)")
        
        if detection_rate < 0.01:  # Less than 1%
            print("❌ MITOCHONDRIA MODEL HAS 0% DETECTION ISSUE!")
            print("   This confirms both models need retraining")
            return False
        else:
            print("✅ Mitochondria model appears to be working")
            print("   Only glomeruli model needs retraining")
            return True
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mitochondria_model_performance()
    if success:
        print("\n🎉 MITOCHONDRIA MODEL IS WORKING!")
        print("Only glomeruli model needs retraining")
    else:
        print("\n💥 BOTH MODELS NEED RETRAINING!")
        print("FastAI v1 vs v2 incompatibility affects both models")
