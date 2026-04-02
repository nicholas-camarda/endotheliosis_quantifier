#!/usr/bin/env python3
"""
Test 512px inference for glomeruli model - matching historical training setup
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, 'src')

def test_512px_inference():
    """Test glomeruli model with 512px images as it was trained"""
    
    print("🔍 Testing 512px inference for glomeruli model...")
    
    try:
        from eq.core import setup_global_functions
        from eq.core.model_loading import load_model_with_historical_support

        # Setup global functions
        setup_global_functions()
        
        # Load the current backup model (not git version)
        model_path = Path('backups/glomerulus_segmentation_model-dynamic_unet-e50_b16_s84.pkl')
        if not model_path.exists():
            print("❌ Backup model not found")
            return False
            
        print(f"🔗 Loading glomeruli model: {model_path.name}")
        model = load_model_with_historical_support(str(model_path))
        assert model is not None, "Model failed to load"
        print("✅ Model loaded successfully")
        
        # Get test images and resize to 512px (as trained)
        test_dir = Path('derived_data/glomeruli_data/testing/image_patches')
        if not test_dir.exists():
            print("❌ Test directory not found")
            return False
            
        image_files = list(test_dir.glob('*.jpg'))[:3]  # Test 3 images
        if not image_files:
            print("❌ No test images found")
            return False
            
        print(f"📸 Testing with {len(image_files)} images at 512px...")
        
        total_pixels = 0
        positive_pixels = 0
        
        for img_path in image_files:
            print(f"  🔮 Testing {img_path.name}...")
            
            # Load and resize to 512px (as trained)
            img = Image.open(img_path).convert('RGB')
            img_512 = img.resize((512, 512), Image.Resampling.LANCZOS)
            
            # FastAI expects PIL Image, not tensor
            print(f"    📏 Input image size: {img_512.size}")
            
            # Get prediction
            with torch.no_grad():
                pred = model.predict(img_512)
                
            # Extract prediction mask
            if hasattr(pred[0], 'data'):  # PILMask
                pred_mask = np.array(pred[0].data)
            elif hasattr(pred[0], 'numpy'):  # Tensor
                pred_mask = pred[0].numpy()
            else:  # Already numpy
                pred_mask = np.array(pred[0])
                
            # Ensure binary prediction
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # Count positive pixels
            img_pixels = pred_mask.size
            img_positive = np.sum(pred_mask > 0)
            
            total_pixels += img_pixels
            positive_pixels += img_positive
            
            print(f"    📊 Shape: {pred_mask.shape}, Positive: {img_positive}/{img_pixels} ({img_positive/img_pixels*100:.1f}%)")
            
        # Overall statistics
        if total_pixels > 0:
            overall_positive_ratio = positive_pixels / total_pixels * 100
            print("\n📈 OVERALL RESULTS:")
            print(f"   Total pixels: {total_pixels:,}")
            print(f"   Positive pixels: {positive_pixels:,}")
            print(f"   Positive ratio: {overall_positive_ratio:.2f}%")
            
            if overall_positive_ratio > 1.0:  # More than 1% positive
                print("✅ SUCCESS: 512px inference produces positive predictions!")
                return True
            else:
                print("❌ FAILURE: Still 0% detection even with 512px")
                return False
        else:
            print("❌ No pixels processed")
            return False
            
    except Exception as e:
        print(f"❌ Error during 512px inference test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_512px_inference()
    sys.exit(0 if success else 1)
