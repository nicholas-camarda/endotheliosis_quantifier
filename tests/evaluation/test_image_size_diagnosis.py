#!/usr/bin/env python3
"""Diagnose image size issues with backup models - READ-ONLY testing."""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def test_glomeruli_model_different_sizes():
    """Test glomeruli model with different image sizes."""
    print("🔍 DIAGNOSING GLOMERULI MODEL IMAGE SIZE SENSITIVITY")
    print("=" * 60)
    
    try:
        import torch
        from PIL import Image

        from eq.core import setup_global_functions
        from eq.core.model_loading import load_model_with_historical_support

        # Setup environment
        setup_global_functions()
        
        # Load model (READ-ONLY)
        model_path = Path('backups/glomerulus_segmentation_model-dynamic_unet-e50_b16_s84.pkl')
        print(f"📂 Loading model (READ-ONLY): {model_path}")
        model = load_model_with_historical_support(str(model_path))
        
        # Get a test image
        test_dir = Path('derived_data/glomeruli_data/testing/image_patches')
        test_images = list(test_dir.glob('*.jpg'))
        
        if len(test_images) == 0:
            print("❌ No test images found")
            return False
            
        test_img_path = test_images[0]
        print(f"🖼️  Testing with: {test_img_path.name}")
        
        # Test different image sizes
        sizes_to_test = [224, 256, 512, 1024]
        
        for size in sizes_to_test:
            try:
                print(f"\n🔬 Testing size: {size}x{size}")
                
                # Load and resize image
                img = Image.open(test_img_path).convert('RGB')
                original_size = img.size
                img_resized = img.resize((size, size))
                
                print(f"  - Original: {original_size} → Resized: {img_resized.size}")
                
                # Get prediction
                with torch.no_grad():
                    pred = model.predict(img_resized)
                    
                    if len(pred) > 0:
                        pred_item = pred[0]
                        
                        # Extract prediction array
                        if hasattr(pred_item, 'data'):
                            pred_array = np.array(pred_item.data)
                        elif hasattr(pred_item, 'numpy'):
                            pred_array = pred_item.numpy()
                        else:
                            pred_array = np.array(pred_item)
                        
                        # Calculate detection
                        pos_pixels = np.sum(pred_array > 0.5)
                        total_pixels = pred_array.size
                        pos_ratio = pos_pixels / total_pixels
                        
                        print(f"  - Prediction shape: {pred_array.shape}")
                        print(f"  - Value range: [{pred_array.min():.3f}, {pred_array.max():.3f}]")
                        print(f"  - Detection rate: {pos_ratio:.3f} ({pos_pixels}/{total_pixels})")
                        
                        if pos_ratio > 0:
                            print(f"  ✅ DETECTION FOUND at {size}px!")
                        else:
                            print(f"  ❌ No detection at {size}px")
                            
            except Exception as e:
                print(f"  ❌ Error at {size}px: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mitochondria_comparison():
    """Compare mitochondria model behavior for reference."""
    print("\n🔬 MITOCHONDRIA MODEL COMPARISON (Known Working)")
    print("=" * 60)
    
    try:
        import torch
        from PIL import Image

        from eq.core import setup_global_functions
        from eq.core.model_loading import load_model_with_historical_support

        # Setup environment
        setup_global_functions()
        
        # Load mitochondria model (READ-ONLY)
        model_path = Path('backups/mito_dynamic_unet_seg_model-e50_b16.pkl')
        print(f"📂 Loading mitochondria model (READ-ONLY): {model_path}")
        model = load_model_with_historical_support(str(model_path))
        
        # Get a test image
        test_dir = Path('derived_data/mitochondria_data/testing/image_patches')
        test_images = list(test_dir.glob('*.jpg'))
        
        if len(test_images) == 0:
            print("❌ No mitochondria test images found")
            return False
            
        test_img_path = test_images[0]
        print(f"🖼️  Testing with: {test_img_path.name}")
        
        # Test the size that works for mitochondria
        img = Image.open(test_img_path).convert('RGB')
        print(f"📏 Original image size: {img.size}")
        
        with torch.no_grad():
            pred = model.predict(img)
            
            if len(pred) > 0:
                pred_item = pred[0]
                
                # Extract prediction array
                if hasattr(pred_item, 'data'):
                    pred_array = np.array(pred_item.data)
                elif hasattr(pred_item, 'numpy'):
                    pred_array = pred_item.numpy()
                else:
                    pred_array = np.array(pred_item)
                
                pos_pixels = np.sum(pred_array > 0.5)
                total_pixels = pred_array.size
                pos_ratio = pos_pixels / total_pixels
                
                print(f"✅ Mitochondria detection: {pos_ratio:.3f} ({pos_pixels}/{total_pixels})")
                print(f"📐 Output shape: {pred_array.shape}")
                
        return True
        
    except Exception as e:
        print(f"❌ Mitochondria test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 BACKUP MODEL IMAGE SIZE DIAGNOSTIC")
    print("=====================================")
    print("🛡️  READ-ONLY testing - backup models safe!")
    print("🎯 Goal: Find why glomeruli model shows 0% vs. your memory of high accuracy")
    
    success = True
    success &= test_mitochondria_comparison()
    success &= test_glomeruli_model_different_sizes()
    
    print("\n" + "=" * 60)
    if success:
        print("🔍 DIAGNOSTIC COMPLETE")
        print("💡 Check results above to see if image size affects detection")
        print("🎯 Look for any size where glomeruli detection > 0%")
    else:
        print("❌ DIAGNOSTIC FAILED")
        
    print("\n🛡️  Backup models remain untouched!")
