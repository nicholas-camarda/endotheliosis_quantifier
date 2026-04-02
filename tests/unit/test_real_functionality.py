#!/usr/bin/env python3
"""Test REAL functionality, not just code existence."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch


def test_real_training():
    """Test if training actually works."""
    print("🧪 === TESTING REAL TRAINING ===")
    
    try:
        from eq.segmentation.train_mitochondria_fastai import train_mitochondria_model

        # Create small test dataset
        train_images = np.random.rand(4, 256, 256, 3).astype(np.float32)
        train_masks = np.random.randint(0, 2, (4, 256, 256, 1)).astype(np.float32)
        val_images = np.random.rand(2, 256, 256, 3).astype(np.float32)
        val_masks = np.random.randint(0, 2, (2, 256, 256, 1)).astype(np.float32)
        
        print("✅ Test data created")
        print(f"   Train: {train_images.shape}, {train_masks.shape}")
        print(f"   Val: {val_images.shape}, {val_masks.shape}")
        
        # Actually train
        result = train_mitochondria_model(
            train_images, train_masks, val_images, val_masks,
            'test_output', 'test_model',
            batch_size=2, epochs=1, learning_rate=0.001, image_size=256
        )
        
        print("✅ Training completed successfully!")
        print(f"   Result type: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def test_real_predictions():
    """Test if predictions actually work."""
    print("\n🧪 === TESTING REAL PREDICTIONS ===")
    
    try:
        from PIL import Image

        from eq.utils.model_loader import load_model_safely

        # Load model
        learn = load_model_safely('backups/mito_dynamic_unet_seg_model-e50_b16.pkl')
        print("✅ Model loaded successfully")
        
        # Create test image
        test_img = np.random.rand(256, 256, 3).astype(np.uint8)
        img_pil = Image.fromarray(test_img)
        print("✅ Test image created")
        
        # Make prediction
        pred = learn.predict(img_pil)
        print("✅ Prediction successful!")
        
        # Extract prediction mask
        pred_mask = pred[1] if isinstance(pred, tuple) else pred
        print(f"   Prediction shape: {pred_mask.shape}")
        print(f"   Prediction range: {pred_mask.min().item():.3f} to {pred_mask.max().item():.3f}")
        
        # Check if prediction has variation
        unique_vals = torch.unique(pred_mask).tolist()
        print(f"   Prediction unique values: {unique_vals}")
        
        if len(unique_vals) > 1:
            print("✅ Model produces varied predictions!")
            return True
        else:
            print("⚠️  Model produces uniform predictions (all zeros or all ones)")
            return False
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False


def test_real_evaluation():
    """Test if evaluation actually works."""
    print("\n🧪 === TESTING REAL EVALUATION ===")
    
    try:
        from eq.evaluation.glomeruli_evaluator import evaluate_glomeruli_model

        # Create mock learner that produces varied predictions
        class MockLearner:
            def __init__(self):
                pass
            def predict(self, img):
                # Return varied predictions (not all zeros)
                pred = np.random.rand(64, 64)
                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0
                return (None, pred, None)
        
        # Create test data
        val_images = np.random.rand(2, 64, 64, 3).astype(np.float32)
        val_masks = np.random.randint(0, 2, (2, 64, 64)).astype(np.uint8)
        
        print("✅ Test data created")
        print(f"   Images: {val_images.shape}")
        print(f"   Masks: {val_masks.shape}")
        
        # Run evaluation
        learn = MockLearner()
        metrics = evaluate_glomeruli_model(
            learn, val_images, val_masks, 
            'test_eval', 'test_model'
        )
        
        print("✅ Evaluation completed successfully!")
        print(f"   Metrics: {metrics}")
        
        # Check if metrics make sense
        if 'dice_mean' in metrics and 'iou_mean' in metrics:
            print("✅ Evaluation metrics look reasonable!")
            return True
        else:
            print("⚠️  Evaluation metrics missing expected keys")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False


def main():
    """Run all real functionality tests."""
    print("🎯 === TESTING REAL FUNCTIONALITY ===")
    
    results = []
    
    # Test 1: Real training
    results.append(("Training", test_real_training()))
    
    # Test 2: Real predictions  
    results.append(("Predictions", test_real_predictions()))
    
    # Test 3: Real evaluation
    results.append(("Evaluation", test_real_evaluation()))
    
    # Summary
    print("\n🎯 === FINAL RESULTS ===")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Your repo actually works!")
    else:
        print("⚠️  Some tests failed. Your repo has issues.")


if __name__ == "__main__":
    main()
