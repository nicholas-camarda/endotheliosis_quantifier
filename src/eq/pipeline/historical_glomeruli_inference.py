#!/usr/bin/env python3
"""
Historical Glomeruli Inference Module

This module provides inference using the correct historical preprocessing approach
identified during model restoration analysis. It can be integrated into the current
repo to restore model functionality.

Key findings from analysis:
- Model loads successfully but needs historical preprocessing
- 512px image sizes required (not 256px)
- Specific get_glom_y and p2c functions needed
- MPS fallback required for Mac compatibility
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastai.vision.all import *

from eq.utils.logger import get_logger

logger = get_logger("eq.historical_inference")

# Historical P2C mapping from successful restoration
HISTORICAL_P2C = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 128, 
    10: 129, 11: 130, 12: 131, 13: 132, 14: 133, 15: 135, 16: 255, 
    17: 56, 18: 57, 19: 134, 20: 58, 21: 187, 22: 188, 23: 184, 
    24: 185, 25: 186, 26: 59, 27: 60, 28: 61, 29: 62, 30: 63, 
    31: 64, 32: 65, 33: 66, 34: 67, 35: 68, 36: 69, 37: 189, 
    38: 190, 39: 191, 40: 192, 41: 193, 42: 194, 43: 195, 44: 196, 
    45: 197, 46: 70, 47: 71, 48: 247, 49: 253, 50: 121, 51: 198, 
    52: 250, 53: 251, 54: 252, 55: 254, 56: 248, 57: 249, 58: 122, 
    59: 123, 60: 124, 61: 125, 62: 126, 63: 127
}

# Global variables needed for model loading
p2c = HISTORICAL_P2C

def get_glom_mask_file(image_file, p2c, thresh=127):
    """
    Historical mask loading function adapted for current data structure.
    
    This function was required for the original model training and must be
    available in global scope for model loading.
    """
    try:
        image_path_str = str(image_file)
        
        # Adapt to current data structure patterns
        if 'image_patches' in image_path_str:
            mask_path = Path(image_path_str.replace('image_patches', 'mask_patches').replace('.jpg', '.png'))
        elif '/data/images/' in image_path_str:
            mask_path = Path(image_path_str.replace('/data/images/', '/data/masks/').replace('.jpg', '_mask.jpg'))
        else:
            # Try generic patterns
            mask_path = Path(str(image_file).replace('.jpg', '_mask.png'))
        
        if not mask_path.exists():
            return None
        
        # Historical processing logic
        msk = np.array(PILMask.create(mask_path))
        msk[msk <= thresh] = 0
        msk[msk > thresh] = 1
        
        return PILMask.create(msk)
        
    except:
        return None

def get_glom_y(o):
    """
    Historical get_y function required for model loading.
    
    This function was used during original training and must be available
    for the model to load properly.
    """
    return get_glom_mask_file(o, p2c)

def setup_historical_environment():
    """
    Set up the environment for historical model loading and inference.
    
    This must be called before loading the model to ensure all required
    functions are available in the global namespace.
    """
    # Set MPS fallback for Mac compatibility (critical for model loading)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Load functions into global namespace for model loading
    globals()['p2c'] = HISTORICAL_P2C
    globals()['get_glom_y'] = get_glom_y
    globals()['get_glom_mask_file'] = get_glom_mask_file
    
    # CRITICAL: Also inject into main module namespace for pickle loading
    import __main__
    if hasattr(__main__, '__dict__'):
        __main__.__dict__['p2c'] = HISTORICAL_P2C
        __main__.__dict__['get_glom_y'] = get_glom_y  
        __main__.__dict__['get_glom_mask_file'] = get_glom_mask_file
        logger.info("✅ Functions injected into main module namespace")
    
    logger.info("✅ Historical environment configured for model loading")

class HistoricalGlomeruliInference:
    """
    Glomeruli inference using historical preprocessing approach.
    
    This class provides the correct preprocessing pipeline identified during
    model restoration analysis to restore proper model functionality.
    """
    
    def __init__(self, model_path: str = "backups/glomerulus_segmentation_model-dynamic_unet-e50_b16_s84.pkl"):
        """
        Initialize historical inference.
        
        Args:
            model_path: Path to the historical glomeruli model
        """
        self.model_path = Path(model_path)
        self.learner = None
        
        # Set up historical environment
        setup_historical_environment()
        
    def load_model(self) -> bool:
        """
        Load the historical model with proper environment.
        
        Returns:
            bool: True if model loaded successfully
        """
        if not self.model_path.exists():
            logger.error(f"Model not found at {self.model_path}")
            return False
        
        try:
            # Load with historical functions available
            self.learner = load_learner(self.model_path)
            logger.info("✅ Historical glomeruli model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def preprocess_image_historical(self, image: PILImage, target_size: int = 512) -> PILImage:
        """
        Apply historical preprocessing to image.
        
        Key insight: Original model was trained with 512px images, not 256px!
        
        Args:
            image: Input PIL image
            target_size: Target size (512px historical, 256px current broken)
            
        Returns:
            Preprocessed PIL image
        """
        # Historical approach: resize to 512px 
        # This was the key difference causing zero predictions
        resized_img = image.resize((target_size, target_size))
        return resized_img
    
    def predict_single(self, image_path: str, use_historical_preprocessing: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Predict on a single image using historical approach.
        
        Args:
            image_path: Path to input image
            use_historical_preprocessing: Whether to use 512px (True) or 256px (False)
            
        Returns:
            Tuple of (prediction_array, metadata)
        """
        if self.learner is None:
            if not self.load_model():
                raise RuntimeError("Model not loaded")
        
        # Load image
        img = PILImage.create(image_path)
        
        if use_historical_preprocessing:
            # Use historical 512px preprocessing (CORRECT)
            processed_img = self.preprocess_image_historical(img, target_size=512)
            approach = "historical_512px"
        else:
            # Use current 256px preprocessing (BROKEN - for comparison)
            processed_img = self.preprocess_image_historical(img, target_size=256)
            approach = "current_256px"
        
        # Predict
        pred = self.learner.predict(processed_img)
        
        if len(pred) > 0:
            pred_array = np.array(pred[0])
            
            # Calculate metrics
            positive_pixels = np.sum(pred_array > 0)
            total_pixels = pred_array.size
            positive_ratio = positive_pixels / total_pixels
            
            metadata = {
                'approach': approach,
                'image_shape': np.array(img).shape,
                'processed_shape': np.array(processed_img).shape,
                'prediction_shape': pred_array.shape,
                'positive_ratio': positive_ratio,
                'unique_values': np.unique(pred_array).tolist(),
                'all_zeros': np.all(pred_array == 0),
                'total_positive_pixels': int(positive_pixels),
                'image_path': str(image_path)
            }
            
            return pred_array, metadata
        else:
            raise RuntimeError("Model produced no prediction output")
    
    def compare_preprocessing_approaches(self, image_path: str) -> dict:
        """
        Compare historical vs current preprocessing on the same image.
        
        This demonstrates the difference in model performance between approaches.
        
        Returns:
            dict: Comparison results
        """
        results = {}
        
        for use_historical, name in [(True, "Historical_512px"), (False, "Current_256px")]:
            try:
                pred_array, metadata = self.predict_single(image_path, use_historical)
                results[name] = metadata
                logger.info(f"{name}: {metadata['positive_ratio']:.4f} positive ratio, all_zeros={metadata['all_zeros']}")
            except Exception as e:
                results[name] = {'error': str(e)}
                logger.error(f"{name} failed: {e}")
        
        return results
    
    def validate_on_test_set(self, test_images: List[str], use_historical: bool = True) -> dict:
        """
        Validate model on a set of test images.
        
        Args:
            test_images: List of test image paths
            use_historical: Whether to use historical preprocessing
            
        Returns:
            dict: Validation results
        """
        results = []
        
        for img_path in test_images:
            try:
                pred_array, metadata = self.predict_single(img_path, use_historical)
                results.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to predict on {img_path}: {e}")
                continue
        
        if not results:
            return {'error': 'No successful predictions'}
        
        # Calculate aggregate metrics
        positive_ratios = [r['positive_ratio'] for r in results]
        all_zeros_count = sum(1 for r in results if r['all_zeros'])
        
        summary = {
            'num_images': len(results),
            'mean_positive_ratio': np.mean(positive_ratios),
            'std_positive_ratio': np.std(positive_ratios),
            'all_zeros_count': all_zeros_count,
            'success_rate': (len(results) - all_zeros_count) / len(results),
            'approach': 'historical_512px' if use_historical else 'current_256px',
            'individual_results': results
        }
        
        return summary

def integrate_into_current_repo():
    """
    Integration guide for current repository.
    
    This function provides concrete steps to integrate the historical approach
    into your existing codebase.
    """
    logger.info("🔧 INTEGRATION GUIDE FOR CURRENT REPOSITORY")
    logger.info("=" * 60)
    
    steps = [
        "1. Update inference modules to use HistoricalGlomeruliInference class",
        "2. Modify image preprocessing to use 512px instead of 256px",
        "3. Ensure PYTORCH_ENABLE_MPS_FALLBACK=1 in environment",
        "4. Add historical functions (get_glom_y, get_glom_mask_file) to global scope before model loading",
        "5. Update evaluation scripts to use historical preprocessing",
        "6. Consider retraining with corrected preprocessing pipeline",
        "7. Update production inference to use historical approach"
    ]
    
    for step in steps:
        logger.info(f"   {step}")
    
    logger.info("\n📊 EVIDENCE FROM ANALYSIS:")
    logger.info("   ✅ Model loads successfully with historical functions")
    logger.info("   ✅ Non-zero predictions achieved with 512px preprocessing")
    logger.info("   ✅ Zero predictions confirmed with 256px preprocessing")
    logger.info("   ✅ Visual evidence saved in logs/phase3_success_*.png")
    
    logger.info("\n🎯 RECOMMENDED NEXT STEPS:")
    logger.info("   1. Test this module on your data")
    logger.info("   2. Compare performance with historical benchmarks")  
    logger.info("   3. Integrate into production pipeline")
    logger.info("   4. Consider improvements to historical approach")

# Example usage and testing
if __name__ == "__main__":
    logger.info("🧪 Testing Historical Glomeruli Inference")
    
    # Initialize inference
    inference = HistoricalGlomeruliInference()
    
    # Test model loading
    if inference.load_model():
        logger.info("✅ Model loaded - ready for inference")
        
        # Find test images
        test_images = []
        for pattern in ["derived_data/glomeruli_data/**/*.jpg", "raw_data/**/*.jpg"]:
            test_images.extend(list(Path(".").glob(pattern)))
        
        if test_images:
            test_img = test_images[0]
            logger.info(f"🧪 Testing preprocessing comparison on: {test_img}")
            
            # Compare approaches
            results = inference.compare_preprocessing_approaches(str(test_img))
            
            logger.info("📊 COMPARISON RESULTS:")
            for approach, result in results.items():
                if 'error' not in result:
                    logger.info(f"   {approach}: {result['positive_ratio']:.4f} positive ratio")
                    logger.info(f"      All zeros: {result['all_zeros']}")
                else:
                    logger.info(f"   {approach}: Failed - {result['error']}")
        else:
            logger.warning("No test images found")
    else:
        logger.error("❌ Model loading failed")
    
    # Show integration guide
    integrate_into_current_repo()
