#!/usr/bin/env python3
"""Test to verify all imports work correctly after legacy import cleanup."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def test_core_imports():
    """Test that core module imports work correctly."""
    print("Testing core module imports...")
    
    try:
        from eq.core import BINARY_P2C, DEFAULT_MASK_THRESHOLD
        print("✅ Core imports successful")
        
        # Verify constants
        assert BINARY_P2C == [0, 1], f"Expected [0, 1], got {BINARY_P2C}"
        assert DEFAULT_MASK_THRESHOLD == 127, f"Expected 127, got {DEFAULT_MASK_THRESHOLD}"
        print("✅ Core constants verified")
        
        return True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False

def test_data_imports():
    """Test that data module imports work correctly."""
    print("\nTesting data module imports...")
    
    try:
        from eq.data import UnifiedDataLoader, load_glomeruli_data, load_mitochondria_patches
        print("✅ Data imports successful")
        
        # Test creating loaders
        mito_loader = UnifiedDataLoader('mitochondria', 'test', 'test')
        glom_loader = UnifiedDataLoader('glomeruli', 'test', 'test')
        
        # Use the imported functions so ruff doesn't remove them
        assert load_mitochondria_patches is not None
        assert load_glomeruli_data is not None
        
        print("✅ Data loader creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Data imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_models_imports():
    """Test that models module imports work correctly."""
    print("\nTesting models module imports...")
    
    try:
        from eq.models.fastai_segmenter import FastaiSegmenter, SegmentationConfig
        from eq.models.train_glomeruli_transfer_learning import train_glomeruli_transfer_learning
        from eq.models.train_mitochondria_fastai import train_mitochondria_model

        # Use the imports so ruff doesn't remove them
        assert FastaiSegmenter is not None
        assert SegmentationConfig is not None
        assert train_mitochondria_model is not None
        assert train_glomeruli_transfer_learning is not None
        
        print("✅ Models imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Models imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processing_imports():
    """Test that processing module imports work correctly."""
    print("\nTesting processing module imports...")
    
    try:
        from eq.processing import (
            convert_tif_to_jpg,
            patchify_image_and_mask_dirs,
            patchify_image_dir,
        )

        # Use the imports so ruff doesn't remove them
        assert patchify_image_dir is not None
        assert patchify_image_and_mask_dirs is not None  
        assert convert_tif_to_jpg is not None
        
        print("✅ Processing imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Processing imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_imports():
    """Test that pipeline imports work correctly after legacy cleanup."""
    print("\nTesting pipeline imports...")
    
    try:
        # Test segmentation pipeline
        # Test other pipeline components
        from eq.pipeline.run_production_pipeline import run_pipeline
        from eq.pipeline.segmentation_pipeline import SegmentationPipeline

        # Use the imports so ruff doesn't remove them
        assert SegmentationPipeline is not None
        assert run_pipeline is not None
        
        print("✅ Segmentation pipeline import successful")
        print("✅ Production pipeline import successful")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utils_imports():
    """Test that utils module imports work correctly."""
    print("\nTesting utils module imports...")
    
    try:
        from eq.utils.backend_manager import BackendManager
        from eq.utils.hardware_detection import get_optimal_batch_size
        from eq.utils.logger import get_logger

        # Use the imports so ruff doesn't remove them
        assert BackendManager is not None
        assert get_optimal_batch_size is not None
        assert get_logger is not None
        
        print("✅ Utils imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Utils imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_imports():
    """Test that evaluation module imports work correctly."""
    print("\nTesting evaluation module imports...")
    
    try:
        from eq.evaluation.segmentation_metrics import dice_coefficient

        # Use the import so ruff doesn't remove it
        assert dice_coefficient is not None
        
        print("✅ Evaluation imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantification_imports():
    """Test that quantification module imports work correctly."""
    print("\nTesting quantification module imports...")
    
    try:
        from eq.quantification import quantify_endotheliosis
        print("✅ Quantification imports successful")
        
        # Test that it's a placeholder
        try:
            quantify_endotheliosis()
        except NotImplementedError:
            print("✅ Quantification placeholder works correctly")
        
        return True
    except Exception as e:
        print(f"❌ Quantification imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("IMPORT VERIFICATION TEST AFTER LEGACY CLEANUP")
    print("=" * 55)
    
    success = True
    success &= test_core_imports()
    success &= test_data_imports()
    success &= test_models_imports()
    success &= test_processing_imports()
    success &= test_pipeline_imports()
    success &= test_utils_imports()
    success &= test_evaluation_imports()
    success &= test_quantification_imports()
    
    print("\n" + "=" * 55)
    if success:
        print("🎉 ALL IMPORTS WORKING CORRECTLY!")
        print("✅ Legacy import cleanup successful")
        print("✅ All modules can be imported")
        print("✅ Core functionality verified")
        print("\nReady to test individual pipeline components!")
    else:
        print("❌ IMPORT ISSUES DETECTED")
        print("Some modules still have import problems...")
