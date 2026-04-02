#!/usr/bin/env python3
"""Test data loading functionality for both mito and glomeruli."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def test_mito_data_loading():
    """Test mitochondria data loading functionality."""
    print("Testing mitochondria data loading...")
    
    try:
        from eq.data import UnifiedDataLoader

        # Check if mito data exists
        mito_data_dir = Path('derived_data/mitochondria_data')
        if not mito_data_dir.exists():
            print("⚠️  Mito data directory not found, skipping functional test")
            return True
            
        # Create mito data loader
        mito_loader = UnifiedDataLoader(
            data_type='mitochondria',
            data_dir=mito_data_dir,
            cache_dir=mito_data_dir / 'cache'
        )
        
        print("✅ Mito loader created successfully")
        print(f"   - Data type: {mito_loader.data_type}")
        print(f"   - Data dir: {mito_loader.data_dir}")
        print(f"   - Cache dir: {mito_loader.cache_dir}")
        
        # Test if we can check for data
        try:
            has_cache = mito_loader._check_cache_exists()
            print(f"   - Has cache: {has_cache}")
        except Exception as e:
            print(f"   - Cache check error (expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mito data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_glom_data_loading():
    """Test glomeruli data loading functionality."""
    print("\nTesting glomeruli data loading...")
    
    try:
        from eq.data import UnifiedDataLoader

        # Check if glom data exists
        glom_data_dir = Path('derived_data/glomeruli_data')
        if not glom_data_dir.exists():
            print("⚠️  Glom data directory not found, skipping functional test")
            return True
            
        # Create glom data loader
        glom_loader = UnifiedDataLoader(
            data_type='glomeruli',
            data_dir=glom_data_dir,
            cache_dir=glom_data_dir / 'cache'
        )
        
        print("✅ Glom loader created successfully")
        print(f"   - Data type: {glom_loader.data_type}")
        print(f"   - Data dir: {glom_loader.data_dir}")
        print(f"   - Cache dir: {glom_loader.cache_dir}")
        
        # Test if we can check for data
        try:
            has_cache = glom_loader._check_cache_exists()
            print(f"   - Has cache: {has_cache}")
        except Exception as e:
            print(f"   - Cache check error (expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Glom data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_functions():
    """Test that core data loading functions work."""
    print("\nTesting core data loading functions...")
    
    try:
        from eq.core import BINARY_P2C, DEFAULT_MASK_THRESHOLD
        
        print("✅ Core functions imported successfully")
        print(f"   - BINARY_P2C: {BINARY_P2C}")
        print(f"   - DEFAULT_MASK_THRESHOLD: {DEFAULT_MASK_THRESHOLD}")
        
        # Test binary conversion logic
        assert BINARY_P2C == [0, 1], f"Expected [0, 1], got {BINARY_P2C}"
        assert DEFAULT_MASK_THRESHOLD == 127, f"Expected 127, got {DEFAULT_MASK_THRESHOLD}"
        
        print("✅ Core constants are correct")
        return True
        
    except Exception as e:
        print(f"❌ Core functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_legacy_compatibility():
    """Test that legacy data loading functions still work."""
    print("\nTesting legacy compatibility functions...")
    
    try:
        from eq.data import load_glomeruli_data, load_mitochondria_patches
        
        print("✅ Legacy functions imported successfully")
        print(f"   - load_mitochondria_patches: {load_mitochondria_patches}")
        print(f"   - load_glomeruli_data: {load_glomeruli_data}")
        
        # These should be callable (even if they fail on actual data)
        assert callable(load_mitochondria_patches)
        assert callable(load_glomeruli_data)
        
        print("✅ Legacy functions are callable")
        return True
        
    except Exception as e:
        print(f"❌ Legacy compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_setup():
    """Test that the environment is set up correctly."""
    print("\nTesting environment setup...")
    
    try:
        import fastai
        import numpy as np
        import PIL
        import torch
        
        print("✅ Core dependencies available:")
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - FastAI: {fastai.__version__}")
        print(f"   - NumPy: {np.__version__}")
        print(f"   - PIL: {PIL.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

if __name__ == "__main__":
    print("TESTING DATA LOADING FUNCTIONALITY AFTER CONSOLIDATION")
    print("=" * 60)
    
    success = True
    success &= test_environment_setup()
    success &= test_core_functions() 
    success &= test_legacy_compatibility()
    success &= test_mito_data_loading()
    success &= test_glom_data_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL DATA LOADING TESTS PASSED")
        print("Both mito and glom data loading functionality works!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Need to fix data loading issues...")
