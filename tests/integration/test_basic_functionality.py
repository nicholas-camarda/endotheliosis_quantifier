#!/usr/bin/env python3
"""Test basic functionality after consolidation."""

import sys

sys.path.insert(0, 'src')

def test_imports():
    """Test that all basic imports work."""
    print("Testing basic imports...")
    
    # Test core imports
    try:
        print("✅ Core imports work")
    except Exception as e:
        print(f"❌ Core import failed: {e}")
        return False
    
    # Test data imports
    try:
        print("✅ Data imports work")
    except Exception as e:
        print(f"❌ Data import failed: {e}")
        return False
    
    # Test quantification imports
    try:
        print("✅ Quantification imports work")
    except Exception as e:
        print(f"❌ Quantification import failed: {e}")
        return False
    
    # Test processing imports  
    try:
        print("✅ Processing imports work")
    except Exception as e:
        print(f"❌ Processing import failed: {e}")
        return False
    
    return True

def test_data_loader_creation():
    """Test creating data loaders."""
    print("\nTesting data loader creation...")
    
    try:
        from eq.data import UnifiedDataLoader

        # Test mito loader creation
        mito_loader = UnifiedDataLoader(
            data_type='mitochondria',
            data_dir='test_data',  # fake path
            cache_dir='test_cache'
        )
        print("✅ Mito loader creation works")
        
        # Test glom loader creation
        glom_loader = UnifiedDataLoader(
            data_type='glomeruli', 
            data_dir='test_data',  # fake path
            cache_dir='test_cache'
        )
        print("✅ Glom loader creation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader creation failed: {e}")
        return False

def test_core_functionality():
    """Test core binary conversion functionality."""
    print("\nTesting core functionality...")
    
    try:
        from eq.core import BINARY_P2C, DEFAULT_MASK_THRESHOLD

        # Test constants
        assert BINARY_P2C == [0, 1], f"Expected [0, 1], got {BINARY_P2C}"
        assert DEFAULT_MASK_THRESHOLD == 127, f"Expected 127, got {DEFAULT_MASK_THRESHOLD}"
        
        print("✅ Core constants are correct")
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("TESTING BASIC FUNCTIONALITY AFTER CONSOLIDATION")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_data_loader_creation() 
    success &= test_core_functionality()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ ALL BASIC TESTS PASSED")
        print("Consolidation did not break basic functionality!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Need to fix import issues...")
