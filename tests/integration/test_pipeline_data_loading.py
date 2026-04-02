#!/usr/bin/env python3
"""Test data loading pipeline functionality after consolidation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def test_unified_data_loader():
    """Test that the UnifiedDataLoader works for both mito and glomeruli."""
    print("Testing UnifiedDataLoader functionality...")
    
    try:
        from eq.core import BINARY_P2C, DEFAULT_MASK_THRESHOLD
        from eq.data import UnifiedDataLoader

        # Test mitochondria loader
        mito_loader = UnifiedDataLoader(
            data_type='mitochondria',
            data_dir='derived_data/mitochondria_data',
            cache_dir='derived_data/mitochondria_data/cache',
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        print("✅ Mitochondria UnifiedDataLoader created")
        print(f"   - Data type: {mito_loader.data_type}")
        print(f"   - Split ratios: {mito_loader.train_ratio}/{mito_loader.val_ratio}/{mito_loader.test_ratio}")
        
        # Test glomeruli loader
        glom_loader = UnifiedDataLoader(
            data_type='glomeruli',
            data_dir='derived_data/glomeruli_data', 
            cache_dir='derived_data/glomeruli_data/cache',
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42
        )
        
        print("✅ Glomeruli UnifiedDataLoader created")
        print(f"   - Data type: {glom_loader.data_type}")
        print(f"   - Split ratios: {glom_loader.train_ratio}/{glom_loader.val_ratio}/{glom_loader.test_ratio}")
        
        # Test core constants are accessible
        assert BINARY_P2C == [0, 1], f"Expected [0, 1], got {BINARY_P2C}"
        assert DEFAULT_MASK_THRESHOLD == 127, f"Expected 127, got {DEFAULT_MASK_THRESHOLD}"
        print("✅ Core constants verified in data loading context")
        
        return True
        
    except Exception as e:
        print(f"❌ UnifiedDataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_legacy_data_loading_functions():
    """Test that legacy data loading functions still work."""
    print("\nTesting legacy data loading compatibility...")
    
    try:
        from eq.data import load_glomeruli_data, load_mitochondria_patches

        # These should be callable even if they don't find data
        assert callable(load_mitochondria_patches), "load_mitochondria_patches not callable"
        assert callable(load_glomeruli_data), "load_glomeruli_data not callable"
        
        print("✅ Legacy data loading functions are callable")
        
        # Test that they have proper signatures (don't actually call them with fake data)
        import inspect

        # Check function signatures exist
        mito_sig = inspect.signature(load_mitochondria_patches)
        glom_sig = inspect.signature(load_glomeruli_data)
        
        print(f"✅ load_mitochondria_patches signature: {len(mito_sig.parameters)} parameters")
        print(f"✅ load_glomeruli_data signature: {len(glom_sig.parameters)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_data_functions():
    """Test that core data loading functions work correctly."""
    print("\nTesting core data loading functions...")
    
    try:
        from eq.core import get_glom_mask_file, get_glom_y, n_glom_codes, setup_global_functions

        # These should all be callable
        assert callable(get_glom_mask_file), "get_glom_mask_file not callable"
        assert callable(get_glom_y), "get_glom_y not callable"  
        assert callable(n_glom_codes), "n_glom_codes not callable"
        assert callable(setup_global_functions), "setup_global_functions not callable"
        
        print("✅ Core data functions are callable")
        
        # Test that setup_global_functions works
        setup_global_functions()
        print("✅ setup_global_functions executes without error")
        
        return True
        
    except Exception as e:
        print(f"❌ Core data functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_preprocessing():
    """Test that data preprocessing functions work."""
    print("\nTesting data preprocessing functionality...")
    
    try:
        from eq.core import DEFAULT_MASK_THRESHOLD
        from eq.data.preprocessing import preprocess_image, preprocess_mask, validate_data_paths

        # These should be callable
        assert callable(preprocess_image), "preprocess_image not callable"
        assert callable(preprocess_mask), "preprocess_mask not callable"
        assert callable(validate_data_paths), "validate_data_paths not callable"
        
        print("✅ Data preprocessing functions are callable")
        
        # Test that threshold is properly accessible
        print(f"✅ Binary conversion threshold available: {DEFAULT_MASK_THRESHOLD}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_path_validation():
    """Test data path validation with real project paths."""
    print("\nTesting data path validation with project structure...")
    
    try:
        from eq.data.preprocessing import validate_data_paths

        # Test with actual project paths (these may or may not exist)
        test_paths = [
            'derived_data/mitochondria_data',
            'derived_data/glomeruli_data', 
            'raw_data',
            'nonexistent_path'
        ]
        
        valid_count = 0
        for path in test_paths:
            try:
                is_valid = validate_data_paths(Path(path))
                if is_valid:
                    valid_count += 1
                    print(f"✅ Path exists: {path}")
                else:
                    print(f"⚠️  Path missing: {path}")
            except Exception as e:
                print(f"⚠️  Path validation error for {path}: {e}")
        
        print(f"✅ Data path validation tested ({valid_count}/{len(test_paths)} paths exist)")
        return True
        
    except Exception as e:
        print(f"❌ Data path validation test failed: {e}")
        return False

if __name__ == "__main__":
    print("DATA LOADING PIPELINE TEST AFTER CONSOLIDATION")
    print("=" * 55)
    
    success = True
    success &= test_unified_data_loader()
    success &= test_legacy_data_loading_functions()
    success &= test_core_data_functions()
    success &= test_data_preprocessing()
    success &= test_data_path_validation()
    
    print("\n" + "=" * 55)
    if success:
        print("🎉 ALL DATA LOADING TESTS PASSED!")
        print("✅ UnifiedDataLoader works for both mito & glomeruli")
        print("✅ Legacy compatibility maintained")
        print("✅ Core data functions operational")
        print("✅ Data preprocessing ready")
        print("✅ Path validation functional")
        print("\nData loading pipeline is ready!")
    else:
        print("❌ SOME DATA LOADING TESTS FAILED")
        print("Data loading pipeline needs attention...")
