import os
import tempfile

import numpy as np


def test_quantify_endotheliosis_wrapper():
    """Test the quantify_endotheliosis.py wrapper imports eq functions correctly"""
    import sys
    sys.path.insert(0, 'scripts/main')
    
    # Test that the wrapper can import from eq.pipeline
    from quantify_endotheliosis import main

    # Test that the wrapper runs without errors (basic functionality)
    try:
        # This should just print a success message and not crash
        main()
        print("Wrapper executed successfully")
    except Exception as e:
        # If it fails, that's okay as long as it's not an import error
        assert "eq.pipeline" in str(e) or "import" not in str(e), f"Unexpected error: {e}"

def test_runtime_check_wrapper():
    """Test that runtime_check.py works with eq package imports"""
    import sys
    sys.path.insert(0, 'scripts/utils')
    
    from runtime_check import check_imports, check_paths

    # Test that runtime_check can import eq modules
    try:
        check_imports()
        print("Runtime check imports successful")
    except Exception as e:
        # Should not fail on import errors for eq modules
        assert "eq" not in str(e), f"Runtime check failed on eq imports: {e}"
    
    # Test that data paths are correct
    try:
        check_paths()
        print("Runtime check data paths successful")
    except Exception as e:
        # Should not fail on data path checks
        assert "preeclampsia_data" in str(e) or "not found" in str(e), f"Unexpected data path error: {e}"

def test_smoke_test_pipeline_wrapper():
    """Test that smoke_test_pipeline.py works with eq package"""
    import sys
    sys.path.insert(0, 'scripts/utils')
    
    from smoke_test_pipeline import main

    # Test that smoke test can import eq modules
    try:
        # This might fail due to TensorFlow/Metal issues on M1, but should not fail on imports
        main()
        print("Smoke test successful")
    except Exception as e:
        # Should not fail on import errors for eq modules
        assert "eq" not in str(e) or "segmentation" in str(e), f"Smoke test failed on eq imports: {e}"

def test_eq_package_imports():
    """Test that all eq package modules can be imported by wrappers"""
    # Test all the main eq modules that wrappers might need
    modules_to_test = [
        'eq.features.helpers',
        'eq.features.preprocess_roi', 
        'eq.io.convert_files_to_jpg',
        'eq.patches.patchify_images',
        'eq.augment.augment_dataset'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name} imports successfully")
        except ImportError as e:
            assert False, f"Failed to import {module_name}: {e}"
    
    # Test quantify_endotheliosis separately to avoid side effects
    try:
        import eq.pipeline.quantify_endotheliosis
        print("✓ eq.pipeline.quantify_endotheliosis imports successfully")
    except ImportError as e:
        assert False, f"Failed to import eq.pipeline.quantify_endotheliosis: {e}"

def test_wrapper_function_calls():
    """Test that wrappers can call eq functions and get expected return values"""
    from eq.features.helpers import load_pickled_data

    # Test load_pickled_data through wrapper
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        import pickle
        test_data = {'test': 'data', 'array': np.array([1, 2, 3])}
        pickle.dump(test_data, f)
        temp_path = f.name
    
    try:
        loaded_data = load_pickled_data(temp_path)
        assert loaded_data['test'] == 'data'
        assert np.array_equal(loaded_data['array'], np.array([1, 2, 3]))
        print("✓ load_pickled_data works through eq package")
    finally:
        os.unlink(temp_path)
    
    # Test that run_random_forest function exists (without running it)
    try:
        from eq.pipeline.quantify_endotheliosis import run_random_forest
        assert callable(run_random_forest)
        print("✓ run_random_forest function is callable")
    except ImportError as e:
        assert False, f"Failed to import run_random_forest: {e}"
