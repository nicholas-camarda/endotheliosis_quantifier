import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def create_dummy_roi_data():
    """Create dummy ROI data for testing"""
    # Create dummy ROI images (256x256x3)
    rois = np.random.randint(0, 255, (10, 256, 256, 3), dtype=np.uint8)
    # Create dummy scores (0-3 scale)
    scores = np.random.uniform(0, 3, 10)
    return rois, scores

def test_feature_extractor_helper_functions():
    """Test key functions from feature_extractor_helper_functions.py"""
    # Import the functions we'll be testing
    import sys
    sys.path.insert(0, 'scripts/main')
    
    from eq.features.helpers import (load_pickled_data,
                                    preprocess_images_to_rois)

    # Test load_pickled_data
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        test_data = {'test': 'data', 'array': np.array([1, 2, 3])}
        pickle.dump(test_data, f)
        temp_path = f.name
    
    try:
        loaded_data = load_pickled_data(temp_path)
        assert loaded_data['test'] == 'data'
        assert np.array_equal(loaded_data['array'], np.array([1, 2, 3]))
    finally:
        os.unlink(temp_path)
    
    # Test preprocess_images_to_rois with dummy data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy image and mask directories
        images_dir = Path(temp_dir) / 'images' / 'T01'
        masks_dir = Path(temp_dir) / 'masks' / 'T01'
        output_dir = Path(temp_dir) / 'rois'
        
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        # Create dummy image and mask files
        import cv2
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        dummy_mask = np.zeros((256, 256), dtype=np.uint8)
        dummy_mask[50:200, 50:200] = 255  # Create a rectangular mask
        
        cv2.imwrite(str(images_dir / 'T01_Image0.jpg'), dummy_img)
        cv2.imwrite(str(masks_dir / 'T01_Image0_mask.jpg'), dummy_mask)
        
        # Test preprocess_images_to_rois
        rois = preprocess_images_to_rois(
            str(images_dir.parent), 
            str(masks_dir.parent), 
            str(output_dir),
            padding=5,
            size=256
        )
        
        # Should produce at least one ROI
        assert rois is not None
        assert len(rois) >= 1
        assert rois[0].shape == (256, 256, 3)

def test_quantify_endotheliosis():
    """Test key functions from 4_quantify_endotheliosis.py"""
    import sys
    sys.path.insert(0, 'scripts/main')
    
    from eq.pipeline.quantify_endotheliosis import (
        run_random_forest,
        load_pickled_data
    )
    
    # Test load_pickled_data (same as above)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        test_data = {'test': 'data', 'array': np.array([1, 2, 3])}
        pickle.dump(test_data, f)
        temp_path = f.name
    
    try:
        loaded_data = load_pickled_data(temp_path)
        assert loaded_data['test'] == 'data'
        assert np.array_equal(loaded_data['array'], np.array([1, 2, 3]))
    finally:
        os.unlink(temp_path)
    
    # Test run_random_forest with dummy data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy training data
        X_train = np.random.rand(100, 10)  # 100 samples, 10 features
        y_train = np.random.uniform(0, 3, 100)  # Scores 0-3
        X_val = np.random.rand(20, 10)  # 20 validation samples
        y_val = np.random.uniform(0, 3, 20)  # Validation scores
        
        # Test random forest training
        model = run_random_forest(
            X_train, y_train, X_val, y_val, 
            model_output_directory=temp_dir,
            n_estimators=10,  # Small number for testing
            n_cpu_jobs=1
        )
        
        # Check that model was trained
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Check that output file was created
        output_file = Path(temp_dir) / "predictions_with_confidence_intervals.csv"
        assert output_file.exists()
        
                # Check output file content
        df = pd.read_csv(output_file)
        assert 'Prediction' in df.columns
        assert "'True' Value" in df.columns
        assert 'Lower_CI' in df.columns
        assert 'Upper_CI' in df.columns
        assert len(df) == len(y_val)  # Should have predictions for all validation samples
