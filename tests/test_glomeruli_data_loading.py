#!/usr/bin/env python3
"""
Tests for glomeruli data loading functionality.

This module tests the existing data loading functions used by the glomeruli
transfer learning pipeline, including annotation loading and score extraction.
"""

import json
import os
import pickle
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from eq.features.data_loader import (
    Annotation,
    find_image_path,
    get_image_size,
    get_scores_from_annotations,
    load_annotations_from_json,
)


class TestAnnotation:
    """Test the Annotation class."""
    
    def test_annotation_creation(self):
        """Test creating an Annotation object."""
        annotation = Annotation("test_image.jpg", [1, 2, 3], 0.5)
        
        assert annotation.image_name == "test_image.jpg"
        assert annotation.rle_mask == [1, 2, 3]
        assert annotation.score == 0.5
    
    def test_annotation_without_score(self):
        """Test creating an Annotation object without a score."""
        annotation = Annotation("test_image.jpg", [1, 2, 3], None)
        
        assert annotation.image_name == "test_image.jpg"
        assert annotation.rle_mask == [1, 2, 3]
        assert annotation.score is None


class TestLoadAnnotationsFromJson:
    """Test loading annotations from JSON files."""
    
    def test_load_annotations_valid_json(self):
        """Test loading annotations from a valid JSON file."""
        # Create test JSON data
        test_data = [
            {
                "file_upload": "test-001.jpg",
                "annotations": [
                    {
                        "result": [
                            {
                                "value": {
                                    "rle": [1, 2, 3, 4],
                                    "choices": ["0.5"]
                                }
                            }
                        ]
                    }
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            annotations = load_annotations_from_json(temp_file)
            
            assert len(annotations) == 1
            assert annotations[0].image_name == "001.jpg"  # After split
            assert annotations[0].rle_mask == [1, 2, 3, 4]
            assert annotations[0].score == 0.5
        finally:
            os.unlink(temp_file)
    
    def test_load_annotations_no_score(self):
        """Test loading annotations without scores."""
        test_data = [
            {
                "file_upload": "test-002.jpg",
                "annotations": [
                    {
                        "result": [
                            {
                                "value": {
                                    "rle": [5, 6, 7, 8]
                                }
                            }
                        ]
                    }
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            annotations = load_annotations_from_json(temp_file)
            
            assert len(annotations) == 1
            assert annotations[0].image_name == "002.jpg"
            assert annotations[0].rle_mask == [5, 6, 7, 8]
            assert annotations[0].score is None
        finally:
            os.unlink(temp_file)
    
    def test_load_annotations_multiple_annotations(self):
        """Test loading annotations with multiple annotation entries."""
        test_data = [
            {
                "file_upload": "test-003.jpg",
                "annotations": [
                    {
                        "result": [
                            {
                                "value": {
                                    "rle": [1, 2, 3]
                                }
                            }
                        ]
                    },
                    {
                        "result": [
                            {
                                "value": {
                                    "choices": ["0.7"]
                                }
                            }
                        ]
                    }
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            annotations = load_annotations_from_json(temp_file)
            
            assert len(annotations) == 1
            assert annotations[0].image_name == "003.jpg"
            assert annotations[0].rle_mask == [1, 2, 3]
            assert annotations[0].score == 0.7
        finally:
            os.unlink(temp_file)
    
    def test_load_annotations_empty_file(self):
        """Test loading annotations from an empty JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            temp_file = f.name
        
        try:
            annotations = load_annotations_from_json(temp_file)
            assert len(annotations) == 0
        finally:
            os.unlink(temp_file)
    
    def test_load_annotations_file_not_found(self):
        """Test loading annotations from a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_annotations_from_json("nonexistent_file.json")


class TestGetScoresFromAnnotations:
    """Test extracting scores from annotations."""
    
    def test_get_scores_from_annotations(self):
        """Test extracting scores from a list of annotations."""
        # Create test annotations
        annotations = [
            Annotation("image1.jpg", [1, 2, 3], 0.5),
            Annotation("image2.jpg", [4, 5, 6], 0.7),
            Annotation("image3.jpg", [7, 8, 9], None)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            scores = get_scores_from_annotations(annotations, temp_dir)
            
            # Check scores
            assert scores["image1"] == 0.5
            assert scores["image2"] == 0.7
            assert scores["image3"] is None  # Actual behavior: None scores remain None
            
            # Check that scores.pickle was created
            scores_file = os.path.join(temp_dir, 'scores.pickle')
            assert os.path.exists(scores_file)
            
            # Verify pickle content
            with open(scores_file, 'rb') as f:
                pickled_scores = pickle.load(f)
                assert pickled_scores == scores
    
    def test_get_scores_sorted(self):
        """Test that scores are returned in sorted order."""
        annotations = [
            Annotation("zebra.jpg", [1, 2, 3], 0.9),
            Annotation("apple.jpg", [4, 5, 6], 0.3),
            Annotation("banana.jpg", [7, 8, 9], 0.6)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            scores = get_scores_from_annotations(annotations, temp_dir)
            
            # Check that keys are sorted alphabetically
            keys = list(scores.keys())
            assert keys == ["apple", "banana", "zebra"]
            
            # Check values
            assert scores["apple"] == 0.3
            assert scores["banana"] == 0.6
            assert scores["zebra"] == 0.9


class TestFindImagePath:
    """Test finding image paths in directory structure."""
    
    def test_find_image_path_existing(self):
        """Test finding an existing image file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test image file
            test_image = os.path.join(temp_dir, "test_image.jpg")
            with open(test_image, 'w') as f:
                f.write("fake image data")
            
            # Test finding the image
            found_path = find_image_path("test_image.jpg", temp_dir)
            assert found_path == test_image
    
    def test_find_image_path_nested(self):
        """Test finding an image in a nested directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory structure
            nested_dir = os.path.join(temp_dir, "nested", "subdir")
            os.makedirs(nested_dir, exist_ok=True)
            
            # Create a test image file in nested directory
            test_image = os.path.join(nested_dir, "nested_image.jpg")
            with open(test_image, 'w') as f:
                f.write("fake image data")
            
            # Test finding the image
            found_path = find_image_path("nested_image.jpg", temp_dir)
            assert found_path == test_image
    
    def test_find_image_path_not_found(self):
        """Test finding a non-existent image file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            found_path = find_image_path("nonexistent.jpg", temp_dir)
            assert found_path is None


class TestGetImageSize:
    """Test getting image dimensions from annotation."""
    
    @patch('cv2.imread')
    def test_get_image_size_success(self, mock_imread):
        """Test successfully getting image dimensions."""
        # Mock cv2.imread to return a test image
        mock_imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        annotation = Annotation("test_image.jpg", [1, 2, 3], 0.5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock image file
            test_image = os.path.join(temp_dir, "test_image.jpg")
            with open(test_image, 'w') as f:
                f.write("fake image data")
            
            # Mock find_image_path to return our test file
            with patch('eq.features.data_loader.find_image_path', return_value=test_image):
                height, width = get_image_size(annotation, temp_dir)
                
                assert height == 480
                assert width == 640
    
    @patch('cv2.imread')
    def test_get_image_size_image_not_found(self, mock_imread):
        """Test getting image size when image file is not found."""
        annotation = Annotation("nonexistent.jpg", [1, 2, 3], 0.5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock find_image_path to return None (file not found)
            with patch('eq.features.data_loader.find_image_path', return_value=None):
                height, width = get_image_size(annotation, temp_dir)
                
                assert height == 0
                assert width == 0
    
    @patch('cv2.imread')
    def test_get_image_size_cv2_error(self, mock_imread):
        """Test getting image size when cv2.imread fails."""
        # Mock cv2.imread to return None (simulating error)
        mock_imread.return_value = None
        
        annotation = Annotation("test_image.jpg", [1, 2, 3], 0.5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock image file
            test_image = os.path.join(temp_dir, "test_image.jpg")
            with open(test_image, 'w') as f:
                f.write("fake image data")
            
            # Mock find_image_path to return our test file
            with patch('eq.features.data_loader.find_image_path', return_value=test_image):
                # The actual implementation will fail when cv2.imread returns None
                # because np.array(None) creates an empty array
                with pytest.raises(ValueError, match="not enough values to unpack"):
                    height, width = get_image_size(annotation, temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
