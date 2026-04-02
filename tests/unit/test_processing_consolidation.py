#!/usr/bin/env python3
"""
Tests for consolidated processing module functionality.

This test suite verifies that all processing functions work correctly
after consolidation from scattered locations into the unified processing module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

from eq.processing import (
    # File conversion
    convert_tif_to_jpg,
    # Image preprocessing
    resize_image_standard,
    resize_image_large,
    preprocess_image_for_model,
    normalize_image_array,
    # Image patchification
    patchify_image_dir,
    patchify_image_and_mask_dirs,
    # Mitochondria patch creation
    create_patches_from_image,
    create_mitochondria_patches,
)


class TestFileConversion:
    """Test file conversion functionality."""
    
    def test_convert_tif_to_jpg_creates_output_directory(self):
        """Test that convert_tif_to_jpg creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / 'input'
            output_dir = Path(temp_dir) / 'output'
            
            # Create input directory with a test TIF file
            input_dir.mkdir()
            test_image = Image.new('RGB', (100, 100), color='red')
            test_tif_path = input_dir / 'test.tif'
            test_image.save(test_tif_path)
            
            # Convert
            convert_tif_to_jpg(str(input_dir), str(output_dir))
            
            # Verify output directory was created
            assert output_dir.exists()
            assert output_dir.is_dir()
    
    def test_convert_tif_to_jpg_converts_files_correctly(self):
        """Test that TIF files are converted to JPG with correct naming."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / 'input'
            output_dir = Path(temp_dir) / 'output'
            
            # Create input directory with test TIF file
            input_dir.mkdir()
            test_image = Image.new('RGB', (100, 100), color='blue')
            test_tif_path = input_dir / 'sample.tif'
            test_image.save(test_tif_path)
            
            # Convert
            convert_tif_to_jpg(str(input_dir), str(output_dir))
            
            # Verify conversion
            expected_jpg = output_dir / 'output_sample.jpg'
            assert expected_jpg.exists()
            
            # Verify it's a valid JPG
            converted_image = Image.open(expected_jpg)
            assert converted_image.format == 'JPEG'


class TestImagePreprocessing:
    """Test image preprocessing functionality."""
    
    def test_resize_image_standard_with_path(self):
        """Test resize_image_standard with file path input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            test_image = Image.new('RGB', (200, 200), color='green')
            test_path = Path(temp_dir) / 'test.png'
            test_image.save(test_path)
            
            # Resize
            result = resize_image_standard(str(test_path))
            
            # Verify result
            assert result.size == (224, 224)  # Default size
    
    def test_resize_image_large_uses_large_size(self):
        """Test resize_image_large uses large size (512px)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            test_image = Image.new('RGB', (100, 100), color='yellow')
            test_path = Path(temp_dir) / 'test.png'
            test_image.save(test_path)
            
            # Resize
            result = resize_image_large(str(test_path))
            
            # Verify result
            assert result.size == (512, 512)  # Large size
    
    def test_preprocess_image_for_model_standard_size(self):
        """Test preprocess_image_for_model with standard size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            test_image = Image.new('RGB', (300, 300), color='purple')
            test_path = Path(temp_dir) / 'test.png'
            test_image.save(test_path)
            
            # Preprocess
            result = preprocess_image_for_model(str(test_path), use_large_size=False)
            
            # Verify result
            assert result.size == (224, 224)  # Standard size
    
    def test_normalize_image_array_zero_one(self):
        """Test normalize_image_array with zero_one method."""
        # Create test array
        test_array = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
        
        # Normalize
        result = normalize_image_array(test_array, method='zero_one')
        
        # Verify result
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.dtype in [np.float32, np.float64]  # Accept either precision


class TestImagePatchification:
    """Test image patchification functionality."""
    
    def test_patchify_image_dir_creates_patches(self):
        """Test patchify_image_dir creates image patches correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / 'input'
            output_dir = Path(temp_dir) / 'output'
            
            # Create input directory with test image
            input_dir.mkdir()
            test_image = Image.new('RGB', (256, 256), color='orange')
            test_path = input_dir / 'test.png'
            test_image.save(test_path)
            
            # Patchify
            patchify_image_dir(128, str(input_dir), str(output_dir))
            
            # Verify patches were created
            patch_files = list(output_dir.glob('*.jpg'))
            assert len(patch_files) == 4  # 2x2 patches of 128x128 from 256x256 image
    
    def test_patchify_image_and_mask_dirs_with_masks(self):
        """Test patchify_image_and_mask_dirs creates both image and mask patches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / 'images'
            mask_dir = Path(temp_dir) / 'masks'
            output_dir = Path(temp_dir) / 'output'
            
            # Create directories
            image_dir.mkdir()
            mask_dir.mkdir()
            
            # Create test image and mask
            test_image = Image.new('RGB', (256, 256), color='cyan')
            test_mask = Image.new('L', (256, 256), color=128)
            
            image_path = image_dir / 'test.png'
            mask_path = mask_dir / 'test_mask.png'
            
            test_image.save(image_path)
            test_mask.save(mask_path)
            
            # Patchify
            patchify_image_and_mask_dirs(128, str(image_dir), str(mask_dir), str(output_dir))
            
            # Verify both image and mask patches were created
            image_patches = list(output_dir.glob('test_*.jpg'))
            mask_patches = list(output_dir.glob('test_*_mask.jpg'))
            
            # Filter out mask patches from image_patches list
            image_patches = [p for p in image_patches if not p.name.endswith('_mask.jpg')]
            
            assert len(image_patches) == 4  # 2x2 patches from 256x256 image
            assert len(mask_patches) == 4   # 2x2 mask patches


class TestMitochondriaPatchCreation:
    """Test mitochondria-specific patch creation functionality."""
    
    def test_create_patches_from_image_returns_patches(self):
        """Test create_patches_from_image returns image and mask patches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image and mask
            test_image = Image.new('L', (512, 512), color=100)
            test_mask = Image.new('L', (512, 512), color=50)
            
            image_path = Path(temp_dir) / 'test_image.png'
            mask_path = Path(temp_dir) / 'test_mask.png'
            
            test_image.save(image_path)
            test_mask.save(mask_path)
            
            # Create patches
            image_patches, mask_patches = create_patches_from_image(
                image_path, mask_path, patch_size=256, overlap=0.1
            )
            
            # Verify patches were created
            assert len(image_patches) > 0
            assert len(mask_patches) > 0
            assert len(image_patches) == len(mask_patches)
            
            # Verify patch shapes
            for patch in image_patches:
                assert patch.shape == (256, 256)
            for patch in mask_patches:
                assert patch.shape == (256, 256)
    
    def test_create_mitochondria_patches_creates_directory_structure(self):
        """Test create_mitochondria_patches creates proper directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / 'mitochondria_data'
            
            # Create input structure
            training_images = input_dir / 'training' / 'images'
            training_masks = input_dir / 'training' / 'masks'
            testing_images = input_dir / 'testing' / 'images'
            testing_masks = input_dir / 'testing' / 'masks'
            
            for dir_path in [training_images, training_masks, testing_images, testing_masks]:
                dir_path.mkdir(parents=True)
            
            # Create test files
            test_image = Image.new('L', (512, 512), color=100)
            test_mask = Image.new('L', (512, 512), color=50)
            
            (training_images / 'train1.tif').write_bytes(test_image.tobytes())
            (training_masks / 'train1_mask.tif').write_bytes(test_mask.tobytes())
            (testing_images / 'test1.tif').write_bytes(test_image.tobytes())
            (testing_masks / 'test1_mask.tif').write_bytes(test_mask.tobytes())
            
            # Create patches
            create_mitochondria_patches(str(input_dir))
            
            # Verify output structure was created
            expected_dirs = [
                input_dir / 'training' / 'image_patches',
                input_dir / 'training' / 'mask_patches',
                input_dir / 'testing' / 'image_patches',
                input_dir / 'testing' / 'mask_patches',
            ]
            
            for dir_path in expected_dirs:
                assert dir_path.exists()


class TestProcessingModuleImports:
    """Test that all processing functions are properly imported."""
    
    def test_processing_module_imports_all_functions(self):
        """Test that the processing module exports all expected functions."""
        from eq.processing import __all__
        
        expected_functions = [
            # File conversion
            'convert_tif_to_jpg',
            # Image preprocessing
            'resize_image_standard',
            'resize_image_large',
            'preprocess_image_for_model',
            'normalize_image_array',
            # Image patchification
            'patchify_image_dir',
            'patchify_image_and_mask_dirs',
            # Mitochondria patch creation
            'create_patches_from_image',
            'create_mitochondria_patches',
        ]
        
        for func_name in expected_functions:
            assert func_name in __all__, f"Function {func_name} not exported from processing module"
    
    def test_processing_module_has_unified_api(self):
        """Test that processing module provides unified API."""
        from eq.processing import (
            convert_tif_to_jpg,
            patchify_image_dir,
            patchify_image_and_mask_dirs,
        )
        
        # Verify functions are callable
        assert callable(convert_tif_to_jpg)
        assert callable(patchify_image_dir)
        assert callable(patchify_image_and_mask_dirs)


if __name__ == '__main__':
    pytest.main([__file__])
