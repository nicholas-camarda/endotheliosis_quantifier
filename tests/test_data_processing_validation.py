#!/usr/bin/env python3
"""
Data Processing and Patching Validation Tests

This module tests the core data processing components to ensure they work correctly
with real medical image data before proceeding with FastAI v2 migration.

Tests cover:
1. Image loading and preprocessing
2. Mask generation and validation
3. Dynamic full-image loader validation
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from eq.core.constants import DEFAULT_IMAGE_SIZE, LARGE_IMAGE_SIZE
from eq.data_management.datablock_loader import build_segmentation_dls_dynamic_patching
from eq.processing.preprocessing import (
    normalize_image_array,
    prepare_image_for_inference,
    preprocess_image_for_model,
    resize_image_large,
    resize_image_standard,
)


class TestDataProcessingComponents:
    """Test core data processing components with real data."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup handled by pytest

    @pytest.fixture
    def sample_image_paths(self, temp_dir):
        """Create sample images for testing."""
        image_paths = []
        
        # Create test images of different sizes
        sizes = [(100, 100), (200, 200), (300, 300)]
        for i, size in enumerate(sizes):
            # Create a simple test image
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            img_path = Path(temp_dir) / f"test_image_{i}_{size[0]}x{size[1]}.jpg"
            cv2.imwrite(str(img_path), img)
            image_paths.append(img_path)
        
        return image_paths

    @pytest.fixture
    def sample_mask_paths(self, temp_dir):
        """Create sample masks for testing."""
        mask_paths = []
        
        # Create test masks of different sizes
        sizes = [(100, 100), (200, 200), (300, 300)]
        for i, size in enumerate(sizes):
            # Create a binary mask
            mask = np.random.randint(0, 2, size, dtype=np.uint8) * 255
            mask_path = Path(temp_dir) / f"test_mask_{i}_{size[0]}x{size[1]}.png"
            cv2.imwrite(str(mask_path), mask)
            mask_paths.append(mask_path)
        
        return mask_paths

    def test_image_loading_with_real_paths(self, sample_image_paths):
        """Test that image loading works with real file paths."""
        print(f"\nTesting image loading with {len(sample_image_paths)} sample images...")
        
        for img_path in sample_image_paths:
            # Test that file exists and can be read
            assert img_path.exists(), f"Image file {img_path} should exist"
            
            # Test OpenCV loading
            img_cv = cv2.imread(str(img_path))
            assert img_cv is not None, f"OpenCV should load {img_path}"
            assert img_cv.shape[2] == 3, f"Image should have 3 channels, got {img_cv.shape[2]}"
            
            print(f"✅ Successfully loaded {img_path.name} with shape {img_cv.shape}")

    def test_image_preprocessing_functions(self, sample_image_paths):
        """Test core preprocessing functions."""
        print("\nTesting image preprocessing functions...")
        
        for img_path in sample_image_paths:
            try:
                # Test resize_image_standard
                resized_std = resize_image_standard(str(img_path))
                assert resized_std.size == (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), (
                    f"Standard resize should be {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}, got {resized_std.size}"
                )
                
                # Test resize_image_large
                resized_large = resize_image_large(str(img_path))
                assert resized_large.size == (LARGE_IMAGE_SIZE, LARGE_IMAGE_SIZE), (
                    f"Large resize should be {LARGE_IMAGE_SIZE}x{LARGE_IMAGE_SIZE}, got {resized_large.size}"
                )
                
                # Test preprocess_image_for_model
                processed = preprocess_image_for_model(str(img_path), use_large_size=False)
                assert processed.size == (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), (
                    f"Model preprocessing should be {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}, got {processed.size}"
                )
                
                print(f"✅ Successfully preprocessed {img_path.name}")
                
            except Exception as e:
                pytest.fail(f"Failed to preprocess {img_path.name}: {e}")

    def test_mask_loading_and_validation(self, sample_mask_paths):
        """Test mask loading and validation."""
        print("\nTesting mask loading and validation...")
        
        for mask_path in sample_mask_paths:
            try:
                # Test OpenCV loading
                mask_cv = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                assert mask_cv is not None, f"OpenCV should load mask {mask_path}"
                
                # Test binary validation
                unique_vals = np.unique(mask_cv)
                assert len(unique_vals) <= 2, f"Mask should be binary, got {len(unique_vals)} values: {unique_vals}"
                
                # Test that values are 0 and 255
                expected_vals = {0, 255}
                actual_vals = set(unique_vals)
                assert actual_vals.issubset(expected_vals), f"Mask values should be 0 or 255, got {actual_vals}"
                
                print(f"✅ Successfully validated mask {mask_path.name}")
                
            except Exception as e:
                pytest.fail(f"Failed to validate mask {mask_path.name}: {e}")

    def test_data_loader_initialization(self, temp_dir):
        """Test that supported dynamic full-image data loader can be initialized."""
        print("\nTesting data loader initialization...")
        
        try:
            root = Path(temp_dir)
            images_dir = root / "images"
            masks_dir = root / "masks"
            images_dir.mkdir()
            masks_dir.mkdir()
            for i in range(8):
                image = np.zeros((96, 96, 3), dtype=np.uint8)
                image[20:60, 20:60, :] = 180
                cv2.imwrite(str(images_dir / f"sample_{i}.jpg"), image)
                mask = np.zeros((96, 96), dtype=np.uint8)
                mask[24:56, 24:56] = 255
                cv2.imwrite(str(masks_dir / f"sample_{i}_mask.png"), mask)

            dls = build_segmentation_dls_dynamic_patching(root, bs=4, num_workers=0, crop_size=64)
            
            assert dls is not None, "Data loader should be created"
            xb, yb = dls.one_batch()
            assert xb.shape[0] == yb.shape[0]
            
            print("✅ Successfully initialized SegmentationDataLoader")
            
        except Exception as e:
            pytest.fail(f"Failed to initialize data loader: {e}")

    def test_normalization_functions(self):
        """Test image normalization functions."""
        print("\nTesting normalization functions...")
        
        # Create test image array
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        try:
            # Test zero_one normalization
            normalized_zero_one = normalize_image_array(test_img, method='zero_one')
            assert np.min(normalized_zero_one) >= 0.0, "Zero-one normalization should be >= 0"
            assert np.max(normalized_zero_one) <= 1.0, "Zero-one normalization should be <= 1"
            
            # Test mean_std normalization
            normalized_mean_std = normalize_image_array(test_img, method='mean_std')
            assert abs(np.mean(normalized_mean_std)) < 0.1, "Mean-std normalization should have mean close to 0"
            assert abs(np.std(normalized_mean_std) - 1.0) < 0.1, "Mean-std normalization should have std close to 1"
            
            print("✅ Successfully tested normalization functions")
            
        except Exception as e:
            pytest.fail(f"Failed to test normalization: {e}")

    def test_inference_preparation(self, sample_image_paths):
        """Test image preparation for inference."""
        print("\nTesting inference preparation...")
        
        for img_path in sample_image_paths:
            try:
                # Test standard preprocessing
                processed_img, metadata = prepare_image_for_inference(str(img_path), use_large_preprocessing=False)
                
                assert processed_img is not None, "Processed image should not be None"
                assert metadata is not None, "Metadata should not be None"
                assert 'original_size' in metadata, "Metadata should contain original_size"
                assert 'processed_size' in metadata, "Metadata should contain processed_size"
                assert 'preprocessing_approach' in metadata, "Metadata should contain preprocessing_approach"
                
                print(f"✅ Successfully prepared {img_path.name} for inference")
                
            except Exception as e:
                pytest.fail(f"Failed to prepare {img_path.name} for inference: {e}")


class TestRealDataValidation:
    """Test with actual project data to ensure real-world compatibility."""

    def test_real_data_availability(self):
        """Test that real project data is available."""
        print("\nTesting real data availability...")
        
        # Check if project data exists using centralized path management
        from eq.utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        data_path = config_manager.global_config.data_path
        raw_data_dir = Path(data_path)
        
        if not raw_data_dir.exists():
            pytest.skip(f"Raw data directory not found: {raw_data_dir}")
        
        # Find the first available project directory with a 'data' subdirectory
        project_data_dir = None
        for project_dir in raw_data_dir.iterdir():
            if project_dir.is_dir():
                candidate_data_path = project_dir / "data"
                if candidate_data_path.exists():
                    project_data_dir = candidate_data_path
                    break
        
        if project_data_dir is None:
            pytest.skip(f"No project with 'data' subdirectory found in: {raw_data_dir}")
        
        assert project_data_dir.exists(), f"Project data directory should exist: {project_data_dir}"
        
        # Check for images (recursively search subdirectories)
        images_dir = project_data_dir / "images"
        if images_dir.exists():
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
                image_files.extend(list(images_dir.rglob(ext)))
            print(f"✅ Found {len(image_files)} image files in project data")
        else:
            print("⚠️  No images directory found in project data")
        
        # Check for masks (recursively search subdirectories)
        masks_dir = project_data_dir / "masks"
        if masks_dir.exists():
            mask_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
                mask_files.extend(list(masks_dir.rglob(ext)))
            print(f"✅ Found {len(mask_files)} mask files in project data")
        else:
            print("⚠️  No masks directory found in project data")

    def test_real_data_loading(self):
        """Test loading actual project data."""
        print("\nTesting real data loading...")
        
        # Use centralized path management and discover actual project structure
        from eq.utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        data_path = config_manager.global_config.data_path
        raw_data_dir = Path(data_path)
        
        if not raw_data_dir.exists():
            pytest.skip(f"Raw data directory not found: {raw_data_dir}")
        
        # Find the first available project directory with a 'data' subdirectory
        project_data_dir = None
        for project_dir in raw_data_dir.iterdir():
            if project_dir.is_dir():
                candidate_data_path = project_dir / "data"
                if candidate_data_path.exists():
                    project_data_dir = candidate_data_path
                    break
        
        if project_data_dir is None:
            pytest.skip(f"No project with 'data' subdirectory found in: {raw_data_dir}")
        
        # Try to load a few real images
        images_dir = project_data_dir / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg"))[:3] + list(images_dir.glob("*.png"))[:3] + list(images_dir.glob("*.tif"))[:3]
            
            for img_file in image_files[:3]:  # Test first 3 images
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        print(f"✅ Successfully loaded real image: {img_file.name} (shape: {img.shape})")
                    else:
                        print(f"⚠️  Failed to load real image: {img_file.name}")
                except Exception as e:
                    print(f"❌ Error loading real image {img_file.name}: {e}")
        else:
            print("⚠️  No images directory found, skipping real data loading test")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
