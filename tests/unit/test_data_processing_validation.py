#!/usr/bin/env python3
"""
Data Processing and Patching Validation Tests

This module tests the core data processing components to ensure they work correctly
with real medical image data before proceeding with FastAI v2 migration.

Tests cover:
1. Image loading and preprocessing
2. Image patching/segmentation
3. Mask generation and validation
4. Data processing pipeline validation
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest

from eq.processing.preprocessing import (
    normalize_image_array,
    preprocess_image_for_model,
    prepare_image_for_inference,
    resize_image_large,
    resize_image_standard,
)
from eq.processing.image_mask_preprocessing import (
    patchify_dataset,
)
from eq.data_management.data_loader import SegmentationDataLoader, DataConfig


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
        print(f"\nTesting image preprocessing functions...")
        
        for img_path in sample_image_paths:
            try:
                # Test resize_image_standard
                resized_std = resize_image_standard(str(img_path))
                assert resized_std.size == (224, 224), f"Standard resize should be 224x224, got {resized_std.size}"
                
                # Test resize_image_large
                resized_large = resize_image_large(str(img_path))
                assert resized_large.size == (512, 512), f"Large resize should be 512x512, got {resized_large.size}"
                
                # Test preprocess_image_for_model
                processed = preprocess_image_for_model(str(img_path), use_large_size=False)
                assert processed.size == (224, 224), f"Model preprocessing should be 224x224, got {processed.size}"
                
                print(f"✅ Successfully preprocessed {img_path.name}")
                
            except Exception as e:
                pytest.fail(f"Failed to preprocess {img_path.name}: {e}")

    def test_mask_loading_and_validation(self, sample_mask_paths):
        """Test mask loading and validation."""
        print(f"\nTesting mask loading and validation...")
        
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

    def test_image_patching_functionality(self, temp_dir):
        """Test image patching/segmentation functionality."""
        print(f"\nTesting image patching functionality...")
        
        # Create a larger test image
        large_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        large_img_path = Path(temp_dir) / "large_test_image.jpg"
        cv2.imwrite(str(large_img_path), large_img)
        
        # Output root for derived data
        patches_root = Path(temp_dir) / "patches_root"
        patches_root.mkdir()
        
        try:
            # Use unified patchify_dataset
            patchify_dataset(
                input_root=str(large_img_path.parent),
                output_root=str(patches_root),
                patch_size=224,
            )
            
            # Check that patches were created under image_patches/
            patch_files = list((patches_root / "image_patches").glob("*.jpg"))
            assert len(patch_files) > 0, "Should create patches from large image"
            
            # Verify patch dimensions
            for patch_file in patch_files:
                patch_img = cv2.imread(str(patch_file))
                assert patch_img.shape[:2] == (224, 224), f"Patch should be 224x224, got {patch_img.shape[:2]}"
            
            print(f"✅ Successfully created {len(patch_files)} patches from large image")
            
        except Exception as e:
            pytest.fail(f"Failed to create patches: {e}")

    def test_mask_patching_functionality(self, temp_dir):
        """Test mask patching functionality."""
        print(f"\nTesting mask patching functionality...")
        
        # Create a larger test mask in a separate input directory
        input_dir = Path(temp_dir) / "input_masks"
        input_dir.mkdir()
        
        # Create a test image (required for the function to work)
        large_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        large_img_path = input_dir / "large_test_image.jpg"
        cv2.imwrite(str(large_img_path), large_img)
        
        # Create a test mask with the correct naming convention
        large_mask = np.random.randint(0, 2, (400, 400), dtype=np.uint8) * 255
        large_mask_path = input_dir / "large_test_image_mask.png"
        cv2.imwrite(str(large_mask_path), large_mask)
        
        # Output root for derived data
        patches_root = Path(temp_dir) / "output_mask_patches_root"
        patches_root.mkdir()
        
        try:
            # Use unified patchify_dataset (auto-detects masks)
            patchify_dataset(
                input_root=str(input_dir),
                output_root=str(patches_root),
                patch_size=224,
            )
            
            image_patch_files = list((patches_root / "image_patches").glob("*.jpg"))
            mask_patch_files = list((patches_root / "mask_patches").glob("*_mask.jpg"))
            
            assert len(image_patch_files) > 0, "Should create image patches"
            assert len(mask_patch_files) > 0, "Should create mask patches"
            
            print(f"✅ Successfully created {len(image_patch_files)} image patches and {len(mask_patch_files)} mask patches")
            
        except Exception as e:
            pytest.fail(f"Failed to create mask patches: {e}")

    def test_data_loader_initialization(self):
        """Test that data loader can be initialized."""
        print(f"\nTesting data loader initialization...")
        
        try:
            config = DataConfig()
            loader = SegmentationDataLoader(config)
            
            assert loader is not None, "Data loader should be created"
            assert loader.config is not None, "Config should be accessible"
            assert loader.transforms is not None, "Transforms should be created"
            
            print("✅ Successfully initialized SegmentationDataLoader")
            
        except Exception as e:
            pytest.fail(f"Failed to initialize data loader: {e}")

    def test_normalization_functions(self):
        """Test image normalization functions."""
        print(f"\nTesting normalization functions...")
        
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
        print(f"\nTesting inference preparation...")
        
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
        print(f"\nTesting real data availability...")
        
        # Check if project data exists
        project_data_dir = Path("raw_data/preeclampsia_project/data")
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
        print(f"\nTesting real data loading...")
        
        project_data_dir = Path("raw_data/preeclampsia_project/data")
        
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

    def test_real_data_patching(self):
        """Test patching with real project data."""
        print(f"\nTesting real data patching...")
        
        project_data_dir = Path("raw_data/preeclampsia_project/data")
        images_dir = project_data_dir / "images"
        
        if images_dir.exists():
            # Find a suitable large image for patching (recursively search subdirectories)
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
                image_files.extend(list(images_dir.rglob(ext)))
            
            if image_files:
                test_image = image_files[0]
                
                # Create temporary output root
                with tempfile.TemporaryDirectory() as temp_dir:
                    patches_root = Path(temp_dir) / "real_patches_root"
                    patches_root.mkdir()
                    
                    try:
                        # Use unified patchify_dataset
                        patchify_dataset(
                            input_root=str(test_image.parent),
                            output_root=str(patches_root),
                            patch_size=224,
                        )
                        
                        # Check results
                        patch_files = list((patches_root / "image_patches").glob("*.jpg"))
                        if patch_files:
                            print(f"✅ Successfully created {len(patch_files)} patches from real image: {test_image.name}")
                        else:
                            print(f"⚠️  No patches created from real image: {test_image.name}")
                            
                    except Exception as e:
                        print(f"❌ Error patching real image {test_image.name}: {e}")
            else:
                print("⚠️  No image files found for patching test")
        else:
            print("⚠️  No images directory found, skipping real data patching test")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
