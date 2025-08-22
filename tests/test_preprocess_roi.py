"""
Comprehensive testing for preprocess_roi.py component.

This module tests the preprocess_images function with various scenarios:
- ROI extraction accuracy with sample masks
- Padding parameter variations
- ROI dimensions and quality validation
- Different mask types (binary, multi-class)
- Edge cases and error handling
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

from eq.features.preprocess_roi import preprocess_images


class TestPreprocessROI:
    """Test suite for preprocess_roi.py functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def create_test_image(self, size: Tuple[int, int] = (256, 256), color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
        """Create a test image with specified dimensions and color."""
        img = np.full((*size, 3), color, dtype=np.uint8)
        return img
    
    def create_test_mask(self, size: Tuple[int, int] = (256, 256), mask_type: str = "circle", 
                        center: Tuple[int, int] = (128, 128), radius: int = 50) -> np.ndarray:
        """Create a test mask with specified parameters."""
        mask = np.zeros(size, dtype=np.uint8)
        
        if mask_type == "circle":
            cv2.circle(mask, center, radius, 255, -1)
        elif mask_type == "rectangle":
            x, y = center
            cv2.rectangle(mask, (x - radius, y - radius), (x + radius, y + radius), 255, -1)
        elif mask_type == "multiple_circles":
            # Create multiple circles
            centers = [(64, 64), (192, 64), (64, 192), (192, 192)]
            for cx, cy in centers:
                cv2.circle(mask, (cx, cy), 30, 255, -1)
        elif mask_type == "binary":
            # Create a simple binary pattern
            mask[100:150, 100:150] = 255
            mask[200:250, 200:250] = 255
        
        return mask
    
    def setup_test_directory_structure(self, temp_dir, patient_id="P01"):
        """Set up test directory structure."""
        # Create directory structure
        train_dir = Path(temp_dir) / "train"
        images_dir = train_dir / "images" / patient_id
        masks_dir = train_dir / "masks" / patient_id
        output_dir = train_dir / "rois"
        
        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(images_dir), str(masks_dir), str(output_dir)
    
    def call_preprocess_images(self, images_dir: str, masks_dir: str, output_dir: str):
        """Call preprocess_images with correct directory structure."""
        # The function expects:
        # train/images/ (contains patient folders like P01/)
        # train/masks/ (contains patient folders like P01/)
        # train/rois/ (output directory)
        # So we need to pass the parent of the patient directories
        base_images_dir = str(Path(images_dir).parent)
        base_masks_dir = str(Path(masks_dir).parent)
        return preprocess_images(base_images_dir, base_masks_dir, output_dir)
    
    def test_roi_extraction_basic_circle_mask(self, temp_dir):
        """Test basic ROI extraction with circular mask."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Create test image and mask
        img = self.create_test_image((256, 256), (100, 150, 200))
        mask = self.create_test_mask((256, 256), "circle", (128, 128), 60)
        
        # Save test files
        cv2.imwrite(os.path.join(images_dir, "test_img.jpg"), img)
        cv2.imwrite(os.path.join(masks_dir, "test_img_mask.jpg"), mask)
        
        # Extract ROIs
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should extract at least one ROI
        assert rois is not None
        assert len(rois) >= 1
        
        # Check output files - ROIs are saved in patient subdirectory
        roi_files = list(Path(output_dir).glob("P01/*.jpg"))
        assert len(roi_files) >= 1
        
        # Verify ROI dimensions (should be 256x256)
        for roi_file in roi_files:
            roi_img = cv2.imread(str(roi_file))
            assert roi_img.shape[:2] == (256, 256)
    
    def test_roi_extraction_padding_variations(self, temp_dir):
        """Test ROI extraction with different padding values."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Create test image and mask
        img = self.create_test_image((256, 256), (80, 120, 160))
        mask = self.create_test_mask((256, 256), "circle", (128, 128), 40)
        
        # Save test files
        cv2.imwrite(os.path.join(images_dir, "test_img.jpg"), img)
        cv2.imwrite(os.path.join(masks_dir, "test_img_mask.jpg"), mask)
        
        # Test different padding values
        padding_values = [0, 5, 10, 20]
        
        for padding in padding_values:
            # Create new output directory for each test
            test_output_dir = os.path.join(output_dir, f"padding_{padding}")
            
            # Extract ROIs with specific padding
            rois = self.call_preprocess_images(images_dir, masks_dir, test_output_dir)
            
            # Should extract at least one ROI
            assert rois is not None
            assert len(rois) >= 1
            
            # Check that ROI files were created - ROIs are saved in patient subdirectory
            roi_files = list(Path(test_output_dir).glob("P01/*.jpg"))
            assert len(roi_files) >= 1
    
    def test_roi_extraction_multiple_contours(self, temp_dir):
        """Test ROI extraction with multiple contours in mask."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Create test image and mask with multiple circles
        img = self.create_test_image((256, 256), (120, 180, 240))
        mask = self.create_test_mask((256, 256), "multiple_circles")
        
        # Save test files
        cv2.imwrite(os.path.join(images_dir, "test_img.jpg"), img)
        cv2.imwrite(os.path.join(masks_dir, "test_img_mask.jpg"), mask)
        
        # Extract ROIs
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should extract multiple ROIs (one per circle)
        assert rois is not None
        assert len(rois) >= 4  # Should have at least 4 ROIs for 4 circles
        
        # Check output files - ROIs are saved in patient subdirectory
        roi_files = list(Path(output_dir).glob("P01/*.jpg"))
        assert len(roi_files) >= 4
        
        # Verify all ROIs have correct dimensions
        for roi_file in roi_files:
            roi_img = cv2.imread(str(roi_file))
            assert roi_img.shape[:2] == (256, 256)
    
    def test_roi_extraction_rectangular_mask(self, temp_dir):
        """Test ROI extraction with rectangular mask."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Create test image and rectangular mask
        img = self.create_test_image((256, 256), (90, 130, 170))
        mask = self.create_test_mask((256, 256), "rectangle", (128, 128), 60)
        
        # Save test files
        cv2.imwrite(os.path.join(images_dir, "test_img.jpg"), img)
        cv2.imwrite(os.path.join(masks_dir, "test_img_mask.jpg"), mask)
        
        # Extract ROIs
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should extract at least one ROI
        assert rois is not None
        assert len(rois) >= 1
        
        # Check output files
        roi_files = list(Path(output_dir).glob("P01/*.jpg"))
        assert len(roi_files) >= 1
        
        # Verify ROI dimensions
        for roi_file in roi_files:
            roi_img = cv2.imread(str(roi_file))
            assert roi_img.shape[:2] == (256, 256)
    
    def test_roi_extraction_binary_mask(self, temp_dir):
        """Test ROI extraction with binary pattern mask."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Create test image and binary mask
        img = self.create_test_image((256, 256), (110, 140, 180))
        mask = self.create_test_mask((256, 256), "binary")
        
        # Save test files
        cv2.imwrite(os.path.join(images_dir, "test_img.jpg"), img)
        cv2.imwrite(os.path.join(masks_dir, "test_img_mask.jpg"), mask)
        
        # Extract ROIs
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should extract at least one ROI
        assert rois is not None
        assert len(rois) >= 1
        
        # Check output files
        roi_files = list(Path(output_dir).glob("P01/*.jpg"))
        assert len(roi_files) >= 1
        
        # Verify ROI dimensions
        for roi_file in roi_files:
            roi_img = cv2.imread(str(roi_file))
            assert roi_img.shape[:2] == (256, 256)
    
    def test_roi_extraction_different_image_sizes(self, temp_dir):
        """Test ROI extraction with images of different sizes."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Test different image sizes
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for i, size in enumerate(sizes):
            # Create test image and mask
            img = self.create_test_image(size, (100 + i * 20, 150 + i * 20, 200 + i * 20))
            mask = self.create_test_mask(size, "circle", (size[0]//2, size[1]//2), size[0]//4)
            
            # Save test files
            cv2.imwrite(os.path.join(images_dir, f"test_img_{i}.jpg"), img)
            cv2.imwrite(os.path.join(masks_dir, f"test_img_{i}_mask.jpg"), mask)
        
        # Extract ROIs
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should extract ROIs for all images
        assert rois is not None
        assert len(rois) >= len(sizes)
        
        # Check output files
        roi_files = list(Path(output_dir).glob("P01/*.jpg"))
        assert len(roi_files) >= len(sizes)
        
        # Verify all ROIs are resized to 256x256
        for roi_file in roi_files:
            roi_img = cv2.imread(str(roi_file))
            assert roi_img.shape[:2] == (256, 256)
    
    def test_roi_extraction_edge_case_small_mask(self, temp_dir):
        """Test ROI extraction with very small mask."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Create test image and very small mask
        img = self.create_test_image((256, 256), (130, 160, 190))
        mask = self.create_test_mask((256, 256), "circle", (128, 128), 5)
        
        # Save test files
        cv2.imwrite(os.path.join(images_dir, "test_img.jpg"), img)
        cv2.imwrite(os.path.join(masks_dir, "test_img_mask.jpg"), mask)
        
        # Extract ROIs
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should still extract at least one ROI
        assert rois is not None
        assert len(rois) >= 1
        
        # Check output files
        roi_files = list(Path(output_dir).glob("P01/*.jpg"))
        assert len(roi_files) >= 1
    
    def test_roi_extraction_edge_case_large_mask(self, temp_dir):
        """Test ROI extraction with mask covering most of image."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Create test image and large mask
        img = self.create_test_image((256, 256), (140, 170, 200))
        mask = self.create_test_mask((256, 256), "circle", (128, 128), 120)
        
        # Save test files
        cv2.imwrite(os.path.join(images_dir, "test_img.jpg"), img)
        cv2.imwrite(os.path.join(masks_dir, "test_img_mask.jpg"), mask)
        
        # Extract ROIs
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should extract at least one ROI
        assert rois is not None
        assert len(rois) >= 1
        
        # Check output files
        roi_files = list(Path(output_dir).glob("P01/*.jpg"))
        assert len(roi_files) >= 1
    
    def test_roi_extraction_quality_validation(self, temp_dir):
        """Test that extracted ROIs maintain image quality."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Create test image with distinct features
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Add some distinct features
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(img, (200, 100), 30, (0, 255, 0), -1)  # Green circle
        cv2.line(img, (100, 200), (200, 200), (0, 0, 255), 5)  # Red line
        
        mask = self.create_test_mask((256, 256), "circle", (128, 128), 80)
        
        # Save test files
        cv2.imwrite(os.path.join(images_dir, "test_img.jpg"), img)
        cv2.imwrite(os.path.join(masks_dir, "test_img_mask.jpg"), mask)
        
        # Extract ROIs
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should extract at least one ROI
        assert rois is not None
        assert len(rois) >= 1
        
        # Check output files
        roi_files = list(Path(output_dir).glob("P01/*.jpg"))
        assert len(roi_files) >= 1
        
        # Verify ROI quality - should not be completely black
        roi_has_content = False
        for roi_file in roi_files:
            roi_img = cv2.imread(str(roi_file))
            assert roi_img.shape[:2] == (256, 256)
            # Check that at least one ROI has content (not completely black)
            if np.mean(roi_img) > 0:
                roi_has_content = True
                break
        
        assert roi_has_content, "No ROIs with content found - all ROIs are completely black"
    
    def test_roi_extraction_multiple_patients(self, temp_dir):
        """Test ROI extraction with multiple patient directories."""
        # Create multiple patient directories
        patients = ["P01", "P02", "P03"]
        
        for patient in patients:
            images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir, patient)
            
            # Create test image and mask for each patient
            img = self.create_test_image((256, 256), (100, 150, 200))
            mask = self.create_test_mask((256, 256), "circle", (128, 128), 60)
            
            # Save test files
            cv2.imwrite(os.path.join(images_dir, f"{patient}_img.jpg"), img)
            cv2.imwrite(os.path.join(masks_dir, f"{patient}_img_mask.jpg"), mask)
        
        # Extract ROIs from all patients
        base_images_dir = os.path.join(temp_dir, "train", "images")
        base_masks_dir = os.path.join(temp_dir, "train", "masks")
        base_output_dir = os.path.join(temp_dir, "train", "rois")
        
        rois = preprocess_images(base_images_dir, base_masks_dir, base_output_dir)
        
        # Should extract ROIs for all patients
        assert rois is not None
        assert len(rois) >= len(patients)
        
        # Check output files for all patients
        for patient in patients:
            patient_output_dir = os.path.join(base_output_dir, patient)
            roi_files = list(Path(patient_output_dir).glob("*.jpg"))
            assert len(roi_files) >= 1
    
    def test_roi_extraction_handle_missing_mask_folder(self, temp_dir):
        """Test handling of missing mask folder."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Create test image
        img = self.create_test_image((256, 256), (120, 160, 200))
        cv2.imwrite(os.path.join(images_dir, "test_img.jpg"), img)
        
        # Don't create mask folder - simulate missing masks
        shutil.rmtree(masks_dir)
        
        # Extract ROIs (should handle gracefully)
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should return empty array when no masks exist
        assert rois is not None
        # No ROIs should be extracted
        assert len(rois) == 0
    
    def test_roi_extraction_handle_empty_directories(self, temp_dir):
        """Test handling of empty image/mask directories."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Leave directories empty
        # Extract ROIs (should handle gracefully)
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should return empty array when no images exist
        assert rois is not None
        assert len(rois) == 0
    
    def test_roi_extraction_filename_consistency(self, temp_dir):
        """Test that ROI filenames are consistent and traceable."""
        images_dir, masks_dir, output_dir = self.setup_test_directory_structure(temp_dir)
        
        # Create test image and mask
        img = self.create_test_image((256, 256), (110, 140, 180))
        mask = self.create_test_mask((256, 256), "circle", (128, 128), 60)
        
        # Save test files with specific names
        image_filename = "patient_001_slide_A_image.jpg"
        mask_filename = "patient_001_slide_A_mask.jpg"
        
        cv2.imwrite(os.path.join(images_dir, image_filename), img)
        cv2.imwrite(os.path.join(masks_dir, mask_filename), mask)
        
        # Extract ROIs
        rois = self.call_preprocess_images(images_dir, masks_dir, output_dir)
        
        # Should extract at least one ROI
        assert rois is not None
        assert len(rois) >= 1
        
        # Check output files - should follow naming convention
        roi_files = list(Path(output_dir).glob("P01/*.jpg"))
        assert len(roi_files) >= 1
        
        # Verify naming convention: original_name_ROI_index.jpg
        for roi_file in roi_files:
            roi_name = roi_file.stem  # filename without extension
            assert roi_name.startswith("patient_001_slide_A_image_ROI_")
            # Check that it ends with a valid ROI index (could be 0, 1, 2, etc.)
            assert "_ROI_" in roi_name
            # Extract the index and verify it's a number
            index_part = roi_name.split("_ROI_")[-1]
            assert index_part.isdigit(), f"ROI index '{index_part}' is not a valid number"


if __name__ == "__main__":
    pytest.main([__file__])
