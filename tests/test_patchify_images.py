"""
Comprehensive testing for patchify_images.py component.

This module tests the patchify_image_dir function with various scenarios:
- Different image sizes and patch sizes
- Patch naming conventions (_ and - separators)
- Recursive directory processing
- Patch dimension validation
- Edge cases (very large images, very small images)
- Various image formats
"""

import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

from eq.patches.patchify_images import patchify_image_dir


class TestPatchifyImages:
    """Test suite for patchify_images.py functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample images of various sizes for testing."""
        images_dir = Path(temp_dir) / "sample_images"
        images_dir.mkdir()
        
        # Create images of different sizes
        sizes = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
        image_paths = []
        
        for i, size in enumerate(sizes):
            # Create a simple pattern image
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            img_path = images_dir / f"sample_{i}_{size[0]}x{size[1]}.jpg"
            cv2.imwrite(str(img_path), img)
            image_paths.append(img_path)
        
        return images_dir, image_paths
    
    def create_test_image(self, size: Tuple[int, int], filename: str, temp_dir: str) -> Path:
        """Create a test image with specified dimensions."""
        img_path = Path(temp_dir) / filename
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        return img_path
    
    def test_patchify_small_image_8x8_patch_4(self, temp_dir):
        """Test patchification of 8x8 image with 4x4 patches."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create 8x8 image
        self.create_test_image((8, 8), "small.jpg", str(input_dir))
        
        # Patchify with 4x4 patches
        patchify_image_dir(4, str(input_dir), str(output_dir))
        
        # Should create 4 patches: (0,0), (0,1), (1,0), (1,1)
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 4
        
        patch_names = {p.name for p in patches}
        expected_names = {
            "small_0_0.jpg", "small_0_1.jpg", 
            "small_1_0.jpg", "small_1_1.jpg"
        }
        assert patch_names == expected_names
        
        # Verify patch dimensions
        for patch_path in patches:
            img = cv2.imread(str(patch_path))
            assert img.shape[:2] == (4, 4)
    
    def test_patchify_medium_image_16x16_patch_8(self, temp_dir):
        """Test patchification of 16x16 image with 8x8 patches."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create 16x16 image
        self.create_test_image((16, 16), "medium.jpg", str(input_dir))
        
        # Patchify with 8x8 patches
        patchify_image_dir(8, str(input_dir), str(output_dir))
        
        # Should create 4 patches
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 4
        
        # Verify patch dimensions
        for patch_path in patches:
            img = cv2.imread(str(patch_path))
            assert img.shape[:2] == (8, 8)
    
    def test_patchify_large_image_128x128_patch_64(self, temp_dir):
        """Test patchification of 128x128 image with 64x64 patches."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create 128x128 image
        self.create_test_image((128, 128), "large.jpg", str(input_dir))
        
        # Patchify with 64x64 patches
        patchify_image_dir(64, str(input_dir), str(output_dir))
        
        # Should create 4 patches
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 4
        
        # Verify patch dimensions
        for patch_path in patches:
            img = cv2.imread(str(patch_path))
            assert img.shape[:2] == (64, 64)
    
    def test_patchify_odd_sized_image(self, temp_dir):
        """Test patchification of image with dimensions not perfectly divisible by patch size."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create 10x10 image (not perfectly divisible by 4)
        self.create_test_image((10, 10), "odd.jpg", str(input_dir))
        
        # Patchify with 4x4 patches
        patchify_image_dir(4, str(input_dir), str(output_dir))
        
        # Should create 4 patches (2x2 grid)
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 4
        
        # Verify patch dimensions (should be 4x4)
        for patch_path in patches:
            img = cv2.imread(str(patch_path))
            assert img.shape[:2] == (4, 4)
    
    def test_patchify_various_image_formats(self, temp_dir):
        """Test patchification with different image formats."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create images in different formats
        formats = ['.jpg', '.jpeg', '.png', '.tif']
        for fmt in formats:
            self.create_test_image((16, 16), f"test{fmt}", str(input_dir))
        
        # Patchify with 8x8 patches
        patchify_image_dir(8, str(input_dir), str(output_dir))
        
        # Should create 4 patches per image = 16 total
        patches = list(output_dir.glob("*"))
        assert len(patches) == 16
        
        # Verify all patches are valid images
        for patch_path in patches:
            img = cv2.imread(str(patch_path))
            assert img is not None
            assert img.shape[:2] == (8, 8)
    
    def test_patchify_recursive_directory_processing(self, temp_dir):
        """Test recursive processing of nested directories."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        
        # Create nested directory structure
        nested_dir = input_dir / "level1" / "level2"
        nested_dir.mkdir(parents=True)
        
        # Create images at different levels
        self.create_test_image((16, 16), "root.jpg", str(input_dir))
        self.create_test_image((16, 16), "nested.jpg", str(nested_dir))
        
        # Patchify
        patchify_image_dir(8, str(input_dir), str(output_dir))
        
        # Should create patches for both images
        root_patches = list(output_dir.glob("root_*.jpg"))
        nested_patches = list((output_dir / "level1" / "level2").glob("nested_*.jpg"))
        
        assert len(root_patches) == 4
        assert len(nested_patches) == 4
        
        # Verify patch dimensions
        for patch_path in root_patches + nested_patches:
            img = cv2.imread(str(patch_path))
            assert img.shape[:2] == (8, 8)
    
    def test_patchify_edge_case_very_small_image(self, temp_dir):
        """Test patchification of very small image (smaller than patch size)."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create 2x2 image (smaller than 4x4 patch)
        self.create_test_image((2, 2), "tiny.jpg", str(input_dir))
        
        # Patchify with 4x4 patches
        patchify_image_dir(4, str(input_dir), str(output_dir))
        
        # Should create no patches (image too small)
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 0
    
    def test_patchify_edge_case_very_large_image(self, temp_dir):
        """Test patchification of very large image."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create 256x256 image
        self.create_test_image((256, 256), "huge.jpg", str(input_dir))
        
        # Patchify with 64x64 patches
        patchify_image_dir(64, str(input_dir), str(output_dir))
        
        # Should create 16 patches (4x4 grid)
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 16
        
        # Verify patch dimensions
        for patch_path in patches:
            img = cv2.imread(str(patch_path))
            assert img.shape[:2] == (64, 64)
    
    def test_patchify_patch_naming_convention_underscore(self, temp_dir):
        """Test patch naming convention with underscore separators."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create image with underscore in name
        self.create_test_image((16, 16), "test_image.jpg", str(input_dir))
        
        # Patchify
        patchify_image_dir(8, str(input_dir), str(output_dir))
        
        # Check naming convention: filename_patchrow_patchcol.ext
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 4
        
        expected_names = {
            "test_image_0_0.jpg", "test_image_0_1.jpg",
            "test_image_1_0.jpg", "test_image_1_1.jpg"
        }
        patch_names = {p.name for p in patches}
        assert patch_names == expected_names
    
    def test_patchify_patch_naming_convention_hyphen(self, temp_dir):
        """Test patch naming convention with hyphen separators."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create image with hyphen in name
        self.create_test_image((16, 16), "test-image.jpg", str(input_dir))
        
        # Patchify
        patchify_image_dir(8, str(input_dir), str(output_dir))
        
        # Check naming convention: filename_patchrow_patchcol.ext
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 4
        
        expected_names = {
            "test-image_0_0.jpg", "test-image_0_1.jpg",
            "test-image_1_0.jpg", "test-image_1_1.jpg"
        }
        patch_names = {p.name for p in patches}
        assert patch_names == expected_names
    
    def test_patchify_output_directory_creation(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Ensure output directory doesn't exist
        assert not output_dir.exists()
        
        # Create test image
        self.create_test_image((16, 16), "test.jpg", str(input_dir))
        
        # Patchify (should create output directory)
        patchify_image_dir(8, str(input_dir), str(output_dir))
        
        # Output directory should now exist
        assert output_dir.exists()
        assert output_dir.is_dir()
        
        # Should contain patches
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 4
    
    def test_patchify_skip_non_image_files(self, temp_dir):
        """Test that non-image files are skipped during processing."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create image file
        self.create_test_image((16, 16), "image.jpg", str(input_dir))
        
        # Create non-image files
        (input_dir / "text.txt").write_text("This is not an image")
        (input_dir / "data.csv").write_text("col1,col2\n1,2")
        (input_dir / "script.py").write_text("print('hello')")
        
        # Patchify
        patchify_image_dir(8, str(input_dir), str(output_dir))
        
        # Should only process image files
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 4
        
        # Verify no non-image files were processed
        non_image_files = list(output_dir.glob("*.txt")) + list(output_dir.glob("*.csv")) + list(output_dir.glob("*.py"))
        assert len(non_image_files) == 0
    
    def test_patchify_preserve_original_extension(self, temp_dir):
        """Test that original file extension is preserved in patch names."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create images with different extensions
        extensions = ['.jpg', '.jpeg', '.png', '.tif']
        for ext in extensions:
            self.create_test_image((16, 16), f"test{ext}", str(input_dir))
        
        # Patchify
        patchify_image_dir(8, str(input_dir), str(output_dir))
        
        # Check that extensions are preserved
        for ext in extensions:
            patches = list(output_dir.glob(f"test_*{ext}"))
            assert len(patches) == 4
            
            # Verify patch names contain original extension
            for patch_path in patches:
                assert patch_path.suffix == ext
    
    def test_patchify_handle_empty_directory(self, temp_dir):
        """Test handling of empty input directory."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Input directory is empty
        assert len(list(input_dir.iterdir())) == 0
        
        # Patchify
        patchify_image_dir(8, str(input_dir), str(output_dir))
        
        # Output directory should exist but be empty
        assert output_dir.exists()
        assert len(list(output_dir.iterdir())) == 0
    
    def test_patchify_handle_single_pixel_image(self, temp_dir):
        """Test handling of 1x1 image (edge case)."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create 1x1 image
        self.create_test_image((1, 1), "single.jpg", str(input_dir))
        
        # Patchify with 4x4 patches
        patchify_image_dir(4, str(input_dir), str(output_dir))
        
        # Should create no patches (image too small)
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 0
    
    def test_patchify_handle_rectangular_image(self, temp_dir):
        """Test handling of rectangular (non-square) image."""
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        
        # Create 16x8 rectangular image
        self.create_test_image((16, 8), "rect.jpg", str(input_dir))
        
        # Patchify with 8x8 patches
        patchify_image_dir(8, str(input_dir), str(output_dir))
        
        # Should create 2 patches (2x1 grid: 2 rows, 1 column)
        patches = list(output_dir.glob("*.jpg"))
        assert len(patches) == 2
        
        # Verify patch dimensions
        for patch_path in patches:
            img = cv2.imread(str(patch_path))
            assert img.shape[:2] == (8, 8)
        
        # Check patch names - for 16x8 image with 8x8 patches:
        # Row 0: i=0, i//8=0, patch: (0,0)
        # Row 1: i=8, i//8=1, patch: (1,0)
        patch_names = {p.name for p in patches}
        expected_names = {"rect_0_0.jpg", "rect_1_0.jpg"}
        assert patch_names == expected_names


if __name__ == "__main__":
    pytest.main([__file__])
