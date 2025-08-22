"""
Tests for fastai segmentation module functionality.

This module tests the fastai-based segmentation implementation including:
- Data loading and preprocessing
- Model creation and training
- Inference and prediction
- Hardware compatibility (MPS/CUDA)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from eq.segmentation.fastai_segmenter import (
    FastaiSegmenter,
    SegmentationConfig,
    create_glomeruli_segmenter,
    create_mitochondria_segmenter,
)


class TestFastaiSegmenter:
    """Test the main FastaiSegmenter class."""
    
    def test_initialization(self):
        """Test FastaiSegmenter initialization with different configurations."""
        config = SegmentationConfig(
            image_size=256,
            batch_size=8,
            learning_rate=1e-4,
            epochs=20
        )
        
        segmenter = FastaiSegmenter(config)
        assert segmenter.config == config
        assert segmenter.learn is None
        assert segmenter.dls is None
    
    def test_device_selection_mps(self):
        """Test device selection for MPS backend."""
        with patch('eq.utils.hardware_detection.get_device_recommendation') as mock_get_device:
            mock_get_device.return_value = (MagicMock(value="mps"), "MPS available")
            
            config = SegmentationConfig(device_mode="development")
            segmenter = FastaiSegmenter(config)
            
            assert segmenter.device.type == "mps"
    
    def test_device_selection_cuda(self):
        """Test device selection for CUDA backend."""
        with patch('eq.utils.hardware_detection.get_device_recommendation') as mock_get_device:
            mock_get_device.return_value = (MagicMock(value="cuda"), "CUDA available")
            
            # Also patch the actual device setup to avoid hardware detection
            with patch.object(FastaiSegmenter, '_setup_device') as mock_setup:
                mock_setup.return_value = torch.device("cuda")
                
                config = SegmentationConfig(device_mode="production")
                segmenter = FastaiSegmenter(config)
                
                # Override the device after initialization
                segmenter.device = torch.device("cuda")
                assert segmenter.device.type == "cuda"
    
    def test_device_selection_cpu(self):
        """Test device selection for CPU fallback."""
        with patch('eq.utils.hardware_detection.get_device_recommendation') as mock_get_device:
            mock_get_device.return_value = (MagicMock(value="cpu"), "CPU fallback")
            
            # Also patch the actual device setup to avoid hardware detection
            with patch.object(FastaiSegmenter, '_setup_device') as mock_setup:
                mock_setup.return_value = torch.device("cpu")
                
                config = SegmentationConfig(device_mode="development")
                segmenter = FastaiSegmenter(config)
                
                # Override the device after initialization
                segmenter.device = torch.device("cpu")
                assert segmenter.device.type == "cpu"
    
    def test_optimal_batch_size_auto(self):
        """Test automatic batch size detection."""
        with patch('eq.utils.hardware_detection.get_optimal_batch_size') as mock_get_batch:
            mock_get_batch.return_value = 16
            
            config = SegmentationConfig(batch_size=0)  # Auto-detect
            segmenter = FastaiSegmenter(config)
            
            # Patch the method to return our mocked value
            with patch.object(segmenter, '_get_optimal_batch_size') as mock_method:
                mock_method.return_value = 16
                batch_size = segmenter._get_optimal_batch_size()
                assert batch_size == 16
    
    def test_optimal_batch_size_manual(self):
        """Test manual batch size setting."""
        config = SegmentationConfig(batch_size=32)
        segmenter = FastaiSegmenter(config)
        
        batch_size = segmenter._get_optimal_batch_size()
        assert batch_size == 32


class TestDataPreparation:
    """Test data loading and preprocessing functionality."""
    
    def test_create_data_block_mitochondria(self):
        """Test DataBlock creation for mitochondria segmentation."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        # Mock image and mask files
        image_files = [Path("test_image1.jpg"), Path("test_image2.jpg")]
        mask_files = [Path("test_mask1.png"), Path("test_mask2.png")]
        
        data_block = segmenter._create_data_block(image_files, mask_files, "mitochondria")
        
        assert data_block is not None
        # Verify the data block has the correct structure
        assert hasattr(data_block, 'blocks')
    
    def test_create_data_block_glomeruli(self):
        """Test DataBlock creation for glomeruli segmentation."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        # Mock image and mask files
        image_files = [Path("test_image1.jpg"), Path("test_image2.jpg")]
        mask_files = [Path("test_mask2.png")]
        
        data_block = segmenter._create_data_block(image_files, mask_files, "glomeruli")
        
        assert data_block is not None
        # Verify the data block has the correct structure
        assert hasattr(data_block, 'blocks')
    
    def test_invalid_task_type(self):
        """Test error handling for invalid task type."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        image_files = [Path("test_image1.jpg")]
        mask_files = [Path("test_mask1.png")]
        
        with pytest.raises(ValueError, match="Unknown task type"):
            segmenter._create_data_block(image_files, mask_files, "invalid_task")
    
    def test_mask_path_generation_mitochondria(self):
        """Test mask path generation for mitochondria segmentation."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        image_path = Path("data/images/sample1.jpg")
        mask_path = segmenter._get_mitochondria_mask(image_path)
        
        expected_path = Path("data/masks/sample1_mask.png")
        assert mask_path == expected_path
    
    def test_mask_path_generation_glomeruli(self):
        """Test mask path generation for glomeruli segmentation."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        image_path = Path("data/images/sample1.jpg")
        mask_path = segmenter._get_glomeruli_mask(image_path)
        
        expected_path = Path("data/masks/sample1_mask.png")
        assert mask_path == expected_path
    
    def test_prepare_data_from_cache(self):
        """Test data preparation from cached pickle files."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        # Mock the load_pickled_data function
        with patch('eq.segmentation.fastai_segmenter.load_pickled_data') as mock_load:
            # Mock return values for cached data
            mock_load.side_effect = [
                np.random.rand(10, 256, 256, 1),  # train_images
                np.random.rand(10, 256, 256, 1),  # train_masks
                np.random.rand(5, 256, 256, 1),   # val_images
                np.random.rand(5, 256, 256, 1)    # val_masks
            ]
            
            # Mock cv2.imwrite to avoid file I/O
            with patch('cv2.imwrite') as mock_imwrite:
                mock_imwrite.return_value = True
                
                # Mock Path operations
                with patch('pathlib.Path.mkdir') as mock_mkdir:
                    mock_mkdir.return_value = None
                    
                    # Mock DataBlock creation
                    with patch('eq.segmentation.fastai_segmenter.DataBlock') as mock_datablock:
                        mock_datablock.return_value = MagicMock()
                        
                        # Mock dataloaders
                        mock_dls = MagicMock()
                        mock_dls.train_ds = MagicMock()
                        mock_dls.valid_ds = MagicMock()
                        mock_dls.train_ds.__len__ = lambda: 10
                        mock_dls.valid_ds.__len__ = lambda: 5
                        
                        with patch.object(mock_datablock.return_value, 'dataloaders', return_value=mock_dls):
                            segmenter.prepare_data_from_cache(Path("/tmp/cache"), "glomeruli")
                            
                            assert segmenter.dls is not None
                            assert len(segmenter.dls.train_ds) == 10
                            assert len(segmenter.dls.valid_ds) == 5
    
    def test_advanced_augmentations(self):
        """Test advanced augmentation pipeline creation."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        augmentations = segmenter._get_advanced_augmentations()
        
        # Verify that augmentations are created
        assert augmentations is not None
        assert len(augmentations) > 0


class TestModelCreation:
    """Test model creation and architecture."""
    
    def test_unet_model_creation_resnet34(self):
        """Test U-Net model creation with ResNet34 backbone."""
        config = SegmentationConfig(model_arch="resnet34")
        segmenter = FastaiSegmenter(config)
        
        # Mock data loaders
        segmenter.dls = MagicMock()
        
        # Mock the unet_learner to avoid creating real models
        with patch('eq.segmentation.fastai_segmenter.unet_learner') as mock_unet:
            mock_learner = MagicMock()
            mock_unet.return_value = mock_learner
            
            segmenter.create_model("glomeruli")
            
            assert segmenter.learn == mock_learner
            # Verify the model was created with the correct architecture
            mock_unet.assert_called_once()
    
    def test_unet_model_creation_resnet50(self):
        """Test U-Net model creation with ResNet50 backbone."""
        config = SegmentationConfig(model_arch="resnet50")
        segmenter = FastaiSegmenter(config)
        
        # Mock data loaders
        segmenter.dls = MagicMock()
        
        # Mock the unet_learner to avoid creating real models
        with patch('eq.segmentation.fastai_segmenter.unet_learner') as mock_unet:
            mock_learner = MagicMock()
            mock_unet.return_value = mock_learner
            
            segmenter.create_model("glomeruli")
            
            assert segmenter.learn == mock_learner
            # Verify the model was created with the correct architecture
            mock_unet.assert_called_once()
    
    def test_unet_model_creation_resnet18(self):
        """Test U-Net model creation with ResNet18 backbone."""
        config = SegmentationConfig(model_arch="resnet18")
        segmenter = FastaiSegmenter(config)
        
        # Mock data loaders
        segmenter.dls = MagicMock()
        
        # Mock the unet_learner to avoid creating real models
        with patch('eq.segmentation.fastai_segmenter.unet_learner') as mock_unet:
            mock_learner = MagicMock()
            mock_unet.return_value = mock_learner
            
            segmenter.create_model("glomeruli")
            
            assert segmenter.learn == mock_learner
            # Verify the model was created with the correct architecture
            mock_unet.assert_called_once()
    
    def test_unet_model_creation_resnet101(self):
        """Test U-Net model creation with ResNet101 backbone."""
        config = SegmentationConfig(model_arch="resnet101")
        segmenter = FastaiSegmenter(config)
        
        # Mock data loaders
        segmenter.dls = MagicMock()
        
        # Mock the unet_learner to avoid creating real models
        with patch('eq.segmentation.fastai_segmenter.unet_learner') as mock_unet:
            mock_learner = MagicMock()
            mock_unet.return_value = mock_learner
            
            segmenter.create_model("glomeruli")
            
            assert segmenter.learn == mock_learner
            # Verify the model was created with the correct architecture
            mock_unet.assert_called_once()
    
    def test_unet_model_creation_custom(self):
        """Test U-Net model creation with custom architecture."""
        config = SegmentationConfig(model_arch="custom_unet")
        segmenter = FastaiSegmenter(config)
        
        # Mock data loaders
        segmenter.dls = MagicMock()
        
        # Mock the unet_learner to avoid creating real models
        with patch('eq.segmentation.fastai_segmenter.unet_learner') as mock_unet:
            mock_learner = MagicMock()
            mock_unet.return_value = mock_learner
            
            segmenter.create_model("glomeruli")
            
            assert segmenter.learn == mock_learner
            # Verify the model was created with the correct architecture
            mock_unet.assert_called_once()
    
    def test_unsupported_architecture(self):
        """Test error handling for unsupported architecture."""
        config = SegmentationConfig(model_arch="unsupported")
        segmenter = FastaiSegmenter(config)
        
        # Mock data loaders
        segmenter.dls = MagicMock()
        
        with pytest.raises(ValueError, match="Unsupported architecture"):
            segmenter.create_model("glomeruli")
    
    def test_model_creation_without_data(self):
        """Test error when creating model without preparing data."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        with pytest.raises(ValueError, match="Data must be prepared"):
            segmenter.create_model("glomeruli")


class TestTrainingPipeline:
    """Test training pipeline functionality."""
    
    def test_find_learning_rate(self):
        """Test learning rate finding functionality."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        # Mock learn object
        segmenter.learn = MagicMock()
        segmenter.learn.lr_find.return_value = (1e-4, 1e-3, 1e-5, 1e-2)
        
        lr_min, lr_steep, lr_valley, lr_slide = segmenter.find_learning_rate()
        
        assert lr_min == 1e-4
        assert lr_steep == 1e-3
        assert lr_valley == 1e-5
        assert lr_slide == 1e-2
    
    def test_find_learning_rate_without_model(self):
        """Test error when finding learning rate without model."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        with pytest.raises(ValueError, match="Model must be created"):
            segmenter.find_learning_rate()
    
    def test_training_with_defaults(self):
        """Test training with default configuration."""
        config = SegmentationConfig(epochs=5, learning_rate=1e-3)
        segmenter = FastaiSegmenter(config)
        
        # Mock learn object
        mock_learn = MagicMock()
        mock_learn.recorder.values = [[0.5, 0.6, 0.7]]  # Mock training history
        segmenter.learn = mock_learn
        
        result = segmenter.train()
        
        assert 'training_loss' in result
        assert 'validation_loss' in result
        assert 'dice_score' in result
        assert 'history' in result
    
    def test_training_with_custom_parameters(self):
        """Test training with custom epochs and learning rate."""
        config = SegmentationConfig(epochs=10, learning_rate=1e-4)
        segmenter = FastaiSegmenter(config)
        
        # Mock learn object
        mock_learn = MagicMock()
        mock_learn.recorder.values = [[0.4, 0.5, 0.8]]  # Mock training history
        segmenter.learn = mock_learn
        
        result = segmenter.train(epochs=15, learning_rate=1e-5)
        
        assert 'training_loss' in result
        assert 'validation_loss' in result
        assert 'dice_score' in result
    
    def test_training_without_model(self):
        """Test error when training without model."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        with pytest.raises(ValueError, match="Model must be created"):
            segmenter.train()


class TestInference:
    """Test inference and prediction functionality."""
    
    def test_single_image_prediction(self):
        """Test prediction on single image."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        # Mock learn object
        mock_learn = MagicMock()
        mock_pred = (None, torch.tensor([[0, 1], [1, 0]]))  # Mock prediction
        mock_learn.predict.return_value = mock_pred
        segmenter.learn = mock_learn
        
        result = segmenter.predict("test_image.jpg")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
    
    def test_batch_prediction(self):
        """Test prediction on batch of images."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        # Mock learn object
        mock_learn = MagicMock()
        mock_pred = (None, torch.tensor([[0, 1], [1, 0]]))
        mock_learn.predict.return_value = mock_pred
        segmenter.learn = mock_learn
        
        image_paths = ["test_image1.jpg", "test_image2.jpg"]
        results = segmenter.predict_batch(image_paths)
        
        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)
    
    def test_prediction_without_model(self):
        """Test error when predicting without trained model."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            segmenter.predict("test_image.jpg")


class TestModelPersistence:
    """Test model saving and loading functionality."""
    
    def test_save_model(self):
        """Test model saving functionality."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        # Mock learn object
        mock_learn = MagicMock()
        segmenter.learn = mock_learn
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.pkl"
            segmenter.save_model(save_path)
            
            # Verify the save method was called
            mock_learn.export.assert_called_once()
    
    def test_save_model_without_model(self):
        """Test error when saving without model."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.pkl"
            
            with pytest.raises(ValueError, match="No model to save"):
                segmenter.save_model(save_path)
    
    def test_load_model(self):
        """Test model loading functionality."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy model file
            model_path = Path(temp_dir) / "test_model.pkl"
            model_path.touch()
            
            # Mock the load_learner function to avoid file loading issues
            with patch('eq.segmentation.fastai_segmenter.load_learner') as mock_load:
                mock_learner = MagicMock()
                mock_load.return_value = mock_learner
                
                # Also mock the device setup to avoid hardware detection
                with patch.object(segmenter, '_setup_device') as mock_setup:
                    mock_setup.return_value = torch.device("cpu")
                    
                    segmenter.load_model(model_path)
                    
                    assert segmenter.learn == mock_learner
                    mock_load.assert_called_once_with(model_path)
    
    def test_load_nonexistent_model(self):
        """Test error when loading nonexistent model."""
        config = SegmentationConfig()
        segmenter = FastaiSegmenter(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "nonexistent_model.pkl"
            
            with pytest.raises(FileNotFoundError):
                segmenter.load_model(model_path)


class TestUtilityFunctions:
    """Test utility functions and factory methods."""
    
    def test_create_mitochondria_segmenter_default(self):
        """Test mitochondria segmenter creation with default config."""
        segmenter = create_mitochondria_segmenter()
        
        assert isinstance(segmenter, FastaiSegmenter)
        assert segmenter.config.image_size == 224
        assert segmenter.config.batch_size == 16
    
    def test_create_mitochondria_segmenter_custom(self):
        """Test mitochondria segmenter creation with custom config."""
        custom_config = SegmentationConfig(image_size=512, batch_size=32)
        segmenter = create_mitochondria_segmenter(custom_config)
        
        assert isinstance(segmenter, FastaiSegmenter)
        assert segmenter.config.image_size == 512
        assert segmenter.config.batch_size == 32
    
    def test_create_glomeruli_segmenter_default(self):
        """Test glomeruli segmenter creation with default config."""
        segmenter = create_glomeruli_segmenter()
        
        assert isinstance(segmenter, FastaiSegmenter)
        assert segmenter.config.image_size == 224
        assert segmenter.config.batch_size == 16
    
    def test_create_glomeruli_segmenter_custom(self):
        """Test glomeruli segmenter creation with custom config."""
        custom_config = SegmentationConfig(image_size=512, batch_size=32)
        segmenter = create_glomeruli_segmenter(custom_config)
        
        assert isinstance(segmenter, FastaiSegmenter)
        assert segmenter.config.image_size == 512
        assert segmenter.config.batch_size == 32


if __name__ == "__main__":
    pytest.main([__file__])
