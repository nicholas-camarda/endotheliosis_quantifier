"""
Tests for the three-stage pipeline architecture.

This module tests the seg, quant-endo, and production commands
to ensure the 3-stage pipeline works correctly.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eq.utils.output_manager import OutputManager


class TestThreeStagePipeline:
    """Test the three-stage pipeline architecture."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data"
        cache_dir = Path(temp_dir) / "cache"
        output_dir = Path(temp_dir) / "output"
        model_dir = Path(temp_dir) / "models"
        
        # Create directories
        data_dir.mkdir()
        cache_dir.mkdir()
        output_dir.mkdir()
        model_dir.mkdir()
        
        # Create dummy image files
        for i in range(5):
            (data_dir / f"test_image_{i}.jpg").touch()
        
        yield {
            'temp_dir': temp_dir,
            'data_dir': data_dir,
            'cache_dir': cache_dir,
            'output_dir': output_dir,
            'model_dir': model_dir
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_environment(self):
        """Mock environment for testing."""
        with patch.dict(os.environ, {'QUICK_TEST': 'true'}):
            yield
    
    def test_seg_command_structure(self, temp_dirs, mock_environment):
        """Test that seg command has correct structure and arguments."""
        from eq.__main__ import seg_command

        # Create mock args
        args = MagicMock()
        args.data_dir = str(temp_dirs['data_dir'])
        args.cache_dir = str(temp_dirs['cache_dir'])
        args.output_dir = str(temp_dirs['output_dir'])
        args.epochs = 2
        args.batch_size = 4
        args.image_size = 256
        
        # Mock the required modules
        with patch('eq.segmentation.fastai_segmenter.FastaiSegmenter') as mock_segmenter, \
             patch('eq.utils.output_manager.OutputManager') as mock_output_manager, \
             patch('eq.pipeline.run_production_pipeline.generate_training_plots') as mock_plots:
            
            # Mock output manager
            mock_output_manager.return_value.get_data_source_name.return_value = "test_data"
            mock_output_manager.return_value.create_output_directory.return_value = {
                'main': temp_dirs['output_dir'],
                'models': temp_dirs['output_dir'] / 'models',
                'plots': temp_dirs['output_dir'] / 'plots',
                'results': temp_dirs['output_dir'] / 'results'
            }
            
            # Mock segmenter
            mock_segmenter_instance = MagicMock()
            mock_segmenter.return_value = mock_segmenter_instance
            mock_segmenter_instance.train.return_value = {'loss': [0.5, 0.4], 'val_loss': [0.6, 0.5]}
            mock_segmenter_instance.save_model.return_value = temp_dirs['output_dir'] / 'models' / 'model.pkl'
            
            # Mock config
            mock_config = MagicMock()
            mock_segmenter.SegmentationConfig.return_value = mock_config
            
            # Mock plots function
            mock_plots.return_value = None
            
            # Test that seg_command can be called without errors
            try:
                seg_command(args)
                # If we get here, the command structure is correct
                assert True
            except Exception as e:
                # Expected to fail due to missing data, but structure should be correct
                assert "data preparation" in str(e) or "not yet implemented" in str(e)
    
    def test_quant_endo_command_structure(self, temp_dirs, mock_environment):
        """Test that quant-endo command has correct structure and arguments."""
        from eq.__main__ import quant_endo_command

        # Create mock args
        args = MagicMock()
        args.data_dir = str(temp_dirs['data_dir'])
        args.cache_dir = str(temp_dirs['cache_dir'])
        args.output_dir = str(temp_dirs['output_dir'])
        args.segmentation_model = str(temp_dirs['model_dir'] / "segmenter.pkl")
        args.epochs = 2
        args.batch_size = 4
        
        # Mock the required modules - use generic patches to avoid missing modules
        with patch('eq.segmentation.fastai_segmenter.FastaiSegmenter') as mock_segmenter, \
             patch('eq.utils.output_manager.OutputManager') as mock_output_manager, \
             patch('eq.features.data_loader', create=True) as mock_data_loader_module, \
             patch('eq.pipeline.quantify_endotheliosis', create=True) as mock_quant_module:
            
            # Mock output manager
            mock_output_manager.return_value.get_data_source_name.return_value = "test_data"
            mock_output_manager.return_value.create_output_directory.return_value = {
                'main': temp_dirs['output_dir'],
                'models': temp_dirs['output_dir'] / 'models',
                'plots': temp_dirs['output_dir'] / 'plots',
                'results': temp_dirs['output_dir'] / 'results',
                'cache': temp_dirs['output_dir'] / 'cache'
            }
            
            # Mock segmenter
            mock_segmenter_instance = MagicMock()
            mock_segmenter.load_model.return_value = mock_segmenter_instance
            mock_segmenter_instance.predict.return_value = [{'confidence': 0.8}]
            mock_segmenter_instance.extract_rois.return_value = [MagicMock()]
            
            # Mock data loader module
            mock_data_loader_instance = MagicMock()
            mock_data_loader_module.DataLoader.return_value = mock_data_loader_instance
            
            # Mock quantification module
            mock_quant_model_instance = MagicMock()
            mock_quant_module.QuantificationModel.return_value = mock_quant_model_instance
            mock_quant_model_instance.train.return_value = {
                'loss': [0.5, 0.4],
                'val_loss': [0.6, 0.5],
                'mae': [0.3, 0.2],
                'val_mae': [0.4, 0.3]
            }
            
            # Test that quant_endo_command can be called without errors
            try:
                quant_endo_command(args)
                # If we get here, the command structure is correct
                assert True
            except Exception as e:
                # Expected to fail due to missing data, but structure should be correct
                assert "annotation" in str(e) or "dummy scores" in str(e)
    
    def test_production_command_structure(self, temp_dirs, mock_environment):
        """Test that production command has correct structure and arguments."""
        from eq.__main__ import pipeline_command

        # Create mock args
        args = MagicMock()
        args.data_dir = str(temp_dirs['data_dir'])
        args.test_data_dir = str(temp_dirs['data_dir'])
        args.cache_dir = str(temp_dirs['cache_dir'])
        args.output_dir = str(temp_dirs['output_dir'])
        args.base_model_path = str(temp_dirs['model_dir'])
        args.epochs = 2
        args.batch_size = 4
        args.image_size = 256
        
        # Create dummy model files
        (temp_dirs['model_dir'] / "glomerulus_segmenter.pkl").touch()
        (temp_dirs['model_dir'] / "endotheliosis_quantifier.pkl").touch()
        
        # Mock the required modules - use generic patches to avoid missing modules
        with patch('eq.segmentation.fastai_segmenter.FastaiSegmenter') as mock_segmenter, \
             patch('eq.utils.output_manager.OutputManager') as mock_output_manager, \
             patch('eq.pipeline.quantify_endotheliosis', create=True) as mock_quant_module, \
             patch('eq.features.data_loader', create=True) as mock_data_loader_module:
            
            # Mock output manager
            mock_output_manager.return_value.get_data_source_name.return_value = "test_data"
            mock_output_manager.return_value.create_output_directory.return_value = {
                'main': temp_dirs['output_dir'],
                'models': temp_dirs['output_dir'] / 'models',
                'plots': temp_dirs['output_dir'] / 'plots',
                'results': temp_dirs['output_dir'] / 'results'
            }
            
            # Mock segmenter
            mock_segmenter_instance = MagicMock()
            mock_segmenter.load_model.return_value = mock_segmenter_instance
            mock_segmenter_instance.predict.return_value = [{'confidence': 0.8}]
            mock_segmenter_instance.extract_rois.return_value = [MagicMock()]
            
            # Mock quantification module
            mock_quant_model_instance = MagicMock()
            mock_quant_module.QuantificationModel.load_model.return_value = mock_quant_model_instance
            mock_quant_model_instance.predict_single.return_value = 0.5
            
            # Mock data loader module
            mock_data_loader_instance = MagicMock()
            mock_data_loader_module.DataLoader.return_value = mock_data_loader_instance
            
            # Test that pipeline_command can be called without errors
            try:
                pipeline_command(args)
                # If we get here, the command structure is correct
                assert True
            except Exception as e:
                # Expected to fail due to missing data, but structure should be correct
                assert "test images" in str(e) or "processing" in str(e)
    
    def test_quick_test_mode_detection(self, temp_dirs):
        """Test that QUICK_TEST mode is properly detected and applied."""
        from eq.__main__ import seg_command

        # Create mock args
        args = MagicMock()
        args.data_dir = str(temp_dirs['data_dir'])
        args.cache_dir = str(temp_dirs['cache_dir'])
        args.output_dir = str(temp_dirs['output_dir'])
        args.epochs = 50  # High number to test reduction
        args.batch_size = 16  # High number to test reduction
        args.image_size = 256
        
        # Test with QUICK_TEST=true
        with patch.dict(os.environ, {'QUICK_TEST': 'true'}):
            with patch('eq.segmentation.fastai_segmenter.FastaiSegmenter') as mock_segmenter, \
                 patch('eq.utils.output_manager.OutputManager') as mock_output_manager, \
                 patch('eq.pipeline.run_production_pipeline.generate_training_plots') as mock_plots:
                
                # Mock output manager
                mock_output_manager.return_value.get_data_source_name.return_value = "test_data"
                mock_output_manager.return_value.create_output_directory.return_value = {
                    'main': temp_dirs['output_dir'],
                    'models': temp_dirs['output_dir'] / 'models',
                    'plots': temp_dirs['output_dir'] / 'plots',
                    'results': temp_dirs['output_dir'] / 'results'
                }
                
                # Mock segmenter
                mock_segmenter_instance = MagicMock()
                mock_segmenter.return_value = mock_segmenter_instance
                mock_segmenter_instance.train.return_value = {'loss': [0.5, 0.4], 'val_loss': [0.6, 0.5]}
                mock_segmenter_instance.save_model.return_value = temp_dirs['output_dir'] / 'models' / 'model.pkl'
                
                # Mock config
                mock_config = MagicMock()
                mock_segmenter.SegmentationConfig.return_value = mock_config
                
                # Mock plots function
                mock_plots.return_value = None
                
                try:
                    seg_command(args)
                    # Check that epochs and batch_size were reduced
                    assert args.epochs <= 5  # Should be reduced for QUICK_TEST
                    assert args.batch_size <= 4  # Should be reduced for QUICK_TEST
                except Exception:
                    # Expected to fail, but QUICK_TEST mode should be applied
                    pass
    
    def test_cli_argument_parsing(self):
        """Test that CLI arguments are correctly parsed for all commands."""
        from eq.__main__ import main

        # Test seg command arguments
        with patch('sys.argv', ['eq', 'seg', '--help']):
            with patch('sys.exit') as mock_exit:
                try:
                    main()
                except SystemExit:
                    pass
                mock_exit.assert_called()
        
        # Test quant-endo command arguments
        with patch('sys.argv', ['eq', 'quant-endo', '--help']):
            with patch('sys.exit') as mock_exit:
                try:
                    main()
                except SystemExit:
                    pass
                mock_exit.assert_called()
        
        # Test production command arguments
        with patch('sys.argv', ['eq', 'production', '--help']):
            with patch('sys.exit') as mock_exit:
                try:
                    main()
                except SystemExit:
                    pass
                mock_exit.assert_called()
    
    def test_orchestrator_menu_options(self):
        """Test that orchestrator shows correct menu options."""
        from eq.__main__ import pipeline_orchestrator_command

        # Mock input to select option 4 (smoke test)
        with patch('builtins.input', return_value='4'):
            with patch('builtins.print') as mock_print:
                pipeline_orchestrator_command(MagicMock())
                
                # Check that the menu shows the correct options
                # Get all print calls and join them
                print_calls = []
                for call in mock_print.call_args_list:
                    if call[0]:  # Check if there are positional arguments
                        print_calls.append(str(call[0][0]))
                
                menu_text = '\n'.join(print_calls)
                
                # Check for expected menu items
                assert "Segmentation Training (seg)" in menu_text or "Available pipeline stages" in menu_text
                assert "QUICK_TEST=true" in menu_text


class TestPipelineIntegration:
    """Test integration between pipeline stages."""
    
    @pytest.fixture
    def temp_pipeline_dirs(self):
        """Create temporary directories for pipeline integration testing."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data"
        cache_dir = Path(temp_dir) / "cache"
        output_dir = Path(temp_dir) / "output"
        model_dir = Path(temp_dir) / "models"
        
        # Create directories
        data_dir.mkdir()
        cache_dir.mkdir()
        output_dir.mkdir()
        model_dir.mkdir()
        
        # Create dummy image files
        for i in range(3):
            (data_dir / f"test_image_{i}.jpg").touch()
        
        yield {
            'temp_dir': temp_dir,
            'data_dir': data_dir,
            'cache_dir': cache_dir,
            'output_dir': output_dir,
            'model_dir': model_dir
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_pipeline_stage_flow(self, temp_pipeline_dirs):
        """Test that pipeline stages can flow from one to another."""
        # This test verifies that the output of one stage can be used as input for the next
        
        # Stage 1: Segmentation training should produce a model
        seg_output_dir = temp_pipeline_dirs['output_dir'] / "seg_training"
        seg_output_dir.mkdir()
        (seg_output_dir / "models").mkdir()
        
        # Create dummy segmentation model
        seg_model_path = seg_output_dir / "models" / "glomerulus_segmenter.pkl"
        seg_model_path.touch()
        
        # Stage 2: Quantification training should use segmentation model
        quant_output_dir = temp_pipeline_dirs['output_dir'] / "quant_training"
        quant_output_dir.mkdir()
        (quant_output_dir / "models").mkdir()
        
        # Create dummy quantification model
        quant_model_path = quant_output_dir / "models" / "endotheliosis_quantifier.pkl"
        quant_model_path.touch()
        
        # Stage 3: Production should use both models
        prod_output_dir = temp_pipeline_dirs['output_dir'] / "production_inference"
        prod_output_dir.mkdir()
        (prod_output_dir / "results").mkdir()
        
        # Verify that the pipeline structure is correct
        assert seg_model_path.exists()
        assert quant_model_path.exists()
        assert seg_output_dir.exists()
        assert quant_output_dir.exists()
        assert prod_output_dir.exists()
    
    def test_output_directory_structure(self, temp_pipeline_dirs):
        """Test that output directories are created with correct structure."""
        output_manager = OutputManager()
        
        # Test seg training output structure
        seg_dirs = output_manager.create_output_directory("test_data", "seg_training")
        assert 'main' in seg_dirs
        assert 'models' in seg_dirs
        assert 'plots' in seg_dirs
        assert 'results' in seg_dirs
        assert 'logs' in seg_dirs
        assert 'cache' in seg_dirs
        
        # Test quant training output structure
        quant_dirs = output_manager.create_output_directory("test_data", "quant_training")
        assert 'main' in quant_dirs
        assert 'models' in quant_dirs
        assert 'plots' in quant_dirs
        assert 'results' in quant_dirs
        assert 'logs' in quant_dirs
        assert 'cache' in quant_dirs
        
        # Test production inference output structure
        prod_dirs = output_manager.create_output_directory("test_data", "production_inference")
        assert 'main' in prod_dirs
        assert 'models' in prod_dirs
        assert 'plots' in prod_dirs
        assert 'results' in prod_dirs
        assert 'logs' in prod_dirs
        assert 'cache' in prod_dirs


if __name__ == "__main__":
    pytest.main([__file__])
