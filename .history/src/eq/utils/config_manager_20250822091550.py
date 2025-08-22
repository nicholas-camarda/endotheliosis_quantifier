#!/usr/bin/env python3
"""Configuration management for the endotheliosis quantifier pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from eq.utils.logger import get_logger


class PipelineConfigManager:
    """Manages pipeline configuration including model selection."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (defaults to pipeline_config.json)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "pipeline_config.json"
        
        self.config_path = Path(config_path)
        self.logger = get_logger("eq.config_manager")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            self.logger.warning(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is not found.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "segmentation": {
                "default_model": "glomeruli",
                "available_models": {
                    "glomeruli": {
                        "name": "Glomerulus Segmentation Model",
                        "path": "segmentation_model_dir/glomerulus_segmentation_model-dynamic_unet-e50_b16_s84.pkl",
                        "description": "Pre-trained model for glomerulus detection in kidney tissue"
                    },
                    "mitochondria": {
                        "name": "Mitochondria Segmentation Model",
                        "path": "segmentation_model_dir/mito_dynamic_unet_seg_model-e50_b16.pkl", 
                        "description": "Pre-trained model for mitochondria detection"
                    }
                }
            },
            "quantification": {
                "default_model": "endotheliosis",
                "available_models": {
                    "endotheliosis": {
                        "name": "Endotheliosis Quantification Model",
                        "path": "models/endotheliosis_quantifier.pkl",
                        "description": "Regression model for endotheliosis scoring"
                    }
                }
            },
            "pipeline": {
                "default_batch_size": 8,
                "default_epochs": 50,
                "image_size": 256,
                "device_mode": "production"
            }
        }
    
    def get_segmentation_model_path(self, model_name: Optional[str] = None) -> str:
        """Get the path to a segmentation model.
        
        Args:
            model_name: Name of the model (defaults to default_model)
            
        Returns:
            Path to the model file
        """
        if model_name is None:
            model_name = self.config["segmentation"]["default_model"]
        
        if model_name not in self.config["segmentation"]["available_models"]:
            self.logger.warning(f"Unknown segmentation model: {model_name}, using default")
            model_name = self.config["segmentation"]["default_model"]
        
        return self.config["segmentation"]["available_models"][model_name]["path"]
    
    def get_segmentation_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a segmentation model.
        
        Args:
            model_name: Name of the model (defaults to default_model)
            
        Returns:
            Model information dictionary
        """
        if model_name is None:
            model_name = self.config["segmentation"]["default_model"]
        
        if model_name not in self.config["segmentation"]["available_models"]:
            self.logger.warning(f"Unknown segmentation model: {model_name}, using default")
            model_name = self.config["segmentation"]["default_model"]
        
        return self.config["segmentation"]["available_models"][model_name]
    
    def get_available_segmentation_models(self) -> Dict[str, Any]:
        """Get all available segmentation models.
        
        Returns:
            Dictionary of available models
        """
        return self.config["segmentation"]["available_models"]
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration.
        
        Returns:
            Pipeline configuration dictionary
        """
        return self.config["pipeline"]
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
        self.logger.info("Configuration updated")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
