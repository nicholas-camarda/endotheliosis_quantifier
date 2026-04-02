"""
Configuration management for dual-environment architecture.

This module provides configuration management for mode-specific settings
and integration with the dual-environment architecture.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .mode_manager import EnvironmentMode, ModeConfig
from .paths import (
    get_cache_path,
    get_data_path,
    get_logs_path,
    get_models_path,
    get_output_path,
    get_repo_root,
)

logger = logging.getLogger(__name__)


@dataclass
class GlobalConfig:
    """Global configuration settings."""

    # Paths
    # Raw inputs live here (unchanged originals)
    data_path: str = field(default_factory=lambda: str(get_data_path()))
    # All generated artifacts go here (processed data, patches, predictions)
    output_path: str = field(default_factory=lambda: str(get_output_path()))
    # Optional cache for intermediate pickles/npy files
    cache_path: str = field(default_factory=lambda: str(get_cache_path()))
    model_path: str = field(default_factory=lambda: str(get_models_path()))

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Environment
    default_mode: str = "auto"
    config_file: str = "~/.eq/config.json"


class ConfigManager:
    """
    Manages configuration for dual-environment architecture.
    
    This class provides configuration management for mode-specific settings,
    environment variables, and integration with the mode and backend managers.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path.home() / ".eq" / "config.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.global_config = self._load_global_config()
        self.mode_configs = self._load_mode_configs()
        
        logger.info(f"ConfigManager initialized with config path: {self.config_path}")
    
    def get_config(self, mode: EnvironmentMode) -> Dict[str, Any]:
        """
        Get configuration for a specific mode.
        
        Args:
            mode: Environment mode
            
        Returns:
            Configuration dictionary
        """
        config = asdict(self.global_config)
        config.update(asdict(self.mode_configs.get(mode.value, ModeConfig())))
        return config
    
    def update_config(self, mode: EnvironmentMode, **kwargs) -> None:
        """
        Update configuration for a specific mode.
        
        Args:
            mode: Environment mode
            **kwargs: Configuration parameters to update
        """
        if mode.value not in self.mode_configs:
            self.mode_configs[mode.value] = ModeConfig()
        
        current_config = self.mode_configs[mode.value]
        
        for key, value in kwargs.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        self._save_configs()
        logger.info(f"Updated configuration for mode: {mode.value}")
    
    def get_environment_config(self) -> Dict[str, str]:
        """
        Get environment-specific configuration.
        
        Returns:
            Dictionary of environment variables
        """
        return {
            "EQ_DATA_PATH": self.global_config.data_path,
            "EQ_OUTPUT_PATH": self.global_config.output_path,
            "EQ_CACHE_PATH": self.global_config.cache_path,
            "EQ_MODEL_PATH": self.global_config.model_path,
            "EQ_LOG_LEVEL": self.global_config.log_level,
            "EQ_DEFAULT_MODE": self.global_config.default_mode
        }
    
    def _load_global_config(self) -> GlobalConfig:
        """Load global configuration."""
        config = GlobalConfig(log_file=str(get_logs_path() / "eq.log"))

        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config_data = json.load(f)

                for key, value in config_data.get("global", {}).items():
                    if value is None or not hasattr(config, key):
                        continue

                    if key in {"data_path", "output_path", "cache_path", "model_path", "log_file"}:
                        value = self._resolve_config_path(value)

                    setattr(config, key, value)
            except Exception as e:
                logger.warning(f"Failed to load global configuration: {e}. Using defaults.")

        if os.getenv("EQ_DATA_PATH"):
            config.data_path = str(get_data_path())
        if os.getenv("EQ_OUTPUT_PATH"):
            config.output_path = str(get_output_path())
        if os.getenv("EQ_CACHE_PATH"):
            config.cache_path = str(get_cache_path())
        if os.getenv("EQ_MODEL_PATH"):
            config.model_path = str(get_models_path())
        if os.getenv("EQ_LOG_PATH") or os.getenv("EQ_LOGS_PATH"):
            config.log_file = str(get_logs_path() / "eq.log")
        if os.getenv("EQ_LOG_LEVEL"):
            config.log_level = os.getenv("EQ_LOG_LEVEL", config.log_level)
        if os.getenv("EQ_DEFAULT_MODE"):
            config.default_mode = os.getenv("EQ_DEFAULT_MODE", config.default_mode)

        return config

    @staticmethod
    def _resolve_config_path(raw_path: str) -> str:
        """Resolve persisted config paths relative to the repository root."""
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return str(path)
        return str((get_repo_root() / path).resolve())
    
    def _load_mode_configs(self) -> Dict[str, ModeConfig]:
        """Load mode-specific configurations."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                mode_configs = {}
                for mode_name, config_dict in config_data.get("modes", {}).items():
                    mode_configs[mode_name] = ModeConfig(**config_dict)
                
                logger.info(f"Loaded mode configurations from {self.config_path}")
                return mode_configs
                
            except Exception as e:
                logger.warning(f"Failed to load mode configurations: {e}. Using defaults.")
        
        # Return default configurations
        return {}
    
    def _save_configs(self) -> None:
        """Save configurations to file."""
        try:
            config_data = {
                "global": asdict(self.global_config),
                "modes": {
                    mode_name: asdict(config) 
                    for mode_name, config in self.mode_configs.items()
                }
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved configurations to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
    
    def get_config_summary(self) -> str:
        """
        Get a summary of all configurations.
        
        Returns:
            Formatted summary string
        """
        summary = "Global Configuration:\n"
        summary += f"  Data Path: {self.global_config.data_path}\n"
        summary += f"  Output Path: {self.global_config.output_path}\n"
        summary += f"  Log Level: {self.global_config.log_level}\n"
        summary += f"  Default Mode: {self.global_config.default_mode}\n"
        
        summary += "\nMode Configurations:\n"
        for mode_name, config in self.mode_configs.items():
            summary += f"  {mode_name.upper()}:\n"
            summary += f"    Device Mode: {config.device_mode}\n"
            summary += f"    Batch Size: {config.batch_size}\n"
            summary += f"    Num Workers: {config.num_workers}\n"
            summary += f"    Debug Mode: {config.debug_mode}\n"
        
        return summary.strip()


class PipelineConfigManager:
    """
    Backwards-compatible pipeline config helper.
    
    Provides simple accessors for commonly used paths and metadata
    expected by pipeline modules after the refactor.
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        self._cfg = config or ConfigManager()

    def get_segmentation_model_path(self, name: Optional[str] = None) -> str:
        """Return a reasonable default segmentation model path.

        If a specific name is provided, it is appended; otherwise a
        default filename is used.
        """
        base = Path(self._cfg.global_config.model_path)
        # Keep a simple, predictable structure
        models_dir = base / 'segmentation' / 'glomeruli'
        filename = f"{name}.pkl" if name else 'glomerulus_segmenter.pkl'
        return str(models_dir / filename)

    def get_segmentation_model_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        path = self.get_segmentation_model_path(name)
        return {
            'name': name or 'glomerulus_segmenter',
            'path': path
        }
