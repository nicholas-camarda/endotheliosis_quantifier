"""
Mode management for dual-environment architecture.

This module provides explicit mode selection between development and production
environments with automatic hardware-based suggestions.
"""

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from eq.utils.hardware_detection import get_device_recommendation, get_hardware_capabilities

logger = logging.getLogger(__name__)


class EnvironmentMode(Enum):
    """Environment modes for dual-environment architecture."""
    
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    AUTO = "auto"


@dataclass
class ModeConfig:
    """Configuration for a specific environment mode."""
    
    # Hardware settings
    device_mode: str = "auto"
    batch_size: int = 0  # 0 means auto-detect
    memory_limit_gb: Optional[float] = None
    
    # Performance settings
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    verbose: bool = False
    
    # Output settings
    save_intermediate: bool = True
    checkpoint_frequency: int = 5
    
    # Development-specific settings
    debug_mode: bool = False
    profile_performance: bool = False
    
    # Production-specific settings
    optimize_memory: bool = True
    use_distributed: bool = False


class ModeManager:
    """
    Manages environment mode selection and configuration.
    
    This class provides explicit mode selection between development and production
    environments, with automatic hardware-based suggestions and mode-specific
    configuration management.
    """
    
    def __init__(self, mode: Optional[EnvironmentMode] = None, config_path: Optional[Path] = None):
        """
        Initialize the mode manager.
        
        Args:
            mode: Initial environment mode (defaults to AUTO)
            config_path: Path to configuration file (defaults to ~/.eq/config.json)
        """
        self.config_path = config_path or Path.home() / ".eq" / "config.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set initial mode
        self._mode = mode or EnvironmentMode.AUTO
        
        # Load or create default configurations
        self._configs = self._load_configurations()
        
        # Validate and set current mode
        self.set_mode(self._mode)
        
        logger.info(f"ModeManager initialized with mode: {self._mode.value}")
    
    @property
    def current_mode(self) -> EnvironmentMode:
        """Get the current environment mode."""
        return self._mode
    
    @property
    def current_config(self) -> ModeConfig:
        """Get the current mode configuration."""
        return self._configs[self._mode.value]
    
    def set_mode(self, mode: EnvironmentMode) -> None:
        """
        Set the environment mode.
        
        Args:
            mode: Environment mode to set
        """
        if not isinstance(mode, EnvironmentMode):
            raise ValueError(f"Invalid mode: {mode}. Must be EnvironmentMode enum.")
        
        # Handle AUTO mode
        if mode == EnvironmentMode.AUTO:
            suggested_mode = self._get_suggested_mode()
            logger.info(f"Auto mode selected. Suggested mode: {suggested_mode.value}")
            mode = suggested_mode
        
        self._mode = mode
        logger.info(f"Environment mode set to: {self._mode.value}")
    
    def get_mode_info(self) -> Dict[str, Any]:
        """
        Get information about the current mode.
        
        Returns:
            Dictionary with mode information
        """
        hardware_capabilities = get_hardware_capabilities()
        
        return {
            "mode": self._mode.value,
            "config": asdict(self.current_config),
            "hardware_capabilities": hardware_capabilities.__dict__,
            "device_recommendation": get_device_recommendation(self.current_config.device_mode)
        }
    
    def switch_mode(self, mode: EnvironmentMode) -> None:
        """
        Switch to a different environment mode.
        
        Args:
            mode: New environment mode
        """
        logger.info(f"Switching from {self._mode.value} to {mode.value}")
        self.set_mode(mode)
        self._save_configurations()
    
    def get_suggested_mode(self) -> EnvironmentMode:
        """
        Get the suggested mode based on hardware capabilities.
        
        Returns:
            Suggested environment mode
        """
        return self._get_suggested_mode()
    
    def validate_mode(self, mode: EnvironmentMode) -> Tuple[bool, str]:
        """
        Validate if a mode is suitable for the current hardware.
        
        Args:
            mode: Mode to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if mode == EnvironmentMode.AUTO:
            return True, "Auto mode is always valid"
        
        hardware_capabilities = get_hardware_capabilities()
        
        if mode == EnvironmentMode.PRODUCTION:
            # Production mode requires significant resources
            if hardware_capabilities.hardware_tier.value == "basic":
                return False, "Production mode requires STANDARD or POWERFUL hardware tier"
            if hardware_capabilities.backend_type and hardware_capabilities.backend_type.value == "CPU":
                return False, "Production mode requires GPU acceleration (MPS or CUDA)"
        
        elif mode == EnvironmentMode.DEVELOPMENT:
            # Development mode is more flexible
            return True, "Development mode is suitable for current hardware"
        
        return True, "Mode is suitable for current hardware"
    
    def _get_suggested_mode(self) -> EnvironmentMode:
        """
        Get the suggested mode based on hardware capabilities.
        
        Returns:
            Suggested environment mode
        """
        hardware_capabilities = get_hardware_capabilities()
        
        # Suggest production for powerful hardware with GPU
        if (hardware_capabilities.hardware_tier.value in ["standard", "powerful"] and 
            hardware_capabilities.backend_type and hardware_capabilities.backend_type.value != "CPU"):
            return EnvironmentMode.PRODUCTION
        
        # Default to development for other cases
        return EnvironmentMode.DEVELOPMENT
    
    def _load_configurations(self) -> Dict[str, ModeConfig]:
        """
        Load mode configurations from file or create defaults.
        
        Returns:
            Dictionary of mode configurations
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                configs = {}
                for mode_name, config_dict in config_data.items():
                    configs[mode_name] = ModeConfig(**config_dict)
                
                logger.info(f"Loaded configurations from {self.config_path}")
                return configs
                
            except Exception as e:
                logger.warning(f"Failed to load configurations: {e}. Using defaults.")
        
        # Create default configurations
        configs = {
            EnvironmentMode.DEVELOPMENT.value: ModeConfig(
                device_mode="auto",
                batch_size=0,
                num_workers=2,
                debug_mode=True,
                profile_performance=True,
                save_intermediate=True,
                log_level="DEBUG"
            ),
            EnvironmentMode.PRODUCTION.value: ModeConfig(
                device_mode="auto", 
                batch_size=0,
                num_workers=8,
                optimize_memory=True,
                use_distributed=False,
                save_intermediate=False,
                log_level="INFO"
            ),
            EnvironmentMode.AUTO.value: ModeConfig(
                device_mode="auto",
                batch_size=0,
                num_workers=4,
                log_level="INFO"
            )
        }
        
        # Save default configurations
        self._save_configurations(configs)
        
        return configs
    
    def _save_configurations(self, configs: Optional[Dict[str, ModeConfig]] = None) -> None:
        """
        Save mode configurations to file.
        
        Args:
            configs: Configurations to save (uses current if None)
        """
        if configs is None:
            configs = self._configs
        
        try:
            config_data = {
                mode_name: asdict(config) 
                for mode_name, config in configs.items()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved configurations to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
    
    def update_config(self, mode: EnvironmentMode, **kwargs) -> None:
        """
        Update configuration for a specific mode.
        
        Args:
            mode: Mode to update
            **kwargs: Configuration parameters to update
        """
        if mode.value not in self._configs:
            raise ValueError(f"Unknown mode: {mode.value}")
        
        current_config = self._configs[mode.value]
        
        # Update configuration
        for key, value in kwargs.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        # Save updated configurations
        self._save_configurations()
        
        logger.info(f"Updated configuration for mode: {mode.value}")
    
    def get_mode_summary(self) -> str:
        """
        Get a summary of the current mode and configuration.
        
        Returns:
            Formatted summary string
        """
        info = self.get_mode_info()
        config = info["config"]
        
        backend_info = info['hardware_capabilities'].get('backend_type', 'Unknown')
        if backend_info and hasattr(backend_info, 'value'):
            backend_info = backend_info.value
        
        summary = f"""
Environment Mode: {info['mode'].upper()}
Device Recommendation: {info['device_recommendation'][0]}
Hardware Tier: {info['hardware_capabilities']['hardware_tier']}
Primary Backend: {backend_info}

Configuration:
- Batch Size: {config['batch_size']} (auto-detect if 0)
- Num Workers: {config['num_workers']}
- Mixed Precision: {config['mixed_precision']}
- Debug Mode: {config['debug_mode']}
- Log Level: {config['log_level']}
"""
        return summary.strip()
