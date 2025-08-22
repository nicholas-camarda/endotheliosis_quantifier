"""
Configuration management for dual-environment architecture.

This module provides configuration management for the dual-environment
architecture with explicit mode selection between development and production.
"""

from .mode_manager import ModeManager, EnvironmentMode
from .backend_manager import BackendManager
from .config_manager import ConfigManager
from .auto_suggestion import AutoSuggestionSystem, Suggestion

__all__ = [
    'ModeManager',
    'EnvironmentMode', 
    'BackendManager',
    'ConfigManager',
    'AutoSuggestionSystem',
    'Suggestion'
]
