"""
Automatic suggestion system for dual-environment architecture.

This module provides automatic suggestions for mode selection, backend choice,
and configuration optimization based on hardware capabilities.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from eq.utils.hardware_detection import get_hardware_capabilities, get_device_recommendation, get_optimal_batch_size
from .mode_manager import EnvironmentMode
from .backend_manager import BackendManager

logger = logging.getLogger(__name__)


@dataclass
class Suggestion:
    """A suggestion for configuration optimization."""
    
    category: str
    title: str
    description: str
    priority: str  # "high", "medium", "low"
    current_value: Any
    suggested_value: Any
    reasoning: str


class AutoSuggestionSystem:
    """
    Automatic suggestion system for dual-environment architecture.
    
    This class provides intelligent suggestions for mode selection,
    backend configuration, and performance optimization based on
    hardware capabilities and current system state.
    """
    
    def __init__(self):
        """Initialize the auto suggestion system."""
        self.hardware_capabilities = get_hardware_capabilities()
        self.backend_manager = BackendManager()
    
    def get_mode_suggestions(self) -> List[Suggestion]:
        """
        Get suggestions for mode selection.
        
        Returns:
            List of mode suggestions
        """
        suggestions = []
        
        # Suggest production mode for powerful hardware
        if (self.hardware_capabilities.hardware_tier.value in ["standard", "powerful"] and 
            self.hardware_capabilities.backend_type and self.hardware_capabilities.backend_type.value != "CPU"):
            
            suggestions.append(Suggestion(
                category="mode",
                title="Consider Production Mode",
                description="Your hardware supports production mode for optimal performance",
                priority="high",
                current_value="development",
                suggested_value="production",
                reasoning=f"Hardware tier: {self.hardware_capabilities.hardware_tier.value}, "
                         f"Backend: {self.hardware_capabilities.primary_backend}"
            ))
        
        # Suggest development mode for basic hardware
        elif self.hardware_capabilities.hardware_tier.value == "basic":
            suggestions.append(Suggestion(
                category="mode",
                title="Use Development Mode",
                description="Development mode is recommended for your hardware configuration",
                priority="high",
                current_value="production",
                suggested_value="development",
                reasoning=f"Hardware tier: {self.hardware_capabilities.hardware_tier.value} "
                         f"may not support production workloads efficiently"
            ))
        
        return suggestions
    
    def get_backend_suggestions(self) -> List[Suggestion]:
        """
        Get suggestions for backend configuration.
        
        Returns:
            List of backend suggestions
        """
        suggestions = []
        current_backend = self.backend_manager.current_backend
        
        # Suggest GPU backends for production
        if (self.hardware_capabilities.hardware_tier.value in ["standard", "powerful"] and
            current_backend.backend_type.value == "CPU"):
            
            if self.hardware_capabilities.backend_type and self.hardware_capabilities.backend_type.value == "MPS":
                suggestions.append(Suggestion(
                    category="backend",
                    title="Enable MPS Backend",
                    description="Apple Silicon GPU detected, enable MPS for better performance",
                    priority="high",
                    current_value="CPU",
                    suggested_value="MPS",
                    reasoning="MPS backend available and hardware supports it"
                ))
            
            elif self.hardware_capabilities.backend_type and self.hardware_capabilities.backend_type.value == "CUDA":
                suggestions.append(Suggestion(
                    category="backend",
                    title="Enable CUDA Backend",
                    description="NVIDIA GPU detected, enable CUDA for better performance",
                    priority="high",
                    current_value="CPU",
                    suggested_value="CUDA",
                    reasoning="CUDA backend available and hardware supports it"
                ))
        
        # Suggest CPU fallback for basic hardware
        elif (self.hardware_capabilities.hardware_tier.value == "basic" and
              current_backend.backend_type.value != "CPU"):
            
            suggestions.append(Suggestion(
                category="backend",
                title="Consider CPU Backend",
                description="CPU backend may be more stable for basic hardware",
                priority="medium",
                current_value=current_backend.backend_type.value,
                suggested_value="CPU",
                reasoning="Basic hardware tier may benefit from CPU stability"
            ))
        
        return suggestions
    
    def get_performance_suggestions(self) -> List[Suggestion]:
        """
        Get suggestions for performance optimization.
        
        Returns:
            List of performance suggestions
        """
        suggestions = []
        
        # Batch size suggestions
        optimal_batch_size = get_optimal_batch_size("auto")
        if optimal_batch_size > 8:
            suggestions.append(Suggestion(
                category="performance",
                title="Increase Batch Size",
                description=f"Hardware supports larger batch size for better throughput",
                priority="medium",
                current_value=4,
                suggested_value=optimal_batch_size,
                reasoning=f"Optimal batch size: {optimal_batch_size} based on hardware capabilities"
            ))
        
        # Memory optimization suggestions
        if self.hardware_capabilities.total_memory_gb < 16:
            suggestions.append(Suggestion(
                category="performance",
                title="Enable Memory Optimization",
                description="Enable memory optimization for limited RAM",
                priority="high",
                current_value=False,
                suggested_value=True,
                reasoning=f"System has {self.hardware_capabilities.total_memory_gb:.1f}GB RAM"
            ))
        
        # Mixed precision suggestions
        if self.hardware_capabilities.primary_backend in ["MPS", "CUDA"]:
            suggestions.append(Suggestion(
                category="performance",
                title="Enable Mixed Precision",
                description="Enable mixed precision for faster training",
                priority="medium",
                current_value=False,
                suggested_value=True,
                reasoning="GPU backend supports mixed precision training"
            ))
        
        return suggestions
    
    def get_all_suggestions(self) -> Dict[str, List[Suggestion]]:
        """
        Get all suggestions organized by category.
        
        Returns:
            Dictionary of suggestions by category
        """
        return {
            "mode": self.get_mode_suggestions(),
            "backend": self.get_backend_suggestions(),
            "performance": self.get_performance_suggestions()
        }
    
    def get_priority_suggestions(self, priority: str = "high") -> List[Suggestion]:
        """
        Get suggestions filtered by priority.
        
        Args:
            priority: Priority level to filter by
            
        Returns:
            List of suggestions with specified priority
        """
        all_suggestions = self.get_all_suggestions()
        priority_suggestions = []
        
        for category_suggestions in all_suggestions.values():
            for suggestion in category_suggestions:
                if suggestion.priority == priority:
                    priority_suggestions.append(suggestion)
        
        return priority_suggestions
    
    def get_suggestions_summary(self) -> str:
        """
        Get a formatted summary of all suggestions.
        
        Returns:
            Formatted summary string
        """
        all_suggestions = self.get_all_suggestions()
        
        summary = "Auto-Suggestion Summary:\n"
        summary += "=" * 50 + "\n"
        
        for category, suggestions in all_suggestions.items():
            if suggestions:
                summary += f"\n{category.upper()} SUGGESTIONS:\n"
                summary += "-" * 30 + "\n"
                
                for suggestion in suggestions:
                    priority_icon = "🔴" if suggestion.priority == "high" else "🟡" if suggestion.priority == "medium" else "🟢"
                    summary += f"{priority_icon} {suggestion.title}\n"
                    summary += f"   {suggestion.description}\n"
                    summary += f"   Current: {suggestion.current_value} → Suggested: {suggestion.suggested_value}\n"
                    summary += f"   Reason: {suggestion.reasoning}\n\n"
        
        if not any(all_suggestions.values()):
            summary += "\n✅ No suggestions at this time. Configuration appears optimal.\n"
        
        return summary.strip()
    
    def apply_suggestions(self, suggestions: List[Suggestion]) -> Dict[str, Any]:
        """
        Apply a list of suggestions and return the changes made.
        
        Args:
            suggestions: List of suggestions to apply
            
        Returns:
            Dictionary of changes made
        """
        changes = {}
        
        for suggestion in suggestions:
            if suggestion.category == "backend":
                # Apply backend changes
                if suggestion.suggested_value == "MPS":
                    success = self.backend_manager.switch_backend(self.backend_manager._backends["MPS"].backend_type)
                    if success:
                        changes["backend"] = "MPS"
                elif suggestion.suggested_value == "CUDA":
                    success = self.backend_manager.switch_backend(self.backend_manager._backends["CUDA"].backend_type)
                    if success:
                        changes["backend"] = "CUDA"
                elif suggestion.suggested_value == "CPU":
                    success = self.backend_manager.switch_backend(self.backend_manager._backends["CPU"].backend_type)
                    if success:
                        changes["backend"] = "CPU"
            
            # Note: Mode and performance suggestions would be applied through the respective managers
            # This is a simplified implementation
        
        return changes
