#!/usr/bin/env python3
"""Output directory management system for the endotheliosis quantifier pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from eq.utils.logger import get_logger


class OutputManager:
    """Manages output directory creation and organization for pipeline runs."""
    
    def __init__(self, base_output_dir: str = "output"):
        """Initialize the output manager.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = get_logger("eq.output_manager")
        
    def create_output_directory(
        self,
        data_source_name: str,
        run_type: str = "production",
        timestamp: Optional[str] = None,
        custom_suffix: Optional[str] = None
    ) -> Dict[str, Path]:
        """Create a structured output directory based on data source and run type.
        
        Args:
            data_source_name: Name of the input data source (e.g., 'preeclampsia_data')
            run_type: Type of run ('quick', 'production', 'smoke', 'development')
            timestamp: Optional timestamp string (defaults to current time)
            custom_suffix: Optional custom suffix for the directory name
            
        Returns:
            Dictionary containing paths to all created directories
        """
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        
        # Create directory name components
        name_parts = [data_source_name, timestamp, run_type]
        if custom_suffix:
            name_parts.append(custom_suffix)
        
        # Create the main output directory name
        output_dir_name = "_".join(name_parts)
        output_dir = self.base_output_dir / output_dir_name
        
        # Create subdirectories
        subdirs = {
            'main': output_dir,
            'models': output_dir / "models",
            'plots': output_dir / "plots", 
            'results': output_dir / "results",
            'reports': output_dir / "reports",
            'logs': output_dir / "logs",
            'cache': output_dir / "cache"
        }
        
        # Create all directories
        for dir_path in subdirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
        
        # Create metadata file
        metadata = {
            'data_source': data_source_name,
            'run_type': run_type,
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat(),
            'custom_suffix': custom_suffix,
            'directory_structure': {k: str(v) for k, v in subdirs.items()}
        }
        
        self._save_metadata(subdirs['main'], metadata)
        
        self.logger.info(f"Created output directory structure: {output_dir}")
        return subdirs
    
    def _save_metadata(self, output_dir: Path, metadata: Dict[str, Any]) -> None:
        """Save metadata about the output directory creation.
        
        Args:
            output_dir: Main output directory
            metadata: Metadata dictionary to save
        """
        import json
        
        metadata_file = output_dir / "run_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Saved metadata to: {metadata_file}")
    
    def get_data_source_name(self, data_path: str) -> str:
        """Extract data source name from data path.
        
        Args:
            data_path: Path to the data directory
            
        Returns:
            Extracted data source name
        """
        data_path = Path(data_path)
        
        # Try to extract meaningful name from path
        if data_path.name:
            # Remove common suffixes and prefixes
            name = data_path.name
            name = name.replace('_data', '').replace('data_', '')
            name = name.replace('_', ' ').title().replace(' ', '_')
            return name
        
        # Fallback to parent directory name
        if data_path.parent.name:
            return data_path.parent.name
        
        # Final fallback
        return "unknown_data_source"
    
    def create_run_summary(self, output_dirs: Dict[str, Path], run_info: Dict[str, Any]) -> None:
        """Create a summary file for the pipeline run.
        
        Args:
            output_dirs: Dictionary of output directory paths
            run_info: Information about the run (config, results, etc.)
        """
        summary_file = output_dirs['reports'] / "run_summary.md"
        
        summary_content = f"""# Pipeline Run Summary

## Run Information
- **Data Source**: {run_info.get('data_source', 'Unknown')}
- **Run Type**: {run_info.get('run_type', 'Unknown')}
- **Timestamp**: {run_info.get('timestamp', 'Unknown')}
- **Created**: {run_info.get('created_at', 'Unknown')}

## Configuration
```json
{run_info.get('config', {})}
```

## Results
```json
{run_info.get('results', {})}
```

## Directory Structure
- **Main**: {output_dirs['main']}
- **Models**: {output_dirs['models']}
- **Plots**: {output_dirs['plots']}
- **Results**: {output_dirs['results']}
- **Reports**: {output_dirs['reports']}
- **Logs**: {output_dirs['logs']}
- **Cache**: {output_dirs['cache']}

## Files Generated
{self._list_generated_files(output_dirs)}

---
Generated by Endotheliosis Quantifier Pipeline
"""
        
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        self.logger.info(f"Created run summary: {summary_file}")
    
    def _list_generated_files(self, output_dirs: Dict[str, Path]) -> str:
        """List all files generated in the output directories.
        
        Args:
            output_dirs: Dictionary of output directory paths
            
        Returns:
            Formatted string listing all files
        """
        file_list = []
        
        for dir_name, dir_path in output_dirs.items():
            if dir_path.exists():
                files = list(dir_path.glob('*'))
                if files:
                    file_list.append(f"\n### {dir_name.title()}")
                    for file_path in files:
                        if file_path.is_file():
                            file_list.append(f"- {file_path.name}")
        
        return '\n'.join(file_list) if file_list else "No files generated yet."
    
    def cleanup_old_runs(self, max_age_days: int = 30) -> None:
        """Clean up old output directories.
        
        Args:
            max_age_days: Maximum age in days for output directories to keep
        """
        import shutil
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        for output_dir in self.base_output_dir.iterdir():
            if output_dir.is_dir():
                # Try to extract timestamp from directory name
                try:
                    # Look for timestamp pattern YYYY-MM-DD_HHMMSS
                    import re
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{6})', output_dir.name)
                    if timestamp_match:
                        timestamp_str = timestamp_match.group(1)
                        dir_date = datetime.strptime(timestamp_str, '%Y-%m-%d_%H%M%S')
                        
                        if dir_date < cutoff_date:
                            shutil.rmtree(output_dir)
                            self.logger.info(f"Cleaned up old output directory: {output_dir}")
                            cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Could not parse timestamp for {output_dir}: {e}")
        
        self.logger.info(f"Cleanup complete: removed {cleaned_count} old output directories")


def create_output_directories(
    data_source_name: str,
    run_type: str = "production",
    timestamp: Optional[str] = None,
    custom_suffix: Optional[str] = None,
    base_output_dir: str = "output"
) -> Dict[str, Path]:
    """Convenience function to create output directories.
    
    Args:
        data_source_name: Name of the input data source
        run_type: Type of run ('quick', 'production', 'smoke', 'development')
        timestamp: Optional timestamp string
        custom_suffix: Optional custom suffix
        base_output_dir: Base directory for outputs
        
    Returns:
        Dictionary containing paths to all created directories
    """
    manager = OutputManager(base_output_dir)
    return manager.create_output_directory(
        data_source_name=data_source_name,
        run_type=run_type,
        timestamp=timestamp,
        custom_suffix=custom_suffix
    )
