#!/usr/bin/env python3
"""Output directory management system for the endotheliosis quantifier pipeline."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

from eq.utils.logger import get_logger
from eq.utils.paths import get_repo_root


class OutputManager:
    """Manages output directory creation and organization for pipeline runs."""
    
    def __init__(self, base_output_dir: str = "output"):
        """Initialize the output manager.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        base_path = Path(base_output_dir).expanduser()
        if not base_path.is_absolute():
            base_path = get_repo_root() / base_path
        self.base_output_dir = base_path
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
        # Create directory name - ONE folder per data source, no timestamps
        output_dir_name = data_source_name.lower().replace(' ', '_')
        output_dir = self.base_output_dir / output_dir_name
        
        # Create subdirectories - SIMPLE structure only, NO LOGS HERE
        subdirs = {
            'main': output_dir,
            'models': output_dir / "models",
            'plots': output_dir / "plots", 
            'results': output_dir / "results",
            'cache': output_dir / "cache"
        }
        
        # Create all directories
        for dir_path in subdirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
        
        # Create metadata file with timestamp info but not in directory name
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            
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
        metadata_file = output_dir / "run_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Saved metadata to: {metadata_file}")
    
    def get_data_source_name(self, data_path: Union[str, Path]) -> str:
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
        summary_file = output_dirs['main'] / "run_summary.md"

        summary_content = "\n".join([
            "# Pipeline Run Summary",
            "",
            "## Run Information",
            f"- **Data Source**: {run_info.get('data_source', 'Unknown')}",
            f"- **Run Type**: {run_info.get('run_type', 'Unknown')}",
            f"- **Timestamp**: {run_info.get('timestamp', 'Unknown')}",
            f"- **Created**: {run_info.get('created_at', 'Unknown')}",
            "",
            "## Configuration",
            "```json",
            json.dumps(run_info.get('config', {}), indent=2, default=str),
            "```",
            "",
            "## Results",
            "```json",
            json.dumps(run_info.get('results', {}), indent=2, default=str),
            "```",
            "",
            "## Directory Structure",
            f"- **Main**: {output_dirs['main']}",
            f"- **Models**: {output_dirs['models']}",
            f"- **Plots**: {output_dirs['plots']}",
            f"- **Results**: {output_dirs['results']}",
            f"- **Cache**: {output_dirs['cache']}",
            "",
            "## Files Generated",
            self._list_generated_files(output_dirs),
            "",
            "---",
            "Generated by Endotheliosis Quantifier Pipeline",
            "",
        ])
        
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

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0

        if not self.base_output_dir.exists():
            self.logger.info(f"Cleanup skipped: base output directory does not exist: {self.base_output_dir}")
            return

        for output_dir in self.base_output_dir.iterdir():
            if not output_dir.is_dir():
                continue

            dir_date = self._get_run_created_at(output_dir)
            if dir_date is None:
                continue

            if dir_date < cutoff_date:
                shutil.rmtree(output_dir)
                self.logger.info(f"Cleaned up old output directory: {output_dir}")
                cleaned_count += 1

        self.logger.info(f"Cleanup complete: removed {cleaned_count} old output directories")

    def _get_run_created_at(self, output_dir: Path) -> Optional[datetime]:
        """Return the recorded creation time for an output directory when available."""
        metadata_file = output_dir / "run_metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
                created_at = metadata.get("created_at")
                if created_at:
                    return datetime.fromisoformat(created_at)
            except Exception as e:
                self.logger.warning(f"Could not read metadata for {output_dir}: {e}")

        try:
            import re

            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{6})', output_dir.name)
            if timestamp_match:
                return datetime.strptime(timestamp_match.group(1), '%Y-%m-%d_%H%M%S')
        except Exception as e:
            self.logger.warning(f"Could not parse timestamp for {output_dir}: {e}")

        return None


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
