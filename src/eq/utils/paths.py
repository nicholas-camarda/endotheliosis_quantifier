"""Path utilities for the eq package."""

from pathlib import Path
from typing import List, Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path() -> Path:
    """
    Get the default data directory path.
    
    Returns:
        Path to the data directory
    """
    return Path("data/preeclampsia_data")


def get_output_path() -> Path:
    """
    Get the default output directory path.
    
    Returns:
        Path to the output directory
    """
    return Path("output")


def get_cache_path() -> Path:
    """
    Get the default cache directory path.
    
    Returns:
        Path to the cache directory
    """
    return get_data_path() / "cache"


def find_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True,
) -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search in
        pattern: File pattern to match (default: "*")
        recursive: Whether to search recursively (default: True)
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def get_relative_path(path: Union[str, Path], base: Union[str, Path] = None) -> str:
    """
    Get a relative path from a base directory.
    
    Args:
        path: Path to make relative
        base: Base directory (default: current working directory)
        
    Returns:
        Relative path string
    """
    path = Path(path)
    if base is None:
        base = Path.cwd()
    else:
        base = Path(base)
    
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
