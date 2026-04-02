"""Path utilities for the eq package."""

import os
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


def get_project_root() -> Path:
    """Return the canonical project root for the eq repository."""
    return Path(__file__).resolve().parents[3]


def get_runtime_root() -> Path:
    """Return the canonical runtime root for the eq project."""
    env_root = os.getenv("EQ_RUNTIME_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.home() / "ProjectsRuntime" / "endotheliosis_quantifier"


def get_cloud_root() -> Path:
    """Return the canonical cloud root for the eq project."""
    env_root = os.getenv("EQ_CLOUD_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" / "SideProjects" / "endotheliosis_quantifier"


def resolve_project_path(path: Union[str, Path]) -> Path:
    """Resolve a path relative to the project root when not already absolute."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return get_project_root() / path


def resolve_runtime_path(path: Union[str, Path]) -> Path:
    """Resolve a path relative to the runtime root when not already absolute."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return get_runtime_root() / path


def get_logs_path() -> Path:
    """Get the default logs directory path."""
    return resolve_runtime_path(os.getenv("EQ_LOG_PATH", "logs"))


def get_raw_data_path() -> Path:
    """Get the default raw-data directory path."""
    return resolve_runtime_path(os.getenv("EQ_RAW_DATA_PATH", "raw_data"))


def get_derived_data_path() -> Path:
    """Get the default derived-data directory path."""
    return resolve_runtime_path(os.getenv("EQ_DERIVED_DATA_PATH", "derived_data"))


def get_data_path() -> Path:
    """
    Get the default data directory path.
    
    Returns:
        Path to the data directory
    """
    return resolve_runtime_path(os.getenv("EQ_DATA_PATH", "data/preeclampsia_data"))


def get_output_path() -> Path:
    """
    Get the default output directory path.
    
    Returns:
        Path to the output directory
    """
    return resolve_runtime_path(os.getenv("EQ_OUTPUT_PATH", "output"))


def get_models_path() -> Path:
    """
    Get the default models directory path.
    
    Returns:
        Path to the models directory
    """
    return resolve_runtime_path(os.getenv("EQ_MODEL_PATH", "models"))


def get_cache_path() -> Path:
    """
    Get the default cache directory path.
    
    Returns:
        Path to the cache directory
    """
    env_cache = os.getenv("EQ_CACHE_PATH")
    if env_cache:
        return resolve_runtime_path(env_cache)
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
        base = get_project_root()
    else:
        base = resolve_project_path(base)
    
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
