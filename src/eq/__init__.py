"""
Endotheliosis Quantifier (EQ) Package

A comprehensive pipeline for quantifying endotheliosis in preeclampsia histology images.
"""

import os
import platform
import subprocess
import sys


def ensure_conda_environment():
    """Ensure the eq conda environment is activated."""
    # Check if we're already in the eq environment
    if 'eq' in sys.executable or ('CONDA_DEFAULT_ENV' in os.environ and os.environ['CONDA_DEFAULT_ENV'] == 'eq'):
        return True
    
    print("🔧 Attempting to activate eq conda environment...")
    
    try:
        # Try to activate the conda environment
        result = subprocess.run([
            'conda', 'run', '-n', 'eq', 'python', '-c', 
            'import sys; print(sys.executable)'
        ], capture_output=True, text=True, check=True)
        
        conda_python = result.stdout.strip()
        print(f"✅ Found eq environment: {conda_python}")
        
        # Restart the script in the conda environment
        print("🔄 Restarting script in eq environment...")
        result = subprocess.run([conda_python] + sys.argv, cwd=os.getcwd())
        sys.exit(result.returncode)
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try mamba
        try:
            result = subprocess.run([
                'mamba', 'run', '-n', 'eq', 'python', '-c',
                'import sys; print(sys.executable)'
            ], capture_output=True, text=True, check=True)
            
            mamba_python = result.stdout.strip()
            print(f"✅ Found eq environment via mamba: {mamba_python}")
            
            print("🔄 Restarting script in eq environment...")
            result = subprocess.run([mamba_python] + sys.argv, cwd=os.getcwd())
            sys.exit(result.returncode)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Could not activate eq environment. Using current Python environment.")
            return False
    
    return True


# Auto-activate conda environment only when explicitly enabled
# Set EQ_AUTO_CONDA=1 to enable auto-activation during CLI runs
if os.environ.get('EQ_AUTO_CONDA', '0') == '1':
    try:
        ensure_conda_environment()
    except Exception:
        # Never fail import due to environment activation issues
        pass

def ensure_directory_structure():
    """Ensure all required directories exist for the package structure."""
    import pathlib
    
    # Get the package directory
    package_dir = pathlib.Path(__file__).parent
    
    # Required directories from spec
    required_dirs = [
        'core',
        'data_management',  # New consolidated data management
        'processing',
        'training',         # New dedicated training scripts
        'inference',        # New dedicated inference scripts
        'models',
        'pipeline',
        'evaluation',
        'utils'
    ]
    
    # Create directories if they don't exist
    for dir_name in required_dirs:
        dir_path = package_dir / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create __init__.py if it doesn't exist
        init_file = dir_path / '__init__.py'
        if not init_file.exists():
            init_file.touch()

# Ensure directory structure exists
ensure_directory_structure()

# Package imports - updated for consolidated structure
from . import core, data_management, evaluation, models, pipeline, processing, utils

# Optional imports for new directories (created on-demand)
try:
    from . import training
except ImportError:
    pass

try:
    from . import inference
except ImportError:
    pass

__all__ = [
    "core",
    "data_management",
    "training",
    "inference",
    "evaluation",
    "models",
    "pipeline",
    "processing",
    "utils"
]
