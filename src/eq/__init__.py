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

# Version info
__version__ = "1.0.0"
__author__ = "Nicholas Camarda"
__email__ = "ncamarda93@gmail.com"

# Package imports - updated for consolidated structure
from . import core, data, evaluation, models, pipeline, processing, quantification, utils

__all__ = [
    "core",
    "data", 
    "evaluation",
    "models",
    "pipeline",
    "processing",
    "quantification",
    "utils"
]
