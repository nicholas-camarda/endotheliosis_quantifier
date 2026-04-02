"""Lightweight package metadata and environment helpers for EQ."""

from __future__ import annotations

import os
import subprocess
import sys

__version__ = '0.1.0'


def ensure_conda_environment() -> bool:
    """Restart the current command inside the ``eq`` conda env when available."""
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env == 'eq' or 'envs/eq' in sys.executable:
        return True

    for launcher in ('conda', 'mamba'):
        try:
            result = subprocess.run(
                [launcher, 'run', '-n', 'eq', 'python', '-c', 'import sys; print(sys.executable)'],
                capture_output=True,
                check=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

        env_python = result.stdout.strip()
        if not env_python:
            continue

        rerun = subprocess.run([env_python, *sys.argv], cwd=os.getcwd(), check=False)
        sys.exit(rerun.returncode)

    return False


if os.environ.get('EQ_AUTO_CONDA', '0') == '1':
    try:
        ensure_conda_environment()
    except Exception:
        # Importing the package should stay safe even if environment activation fails.
        pass


__all__ = ['__version__', 'ensure_conda_environment']
