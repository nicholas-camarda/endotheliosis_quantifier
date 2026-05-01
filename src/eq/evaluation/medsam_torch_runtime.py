"""MedSAM SAM subprocess runtime: device env and Python interpreter selection (MPS/CUDA)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def medsam_subprocess_extra_env(*, device: str) -> dict[str, str]:
    """Environment variables merged into MedSAM SAM batch/training subprocesses."""
    env: dict[str, str] = {
        # So epoch progress and prints appear promptly when stdout/stderr are piped (e.g. tee).
        "PYTHONUNBUFFERED": "1",
    }
    name = str(device).strip().lower()
    if name == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK", "1"
        )
    return env


def resolve_medsam_torch_python(config: dict[str, Any]) -> Path:
    """Interpreter for SAM inference and optional training adapters.

    On macOS with ``medsam.device: mps``, defaults to ``run.python`` (eq-mac) when
    ``medsam.inference_python`` is unset, so subprocesses use the PyTorch Metal build.

    Override with ``medsam.inference_python`` when you need a different env.
    """
    medsam = config.get("medsam") if isinstance(config.get("medsam"), dict) else {}
    explicit = str(medsam.get("inference_python", "")).strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"medsam.inference_python does not exist: {path}")
        return path

    default_raw = str(medsam.get("python", "")).strip()
    default_py = Path(default_raw).expanduser() if default_raw else Path()
    device = str(medsam.get("device", "cpu")).strip().lower()
    run_cfg = config.get("run") if isinstance(config.get("run"), dict) else {}
    runtime_raw = str(run_cfg.get("python", "")).strip()
    runtime_py = Path(runtime_raw).expanduser() if runtime_raw else None

    if sys.platform == "darwin" and device == "mps":
        if runtime_py and runtime_py.exists():
            return runtime_py
        if default_raw and default_py.exists():
            return default_py
        raise FileNotFoundError(
            "macOS+MPS requires an existing run.python (eq-mac) or medsam.python; "
            f"got run.python={runtime_raw!r}, medsam.python={default_raw!r}"
        )

    if default_raw and default_py.exists():
        return default_py
    raise FileNotFoundError(
        f"MedSAM Python interpreter does not exist: {default_py or default_raw!r}"
    )


def training_backend(config: dict[str, Any]) -> str:
    """Select MedSAM fine-tuning command family.

    * ``eq_native_adapter`` — in-repo ``medsam_glomeruli_adapter`` (MPS-tested on macOS).
    * ``upstream_medsam`` — vendor ``train_one_gpu.py`` (CUDA-oriented).

    Default: ``eq_native_adapter`` on macOS when ``medsam.device`` is ``mps``, else upstream.
    """
    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    explicit = str(training.get("backend", "")).strip().lower()
    if explicit in ("eq_native_adapter", "upstream_medsam"):
        return explicit
    medsam = config.get("medsam") if isinstance(config.get("medsam"), dict) else {}
    device = str(medsam.get("device", "cpu")).strip().lower()
    if sys.platform == "darwin" and device == "mps":
        return "eq_native_adapter"
    return "upstream_medsam"
