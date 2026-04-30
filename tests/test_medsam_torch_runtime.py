"""Tests for MedSAM subprocess interpreter and training-backend resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eq.evaluation.medsam_torch_runtime import (
    resolve_medsam_torch_python,
    training_backend,
)


def test_resolve_prefers_inference_python_override(tmp_path: Path) -> None:
    explicit = tmp_path / "special_py"
    explicit.write_bytes(b"")
    cfg = {
        "medsam": {"inference_python": str(explicit), "python": "", "device": "cpu"},
        "run": {},
    }
    assert resolve_medsam_torch_python(cfg) == explicit


def test_training_backend_explicit_upstream_overrides_darwin_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    cfg = {
        "medsam": {"device": "mps"},
        "training": {"backend": "upstream_medsam"},
    }
    assert training_backend(cfg) == "upstream_medsam"


def test_resolve_darwin_mps_prefers_run_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    run_py = tmp_path / "eq-mac/bin/python"
    run_py.parent.mkdir(parents=True)
    run_py.write_bytes(b"")
    medsam_py = tmp_path / "medsam/bin/python"
    medsam_py.parent.mkdir(parents=True)
    medsam_py.write_bytes(b"")
    cfg = {
        "run": {"python": str(run_py)},
        "medsam": {"python": str(medsam_py), "device": "mps"},
    }
    assert resolve_medsam_torch_python(cfg) == run_py


def test_training_backend_defaults_native_adapter_on_darwin_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    cfg = {"medsam": {"device": "mps"}, "training": {}}
    assert training_backend(cfg) == "eq_native_adapter"


def test_training_backend_defaults_upstream_off_macos_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = {"medsam": {"device": "mps"}, "training": {}}
    assert training_backend(cfg) == "upstream_medsam"
