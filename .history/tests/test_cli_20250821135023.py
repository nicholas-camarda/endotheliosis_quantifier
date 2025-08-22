"""Test CLI functionality."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_help():
    """Test that CLI help works."""
    result = subprocess.run(
        [sys.executable, "-m", "eq", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    assert result.returncode == 0
    assert "Endotheliosis Quantifier Pipeline" in result.stdout
    assert "data-load" in result.stdout
    assert "train-segmenter" in result.stdout
    assert "extract-features" in result.stdout
    assert "quantify" in result.stdout
    assert "pipeline" in result.stdout


def test_cli_data_load_help():
    """Test that data-load help works."""
    result = subprocess.run(
        [sys.executable, "-m", "eq", "data-load", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    assert result.returncode == 0
    assert "--data-dir" in result.stdout
    assert "--test-data-dir" in result.stdout
    assert "--cache-dir" in result.stdout


def test_cli_no_command():
    """Test that CLI shows help when no command is given."""
    result = subprocess.run(
        [sys.executable, "-m", "eq"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    assert result.returncode == 1
    assert "Endotheliosis Quantifier Pipeline" in result.stdout


def test_cli_invalid_command():
    """Test that CLI handles invalid commands gracefully."""
    result = subprocess.run(
        [sys.executable, "-m", "eq", "invalid-command"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    assert result.returncode == 2
    assert "error:" in result.stderr.lower()
