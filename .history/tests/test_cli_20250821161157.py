"""Test CLI functionality."""

import subprocess
import sys
from pathlib import Path


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


def test_cli_capabilities_help():
    result = subprocess.run(
        [sys.executable, "-m", "eq", "capabilities", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    assert result.returncode == 0
    assert "Show hardware capabilities" in result.stdout


def test_cli_mode_show_help():
    result = subprocess.run(
        [sys.executable, "-m", "eq", "mode", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    assert result.returncode == 0
    assert "Inspect and manage environment mode" in result.stdout


def test_cli_global_mode_flag_shows_in_help():
    result = subprocess.run(
        [sys.executable, "-m", "eq", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    assert result.returncode == 0
    assert "--mode" in result.stdout


def test_cli_mode_show_runs():
    result = subprocess.run(
        [sys.executable, "-m", "eq", "mode", "--show"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    # It should run and print a summary, not error
    assert result.returncode == 0
    assert "Environment Mode:" in result.stdout


def test_cli_capabilities_runs():
    result = subprocess.run(
        [sys.executable, "-m", "eq", "capabilities"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    # It should run and print a capability report header
    assert result.returncode == 0
    assert "Hardware Capability Report" in result.stdout
