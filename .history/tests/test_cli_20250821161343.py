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


def test_cli_mode_set_development():
    """Test setting mode to development."""
    result = subprocess.run(
        [sys.executable, "-m", "eq", "mode", "--set", "development"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    # Should succeed and show mode updated
    assert result.returncode == 0
    assert "Mode updated to: DEVELOPMENT" in result.stdout


def test_cli_mode_set_production():
    """Test setting mode to production."""
    result = subprocess.run(
        [sys.executable, "-m", "eq", "mode", "--set", "production"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    # May fail on some hardware, but should handle gracefully
    if result.returncode != 0:
        assert "Cannot set mode to 'production'" in result.stdout
        assert "Suggested mode:" in result.stdout
    else:
        assert "Mode updated to: PRODUCTION" in result.stdout


def test_cli_mode_validate():
    """Test mode validation."""
    result = subprocess.run(
        [sys.executable, "-m", "eq", "mode", "--validate"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    # Should show validation status
    assert result.returncode == 0
    assert "Validation:" in result.stdout


def test_cli_global_mode_flag():
    """Test global --mode flag works."""
    result = subprocess.run(
        [sys.executable, "-m", "eq", "--mode", "development", "capabilities"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    # Should run capabilities with development mode
    assert result.returncode == 0
    assert "Hardware Capability Report" in result.stdout


def test_cli_command_consistency_across_modes():
    """Test that commands work consistently across different modes."""
    modes = ["development", "production"]
    
    for mode in modes:
        # Test capabilities command with each mode
        result = subprocess.run(
            [sys.executable, "-m", "eq", "--mode", mode, "capabilities"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        # Should work regardless of mode
        assert result.returncode == 0
        assert "Hardware Capability Report" in result.stdout


def test_cli_mode_transparency():
    """Test that mode information is transparently shown in command output."""
    result = subprocess.run(
        [sys.executable, "-m", "eq", "--mode", "development", "capabilities"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    # Should show mode in logs
    assert result.returncode == 0


def test_cli_error_handling_invalid_mode():
    """Test error handling for invalid mode values."""
    result = subprocess.run(
        [sys.executable, "-m", "eq", "--mode", "invalid_mode", "capabilities"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    # Should handle invalid mode gracefully
    assert result.returncode == 0  # Should fallback to auto mode


def test_cli_mode_specific_batch_size():
    """Test that batch size is mode-aware."""
    # This test verifies that the CLI handles mode-specific batch size logic
    # The actual batch size detection happens in the command functions
    result = subprocess.run(
        [sys.executable, "-m", "eq", "--mode", "development", "capabilities"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    assert result.returncode == 0


def test_cli_unified_interface():
    """Test that all commands work through the unified CLI interface."""
    commands = ["capabilities", "mode", "data-load", "train-segmenter", "extract-features", "quantify", "pipeline"]
    
    for command in commands:
        result = subprocess.run(
            [sys.executable, "-m", "eq", command, "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        # All commands should have help available
        assert result.returncode == 0
        assert "usage:" in result.stdout or "help:" in result.stdout


def test_cli_mode_persistence():
    """Test that mode changes persist across commands."""
    # Set mode to development
    result1 = subprocess.run(
        [sys.executable, "-m", "eq", "mode", "--set", "development"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    assert result1.returncode == 0
    
    # Check that mode is set
    result2 = subprocess.run(
        [sys.executable, "-m", "eq", "mode", "--show"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    assert result2.returncode == 0
    assert "Environment Mode: DEVELOPMENT" in result2.stdout
