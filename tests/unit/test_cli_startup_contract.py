import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import eq.__main__ as eq_main
from eq.utils.mode_manager import EnvironmentMode


def test_runtime_directory_creation_failure_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    blocked = tmp_path / 'runtime_blocker'
    blocked.write_text('not a directory', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['eq', 'mode', '--show'])
    monkeypatch.setattr(eq_main, 'get_runtime_raw_data_path', lambda: blocked / 'raw_data')

    with pytest.raises(OSError):
        eq_main.main()


def test_invalid_global_mode_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, 'argv', ['eq', '--mode', 'invalid', 'mode', '--show'])

    with pytest.raises(SystemExit) as exc_info:
        eq_main.main()

    assert exc_info.value.code == 2


def test_darwin_auto_setup_does_not_set_global_mps_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeModeManager:
        def __init__(self):
            self.current_mode = EnvironmentMode.DEVELOPMENT
            self.current_config = SimpleNamespace(batch_size=None)

        def get_suggested_mode(self):
            return EnvironmentMode.DEVELOPMENT

        def set_mode(self, mode):
            self.current_mode = mode

    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK', raising=False)
    monkeypatch.setattr(eq_main, 'ModeManager', FakeModeManager)
    monkeypatch.setattr(
        'eq.utils.hardware_detection.get_hardware_capabilities',
        lambda: SimpleNamespace(platform='Darwin'),
    )

    eq_main.auto_setup_environment()

    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None
