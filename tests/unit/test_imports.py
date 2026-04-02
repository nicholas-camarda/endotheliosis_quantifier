import importlib


def test_can_import_eq_and_subpackages():
    # Expected top-level package
    assert importlib.import_module('eq') is not None

    # Current clean subpackages in the new architecture
    for sub in [
        'core', 'data_management', 'processing', 'pipeline', 'evaluation', 'training', 'inference', 'models', 'utils',
    ]:
        mod = importlib.import_module(f'eq.{sub}')
        assert mod is not None, f'Failed to import eq.{sub}'



