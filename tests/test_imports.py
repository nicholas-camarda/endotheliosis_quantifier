import importlib


def test_can_import_eq_and_subpackages():
    # Expected top-level package
    assert importlib.import_module('eq') is not None

    # Subpackages we plan to create
    for sub in [
        'io', 'augment', 'patches', 'models', 'segmentation', 'features', 'metrics', 'pipeline', 'utils',
    ]:
        mod = importlib.import_module(f'eq.{sub}')
        assert mod is not None, f'Failed to import eq.{sub}'



