import importlib


def test_can_import_eq_and_subpackages():
    assert importlib.import_module('eq') is not None

    for subpackage in [
        'core',
        'data_management',
        'processing',
        'pipeline',
        'evaluation',
        'training',
        'inference',
        'models',
        'utils',
    ]:
        module = importlib.import_module(f'eq.{subpackage}')
        assert module is not None, f'Failed to import eq.{subpackage}'
