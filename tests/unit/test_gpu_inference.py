import time
from unittest.mock import MagicMock

import pytest
import torch

from eq.inference.gpu_inference import GPUGlomeruliInference


def make_inference() -> GPUGlomeruliInference:
    inference = GPUGlomeruliInference.__new__(GPUGlomeruliInference)
    inference.logger = MagicMock()
    return inference


def test_auto_device_prefers_cuda(monkeypatch):
    inference = make_inference()

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'Test CUDA')
    monkeypatch.setattr(
        torch.cuda,
        'get_device_properties',
        lambda *args: MagicMock(total_memory=8 * 1024**3),
    )
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)

    assert inference._select_device('auto') == 'cuda'


def test_auto_device_uses_mps_when_cuda_unavailable(monkeypatch):
    inference = make_inference()

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)

    assert inference._select_device('auto') == 'mps'


def test_auto_device_uses_cpu_when_no_accelerator(monkeypatch):
    inference = make_inference()

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)

    assert inference._select_device('auto') == 'cpu'


def test_explicit_mps_requires_mps_availability(monkeypatch):
    inference = make_inference()

    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)

    with pytest.raises(ValueError, match='MPS was requested'):
        inference._select_device('mps')


def test_explicit_cuda_requires_cuda_availability(monkeypatch):
    inference = make_inference()

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    with pytest.raises(ValueError, match='CUDA was requested'):
        inference._select_device('cuda')


def test_benchmark_mps_does_not_call_cuda_runtime(monkeypatch):
    inference = make_inference()
    inference.device = 'mps'
    inference.expected_size = 8
    inference.learn = MagicMock()
    inference.learn.model = lambda tensor: tensor
    inference.predict_batch = lambda images, batch_size=8: []

    monkeypatch.setattr(torch.Tensor, 'to', lambda self, *args, **kwargs: self)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(time, 'time', lambda: next(clock))
    monkeypatch.setattr(
        torch.cuda,
        'synchronize',
        MagicMock(side_effect=AssertionError('cuda synchronize called')),
    )
    monkeypatch.setattr(
        torch.cuda,
        'memory_allocated',
        MagicMock(side_effect=AssertionError('cuda memory called')),
    )
    monkeypatch.setattr(
        torch.cuda,
        'get_device_name',
        MagicMock(side_effect=AssertionError('cuda name called')),
    )
    monkeypatch.setattr(
        torch.cuda,
        'get_device_properties',
        MagicMock(side_effect=AssertionError('cuda properties called')),
    )

    result = inference.benchmark_performance([MagicMock()], num_runs=1)

    assert result['gpu_memory_gb'] == 0.0


def test_benchmark_cpu_does_not_call_cuda_runtime(monkeypatch):
    inference = make_inference()
    inference.device = 'cpu'
    inference.expected_size = 8
    inference.learn = MagicMock()
    inference.learn.model = lambda tensor: tensor
    inference.predict_batch = lambda images, batch_size=8: []
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(time, 'time', lambda: next(clock))

    monkeypatch.setattr(
        torch.cuda,
        'synchronize',
        MagicMock(side_effect=AssertionError('cuda synchronize called')),
    )
    monkeypatch.setattr(
        torch.cuda,
        'memory_allocated',
        MagicMock(side_effect=AssertionError('cuda memory called')),
    )
    monkeypatch.setattr(
        torch.cuda,
        'get_device_name',
        MagicMock(side_effect=AssertionError('cuda name called')),
    )
    monkeypatch.setattr(
        torch.cuda,
        'get_device_properties',
        MagicMock(side_effect=AssertionError('cuda properties called')),
    )

    result = inference.benchmark_performance([MagicMock()], num_runs=1)

    assert result['gpu_memory_gb'] == 0.0
