import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from eq.inference.gpu_inference import GPUGlomeruliInference
from eq.inference.prediction_core import (
    DEFAULT_PREDICTION_THRESHOLD,
    create_prediction_core,
)


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


def test_gpu_preprocess_uses_shared_imagenet_normalization(monkeypatch):
    inference = make_inference()
    inference.expected_size = 4
    inference.device = 'cpu'
    image = Image.fromarray(np.ones((4, 4, 3), dtype=np.uint8) * 128)

    monkeypatch.setattr(torch.Tensor, 'to', lambda self, *args, **kwargs: self)

    expected = create_prediction_core(4).preprocess_image_imagenet_normalized(image)
    observed = inference.preprocess_image(image)

    assert torch.allclose(observed, expected)


def test_predict_single_defaults_to_shared_threshold(monkeypatch):
    inference = make_inference()
    inference.expected_size = 2
    inference.device = 'cpu'
    inference.learn = MagicMock()
    inference.learn.model = lambda tensor: torch.zeros((1, 1, 2, 2))
    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))

    monkeypatch.setattr(torch.Tensor, 'to', lambda self, *args, **kwargs: self)

    _, pred_mask, default_binary = inference.predict_single(image)
    _, _, explicit_binary = inference.predict_single(
        image, threshold=DEFAULT_PREDICTION_THRESHOLD - 0.01
    )

    assert pred_mask.eq(DEFAULT_PREDICTION_THRESHOLD).all()
    assert default_binary.sum().item() == 0
    assert explicit_binary.sum().item() == 4


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
