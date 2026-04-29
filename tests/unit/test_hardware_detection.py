from unittest.mock import MagicMock, patch

from eq.utils.hardware_detection import (
    BackendType,
    HardwareCapabilities,
    HardwareDetector,
    HardwareTier,
    get_capability_report,
    get_device_info,
    get_device_recommendation,
    get_hardware_capabilities,
    get_optimal_batch_size,
    get_segmentation_training_batch_size,
)


def make_capabilities(
    *,
    platform: str = 'Linux',
    architecture: str = 'x86_64',
    cpu_count: int = 16,
    total_memory_gb: float = 16.0,
    available_memory_gb: float = 8.0,
    backend_type: BackendType = BackendType.CUDA,
    gpu_name: str | None = 'NVIDIA GeForce RTX 3080',
    gpu_memory_gb: float | None = 10.0,
    hardware_tier: HardwareTier = HardwareTier.POWERFUL,
    mps_available: bool = False,
    mps_built: bool = False,
    cuda_available: bool = True,
    cuda_device_count: int = 1,
) -> HardwareCapabilities:
    return HardwareCapabilities(
        platform=platform,
        architecture=architecture,
        cpu_count=cpu_count,
        total_memory_gb=total_memory_gb,
        available_memory_gb=available_memory_gb,
        backend_type=backend_type,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        hardware_tier=hardware_tier,
        mps_available=mps_available,
        mps_built=mps_built,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
    )


def test_primary_backend_preference_order():
    detector = HardwareDetector()

    assert detector._determine_primary_backend(True, True) == BackendType.CUDA
    assert detector._determine_primary_backend(True, False) == BackendType.MPS
    assert detector._determine_primary_backend(False, False) == BackendType.CPU


def test_hardware_tier_classification():
    detector = HardwareDetector()

    assert (
        detector._classify_hardware_tier(4, 4.0, BackendType.CPU, None)
        == HardwareTier.BASIC
    )
    assert (
        detector._classify_hardware_tier(4, 8.0, BackendType.CPU, None)
        == HardwareTier.STANDARD
    )
    assert (
        detector._classify_hardware_tier(8, 16.0, BackendType.CUDA, 10.0)
        == HardwareTier.POWERFUL
    )


def test_device_recommendation_by_mode():
    detector = HardwareDetector()
    detector._capabilities = make_capabilities()

    backend, explanation = detector.get_device_recommendation('production')
    assert backend == BackendType.CUDA
    assert 'Production mode' in explanation

    backend, explanation = detector.get_device_recommendation('auto')
    assert backend == BackendType.CUDA
    assert 'Auto mode' in explanation

    detector._capabilities = make_capabilities(
        platform='Darwin',
        architecture='arm64',
        backend_type=BackendType.MPS,
        gpu_name='Apple M2 GPU',
        gpu_memory_gb=None,
        mps_available=True,
        mps_built=True,
        cuda_available=False,
        cuda_device_count=0,
    )
    backend, explanation = detector.get_device_recommendation('development')
    assert backend == BackendType.MPS
    assert 'Development mode' in explanation


def test_optimal_batch_size_uses_backend_and_memory():
    detector = HardwareDetector()
    detector._capabilities = make_capabilities(gpu_memory_gb=12.0)
    assert detector.get_optimal_batch_size('production') == 16

    detector._capabilities = make_capabilities(
        platform='Darwin',
        architecture='arm64',
        backend_type=BackendType.MPS,
        gpu_name='Apple M2 GPU',
        gpu_memory_gb=None,
        total_memory_gb=16.0,
        mps_available=True,
        mps_built=True,
        cuda_available=False,
        cuda_device_count=0,
    )
    assert detector.get_optimal_batch_size('auto') == 8

    detector._capabilities = make_capabilities(
        backend_type=BackendType.CPU,
        gpu_name=None,
        gpu_memory_gb=None,
        cuda_available=False,
        cuda_device_count=0,
    )
    assert detector.get_optimal_batch_size('production') == 2


def test_segmentation_training_batch_size_uses_powerful_mps_stage_defaults():
    detector = HardwareDetector()
    detector._capabilities = make_capabilities(
        platform='Darwin',
        architecture='arm64',
        backend_type=BackendType.MPS,
        gpu_name='Apple Silicon GPU',
        gpu_memory_gb=19.2,
        total_memory_gb=64.0,
        mps_available=True,
        mps_built=True,
        cuda_available=False,
        cuda_device_count=0,
    )

    assert (
        detector.get_segmentation_training_batch_size('mitochondria', image_size=256)
        == 24
    )
    assert (
        detector.get_segmentation_training_batch_size(
            'glomeruli', image_size=256, crop_size=512
        )
        == 12
    )


def test_segmentation_training_batch_size_respects_explicit_override():
    detector = HardwareDetector()
    detector._capabilities = make_capabilities(
        platform='Darwin',
        architecture='arm64',
        backend_type=BackendType.MPS,
        gpu_name='Apple Silicon GPU',
        gpu_memory_gb=19.2,
        total_memory_gb=64.0,
        mps_available=True,
        mps_built=True,
        cuda_available=False,
        cuda_device_count=0,
    )

    assert (
        detector.get_segmentation_training_batch_size(
            'mitochondria', image_size=256, requested_batch_size=32
        )
        == 32
    )


def test_capability_report_contains_expected_sections():
    detector = HardwareDetector()
    detector._capabilities = make_capabilities()
    report = detector.get_capability_report()

    assert 'Hardware Capability Report' in report
    assert 'Backend Availability' in report
    assert 'Mode Recommendations' in report
    assert 'NVIDIA GeForce RTX 3080' in report


def test_capability_report_uses_mps_when_cuda_unavailable():
    detector = HardwareDetector()
    detector._capabilities = make_capabilities(
        platform='Darwin',
        architecture='arm64',
        backend_type=BackendType.MPS,
        gpu_name='Apple Silicon GPU',
        gpu_memory_gb=19.2,
        mps_available=True,
        mps_built=True,
        cuda_available=False,
        cuda_device_count=0,
    )

    report = detector.get_capability_report()

    assert 'MPS Available: True' in report
    assert 'CUDA Available: False' in report
    assert 'Type: MPS' in report


@patch('eq.utils.hardware_detection.hardware_detector')
def test_global_hardware_detection_wrappers(mock_detector: MagicMock):
    capabilities = make_capabilities()
    mock_detector.detect_capabilities.return_value = capabilities
    mock_detector.get_device_recommendation.return_value = (BackendType.CUDA, 'ok')
    mock_detector.get_capability_report.return_value = 'report'
    mock_detector.get_optimal_batch_size.return_value = 16
    mock_detector.get_segmentation_training_batch_size.return_value = 24

    assert get_hardware_capabilities() == capabilities
    assert get_device_recommendation('production') == (BackendType.CUDA, 'ok')
    assert get_capability_report() == 'report'
    assert get_optimal_batch_size('production') == 16
    assert get_segmentation_training_batch_size('mitochondria', image_size=256) == 24


@patch('eq.utils.hardware_detection.get_hardware_capabilities')
@patch('eq.utils.hardware_detection.get_device_recommendation')
def test_get_device_info_respects_availability(
    mock_recommendation: MagicMock, mock_capabilities: MagicMock
):
    mock_capabilities.return_value = make_capabilities()
    mock_recommendation.return_value = (BackendType.CUDA, 'best')

    auto_info = get_device_info()
    assert auto_info['device'] == 'cuda'
    assert auto_info['backend'] == 'cuda'

    mock_capabilities.return_value = make_capabilities(
        backend_type=BackendType.CPU,
        gpu_name=None,
        gpu_memory_gb=None,
        cuda_available=False,
        cuda_device_count=0,
    )
    fallback_info = get_device_info('cuda')
    assert fallback_info['device'] == 'cpu'
    assert fallback_info['backend'] == 'cpu'


@patch('eq.utils.hardware_detection.get_hardware_capabilities')
@patch('eq.utils.hardware_detection.get_device_recommendation')
def test_get_device_info_auto_can_select_mps_without_cuda(
    mock_recommendation: MagicMock, mock_capabilities: MagicMock
):
    mock_capabilities.return_value = make_capabilities(
        platform='Darwin',
        architecture='arm64',
        backend_type=BackendType.MPS,
        gpu_name='Apple Silicon GPU',
        gpu_memory_gb=19.2,
        mps_available=True,
        mps_built=True,
        cuda_available=False,
        cuda_device_count=0,
    )
    mock_recommendation.return_value = (BackendType.MPS, 'best')

    auto_info = get_device_info()

    assert auto_info['device'] == 'mps'
    assert auto_info['backend'] == 'mps'
