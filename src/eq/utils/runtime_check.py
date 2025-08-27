import importlib
import sys
import warnings
from pathlib import Path

# Suppress pkg_resources deprecation warning emitted by tensorflow_hub
warnings.filterwarnings(
    'ignore',
    message=r'.*pkg_resources is deprecated as an API.*',
    category=UserWarning,
)

REQUIRED_IMPORTS = [
    ('numpy', None),
    ('scipy', None),
    ('sklearn', None),
    ('matplotlib', None),
    ('cv2', None),
    ('lightgbm', None),
    ('xgboost', None),
    ('torch', None),
    ('fastai', None),
]

DATA_PATHS = [
    'data/preeclampsia_data/train/images',
    'data/preeclampsia_data/train/masks',
    'data/preeclampsia_data/train/rois',
    'data/preeclampsia_data/cache',
]


def check_imports() -> bool:
    print('Checking required imports...')
    ok = True
    for module_name, attr in REQUIRED_IMPORTS:
        try:
            mod = importlib.import_module(module_name)
            if attr is not None and not hasattr(mod, attr):
                print(f'- {module_name}: MISSING attribute {attr}')
                ok = False
            else:
                print(f'- {module_name}: OK')
        except Exception as e:
            print(f'- {module_name}: IMPORT FAILED -> {e}')
            ok = False
    return ok


def check_hardware_capabilities() -> bool:
    """Check hardware capabilities and backend availability."""
    print('\nChecking hardware capabilities...')
    try:
        from eq.utils.hardware_detection import get_capability_report, get_device_recommendation
        print(get_capability_report())
        
        # Test device recommendations
        dev_backend, dev_explanation = get_device_recommendation("development")
        prod_backend, prod_explanation = get_device_recommendation("production")
        
        print('\nDevice Recommendations:')
        print(f'- Development: {dev_backend.value.upper()} - {dev_explanation}')
        print(f'- Production: {prod_backend.value.upper()} - {prod_explanation}')
        
        return True
    except Exception as e:
        print(f'- Hardware detection: FAILED -> {e}')
        return False


def check_paths() -> bool:
    print('\nChecking expected data paths...')
    ok = True
    for p in DATA_PATHS:
        path = Path(p)
        if not path.exists():
            print(f'- {p}: MISSING')
            ok = False
        else:
            print(f'- {p}: OK')
    return ok


def main() -> int:
    imports_ok = check_imports()
    hardware_ok = check_hardware_capabilities()
    paths_ok = check_paths()
    
    if not imports_ok:
        print(
            '\nOne or more required imports failed. Please review environment.yml and install the environment.',
        )
    if not hardware_ok:
        print(
            '\nHardware capability detection failed. Check PyTorch installation and system compatibility.',
        )
    if not paths_ok:
        print(
            '\nOne or more expected data paths are missing. Adjust data paths or place data accordingly.',
        )
    return 0 if (imports_ok and hardware_ok and paths_ok) else 1


if __name__ == '__main__':
    sys.exit(main())
