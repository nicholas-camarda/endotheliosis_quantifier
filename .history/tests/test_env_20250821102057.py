import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.env_check import read_environment_name_from_yaml, verify_environment_name


def test_environment_yml_name_is_eq():
    name = read_environment_name_from_yaml(PROJECT_ROOT / 'environment.yml')
    assert name == 'eq', f"Expected environment name 'eq', found '{name}'"


def test_conda_env_list_contains_eq():
    ok, message = verify_environment_name(expected_name='eq', environment_file_path=PROJECT_ROOT / 'environment.yml')
    assert ok, message


