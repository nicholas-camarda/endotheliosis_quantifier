import subprocess
import sys
from pathlib import Path


def read_environment_name_from_yaml(environment_file_path: str = "environment.yml") -> str:
    environment_path = Path(environment_file_path)
    if not environment_path.exists():
        raise FileNotFoundError(f"Environment file not found: {environment_file_path}")

    with environment_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("name:"):
                # format: name: eq
                return stripped.split(":", 1)[1].strip()

    raise ValueError("No 'name:' field found in environment.yml")


def list_conda_env_names() -> list[str]:
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Conda is not installed or not available on PATH."
        ) from exc

    env_names: list[str] = []
    for line in result.stdout.splitlines():
        # Typical lines look like:
        # base                  *  /Users/you/miniconda3
        # eq                       /Users/you/miniconda3/envs/eq
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 0:
            continue
        # If an asterisk is in the second column, the name is in the first column
        # Otherwise the first token is the env name
        env_name = parts[0]
        env_names.append(env_name)
    return env_names


def verify_environment_name(expected_name: str = "eq", environment_file_path: str = "environment.yml") -> tuple[bool, str]:
    yaml_name = read_environment_name_from_yaml(environment_file_path)
    if yaml_name != expected_name:
        return False, f"environment.yml name is '{yaml_name}', expected '{expected_name}'"

    try:
        envs = list_conda_env_names()
    except RuntimeError as e:
        return False, str(e)

    if expected_name not in envs:
        return False, f"Conda environment '{expected_name}' not found in 'conda env list'"

    return True, "Environment verification passed: name matches and env exists."


def main() -> int:
    ok, message = verify_environment_name()
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


