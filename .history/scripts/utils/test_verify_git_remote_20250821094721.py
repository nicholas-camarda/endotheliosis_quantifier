import subprocess
import sys


EXPECTED_OLD = "https://github.com/nicholas-camarda/endotheliosisQuantifier.git"
EXPECTED_NEW = "https://github.com/nicholas-camarda/endotheliosis_quantifier.git"


def get_git_remote_url(name: str) -> str:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", name],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr)
        return ""
    return result.stdout.strip()


def main() -> int:
    origin_url = get_git_remote_url("origin")
    if not origin_url:
        print("Could not retrieve origin URL.")
        return 1

    print(f"origin: {origin_url}")

    if origin_url == EXPECTED_NEW:
        print("OK: Origin matches new underscore repo name.")
        return 0
    if origin_url == EXPECTED_OLD:
        print("NOTE: Origin matches old camelCase repo name; update recommended.")
        print(f"Run: git remote set-url origin {EXPECTED_NEW}")
        return 0

    print("FAIL: Origin URL does not match known expected values.")
    print(f"Expected one of: {EXPECTED_NEW} OR {EXPECTED_OLD}")
    return 1


if __name__ == "__main__":
    sys.exit(main())


