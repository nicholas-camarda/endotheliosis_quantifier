from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def test_old_repo_names_absent():
    targets = [
        PROJECT_ROOT / 'README.md',
        PROJECT_ROOT / '.agent-os' / 'product' / 'README.md',
    ]

    stale_markers = [
        'endotheliosisQuantifier_LEARN',
        'endotheliosisQuantifier',
    ]

    for file_path in targets:
        content = read_text(file_path)
        for marker in stale_markers:
            assert marker not in content, f"Found stale repo name '{marker}' in {file_path}"


def test_readme_has_correct_repo_and_mamba_commands():
    readme = read_text(PROJECT_ROOT / 'README.md')

    assert 'https://github.com/nicholas-camarda/endotheliosis_quantifier.git' in readme
    assert 'mamba env create -f environment.yml' in readme
    assert 'mamba env update -f environment.yml --prune' in readme
    assert 'conda activate eq' in readme or 'mamba activate eq' in readme


