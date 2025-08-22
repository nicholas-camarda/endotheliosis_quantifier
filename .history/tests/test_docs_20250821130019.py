from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def test_old_repo_names_absent():
    # Validate no stale references remain in docs
    targets = [PROJECT_ROOT / 'README.md', PROJECT_ROOT / '.agent-os' / 'product' / 'README.md']

    stale_markers = ['endotheliosisQuantifier_LEARN', 'endotheliosisQuantifier']

    for file_path in targets:
        content = read_text(file_path)
        for marker in stale_markers:
            assert marker not in content, f"Found stale repo name '{marker}' in {file_path}"


def test_readme_has_correct_repo_and_mamba_commands():
    readme = read_text(PROJECT_ROOT / 'README.md')

    # Repository URL
    assert 'https://github.com/nicholas-camarda/endotheliosis_quantifier.git' in readme

    # mamba create/update commands
    assert 'mamba env create -f environment.yml' in readme
    assert 'mamba env update -f environment.yml --prune' in readme

    # activation guidance
    assert 'conda activate eq' in readme or 'mamba activate eq' in readme


def test_readme_examples_reference_eq_package():
    """Test that README examples use eq.* package imports."""
    readme = read_text(PROJECT_ROOT / 'README.md')
    
    # Check for eq package usage in examples
    assert 'from eq.features.preprocess_roi import' in readme
    assert 'from eq.features.helpers import' in readme
    assert 'from eq.pipeline.quantify_endotheliosis import' in readme
    
    # Check for package structure documentation
    assert 'eq.features.helpers' in readme
    assert 'eq.features.preprocess_roi' in readme
    assert 'eq.pipeline.quantify_endotheliosis' in readme
    assert 'eq.io.convert_files_to_jpg' in readme
    assert 'eq.patches.patchify_images' in readme
    assert 'eq.augment.augment_dataset' in readme


def test_no_stale_paths_in_docs():
    """Test that no stale paths remain in documentation."""
    readme = read_text(PROJECT_ROOT / 'README.md')
    
    # Check for old data directory references
    stale_paths = [
        'Lauren_PreEclampsia_Data',
        'scripts/utils/convert_files_to_jpg.py',
        'scripts/utils/patchify_images.py', 
        'scripts/utils/generate_augmented_dataset.py',
        'scripts/utils/preprocess_ROI_then_extract_features.py',
        'scripts/main/feature_extractor_helper_functions.py',
        'scripts/main/4_quantify_endotheliosis.py'
    ]
    
    for stale_path in stale_paths:
        assert stale_path not in readme, f"Found stale path '{stale_path}' in README"


def test_readme_has_professional_structure():
    """Test that README has professional structure without excessive emojis."""
    readme = read_text(PROJECT_ROOT / 'README.md')
    
    # Check for professional sections
    assert '## Environment Setup' in readme
    assert '## Using the eq Package' in readme
    
    # Check that emoji usage is minimal (not excessive)
    emoji_count = readme.count('✅') + readme.count('🚀') + readme.count('🎯') + readme.count('🔥')
    assert emoji_count <= 3, f"Too many emojis found in README: {emoji_count}"
