from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ACTIVE_DOCS = (
    'README.md',
    'docs/README.md',
    'docs/INTEGRATION_GUIDE.md',
    'docs/PIPELINE_INTEGRATION_PLAN.md',
    'docs/OUTPUT_STRUCTURE.md',
    'docs/SEGMENTATION_ENGINEERING_GUIDE.md',
    'docs/ONBOARDING_GUIDE.md',
    'docs/TECHNICAL_LAB_NOTEBOOK.md',
)

BLOCKED_ACTIVE_PHRASES = (
    'eq.inference.historical_glomeruli_inference',
    'HistoricalGlomeruliInference',
    'setup_historical_environment',
    'use_historical_preprocessing',
    'historical_glomeruli_inference.py',
    'historical fallback',
    'automatic fallback loading',
    'historical approach as fallback',
    'copy historical inference',
)


def test_active_docs_do_not_contain_historical_fallback_operations():
    violations = []
    for relative_path in ACTIVE_DOCS:
        path = REPO_ROOT / relative_path
        text = path.read_text(encoding='utf-8')
        lowered = text.lower()
        for phrase in BLOCKED_ACTIVE_PHRASES:
            if phrase.lower() in lowered:
                violations.append(f'{relative_path}: {phrase}')

    assert violations == []


def test_archived_historical_docs_are_reference_only_and_indexed():
    historical_notes = (REPO_ROOT / 'docs' / 'HISTORICAL_NOTES.md').read_text(
        encoding='utf-8'
    )
    archive_paths = sorted((REPO_ROOT / 'docs' / 'archive').glob('*.md'))
    assert archive_paths

    violations = []
    for path in archive_paths:
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding='utf-8')
        header = text[:500].lower()
        if relative_path.removeprefix('docs/') not in historical_notes:
            violations.append(f'{relative_path}: not indexed from docs/HISTORICAL_NOTES.md')
        if 'reference-only' not in header and 'reference only' not in header:
            violations.append(f'{relative_path}: missing reference-only header')
        if 'not current operational guidance' not in header:
            violations.append(
                f'{relative_path}: missing current-guidance exclusion header'
            )

    assert violations == []
