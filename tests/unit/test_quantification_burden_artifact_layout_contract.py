from pathlib import Path

FORBIDDEN_ACTIVE_PATHS = (
    'burden_model/primary_model/',
    'burden_model/validation/',
    'burden_model/calibration/',
    'burden_model/summaries/',
    'burden_model/evidence/',
    'burden_model/candidates/',
    'burden_model/diagnostics/',
    'burden_model/feature_sets/',
    '../burden_model/evidence/',
)


def test_active_docs_and_specs_do_not_reintroduce_old_burden_paths():
    repo_root = Path(__file__).resolve().parents[2]
    checked_paths = [
        repo_root / 'README.md',
        repo_root / 'docs' / 'OUTPUT_STRUCTURE.md',
        repo_root / 'docs' / 'ONBOARDING_GUIDE.md',
        repo_root / 'openspec' / 'specs' / 'endotheliosis-burden-index' / 'spec.md',
        repo_root
        / 'openspec'
        / 'specs'
        / 'morphology-aware-quantification-features'
        / 'spec.md',
        repo_root
        / 'openspec'
        / 'specs'
        / 'quantification-burden-artifact-layout'
        / 'spec.md',
    ]

    violations = []
    for path in checked_paths:
        text = path.read_text(encoding='utf-8')
        for forbidden in FORBIDDEN_ACTIVE_PATHS:
            if forbidden in text:
                violations.append(f'{path.relative_to(repo_root)}: {forbidden}')

    assert violations == []
