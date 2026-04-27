from pathlib import Path


def test_overcoverage_audit_results_record_real_audit_attempts():
    openspec_changes = Path(__file__).resolve().parents[1] / 'openspec' / 'changes'
    active_results_path = (
        openspec_changes
        / 'p0-calibrate-glomeruli-overcoverage-controls'
        / 'audit-results.md'
    )
    archived_results = sorted(
        (openspec_changes / 'archive').glob(
            '*-p0-calibrate-glomeruli-overcoverage-controls/audit-results.md'
        )
    )
    results_path = (
        active_results_path if active_results_path.exists() else archived_results[-1]
    )

    text = results_path.read_text(encoding='utf-8')

    assert 'p0_negative_background_quick_5epoch_no_training_audit' in text
    assert 'latest_nonquick_e20_e25_no_training_audit' in text
    assert 'env PYTORCH_ENABLE_MPS_FALLBACK=1' in text
    assert 'threshold_policy_artifact' in text
    assert 'threshold=0.5' in text
    assert 'Status: not yet run' not in text
