from pathlib import Path


def test_overcoverage_audit_results_record_real_audit_attempts():
    results_path = (
        Path(__file__).resolve().parents[1]
        / "openspec"
        / "changes"
        / "p0-calibrate-glomeruli-overcoverage-controls"
        / "audit-results.md"
    )

    text = results_path.read_text(encoding="utf-8")

    assert "p0_negative_background_quick_5epoch_no_training_audit" in text
    assert "latest_nonquick_e20_e25_no_training_audit" in text
    assert "env PYTORCH_ENABLE_MPS_FALLBACK=1" in text
    assert "threshold_policy_artifact" in text
    assert "threshold=0.5" in text
    assert "Status: not yet run" not in text
