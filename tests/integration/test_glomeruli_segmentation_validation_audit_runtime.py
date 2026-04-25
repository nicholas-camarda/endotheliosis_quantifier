"""Optional local-runtime audit for current glomeruli segmentation artifacts.

This test attempts the real local ProjectsRuntime artifact path by default.
It is the pytest surface for checking runtime artifacts before promotion or
README-facing claims, not a user-facing workflow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from eq.utils.paths import get_active_runtime_root
from eq.training.segmentation_validation_audit import (
    PROMOTION_ELIGIBLE,
    audit_manifest_rows,
    audit_prediction_shapes,
    audit_split_overlap,
    documentation_claim_audit,
)


def _runtime_root() -> Path:
    path = Path(os.environ.get("EQ_RUNTIME_ROOT") or get_active_runtime_root()).expanduser()
    if not path.exists():
        pytest.skip(f"EQ_RUNTIME_ROOT does not exist: {path}")
    return path


def _comparison_report_path(runtime_root: Path) -> Path:
    explicit = os.environ.get("EQ_GLOMERULI_PROMOTION_REPORT")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            pytest.skip(f"EQ_GLOMERULI_PROMOTION_REPORT does not exist: {path}")
        return path
    report = (
        runtime_root
        / "output"
        / "segmentation_evaluation"
        / "glomeruli_candidate_comparison"
        / "latest_run"
        / "promotion_report.json"
    )
    if not report.exists():
        pytest.skip(f"No default promotion report exists at {report}")
    return report


def test_runtime_glomeruli_promotion_report_has_auditable_claim_evidence():
    runtime_root = _runtime_root()
    report_path = _comparison_report_path(runtime_root)
    report = json.loads(report_path.read_text())

    manifest = report.get("candidate_manifest") or []
    if not manifest and report.get("manifest_path"):
        manifest_path = Path(report["manifest_path"])
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
    if not manifest:
        decision = report.get("decision", {})
        assert decision.get("promotion_evidence_status") == "insufficient_evidence_for_promotion"
        assert decision.get("decision_reason") == "insufficient_heldout_category_support"
        assert report.get("manifest_audit", {}).get("promotion_evidence_status") == "insufficient_evidence_for_promotion"
        for summary in report.get("candidate_summaries", []):
            assert summary.get("runtime_use_status") == "available_research_use"
            assert "insufficient_heldout_category_support" in summary.get("gate", {}).get("reasons", [])
        return

    rows = []
    manifest_csv = runtime_root / "raw_data" / "cohorts" / "manifest.csv"
    if manifest_csv.exists():
        import pandas as pd

        rows = pd.read_csv(manifest_csv).fillna("").to_dict(orient="records")
        manifest_audit = audit_manifest_rows(rows, runtime_root=runtime_root)
        assert manifest_audit["ok"], manifest_audit["missing_pairs"][:5]

    split_audits = []
    prediction_rows = []
    for summary in report.get("candidate_summaries", []):
        provenance = summary.get("provenance", {})
        split_audits.append(
            audit_split_overlap(
                train_images=provenance.get("train_images"),
                valid_images=provenance.get("valid_images"),
                promotion_manifest=manifest,
            )
        )
        prediction_rows.extend(summary.get("prediction_rows", []))

    assert split_audits, "Promotion report has no candidate split provenance to audit."
    assert all(audit["promotion_evidence_status"] == PROMOTION_ELIGIBLE for audit in split_audits)

    shape_audit = audit_prediction_shapes(prediction_rows)
    assert not shape_audit["blocked"], shape_audit["family_status"]

    readme_path = Path("README.md")
    onboarding_path = Path("docs/ONBOARDING_GUIDE.md")
    docs = {
        str(readme_path): readme_path.read_text(encoding="utf-8"),
        str(onboarding_path): onboarding_path.read_text(encoding="utf-8"),
    }
    claim_audit = documentation_claim_audit(
        docs,
        cited_report_status=report.get("decision", {}).get("promotion_evidence_status", "audit_missing"),
        cited_report_path=report_path,
    )
    assert not claim_audit["blocked"], claim_audit["rows"]
