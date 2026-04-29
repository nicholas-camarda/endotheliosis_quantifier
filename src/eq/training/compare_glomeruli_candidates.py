#!/usr/bin/env python3
"""Compare glomeruli transfer and scratch candidates under one deterministic promotion contract."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from eq.core.constants import DEFAULT_VAL_RATIO
from eq.data_management.datablock_loader import (
    get_items_full_images,
    training_item_image_path,
    training_item_mask_path,
    validate_supported_segmentation_training_root,
)
from eq.data_management.model_loading import load_model_safely
from eq.data_management.standard_getters import get_y_full
from eq.evaluation.segmentation_metrics import pixel_accuracy
from eq.inference.prediction_core import create_prediction_core
from eq.training.promotion_gates import (
    audit_manifest_crops,
    binary_dice_jaccard,
    binary_precision_recall,
    deterministic_validation_manifest,
    evaluate_glomeruli_promotion_candidate,
    resolve_image_path_for_mask,
    trivial_baseline_metrics,
)
from eq.training.segmentation_validation_audit import (
    PROMOTION_AUDIT_MISSING,
    PROMOTION_ELIGIBLE,
    PROMOTION_INSUFFICIENT,
    PROMOTION_NOT_ELIGIBLE,
    RUNTIME_USE_AVAILABLE,
    aggregate_metric_by_category,
    audit_prediction_shapes,
    audit_preprocessing_parity,
    audit_resize_policy_parity,
    audit_split_overlap,
    classify_artifact_status,
    classify_root_causes,
    documentation_claim_audit,
    failure_reproduction_row,
    resize_policy_record,
    write_csv_rows,
    write_documentation_claim_audit,
)
from eq.utils.execution_logging import (
    direct_execution_log_context,
    run_logged_subprocess,
)
from eq.utils.logger import get_logger, setup_logging
from eq.utils.paths import (
    get_runtime_models_path,
    get_runtime_segmentation_evaluation_path,
)
from eq.utils.run_io import metadata_path_for_model

TIE_MARGIN = 0.02
DEFAULT_EXAMPLES_PER_CATEGORY = 2
COMPARE_PREDICTION_THRESHOLD = 0.01
THRESHOLD_POLICY_UNVERIFIED = "threshold_policy_unverified"
THRESHOLD_POLICY_FIXED_REVIEW = "fixed_review_threshold"
THRESHOLD_POLICY_AUDIT_BACKED_FIXED = "audit_backed_fixed_threshold"
THRESHOLD_POLICY_VALIDATION_DERIVED = "validation_derived_threshold"
PROMOTION_READY_THRESHOLD_POLICIES = {
    THRESHOLD_POLICY_AUDIT_BACKED_FIXED,
    THRESHOLD_POLICY_VALIDATION_DERIVED,
}
BACKGROUND_FALSE_POSITIVE_LIMIT = 0.02
RESIZE_SCREEN_MATERIAL_MARGIN = 0.02
RESIZE_DECISION_CURRENT_CLEARED = "current_policy_cleared"
RESIZE_DECISION_SELECTED_LESS_DOWNSAMPLED = "selected_less_downsampled_policy"
RESIZE_DECISION_INFEASIBLE = "resize_screen_infeasible"
RESIZE_DECISION_UNRESOLVED = "resize_screen_unresolved"
VALIDATION_ADJUDICATION_REQUIRED_COLUMNS = {
    "adjudication_id",
    "candidate_family",
    "category",
    "manifest_index",
    "image_path",
    "crop_box",
    "original_gate_failure",
    "adjudication_label",
    "adjudication_decision",
    "requires_mask_correction",
    "counts_as_model_failure_after_review",
}
VALIDATION_ADJUDICATION_BOOLEAN_COLUMNS = {
    "requires_mask_correction",
    "counts_as_model_failure_after_review",
}
VALIDATION_ADJUDICATION_NONBLOCKING = "applied_nonblocking"
VALIDATION_ADJUDICATION_BLOCKING = "applied_blocking"
VALIDATION_ADJUDICATION_NOT_REVIEWED = "not_reviewed"
FRONT_FACING_PANEL_CATEGORIES = ("background", "boundary", "positive")
FRONT_FACING_PANEL_TILE_SIZE = 320

logger = get_logger("eq.glomeruli_candidate_comparison")


def _generated_run_id(seed: int) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}_all_manual_mask_glomeruli_seed{seed}"


@dataclass
class CandidateRuntime:
    family: str
    role: str
    model_path: Path | None
    seed: int | None
    command: list[str] | None
    status: str
    error: str | None = None


def _read_metadata_if_available(model_path: Path | None) -> Dict[str, Any]:
    if model_path is None:
        return {}
    metadata_path = metadata_path_for_model(model_path)
    if not metadata_path.exists():
        return {}
    with metadata_path.open() as handle:
        return json.load(handle)


def _read_overcoverage_audit(audit_dir: str | Path | None) -> Dict[str, Any]:
    if not audit_dir:
        return {}
    audit_path = Path(audit_dir).expanduser()
    summary_path = audit_path / "audit_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Overcoverage audit summary not found: {summary_path}")
    with summary_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["audit_dir"] = str(audit_path)
    payload["audit_summary_path"] = str(summary_path)
    return payload


def _parse_bool_field(value: Any, *, column: str, adjudication_id: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(
        f"Invalid boolean value for {column} in validation adjudication {adjudication_id}: {value!r}"
    )


def _normalize_crop_box(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return json.dumps([int(item) for item in value])
    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, list):
        return json.dumps([int(item) for item in parsed])
    return text


def _adjudication_match_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("candidate_family") or row.get("family") or "").strip(),
        str(row.get("category") or "").strip(),
        str(row.get("manifest_index") or "").strip(),
        str(row.get("image_path") or "").strip(),
        _normalize_crop_box(row.get("crop_box")),
    )


def _read_validation_adjudications(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {
            "status": "not_supplied",
            "path": None,
            "rows": [],
            "row_count": 0,
            "applied_count": 0,
            "nonblocking_count": 0,
            "blocking_count": 0,
            "unused_count": 0,
        }
    adjudication_path = Path(path).expanduser()
    if not adjudication_path.exists():
        raise FileNotFoundError(f"Validation adjudication file not found: {adjudication_path}")
    with adjudication_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(VALIDATION_ADJUDICATION_REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"Validation adjudication file {adjudication_path} is missing required columns: "
                f"{', '.join(missing)}"
            )
        rows = []
        for raw_row in reader:
            row = dict(raw_row)
            adjudication_id = str(row.get("adjudication_id") or "").strip()
            if not adjudication_id:
                raise ValueError(f"Validation adjudication file {adjudication_path} has a row without adjudication_id")
            for column in VALIDATION_ADJUDICATION_BOOLEAN_COLUMNS:
                row[column] = _parse_bool_field(row.get(column), column=column, adjudication_id=adjudication_id)
            row["candidate_family"] = str(row.get("candidate_family") or "").strip()
            row["category"] = str(row.get("category") or "").strip()
            row["manifest_index"] = str(row.get("manifest_index") or "").strip()
            row["image_path"] = str(row.get("image_path") or "").strip()
            row["crop_box"] = _normalize_crop_box(row.get("crop_box"))
            row["original_gate_failure"] = str(row.get("original_gate_failure") or "").strip()
            if not row["candidate_family"] or not row["category"] or not row["manifest_index"]:
                raise ValueError(
                    f"Validation adjudication {adjudication_id} must include candidate_family, category, and manifest_index"
                )
            rows.append(row)
    return {
        "status": "supplied",
        "path": str(adjudication_path),
        "rows": rows,
        "row_count": len(rows),
        "applied_count": 0,
        "nonblocking_count": 0,
        "blocking_count": 0,
        "unused_count": len(rows),
    }


def _adjudication_applies_to_gate(adjudication: Mapping[str, Any], gate_row: Mapping[str, Any]) -> bool:
    if _adjudication_match_key(adjudication) != _adjudication_match_key(gate_row):
        return False
    original_failure = str(adjudication.get("original_gate_failure") or "").strip()
    gate_failure = str(gate_row.get("failure_reason") or "").strip()
    if original_failure == gate_failure:
        return True
    return (
        str(adjudication.get("adjudication_label") or "") == "ground_truth_omission"
        and str(gate_row.get("category") or "") == "background"
        and bool(gate_failure)
    )


def _recompute_category_gate_status(category_gate_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    reasons_by_family: Dict[str, set[str]] = defaultdict(set)
    categories_by_family: Dict[str, set[str]] = defaultdict(set)
    families = sorted({str(row.get("family", "unknown")) for row in category_gate_rows})
    for row in category_gate_rows:
        family = str(row.get("family", "unknown"))
        categories_by_family[family].add(str(row.get("category", "unknown")))
        if row.get("gate_passed") is False or str(row.get("gate_passed")) == "False":
            reason = str(row.get("failure_reason") or "").strip()
            if reason:
                reasons_by_family[family].add(reason)
    return {
        family: {
            "promotion_evidence_status": PROMOTION_NOT_ELIGIBLE if reasons_by_family.get(family) else PROMOTION_ELIGIBLE,
            "reasons": sorted(reasons_by_family.get(family, set())),
            "categories_evaluated": sorted(categories_by_family.get(family, set())),
            "failed_gate_count": sum(
                1
                for row in category_gate_rows
                if str(row.get("family", "unknown")) == family
                and (row.get("gate_passed") is False or str(row.get("gate_passed")) == "False")
            ),
            "gate_count": sum(1 for row in category_gate_rows if str(row.get("family", "unknown")) == family),
        }
        for family in families
    }


def _recompute_prediction_shape_from_category_gate(
    prediction_shape: Mapping[str, Any],
    category_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    failed_reasons_by_key: Dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for gate_row in category_gate.get("rows", []):
        if gate_row.get("gate_passed") is False or str(gate_row.get("gate_passed")) == "False":
            key = (
                str(gate_row.get("family", "unknown")),
                str(gate_row.get("category", "unknown")),
                str(gate_row.get("manifest_index") or ""),
            )
            reason = str(gate_row.get("failure_reason") or gate_row.get("gate_name") or "").strip()
            if reason:
                failed_reasons_by_key[key].add(reason)

    shape_rows = []
    reasons_by_family: Dict[str, set[str]] = defaultdict(set)
    for row in prediction_shape.get("rows", []):
        key = (
            str(row.get("family", "unknown")),
            str(row.get("category", "unknown")),
            str(row.get("manifest_index") or ""),
        )
        reasons = sorted(failed_reasons_by_key.get(key, set()))
        if reasons:
            reasons.append("category_metric_failure")
        for reason in reasons:
            reasons_by_family[key[0]].add(reason)
        shape_rows.append(
            {
                **dict(row),
                "shape_gate_failed": bool(reasons),
                "shape_gate_reasons": "|".join(reasons),
            }
        )
    families = sorted({str(row.get("family", "unknown")) for row in shape_rows} | set(reasons_by_family))
    family_status = {
        family: {
            "promotion_evidence_status": PROMOTION_NOT_ELIGIBLE if reasons_by_family.get(family) else PROMOTION_ELIGIBLE,
            "reasons": sorted(reasons_by_family.get(family, set())),
        }
        for family in families
    }
    return {
        **dict(prediction_shape),
        "rows": shape_rows,
        "family_status": family_status,
        "blocked": any(status["reasons"] for status in family_status.values()),
        "category_gate": category_gate,
    }


def _apply_validation_adjudication_to_audit(
    audit: Dict[str, Any],
    adjudication_payload: Dict[str, Any],
) -> list[dict[str, Any]]:
    category_gate = audit.get("category_gate", {})
    original_rows = list(category_gate.get("rows", []))
    if not original_rows:
        return []
    audit_families = {str(row.get("family", "unknown")) for row in original_rows}
    adjudications = [
        row
        for row in list(adjudication_payload.get("rows") or [])
        if str(row.get("candidate_family") or "") in audit_families
    ]
    applied: list[dict[str, Any]] = []
    used_adjudication_ids: set[str] = set()
    updated_rows: list[dict[str, Any]] = []
    for gate_row in original_rows:
        updated = dict(gate_row)
        gate_passed_before = not (
            gate_row.get("gate_passed") is False or str(gate_row.get("gate_passed")) == "False"
        )
        updated["gate_passed_before_adjudication"] = gate_passed_before
        updated["failure_reason_before_adjudication"] = gate_row.get("failure_reason")
        updated["validation_adjudication_status"] = VALIDATION_ADJUDICATION_NOT_REVIEWED
        updated["validation_adjudication_id"] = ""
        updated["validation_adjudication_label"] = ""
        updated["validation_adjudication_decision"] = ""
        updated["validation_adjudication_requires_mask_correction"] = ""
        updated["validation_adjudication_counts_as_model_failure"] = ""
        updated["validation_adjudication_reviewer_note"] = ""
        updated["validation_adjudication_next_action"] = ""
        updated["gate_passed_after_adjudication"] = gate_passed_before
        if not gate_passed_before:
            matching = [row for row in adjudications if _adjudication_applies_to_gate(row, gate_row)]
            if len(matching) > 1:
                ids = ", ".join(str(row.get("adjudication_id")) for row in matching)
                raise ValueError(
                    "Multiple validation adjudications match one category gate row: "
                    f"family={gate_row.get('family')} manifest_index={gate_row.get('manifest_index')} ids={ids}"
                )
            if matching:
                adjudication = matching[0]
                used_adjudication_ids.add(str(adjudication.get("adjudication_id")))
                counts_as_failure = bool(adjudication["counts_as_model_failure_after_review"])
                status = VALIDATION_ADJUDICATION_BLOCKING if counts_as_failure else VALIDATION_ADJUDICATION_NONBLOCKING
                updated["validation_adjudication_status"] = status
                updated["validation_adjudication_id"] = adjudication.get("adjudication_id")
                updated["validation_adjudication_label"] = adjudication.get("adjudication_label")
                updated["validation_adjudication_decision"] = adjudication.get("adjudication_decision")
                updated["validation_adjudication_requires_mask_correction"] = adjudication.get("requires_mask_correction")
                updated["validation_adjudication_counts_as_model_failure"] = counts_as_failure
                updated["validation_adjudication_reviewer_note"] = adjudication.get("reviewer_note", "")
                updated["validation_adjudication_next_action"] = adjudication.get("next_action", "")
                updated["gate_passed_after_adjudication"] = not counts_as_failure
                if not counts_as_failure:
                    updated["gate_passed"] = True
                    updated["failure_reason"] = ""
                applied.append(
                    {
                        **dict(adjudication),
                        "family": gate_row.get("family"),
                        "gate_name": gate_row.get("gate_name"),
                        "metric_name": gate_row.get("metric_name"),
                        "failure_reason_before_adjudication": gate_row.get("failure_reason"),
                        "gate_passed_before_adjudication": gate_passed_before,
                        "gate_passed_after_adjudication": updated["gate_passed_after_adjudication"],
                        "validation_adjudication_status": status,
                    }
                )
        updated_rows.append(updated)
    unused = [
        str(row.get("adjudication_id"))
        for row in adjudications
        if str(row.get("adjudication_id")) not in used_adjudication_ids
    ]
    if adjudications and unused:
        raise ValueError(
            "Validation adjudication row(s) did not match any failing category gate: "
            + ", ".join(sorted(unused))
        )
    family_status = _recompute_category_gate_status(updated_rows)
    updated_category_gate = {
        **dict(category_gate),
        "rows": updated_rows,
        "family_status": family_status,
        "blocked": any(status["reasons"] for status in family_status.values()),
    }
    audit["category_gate"] = updated_category_gate
    audit["prediction_shape"] = _recompute_prediction_shape_from_category_gate(
        audit.get("prediction_shape", {}),
        updated_category_gate,
    )
    adjudication_payload["applied_count"] = len(applied)
    adjudication_payload["nonblocking_count"] = sum(
        1 for row in applied if row.get("validation_adjudication_status") == VALIDATION_ADJUDICATION_NONBLOCKING
    )
    adjudication_payload["blocking_count"] = sum(
        1 for row in applied if row.get("validation_adjudication_status") == VALIDATION_ADJUDICATION_BLOCKING
    )
    adjudication_payload["unused_count"] = 0 if adjudications else 0
    return applied


def _refresh_root_cause_after_adjudication(audit: Dict[str, Any], summary: Mapping[str, Any]) -> None:
    family = str(summary.get("family") or "")
    shape_status = audit.get("prediction_shape", {}).get("family_status", {}).get(family, {})
    shape_reasons = list(shape_status.get("reasons") or [])
    has_background_false_positive = any("background_false_positive" in reason for reason in shape_reasons)
    has_negative_background_supervision = (
        str(summary.get("provenance", {}).get("negative_crop_supervision_status") or "").strip().lower() == "present"
    )
    requires_transfer_base = _candidate_requires_transfer_base(
        CandidateRuntime(
            family=family,
            role=str(summary.get("comparison_role") or "candidate"),
            model_path=Path(str(summary["artifact_path"])) if summary.get("artifact_path") else None,
            seed=summary.get("seed"),
            command=summary.get("command"),
            status=str(summary.get("status") or "available"),
        ),
        dict(summary.get("provenance") or {}),
    )
    audit["root_cause"] = classify_root_causes(
        {
            "split_or_panel_bias": audit["split_integrity"]["promotion_evidence_status"] != PROMOTION_ELIGIBLE,
            "resize_policy_artifact": audit["resize_policy_parity"]["promotion_evidence_status"] != PROMOTION_ELIGIBLE
            or audit["resize_sensitivity"]["promotion_evidence_status"] != PROMOTION_ELIGIBLE,
            "class_channel_or_threshold_error": not audit["preprocessing_parity"]["ok"],
            "negative_background_supervision_missing": has_background_false_positive
            and not has_negative_background_supervision,
            "training_signal_insufficient": bool(summary.get("gate", {}).get("blocked")) or bool(shape_reasons),
            "mitochondria_base_defect": (
                requires_transfer_base
                and audit["artifact_status"]["promotion_evidence_status"] == PROMOTION_AUDIT_MISSING
            ),
        }
    )


def _threshold_sweep_rows(audit: Dict[str, Any]) -> list[dict[str, Any]]:
    sweep_path = Path(str(audit.get("artifacts", {}).get("threshold_sweep") or ""))
    if not sweep_path.exists():
        return []
    with sweep_path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _threshold_metric_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    selected_threshold: float,
) -> tuple[float | None, float | None]:
    background_values: list[float] = []
    recall_values: list[float] = []
    for row in rows:
        if abs(float(row.get("threshold") or "nan") - float(selected_threshold)) > 1e-9:
            continue
        category = str(row.get("category") or "")
        if category == "background" and row.get("false_positive_foreground_fraction") not in (None, ""):
            background_values.append(float(row["false_positive_foreground_fraction"]))
        if category in {"positive", "boundary"} and row.get("recall") not in (None, ""):
            recall_values.append(float(row["recall"]))
    background_fp = float(np.mean(background_values)) if background_values else None
    positive_boundary_recall = float(np.mean(recall_values)) if recall_values else None
    return background_fp, positive_boundary_recall


def _select_validation_threshold(
    rows: Sequence[Dict[str, Any]],
    *,
    background_limit: float = BACKGROUND_FALSE_POSITIVE_LIMIT,
) -> dict[str, Any]:
    thresholds = sorted({float(row["threshold"]) for row in rows if row.get("threshold") not in (None, "")})
    if not thresholds:
        raise ValueError("Cannot derive a glomeruli comparison threshold because threshold_sweep.csv has no threshold rows.")

    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_rows = [row for row in rows if abs(float(row.get("threshold") or "nan") - threshold) <= 1e-9]
        families = sorted({str(row.get("candidate_family") or "") for row in threshold_rows if row.get("candidate_family")})
        background_by_family: dict[str, list[float]] = defaultdict(list)
        recall_values: list[float] = []
        for row in threshold_rows:
            family = str(row.get("candidate_family") or "")
            category = str(row.get("category") or "")
            if category == "background" and row.get("false_positive_foreground_fraction") not in (None, ""):
                background_by_family[family].append(float(row["false_positive_foreground_fraction"]))
            if category in {"positive", "boundary"} and row.get("recall") not in (None, ""):
                recall_values.append(float(row["recall"]))
        if not families or any(not background_by_family.get(family) for family in families):
            continue
        family_background = {
            family: float(np.mean(values))
            for family, values in background_by_family.items()
        }
        max_background = max(family_background.values())
        if max_background > background_limit:
            continue
        candidates.append(
            {
                "threshold": threshold,
                "max_background_false_positive_foreground_fraction": max_background,
                "mean_background_false_positive_foreground_fraction": float(np.mean(list(family_background.values()))),
                "positive_boundary_recall": float(np.mean(recall_values)) if recall_values else None,
            }
        )

    if not candidates:
        raise ValueError(
            "Cannot derive a glomeruli comparison threshold because no threshold in threshold_sweep.csv "
            f"keeps every candidate family background false-positive foreground fraction <= {background_limit}."
        )

    def score(row: dict[str, Any]) -> tuple[float, float, float]:
        recall = -1.0 if row["positive_boundary_recall"] is None else float(row["positive_boundary_recall"])
        return (
            recall,
            -float(row["mean_background_false_positive_foreground_fraction"]),
            -float(row["threshold"]),
        )

    selected = max(candidates, key=score)
    selected["selection_rule"] = (
        "Choose the threshold in the audit sweep that keeps every candidate family's mean background "
        f"false-positive foreground fraction <= {background_limit}, then maximize mean positive/boundary recall; "
        "ties prefer lower background foreground fraction and then the lower threshold."
    )
    selected["background_false_positive_limit"] = background_limit
    return selected


def _threshold_policy_from_audit(
    audit: Dict[str, Any],
    *,
    selected_threshold: float | None,
    explicit_threshold: bool = True,
) -> Dict[str, Any]:
    if not audit:
        threshold = COMPARE_PREDICTION_THRESHOLD if selected_threshold is None else float(selected_threshold)
        status = THRESHOLD_POLICY_FIXED_REVIEW if explicit_threshold else THRESHOLD_POLICY_UNVERIFIED
        rationale = (
            "operator_supplied_fixed_threshold_without_overcoverage_audit"
            if explicit_threshold
            else "legacy_underconfident_threshold_without_overcoverage_audit"
        )
        return {
            "threshold_policy_status": status,
            "threshold": threshold,
            "threshold_rationale": rationale,
            "threshold_selection_rule": None,
            "threshold_is_promotion_ready": False,
            "threshold_grid": [],
            "overcoverage_audit_dir": None,
            "overcoverage_audit_summary_path": None,
            "background_false_positive_foreground_fraction": None,
            "positive_boundary_recall": None,
        }

    sweep_rows = _threshold_sweep_rows(audit)
    if not sweep_rows:
        raise ValueError("Overcoverage audit is missing usable threshold_sweep.csv rows; cannot set threshold policy.")

    threshold_selection: dict[str, Any] | None = None
    if selected_threshold is None:
        threshold_selection = _select_validation_threshold(sweep_rows)
        threshold = float(threshold_selection["threshold"])
        status = THRESHOLD_POLICY_VALIDATION_DERIVED
        rationale = "selected_from_overcoverage_threshold_sweep"
        selection_rule = threshold_selection["selection_rule"]
    else:
        threshold = float(selected_threshold)
        status = THRESHOLD_POLICY_AUDIT_BACKED_FIXED
        rationale = "operator_supplied_fixed_threshold_with_threshold_sweep_evidence"
        selection_rule = "Operator supplied a fixed threshold and attached the overcoverage audit sweep as supporting evidence."

    background_fp, positive_boundary_recall = _threshold_metric_summary(
        sweep_rows,
        selected_threshold=threshold,
    )
    return {
        "threshold_policy_status": status,
        "threshold": threshold,
        "threshold_rationale": rationale,
        "threshold_selection_rule": selection_rule,
        "threshold_selection": threshold_selection,
        "threshold_is_promotion_ready": status in PROMOTION_READY_THRESHOLD_POLICIES,
        "threshold_grid": audit.get("thresholds") or [],
        "overcoverage_audit_dir": audit.get("audit_dir") or audit.get("output_dir"),
        "overcoverage_audit_summary_path": audit.get("audit_summary_path"),
        "background_false_positive_foreground_fraction": background_fp,
        "positive_boundary_recall": positive_boundary_recall,
    }


def _read_resize_screening_summary(summary_path: str | Path | None) -> list[dict[str, Any]]:
    if not summary_path:
        return []
    path = Path(summary_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Resize-screening summary not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def resize_screening_decision_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    reference_run_id: str = "p0_resize_screen_current_512to256",
    primary_run_id: str = "p0_resize_screen_512to512",
    fallback_run_id: str = "p0_resize_screen_512to384",
    background_limit: float = BACKGROUND_FALSE_POSITIVE_LIMIT,
    material_margin: float = RESIZE_SCREEN_MATERIAL_MARGIN,
) -> dict[str, Any]:
    """Derive the resize-screen decision from per-family screening rows."""
    attempt_rows = [row for row in rows if str(row.get("row_type") or "attempt") == "attempt"]
    current_rows = [
        row
        for row in attempt_rows
        if str(row.get("run_id") or "") == reference_run_id
        and str(row.get("runtime_status") or "") == "completed"
    ]
    completed_primary = [
        row
        for row in attempt_rows
        if str(row.get("run_id") or "") == primary_run_id
        and str(row.get("runtime_status") or "") == "completed"
    ]
    completed_fallback = [
        row
        for row in attempt_rows
        if str(row.get("run_id") or "") == fallback_run_id
        and str(row.get("runtime_status") or "") == "completed"
    ]
    comparator_rows = completed_primary or completed_fallback
    comparator_run_id = primary_run_id if completed_primary else (fallback_run_id if completed_fallback else None)
    if not current_rows:
        return {
            "row_type": "decision",
            "resize_decision_status": RESIZE_DECISION_UNRESOLVED,
            "resize_decision_reason": "reference_resize_screen_missing_or_failed",
            "selected_run_id": "",
            "selected_crop_size": "",
            "selected_image_size": "",
        }
    if not comparator_rows:
        return {
            "row_type": "decision",
            "resize_decision_status": RESIZE_DECISION_INFEASIBLE,
            "resize_decision_reason": "no_less_downsampled_or_no_downsample_screen_completed",
            "selected_run_id": "",
            "selected_crop_size": "",
            "selected_image_size": "",
        }

    def mean_metric(metric: str, metric_rows: Sequence[dict[str, Any]]) -> float | None:
        values = [_float_or_none(row.get(metric)) for row in metric_rows]
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None

    current_dice = mean_metric("positive_boundary_dice", current_rows)
    comparator_dice = mean_metric("positive_boundary_dice", comparator_rows)
    current_recall = mean_metric("positive_boundary_recall", current_rows)
    comparator_recall = mean_metric("positive_boundary_recall", comparator_rows)
    current_ratio_error = mean_metric("positive_boundary_prediction_to_truth_ratio_abs_error", current_rows)
    comparator_ratio_error = mean_metric("positive_boundary_prediction_to_truth_ratio_abs_error", comparator_rows)
    comparator_background = mean_metric("background_false_positive_foreground_fraction", comparator_rows)
    background_ok = comparator_background is None or comparator_background <= background_limit
    category_gate_ok = all(
        str(row.get("category_gate_status") or "") in {"", PROMOTION_ELIGIBLE}
        and str(row.get("category_gate_failed_count") or "0") in {"", "0", "0.0"}
        for row in comparator_rows
    )
    dice_improved = (
        current_dice is not None
        and comparator_dice is not None
        and comparator_dice > current_dice + material_margin
    )
    recall_improved = (
        current_recall is not None
        and comparator_recall is not None
        and comparator_recall > current_recall + material_margin
    )
    ratio_improved = (
        current_ratio_error is not None
        and comparator_ratio_error is not None
        and comparator_ratio_error < current_ratio_error - material_margin
    )
    selected_row = comparator_rows[0]
    if background_ok and category_gate_ok and (dice_improved or recall_improved or ratio_improved):
        return {
            "row_type": "decision",
            "resize_decision_status": RESIZE_DECISION_SELECTED_LESS_DOWNSAMPLED,
            "resize_decision_reason": "less_downsampled_policy_materially_improved_positive_or_boundary_metrics",
            "selected_run_id": comparator_run_id or "",
            "selected_crop_size": selected_row.get("crop_size") or "",
            "selected_image_size": selected_row.get("image_size") or selected_row.get("output_size") or "",
            "current_positive_boundary_dice": current_dice,
            "selected_positive_boundary_dice": comparator_dice,
            "current_positive_boundary_recall": current_recall,
            "selected_positive_boundary_recall": comparator_recall,
            "current_positive_boundary_prediction_to_truth_ratio_abs_error": current_ratio_error,
            "selected_positive_boundary_prediction_to_truth_ratio_abs_error": comparator_ratio_error,
            "selected_background_false_positive_foreground_fraction": comparator_background,
        }
    current_row = current_rows[0]
    return {
        "row_type": "decision",
        "resize_decision_status": RESIZE_DECISION_CURRENT_CLEARED,
        "resize_decision_reason": (
            "less_downsampled_policy_failed_category_gates"
            if not category_gate_ok
            else "less_downsampled_policy_not_materially_better_than_current_policy"
        ),
        "selected_run_id": reference_run_id,
        "selected_crop_size": current_row.get("crop_size") or "",
        "selected_image_size": current_row.get("image_size") or current_row.get("output_size") or "",
        "current_positive_boundary_dice": current_dice,
        "selected_positive_boundary_dice": comparator_dice,
        "current_positive_boundary_recall": current_recall,
        "selected_positive_boundary_recall": comparator_recall,
        "current_positive_boundary_prediction_to_truth_ratio_abs_error": current_ratio_error,
        "selected_positive_boundary_prediction_to_truth_ratio_abs_error": comparator_ratio_error,
        "selected_background_false_positive_foreground_fraction": comparator_background,
    }


def _resize_sensitivity_from_screening(
    rows: Sequence[dict[str, Any]],
    *,
    crop_size: int,
    output_size: int,
    summary_path: str | Path | None,
) -> dict[str, Any] | None:
    if not rows:
        return None
    decision_rows = [row for row in rows if str(row.get("row_type") or "") == "decision"]
    decision = decision_rows[-1] if decision_rows else resize_screening_decision_from_rows(rows)
    status = str(decision.get("resize_decision_status") or RESIZE_DECISION_UNRESOLVED)
    selected_crop_size = _int_or_none(decision.get("selected_crop_size"))
    selected_output_size = _int_or_none(decision.get("selected_image_size") or decision.get("selected_output_size"))
    selected_current_policy = selected_crop_size == int(crop_size) and selected_output_size == int(output_size)
    if status == RESIZE_DECISION_CURRENT_CLEARED:
        promotion_status = PROMOTION_ELIGIBLE if selected_current_policy else PROMOTION_INSUFFICIENT
        gate_status = "resize_benefit_cleared_current_policy" if selected_current_policy else "resize_policy_not_selected"
    elif status == RESIZE_DECISION_SELECTED_LESS_DOWNSAMPLED:
        promotion_status = PROMOTION_ELIGIBLE if selected_current_policy else PROMOTION_INSUFFICIENT
        gate_status = "resize_policy_selected" if selected_current_policy else "resize_policy_not_selected"
    elif status == RESIZE_DECISION_INFEASIBLE:
        promotion_status = PROMOTION_INSUFFICIENT
        gate_status = RESIZE_DECISION_INFEASIBLE
    else:
        promotion_status = PROMOTION_INSUFFICIENT
        gate_status = RESIZE_DECISION_UNRESOLVED
    return {
        "status": gate_status,
        "promotion_evidence_status": promotion_status,
        "required_comparison": None if promotion_status == PROMOTION_ELIGIBLE else "current_policy_vs_no_downsample_or_less_downsample",
        "infeasibility_reason": (
            decision.get("resize_decision_reason")
            if status in {RESIZE_DECISION_INFEASIBLE, RESIZE_DECISION_UNRESOLVED}
            else None
        ),
        "resize_screening_summary_path": str(Path(summary_path).expanduser()) if summary_path else None,
        "resize_decision_status": status,
        "resize_decision_reason": decision.get("resize_decision_reason"),
        "selected_run_id": decision.get("selected_run_id"),
        "selected_crop_size": selected_crop_size,
        "selected_output_size": selected_output_size,
    }


def _split_path_for_model(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_splits.json")


def _validation_trace_path_for_model(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_validation_prediction_trace.csv")


def _read_validation_trace_rows(
    model_path: str | Path | None,
    *,
    root_causes: Sequence[str],
    remediation_path: str | None,
) -> list[Dict[str, Any]]:
    if not model_path:
        return []
    trace_path = _validation_trace_path_for_model(Path(model_path).expanduser())
    if not trace_path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with trace_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = dict(row)
            record["trace_source"] = str(trace_path)
            if not str(record.get("root_causes") or "").strip() and root_causes:
                record["root_causes"] = "|".join(root_causes)
            if not str(record.get("remediation_path") or "").strip() and remediation_path:
                record["remediation_path"] = remediation_path
            rows.append(record)
    return rows


def _merge_split_sidecar(
    provenance: Dict[str, Any],
    *,
    model_path: Path | None,
) -> Dict[str, Any]:
    if model_path is None:
        return provenance
    split_path = _split_path_for_model(model_path)
    if not split_path.exists():
        return provenance
    with split_path.open() as handle:
        split_metadata = json.load(handle)
    merged = dict(provenance)
    for key in (
        "train_images",
        "valid_images",
        "train_subjects",
        "valid_subjects",
        "split_seed",
        "splitter_name",
        "data_root",
        "training_mode",
        "crop_size",
        "output_size",
        "image_size",
        "counts",
    ):
        if key not in split_metadata:
            continue
        current = merged.get(key)
        incoming = split_metadata[key]
        if current not in (None, "", [], {}) and incoming not in (None, "", [], {}) and current != incoming:
            raise ValueError(
                f"Conflicting {key!r} between run metadata and split sidecar for {model_path}: "
                f"{current!r} != {incoming!r}"
            )
        merged[key] = incoming if current in (None, "", [], {}) else current
    merged["split_sidecar_path"] = str(split_path)
    return merged


def _build_fallback_provenance(runtime: CandidateRuntime) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "family": runtime.family,
        "comparison_role": runtime.role,
        "artifact_path": str(runtime.model_path) if runtime.model_path else None,
        "artifact_status": "comparison_only_missing_metadata" if runtime.model_path else "unavailable",
        "scientific_promotion_status": "not_evaluated" if runtime.model_path else "unavailable",
        "training_mode": "unknown",
        "seed": runtime.seed,
    }
    if runtime.command:
        provenance["comparison_command"] = runtime.command
    if runtime.error:
        provenance["error"] = runtime.error
    return provenance


def _merged_provenance(runtime: CandidateRuntime) -> Dict[str, Any]:
    metadata = _read_metadata_if_available(runtime.model_path)
    if not metadata:
        return _merge_split_sidecar(
            _build_fallback_provenance(runtime),
            model_path=runtime.model_path,
        )
    provenance = dict(metadata)
    provenance.setdefault("artifact_path", str(runtime.model_path) if runtime.model_path else None)
    provenance.setdefault("artifact_status", metadata.get("artifact_status", "supported_runtime"))
    provenance.setdefault(
        "scientific_promotion_status",
        metadata.get("scientific_promotion_status", "not_evaluated"),
    )
    invocation = provenance.get("invocation", {})
    if runtime.command and "comparison_command" not in provenance:
        provenance["comparison_command"] = runtime.command
    provenance["comparison_role"] = runtime.role
    provenance["family"] = runtime.family
    provenance["seed"] = invocation.get("seed", runtime.seed)
    return _merge_split_sidecar(provenance, model_path=runtime.model_path)


def _candidate_requires_transfer_base(runtime: CandidateRuntime, provenance: Dict[str, Any]) -> bool:
    family = str(runtime.family)
    candidate_family = str(provenance.get("candidate_family", ""))
    return family == "transfer" or candidate_family == "mitochondria_transfer"


def _promotion_status_from_reasons(reasons: Sequence[str]) -> str:
    if any("missing" in reason or "audit" in reason for reason in reasons):
        return PROMOTION_AUDIT_MISSING
    if reasons:
        return PROMOTION_NOT_ELIGIBLE
    return PROMOTION_INSUFFICIENT


def _heldout_mask_paths_for_runtimes(runtimes: Sequence[CandidateRuntime]) -> List[Path] | None:
    valid_sets: list[set[str]] = []
    train_subject_sets: list[set[str]] = []
    for runtime in runtimes:
        provenance = _merged_provenance(runtime)
        valid_images = provenance.get("valid_images") or []
        if not valid_images:
            return None
        valid_sets.append({str(Path(path).expanduser()) for path in valid_images})
        train_subject_sets.append(
            {_subject_id_from_image_path(Path(path).expanduser()) for path in provenance.get("train_images") or []}
        )
    if not valid_sets:
        return None
    shared_images = sorted(set.intersection(*valid_sets))
    if not shared_images:
        return []
    train_subjects = set.union(*train_subject_sets) if train_subject_sets else set()
    mask_paths: List[Path] = []
    for image_path in shared_images:
        if _subject_id_from_image_path(Path(image_path).expanduser()) in train_subjects:
            continue
        try:
            mask_paths.append(Path(get_y_full(Path(image_path))))
        except Exception:
            continue
    return mask_paths


def _build_deterministic_manifest(
    *,
    data_root: Path,
    runtimes: Sequence[CandidateRuntime],
    crop_size: int,
    examples_per_category: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    heldout_mask_paths = _heldout_mask_paths_for_runtimes(runtimes)
    manifest_source = "shared_candidate_validation_split"
    if heldout_mask_paths is None:
        heldout_mask_paths = _validation_mask_paths(data_root)
        manifest_source = "full_data_root_audit_missing_split_provenance"
    elif not heldout_mask_paths:
        return [], {
            "manifest_source": manifest_source,
            "promotion_evidence_status": PROMOTION_INSUFFICIENT,
            "decision_reason": "insufficient_heldout_category_support",
            "count": 0,
            "categories": {},
        }
    try:
        manifest = deterministic_validation_manifest(
            heldout_mask_paths,
            crop_size=crop_size,
            examples_per_category=examples_per_category,
        )
        audit = audit_manifest_crops(manifest)
        audit["manifest_source"] = manifest_source
        audit["promotion_evidence_status"] = (
            PROMOTION_ELIGIBLE
            if manifest_source == "shared_candidate_validation_split"
            else PROMOTION_AUDIT_MISSING
        )
        return manifest, audit
    except ValueError as exc:
        return [], {
            "manifest_source": manifest_source,
            "promotion_evidence_status": PROMOTION_INSUFFICIENT,
            "decision_reason": "insufficient_heldout_category_support",
            "error": str(exc),
            "count": 0,
            "categories": {},
        }


def _manifest_metadata_index(data_root: Path) -> Dict[str, Dict[str, str]]:
    manifest_path = data_root / "manifest.csv" if data_root.name == "cohorts" else data_root.parent / "manifest.csv"
    if not manifest_path.exists():
        return {}
    runtime_root = data_root.parents[1] if data_root.name == "cohorts" and data_root.parent.name == "raw_data" else data_root.parents[2]
    index: Dict[str, Dict[str, str]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata = {
                "cohort_id": str(row.get("cohort_id") or ""),
                "lane_assignment": str(row.get("lane_assignment") or ""),
            }
            for key in ("image_path", "mask_path"):
                value = str(row.get(key) or "")
                if not value:
                    continue
                path = Path(value).expanduser()
                absolute = path if path.is_absolute() else runtime_root / path
                index[str(absolute)] = metadata
                index[value] = metadata
    return index


def _annotate_manifest_with_context(manifest: Sequence[Dict[str, Any]], data_root: Path) -> List[Dict[str, Any]]:
    context = _manifest_metadata_index(data_root)
    if not context:
        return [dict(row) for row in manifest]
    annotated: List[Dict[str, Any]] = []
    for row in manifest:
        record = dict(row)
        metadata = context.get(str(row.get("image_path"))) or context.get(str(row.get("mask_path"))) or {}
        if metadata:
            record.setdefault("cohort_id", metadata.get("cohort_id"))
            record.setdefault("lane_assignment", metadata.get("lane_assignment"))
        annotated.append(record)
    return annotated


def _provenance_resize_policy(provenance: Dict[str, Any], *, fallback_crop_size: int, fallback_output_size: int) -> Dict[str, Any]:
    policy = provenance.get("resize_policy")
    if isinstance(policy, dict):
        return dict(policy)
    return resize_policy_record(
        crop_size=int(provenance.get("crop_size") or fallback_crop_size),
        output_size=int(provenance.get("output_size") or provenance.get("image_size") or fallback_output_size),
    )


def _candidate_audit_rows(
    *,
    summary: Dict[str, Any],
    manifest: Sequence[Dict[str, Any]],
    expected_size: int,
    prediction_threshold: float,
    threshold_policy: Dict[str, Any],
    resize_screening_rows: Sequence[dict[str, Any]] | None = None,
    resize_screening_summary_path: str | Path | None = None,
) -> Dict[str, Any]:
    provenance = summary.get("provenance", {})
    prediction_rows = [
        {
            **dict(row),
            "threshold_policy_status": threshold_policy.get("threshold_policy_status"),
        }
        for row in summary.get("prediction_rows", [])
    ]
    summary["prediction_rows"] = prediction_rows
    split_integrity = audit_split_overlap(
        train_images=provenance.get("train_images"),
        valid_images=provenance.get("valid_images"),
        promotion_manifest=manifest,
    )
    training_policy = _provenance_resize_policy(
        provenance,
        fallback_crop_size=expected_size,
        fallback_output_size=expected_size,
    )
    resize_rows = []
    resize_ratio = float(training_policy.get("crop_to_output_resize_ratio") or 1.0)
    resize_sensitivity_status = (
        "resize_benefit_unproven"
        if resize_ratio != 1.0
        else "no_downsample_policy"
    )
    resize_infeasibility_reason = (
        "no_no_downsample_or_less_downsample_candidate_artifact_available"
        if resize_ratio != 1.0
        else ""
    )
    for row in prediction_rows:
        resize_rows.append(
            {
                "family": summary["family"],
                "resize_sensitivity_status": resize_sensitivity_status,
                "resize_sensitivity_comparison": "current_policy_vs_no_downsample_or_less_downsample",
                "resize_sensitivity_infeasibility_reason": resize_infeasibility_reason,
                "heldout_dice": row.get("dice"),
                "heldout_jaccard": row.get("jaccard"),
                "truth_foreground_fraction": row.get("truth_foreground_fraction"),
                "prediction_foreground_fraction": row.get("prediction_foreground_fraction"),
                **resize_policy_record(
                    crop_size=int(training_policy.get("crop_size") or expected_size),
                    output_size=int(training_policy.get("output_size") or expected_size),
                    category=row.get("category"),
                    cohort_id=row.get("cohort_id"),
                    lane_assignment=row.get("lane_assignment"),
                    split="deterministic_promotion",
                ),
            }
        )
    if not resize_rows:
        resize_rows.append(
            {
                "family": summary["family"],
                "resize_sensitivity_status": resize_sensitivity_status,
                "resize_sensitivity_comparison": "current_policy_vs_no_downsample_or_less_downsample",
                "resize_sensitivity_infeasibility_reason": (
                    resize_infeasibility_reason or "deterministic_promotion_manifest_unavailable"
                ),
                "heldout_dice": None,
                "heldout_jaccard": None,
                "truth_foreground_fraction": None,
                "prediction_foreground_fraction": None,
                **resize_policy_record(
                    crop_size=int(training_policy.get("crop_size") or expected_size),
                    output_size=int(training_policy.get("output_size") or expected_size),
                    split="deterministic_promotion_unavailable",
                ),
            }
        )
    eval_policy = resize_policy_record(
        crop_size=int(training_policy.get("crop_size") or expected_size),
        output_size=int(training_policy.get("output_size") or expected_size),
    )
    resize_parity = audit_resize_policy_parity(training_policy, eval_policy)
    preprocessing_parity = audit_preprocessing_parity(
        learner_consistent_preprocessing=True,
        supported_threshold_semantics=True,
        training_policy=training_policy,
        evaluation_policy=eval_policy,
    )
    prediction_shape = audit_prediction_shapes(prediction_rows)
    category_gate = prediction_shape.get("category_gate", {})
    resize_sensitivity = {
        "status": resize_sensitivity_status,
        "promotion_evidence_status": PROMOTION_INSUFFICIENT if resize_ratio != 1.0 else PROMOTION_ELIGIBLE,
        "required_comparison": (
            "current_policy_vs_no_downsample_or_less_downsample"
            if resize_ratio != 1.0
            else None
        ),
        "infeasibility_reason": resize_infeasibility_reason or None,
    }
    screened_resize_sensitivity = _resize_sensitivity_from_screening(
        list(resize_screening_rows or []),
        crop_size=int(training_policy.get("crop_size") or expected_size),
        output_size=int(training_policy.get("output_size") or expected_size),
        summary_path=resize_screening_summary_path,
    )
    if screened_resize_sensitivity is not None:
        resize_sensitivity = screened_resize_sensitivity
    artifact_status = classify_artifact_status(
        provenance,
        loadable=bool(summary.get("available")),
        requires_transfer_base=_candidate_requires_transfer_base(
            CandidateRuntime(
                family=summary["family"],
                role=summary["comparison_role"],
                model_path=Path(summary["artifact_path"]) if summary.get("artifact_path") else None,
                seed=summary.get("seed"),
                command=summary.get("command"),
                status=summary.get("status", "unavailable"),
            ),
            provenance,
        ),
    )
    transfer_base_metadata = provenance.get("transfer_base_metadata") or {}
    transfer_base_report = {
        "artifact_path": provenance.get("transfer_base_artifact_path") or provenance.get("base_model_path"),
        "mitochondria_training_scope": provenance.get("transfer_base_mitochondria_training_scope")
        or transfer_base_metadata.get("mitochondria_training_scope"),
        "mitochondria_inference_claim_status": transfer_base_metadata.get("mitochondria_inference_claim_status"),
        "physical_training_image_count": transfer_base_metadata.get("mitochondria_physical_training_image_count"),
        "physical_testing_image_count": transfer_base_metadata.get("mitochondria_physical_testing_image_count"),
        "actual_fitted_image_count": len(transfer_base_metadata.get("actual_pretraining_image_paths") or []),
        "actual_fitted_mask_count": len(transfer_base_metadata.get("actual_pretraining_mask_paths") or []),
        "resize_policy": transfer_base_metadata.get("resize_policy"),
    }
    has_background_false_positive = any(
        "background_false_positive" in reason
        for status in prediction_shape["family_status"].values()
        for reason in status["reasons"]
    )
    has_negative_background_supervision = (
        str(provenance.get("negative_crop_supervision_status") or "").strip().lower() == "present"
    )
    root_cause = classify_root_causes(
        {
            "split_or_panel_bias": split_integrity["promotion_evidence_status"] != PROMOTION_ELIGIBLE,
            "resize_policy_artifact": resize_parity["promotion_evidence_status"] != PROMOTION_ELIGIBLE
            or resize_sensitivity["promotion_evidence_status"] != PROMOTION_ELIGIBLE,
            "class_channel_or_threshold_error": not preprocessing_parity["ok"],
            "negative_background_supervision_missing": has_background_false_positive
            and not has_negative_background_supervision,
            "training_signal_insufficient": bool(summary.get("gate", {}).get("blocked"))
            or has_background_false_positive,
            "mitochondria_base_defect": (
                _candidate_requires_transfer_base(
                    CandidateRuntime(
                        family=summary["family"],
                        role=summary["comparison_role"],
                        model_path=Path(summary["artifact_path"]) if summary.get("artifact_path") else None,
                        seed=summary.get("seed"),
                        command=summary.get("command"),
                        status=summary.get("status", "unavailable"),
                    ),
                    provenance,
                )
                and artifact_status["promotion_evidence_status"] == PROMOTION_AUDIT_MISSING
            ),
        }
    )
    reproduction_rows = [
        failure_reproduction_row(
            candidate_family=summary["family"],
            panel_id=f"{summary['family']}-{row.get('manifest_index', index)}",
            artifact_path=summary.get("artifact_path"),
            image_path=row.get("image_path"),
            mask_path=row.get("mask_path"),
            crop_box=row.get("crop_box"),
            resize_policy=training_policy,
            threshold=prediction_threshold,
            prediction_tensor_shape=row.get("prediction_tensor_shape"),
            overlay_path=row.get("review_panel_path"),
            root_causes=root_cause["root_causes"],
            remediation_path=root_cause["remediation_path"],
        )
        for index, row in enumerate(summary.get("prediction_rows", []))
    ]
    reproduction_rows.extend(
        _read_validation_trace_rows(
            summary.get("artifact_path"),
            root_causes=root_cause["root_causes"],
            remediation_path=root_cause["remediation_path"],
        )
    )
    reproduction_rows.extend(
        _read_validation_trace_rows(
            transfer_base_report.get("artifact_path"),
            root_causes=[],
            remediation_path="audit_trace_only_not_a_root_cause_classification",
        )
    )
    return {
        "split_integrity": split_integrity,
        "resize_policy": resize_rows,
        "resize_policy_parity": resize_parity,
        "resize_sensitivity": resize_sensitivity,
        "preprocessing_parity": preprocessing_parity,
        "prediction_shape": prediction_shape,
        "category_gate": category_gate,
        "artifact_status": artifact_status,
        "transfer_base_report": transfer_base_report,
        "root_cause": root_cause,
        "failure_reproduction": reproduction_rows,
    }


def _discover_model_path(family_dir: Path) -> Path:
    candidates = sorted(family_dir.rglob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model artifact produced under {family_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_training_command(command: list[str], family: str, role: str, model_root: Path, seed: int) -> CandidateRuntime:
    try:
        logger.info("Running candidate training command: %s", " ".join(command))
        run_logged_subprocess(command, logger=logger)
        family_dir = model_root / family
        return CandidateRuntime(
            family=family,
            role=role,
            model_path=_discover_model_path(family_dir),
            seed=seed,
            command=command,
            status="available",
        )
    except Exception as exc:
        return CandidateRuntime(
            family=family,
            role=role,
            model_path=None,
            seed=seed,
            command=command,
            status="unavailable",
            error=str(exc),
        )


def _candidate_command(
    *,
    data_dir: Path,
    model_dir: Path,
    model_name: str,
    epochs: int,
    learning_rate: float,
    image_size: int,
    crop_size: int,
    batch_size: int | None,
    loss_name: str | None,
    seed: int,
    from_scratch: bool,
    base_model: Path | None = None,
    split_manifest_path: Path | None = None,
    negative_crop_manifest_path: Path | None = None,
    negative_crop_sampler_weight: float = 0.0,
    augmentation_variant: str = "fastai_default",
    device: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "eq.training.train_glomeruli",
        "--data-dir",
        str(data_dir),
        "--model-dir",
        str(model_dir),
        "--model-name",
        model_name,
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
        "--image-size",
        str(image_size),
        "--crop-size",
        str(crop_size),
        "--seed",
        str(seed),
    ]
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    if loss_name:
        command.extend(["--loss", loss_name])
    if split_manifest_path is not None:
        command.extend(["--split-manifest", str(split_manifest_path)])
    if negative_crop_manifest_path is not None:
        command.extend(
            [
                "--negative-crop-manifest",
                str(negative_crop_manifest_path),
                "--negative-crop-sampler-weight",
                str(negative_crop_sampler_weight),
            ]
        )
    command.extend(["--augmentation-variant", augmentation_variant])
    if device:
        command.extend(["--device", device])
    if from_scratch:
        command.append("--from-scratch")
    else:
        if base_model is None:
            raise ValueError("Transfer candidate requires an explicit base_model path.")
        command.extend(["--base-model", str(base_model)])
    return command


def _subject_id_from_image_path(image_path: Path) -> str:
    if image_path.parent != image_path and image_path.parent.name not in {"images", "masks"}:
        return image_path.parent.name
    stem = image_path.stem.removesuffix("_mask")
    match = re.match(r"(?P<subject>.+?)_image\d+$", stem, flags=re.IGNORECASE)
    if match:
        return match.group("subject")
    return stem


def _write_shared_training_split(data_root: Path, output_root: Path, *, seed: int) -> Path:
    """Persist the one train/validation split used by fresh comparison candidates."""
    image_paths = [str(training_item_image_path(item)) for item in get_items_full_images(data_root)]
    if len(image_paths) < 2:
        raise ValueError("Candidate comparison requires at least two image/mask pairs for an explicit split.")
    subject_to_images: dict[str, list[str]] = defaultdict(list)
    for image_path in image_paths:
        subject_to_images[_subject_id_from_image_path(Path(image_path))].append(image_path)
    if len(subject_to_images) < 2:
        raise ValueError(
            "Candidate comparison requires at least two resolved subjects for a subject-held-out split."
        )
    rng = np.random.default_rng(int(seed))
    subjects = np.asarray(sorted(subject_to_images))
    rng.shuffle(subjects)
    target_valid_count = max(1, int(round(len(image_paths) * float(DEFAULT_VAL_RATIO))))
    valid_subjects: list[str] = []
    valid_image_count = 0
    for subject in subjects:
        if len(valid_subjects) >= len(subjects) - 1:
            break
        valid_subjects.append(str(subject))
        valid_image_count += len(subject_to_images[str(subject)])
        if valid_image_count >= target_valid_count:
            break
    valid_subject_set = set(valid_subjects)
    train_subjects = [subject for subject in sorted(subject_to_images) if subject not in valid_subject_set]
    train_images = [
        image
        for subject in train_subjects
        for image in sorted(subject_to_images[subject])
    ]
    valid_images = [
        image
        for subject in sorted(valid_subject_set)
        for image in sorted(subject_to_images[subject])
    ]
    payload = {
        "split_seed": int(seed),
        "splitter_name": "explicit_shared_subject_split",
        "valid_pct": float(DEFAULT_VAL_RATIO),
        "train_subjects": train_subjects,
        "valid_subjects": sorted(valid_subject_set),
        "train_images": train_images,
        "valid_images": valid_images,
        "counts": {"train": len(train_images), "valid": len(valid_images)},
    }
    split_path = output_root / "shared_candidate_training_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return split_path


def _crop_array(arr: np.ndarray, crop_box: Sequence[int]) -> np.ndarray:
    left, top, right, bottom = [int(value) for value in crop_box]
    return arr[top:bottom, left:right]


def _load_crop_bundle(item: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_path = Path(item.get("image_path") or resolve_image_path_for_mask(item["mask_path"]))
    mask_path = Path(item["mask_path"])
    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    truth_crop = (_crop_array(mask, item["crop_box"]) > 0).astype(np.uint8)
    image_crop = _crop_array(image, item["crop_box"])
    return image, image_crop, truth_crop


def _predict_crop_with_audit(
    learn: Any,
    image_crop: np.ndarray,
    truth_shape: tuple[int, int],
    expected_size: int,
    threshold: float = COMPARE_PREDICTION_THRESHOLD,
) -> tuple[np.ndarray, Dict[str, Any]]:
    core = create_prediction_core(expected_size)
    device = next(learn.model.parameters()).device
    pil_image = Image.fromarray(image_crop).convert("RGB")
    resized = pil_image.resize((expected_size, expected_size), Image.Resampling.BILINEAR)
    image_np = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.no_grad():
        raw_output = learn.model(tensor.to(device))
    if raw_output.shape[1] == 2:
        pred_prob = torch.softmax(raw_output, dim=1)[:, 1]
    else:
        pred_prob = torch.sigmoid(raw_output)
    pred_np = pred_prob.squeeze().detach().cpu().numpy()
    pred_resized = core.resize_prediction_to_match(pred_np, truth_shape)
    return (pred_resized > float(threshold)).astype(np.uint8), {
        "input_tensor_shape": [int(value) for value in tensor.shape],
        "raw_output_shape": [int(value) for value in raw_output.shape],
        "prediction_probability_shape": [int(value) for value in pred_np.shape],
        "prediction_resized_shape": [int(value) for value in pred_resized.shape],
        "threshold": float(threshold),
        "inference_preprocessing": "deterministic_resize_imagenet_normalize",
        "threshold_resize_order": "resize_probability_then_threshold",
    }


def _predict_crop(learn: Any, image_crop: np.ndarray, truth_shape: tuple[int, int], expected_size: int) -> np.ndarray:
    pred, _audit = _predict_crop_with_audit(
        learn,
        image_crop,
        truth_shape,
        expected_size,
        threshold=COMPARE_PREDICTION_THRESHOLD,
    )
    return pred


def _overlay_mask(image_crop: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = image_crop.astype(np.float32).copy()
    overlay = np.zeros_like(base)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    active = mask.astype(bool)
    base[active] = (0.6 * base[active]) + (0.4 * overlay[active])
    return np.clip(base, 0, 255).astype(np.uint8)


def _save_review_panel(
    *,
    asset_path: Path,
    image_crop: np.ndarray,
    truth_crop: np.ndarray,
    pred_crop: np.ndarray,
) -> None:
    truth_overlay = _overlay_mask(image_crop, truth_crop, (0, 255, 0))
    pred_overlay = _overlay_mask(image_crop, pred_crop, (255, 0, 0))
    panel = np.concatenate([image_crop, truth_overlay, pred_overlay], axis=1)
    Image.fromarray(panel).save(asset_path)


def _front_facing_panel_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _resize_panel_rgb(arr: np.ndarray, size: int = FRONT_FACING_PANEL_TILE_SIZE) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)


def _model_input_panel(image_crop: np.ndarray, expected_size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image_crop.astype(np.uint8)).resize(
            (expected_size, expected_size),
            Image.Resampling.BILINEAR,
        )
    )


def _raw_context_panel(source_image: np.ndarray, crop_box: Sequence[int], size: int = FRONT_FACING_PANEL_TILE_SIZE) -> Image.Image:
    image = Image.fromarray(source_image.astype(np.uint8)).convert("RGB")
    source_width, source_height = image.size
    scale = min(size / source_width, size / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), "white")
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    left, top, right, bottom = [int(value) for value in crop_box]
    box = (
        offset_x + int(round(left * scale)),
        offset_y + int(round(top * scale)),
        offset_x + int(round(right * scale)),
        offset_y + int(round(bottom * scale)),
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(255, 220, 0), width=max(3, size // 90))
    return canvas


def _error_overlay(image_crop: np.ndarray, truth_crop: np.ndarray, pred_crop: np.ndarray) -> np.ndarray:
    base = image_crop.astype(np.float32).copy()
    truth = truth_crop.astype(bool)
    pred = pred_crop.astype(bool)
    overlay = np.zeros_like(base)
    overlay[truth & pred] = (0, 255, 0)
    overlay[pred & ~truth] = (255, 0, 0)
    overlay[truth & ~pred] = (0, 0, 255)
    active = truth | pred
    base[active] = (0.45 * base[active]) + (0.55 * overlay[active])
    return np.clip(base, 0, 255).astype(np.uint8)


def _prediction_metric_sort_value(example: Dict[str, Any]) -> tuple[float, ...]:
    row = example["row"]
    category = str(row.get("category", ""))
    dice = float(row.get("dice") or 0.0)
    jaccard = float(row.get("jaccard") or 0.0)
    pixel_accuracy_value = float(row.get("pixel_accuracy") or 0.0)
    pred_fg = float(row.get("prediction_foreground_fraction") or 0.0)
    truth_fg = float(row.get("truth_foreground_fraction") or 0.0)
    foreground_delta = abs(pred_fg - truth_fg)
    if category == "background":
        return (pred_fg, -pixel_accuracy_value, -dice)
    return (-dice, -jaccard, foreground_delta)


def _select_front_facing_examples(examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used_manifest_indices: set[int] = set()
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_category[str(example["row"].get("category"))].append(example)

    for category in FRONT_FACING_PANEL_CATEGORIES:
        category_examples = sorted(by_category.get(category, []), key=_prediction_metric_sort_value)
        if category_examples:
            selected_example = category_examples[0]
            selected.append(selected_example)
            used_manifest_indices.add(int(selected_example["row"].get("manifest_index", -1)))

    if len(selected) < len(FRONT_FACING_PANEL_CATEGORIES):
        for example in sorted(examples, key=_prediction_metric_sort_value):
            manifest_index = int(example["row"].get("manifest_index", -1))
            if manifest_index in used_manifest_indices:
                continue
            selected.append(example)
            used_manifest_indices.add(manifest_index)
            if len(selected) == len(FRONT_FACING_PANEL_CATEGORIES):
                break
    return selected


def _draw_panel_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: Sequence[str], font: ImageFont.ImageFont) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=(20, 20, 20), font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + 4


def _save_front_facing_validation_panel(
    *,
    asset_path: Path,
    family: str,
    model_path: str | None,
    examples: Sequence[Dict[str, Any]],
    expected_size: int,
    prediction_threshold: float,
) -> List[Dict[str, Any]]:
    selected = _select_front_facing_examples(examples)
    if not selected:
        return []

    tile = FRONT_FACING_PANEL_TILE_SIZE
    gap = 22
    top = 118
    label_height = 82
    columns = ("Raw source", f"Input ({expected_size}px)", "Ground truth", "Prediction", "Overlay")
    width = (tile * len(columns)) + (gap * (len(columns) + 1))
    height = top + len(selected) * (tile + label_height + gap) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _front_facing_panel_font(22)
    label_font = _front_facing_panel_font(14)
    small_font = _front_facing_panel_font(12)
    draw.text(
        (gap, 18),
        f"Glomeruli Validation Examples - {family} candidate",
        fill=(20, 20, 20),
        font=title_font,
    )
    draw.text(
        (gap, 52),
        f"Source: deterministic candidate-comparison manifest | threshold={prediction_threshold:.2f} | model_input={expected_size}px",
        fill=(45, 45, 45),
        font=label_font,
    )
    if model_path:
        draw.text((gap, 78), f"Model: {Path(model_path).name}", fill=(80, 80, 80), font=small_font)

    for column_index, column in enumerate(columns):
        x = gap + column_index * (tile + gap)
        draw.text((x, top - 28), column, fill=(20, 20, 20), font=label_font)

    for row_index, example in enumerate(selected):
        row = example["row"]
        source_image = example["source_image"]
        image_crop = example["image_crop"]
        truth_crop = example["truth_crop"]
        pred_crop = example["pred_crop"]
        crop_box_values = json.loads(str(row.get("crop_box", "[]")))
        y = top + row_index * (tile + label_height + gap)
        truth_overlay = _overlay_mask(image_crop, truth_crop, (0, 255, 0))
        pred_overlay = _overlay_mask(image_crop, pred_crop, (255, 0, 0))
        error_overlay = _error_overlay(image_crop, truth_crop, pred_crop)
        tiles = (
            _raw_context_panel(source_image, crop_box_values, tile),
            _resize_panel_rgb(_model_input_panel(image_crop, expected_size), tile),
            _resize_panel_rgb(truth_overlay, tile),
            _resize_panel_rgb(pred_overlay, tile),
            _resize_panel_rgb(error_overlay, tile),
        )
        for column_index, tile_image in enumerate(tiles):
            x = gap + column_index * (tile + gap)
            canvas.paste(tile_image, (x, y))

        category = str(row.get("category", ""))
        image_name = str(row.get("image_name", ""))
        crop_box = str(row.get("crop_box", ""))
        metrics_line = (
            f"{category} | Dice {float(row.get('dice') or 0.0):.3f} | "
            f"Jaccard {float(row.get('jaccard') or 0.0):.3f} | "
            f"Precision {float(row.get('precision') or 0.0):.3f} | Recall {float(row.get('recall') or 0.0):.3f}"
        )
        foreground_line = (
            f"truth_fg {float(row.get('truth_foreground_fraction') or 0.0):.3f} | "
            f"pred_fg {float(row.get('prediction_foreground_fraction') or 0.0):.3f} | "
            f"crop {crop_box} | input_resize={expected_size}px"
        )
        label_y = y + tile + 10
        _draw_panel_label(
            draw,
            (gap, label_y),
            (
                f"{row_index + 1}. {image_name}",
                metrics_line,
                foreground_line,
            ),
            small_font,
        )

    legend_y = height - 26
    draw.text(
        (gap, legend_y),
        "Raw source shows the selected crop box. Overlay colors: green=TP, red=FP, blue=FN.",
        fill=(45, 45, 45),
        font=small_font,
    )
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(asset_path)
    return [dict(example["row"]) for example in selected]


def _safe_slug(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in value]
    slug = "".join(chars).strip("_")
    return slug or "item"


def _aggregate_metrics(truth_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray]) -> Dict[str, float]:
    truth = np.concatenate([mask.astype(bool).reshape(-1) for mask in truth_masks])
    pred = np.concatenate([mask.astype(bool).reshape(-1) for mask in pred_masks])
    metrics = binary_dice_jaccard(truth, pred)
    metrics.update(binary_precision_recall(truth, pred))
    metrics["pixel_accuracy"] = pixel_accuracy(pred.astype(np.uint8), truth.astype(np.uint8))
    return metrics


def _evaluate_runtime(
    runtime: CandidateRuntime,
    manifest: Sequence[Dict[str, Any]],
    asset_dir: Path,
    expected_size: int,
    prediction_threshold: float = COMPARE_PREDICTION_THRESHOLD,
) -> Dict[str, Any]:
    provenance = _merged_provenance(runtime)
    summary: Dict[str, Any] = {
        "family": runtime.family,
        "comparison_role": runtime.role,
        "available": runtime.status == "available" and runtime.model_path is not None,
        "status": runtime.status,
        "artifact_path": str(runtime.model_path) if runtime.model_path else None,
        "seed": provenance.get("seed", runtime.seed),
        "provenance": provenance,
        "command": runtime.command,
        "error": runtime.error,
    }
    if not summary["available"]:
        summary["gate"] = {"blocked": True, "reasons": ["candidate_family_unavailable"]}
        summary["metrics"] = {}
        summary["prediction_rows"] = []
        return summary

    prediction_rows: List[Dict[str, Any]] = []
    if not manifest:
        summary["gate"] = {
            "blocked": True,
            "reasons": ["insufficient_heldout_category_support"],
        }
        summary["metrics"] = {}
        summary["baselines"] = {}
        summary["prediction_rows"] = prediction_rows
        summary["runtime_use_status"] = RUNTIME_USE_AVAILABLE
        summary["promotion_evidence_status"] = PROMOTION_INSUFFICIENT
        return summary

    learn = load_model_safely(str(runtime.model_path), model_type="glomeruli")
    learn.model.eval()
    truth_masks: List[np.ndarray] = []
    pred_masks: List[np.ndarray] = []
    front_facing_examples: List[Dict[str, Any]] = []
    for index, item in enumerate(manifest):
        source_image, image_crop, truth_crop = _load_crop_bundle(item)
        pred_crop, predict_audit = _predict_crop_with_audit(
            learn,
            image_crop,
            truth_crop.shape,
            expected_size=expected_size,
            threshold=prediction_threshold,
        )
        truth_masks.append(truth_crop)
        pred_masks.append(pred_crop)
        metrics = binary_dice_jaccard(truth_crop, pred_crop)
        metrics.update(binary_precision_recall(truth_crop, pred_crop))
        metrics["pixel_accuracy"] = pixel_accuracy(
            np.asarray(pred_crop).astype(np.uint8),
            np.asarray(truth_crop).astype(np.uint8),
        )
        image_name = str(item.get("image_name") or Path(str(item.get("image_path", ""))).name)
        subject_id = str(item.get("subject_id") or Path(str(item.get("image_path", ""))).parent.name)
        category = str(item.get("category"))
        asset_path = asset_dir / (
            f"{runtime.family}_{index:02d}_{_safe_slug(category)}_{_safe_slug(subject_id)}_{_safe_slug(Path(image_name).stem)}.png"
        )
        _save_review_panel(
            asset_path=asset_path,
            image_crop=image_crop,
            truth_crop=truth_crop,
            pred_crop=pred_crop,
        )
        prediction_row = {
            "family": runtime.family,
            "comparison_role": runtime.role,
            "seed": provenance.get("seed", runtime.seed),
            "image_path": item.get("image_path"),
            "image_name": image_name,
            "subject_id": subject_id,
            "mask_path": item.get("mask_path"),
            "category": category,
            "cohort_id": item.get("cohort_id"),
            "lane_assignment": item.get("lane_assignment"),
            "manifest_index": index,
            "crop_box": json.dumps(item.get("crop_box")),
            "truth_foreground_fraction": float(truth_crop.mean()),
            "prediction_foreground_fraction": float(pred_crop.mean()),
            "prediction_tensor_shape": predict_audit["raw_output_shape"],
            "prediction_probability_shape": predict_audit["prediction_probability_shape"],
            "prediction_resized_shape": predict_audit["prediction_resized_shape"],
            "threshold": predict_audit["threshold"],
            "threshold_resize_order": predict_audit["threshold_resize_order"],
            "dice": metrics["dice"],
            "jaccard": metrics["jaccard"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "review_panel_path": str(asset_path),
        }
        prediction_rows.append(prediction_row)
        front_facing_examples.append(
            {
                "row": prediction_row,
                "source_image": source_image,
                "image_crop": image_crop,
                "truth_crop": truth_crop,
                "pred_crop": pred_crop,
            }
        )

    gate = evaluate_glomeruli_promotion_candidate(
        truth_masks,
        pred_masks,
        manifest,
        provenance,
    )
    baselines = trivial_baseline_metrics(truth_masks)
    summary["gate"] = gate
    summary["metrics"] = _aggregate_metrics(truth_masks, pred_masks)
    summary["baselines"] = baselines
    summary["prediction_rows"] = prediction_rows
    front_facing_panel_path = asset_dir / f"{runtime.family}_validation_predictions.png"
    front_facing_panel_rows = _save_front_facing_validation_panel(
        asset_path=front_facing_panel_path,
        family=runtime.family,
        model_path=summary.get("artifact_path"),
        examples=front_facing_examples,
        expected_size=expected_size,
        prediction_threshold=prediction_threshold,
    )
    summary["front_facing_validation_panel_path"] = str(front_facing_panel_path)
    summary["front_facing_validation_panel_selection_rule"] = (
        "one example per category; background=min_prediction_foreground_fraction; "
        "boundary_positive=max_dice_then_jaccard_then_foreground_fraction_match"
    )
    summary["front_facing_validation_panel_manifest_indices"] = [
        row.get("manifest_index") for row in front_facing_panel_rows
    ]
    return summary


def _within_tie_margin(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any], *, margin: float = TIE_MARGIN) -> bool:
    return (
        abs(float(metrics_a.get("dice", 0.0)) - float(metrics_b.get("dice", 0.0))) <= margin
        and abs(float(metrics_a.get("jaccard", 0.0)) - float(metrics_b.get("jaccard", 0.0))) <= margin
    )


def _strictly_better_on_both(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any], *, margin: float = TIE_MARGIN) -> bool:
    return (
        float(metrics_a.get("dice", -1.0)) > float(metrics_b.get("dice", -1.0)) + margin
        and float(metrics_a.get("jaccard", -1.0)) > float(metrics_b.get("jaccard", -1.0)) + margin
    )


def determine_promotion_decision(
    candidate_summaries: Sequence[Dict[str, Any]],
    compatibility_summary: Dict[str, Any] | None = None,
    *,
    tie_margin: float = TIE_MARGIN,
) -> Dict[str, Any]:
    transfer = next((row for row in candidate_summaries if row["family"] == "transfer"), None)
    scratch = next((row for row in candidate_summaries if row["family"] == "scratch"), None)
    unavailable = [
        row["family"]
        for row in candidate_summaries
        if row["family"] in {"transfer", "scratch"} and not row["available"]
    ]
    promotable = [
        row
        for row in candidate_summaries
        if row["family"] in {"transfer", "scratch"}
        and row["available"]
        and not row["gate"]["blocked"]
        and row.get("promotion_evidence_status") in {None, PROMOTION_ELIGIBLE}
    ]

    result: Dict[str, Any] = {
        "decision_state": None,
        "decision_reason": None,
        "promoted_family": None,
        "research_use_candidates": [],
        "tie_margin": {"dice": tie_margin, "jaccard": tie_margin},
    }

    if unavailable:
        result["decision_state"] = "insufficient_evidence"
        result["decision_reason"] = f"candidate_family_unavailable:{','.join(sorted(unavailable))}"
        result["research_use_candidates"] = [
            {"family": row["family"], "artifact_path": row["artifact_path"]}
            for row in promotable
        ]
        return result

    if not promotable:
        if any(
            row.get("promotion_evidence_status") == PROMOTION_INSUFFICIENT
            or "insufficient_heldout_category_support" in row.get("gate", {}).get("reasons", [])
            for row in candidate_summaries
        ):
            result["decision_state"] = "insufficient_evidence"
            result["decision_reason"] = "insufficient_heldout_category_support"
        else:
            result["decision_state"] = "not_promotion_eligible"
            result["decision_reason"] = "no_candidate_cleared_promotion_gates"
        result["research_use_candidates"] = [
            {"family": row["family"], "artifact_path": row["artifact_path"]}
            for row in candidate_summaries
            if row["family"] in {"transfer", "scratch"} and row.get("runtime_use_status") == RUNTIME_USE_AVAILABLE
        ]
        return result

    if compatibility_summary and compatibility_summary.get("available") and compatibility_summary.get("metrics"):
        filtered: List[Dict[str, Any]] = []
        for row in promotable:
            if _within_tie_margin(row["metrics"], compatibility_summary["metrics"], margin=tie_margin):
                row.setdefault("decision_notes", []).append(
                    "candidate_does_not_materially_exceed_compatibility_reference"
                )
            else:
                filtered.append(row)
        if not filtered:
            result["decision_state"] = "not_promotion_eligible"
            result["decision_reason"] = "no_candidate_materially_exceeded_compatibility_reference"
            return result
        promotable = filtered

    if len(promotable) == 1:
        winner = promotable[0]
        result["decision_state"] = "promoted"
        result["decision_reason"] = "exactly_one_candidate_cleared_promotion_gates"
        result["promoted_family"] = winner["family"]
        return result

    first, second = promotable[0], promotable[1]
    if _within_tie_margin(first["metrics"], second["metrics"], margin=tie_margin):
        result["decision_state"] = "insufficient_evidence"
        result["decision_reason"] = "transfer_and_scratch_are_within_practical_tie_margin"
        result["research_use_candidates"] = [
            {"family": row["family"], "artifact_path": row["artifact_path"]}
            for row in promotable
        ]
        return result

    if _strictly_better_on_both(first["metrics"], second["metrics"], margin=tie_margin):
        winner = first
    elif _strictly_better_on_both(second["metrics"], first["metrics"], margin=tie_margin):
        winner = second
    else:
        result["decision_state"] = "insufficient_evidence"
        result["decision_reason"] = "mixed_metric_signal_between_transfer_and_scratch"
        result["research_use_candidates"] = [
            {"family": row["family"], "artifact_path": row["artifact_path"]}
            for row in promotable
        ]
        return result

    result["decision_state"] = "promoted"
    result["decision_reason"] = "one_candidate_outperformed_the_other_on_shared_manifest_metrics"
    result["promoted_family"] = winner["family"]
    return result


def _decision_promotion_status(decision: Dict[str, Any], candidate_summaries: Sequence[Dict[str, Any]]) -> str:
    statuses = [row.get("promotion_evidence_status") for row in candidate_summaries]
    if PROMOTION_AUDIT_MISSING in statuses:
        return PROMOTION_AUDIT_MISSING
    if decision.get("decision_state") == "promoted":
        return PROMOTION_ELIGIBLE
    if decision.get("decision_state") == "insufficient_evidence":
        return PROMOTION_INSUFFICIENT
    return PROMOTION_NOT_ELIGIBLE


def _write_manifest(manifest: Sequence[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w") as handle:
        json.dump(list(manifest), handle, indent=2, sort_keys=True)


def _write_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_report(
    *,
    output_path: Path,
    decision: Dict[str, Any],
    manifest_audit: Dict[str, Any],
    candidate_summaries: Sequence[Dict[str, Any]],
    compatibility_summary: Dict[str, Any] | None,
) -> None:
    manifest_categories = manifest_audit.get("categories", {})
    lines = [
        "# Glomeruli Candidate Comparison Report",
        "",
        f"- Decision: `{decision['decision_state']}`",
        f"- Reason: `{decision['decision_reason']}`",
        f"- Promoted family: `{decision.get('promoted_family')}`",
        "",
        "## Manifest",
        "",
        f"- Total crops: `{manifest_audit.get('count')}`",
        f"- Unique images: `{manifest_audit.get('unique_images')}`",
        f"- Unique subjects: `{manifest_audit.get('unique_subjects')}`",
        (
            f"- Category counts: `background={manifest_categories.get('background', 0)}` "
            f"`boundary={manifest_categories.get('boundary', 0)}` "
            f"`positive={manifest_categories.get('positive', 0)}`"
        ),
        "- Review panels show: `raw crop | truth overlay (green) | prediction overlay (red)`",
        "",
        "## Candidates",
        "",
    ]
    for summary in candidate_summaries:
        metrics = summary.get("metrics", {})
        split_integrity = summary.get("split_integrity", {})
        root_cause = summary.get("root_cause", {})
        resize_sensitivity = summary.get("resize_sensitivity", {})
        provenance = summary.get("provenance", {})
        transfer_base_report = summary.get("transfer_base_report", {})
        threshold_policy = summary.get("threshold_policy", {})
        category_gate_status = summary.get("category_gate_status", {})
        validation_adjudication = summary.get("validation_adjudication", {})
        lines.extend(
            [
                f"### {summary['family']}",
                f"- Available: `{summary['available']}`",
                f"- Runtime use status: `{summary.get('runtime_use_status')}`",
                f"- Promotion evidence status: `{summary.get('promotion_evidence_status')}`",
                f"- Role: `{summary['comparison_role']}`",
                f"- Artifact: `{summary['artifact_path']}`",
                f"- Front-facing validation panel: `{summary.get('front_facing_validation_panel_path')}`",
                f"- Front-facing validation panel selection: `{summary.get('front_facing_validation_panel_selection_rule')}` "
                f"`manifest_indices={summary.get('front_facing_validation_panel_manifest_indices')}`",
                f"- Seed: `{summary.get('seed')}`",
                f"- Gate blocked: `{summary['gate']['blocked']}`",
                f"- Gate reasons: `{', '.join(summary['gate'].get('reasons', []))}`",
                f"- Split integrity: `{split_integrity.get('reason')}` "
                f"`train_image_overlap={split_integrity.get('train_image_overlap_count')}` "
                f"`subject_overlap={split_integrity.get('subject_overlap_count')}`",
                f"- Transfer base artifact: `{transfer_base_report.get('artifact_path') or provenance.get('transfer_base_artifact_path') or provenance.get('base_model_path')}`",
                f"- Transfer base mitochondria scope: `{transfer_base_report.get('mitochondria_training_scope')}`",
                f"- Transfer base inference claim status: `{transfer_base_report.get('mitochondria_inference_claim_status')}`",
                f"- Transfer base physical counts: `training={transfer_base_report.get('physical_training_image_count')}` `testing={transfer_base_report.get('physical_testing_image_count')}`",
                f"- Transfer base fitted counts: `images={transfer_base_report.get('actual_fitted_image_count')}` `masks={transfer_base_report.get('actual_fitted_mask_count')}`",
                f"- Transfer base resize policy: `{json.dumps(transfer_base_report.get('resize_policy'), sort_keys=True) if transfer_base_report.get('resize_policy') is not None else None}`",
                f"- Resize sensitivity status: `{resize_sensitivity.get('status')}`",
                f"- Resize screening summary: `{resize_sensitivity.get('resize_screening_summary_path')}`",
                f"- Resize decision status: `{resize_sensitivity.get('resize_decision_status')}`",
                f"- Resize decision reason: `{resize_sensitivity.get('resize_decision_reason')}`",
                f"- Selected resize run: `{resize_sensitivity.get('selected_run_id')}`",
                f"- Selected resize policy: `crop_size={resize_sensitivity.get('selected_crop_size')}` "
                f"`output_size={resize_sensitivity.get('selected_output_size')}`",
                f"- Negative crop supervision status: `{provenance.get('negative_crop_supervision_status', 'absent')}`",
                f"- Negative crop manifest: `{provenance.get('negative_crop_manifest_path')}`",
                f"- Negative crop counts: `total={provenance.get('negative_crop_count', 0)}` "
                f"`mask_derived_background={provenance.get('mask_derived_background_crop_count', 0)}` "
                f"`curated={provenance.get('curated_negative_crop_count', 0)}`",
                f"- Negative crop sampler weight: `{provenance.get('negative_crop_sampler_weight', 0.0)}`",
                f"- Augmentation policy: `{json.dumps(provenance.get('augmentation_policy', {}), sort_keys=True)}`",
                f"- Threshold policy status: `{threshold_policy.get('threshold_policy_status')}`",
                f"- Threshold: `{threshold_policy.get('threshold')}`",
                f"- Threshold rationale: `{threshold_policy.get('threshold_rationale')}`",
                f"- Threshold selection rule: `{threshold_policy.get('threshold_selection_rule')}`",
                f"- Threshold promotion-ready: `{threshold_policy.get('threshold_is_promotion_ready')}`",
                f"- Threshold grid: `{threshold_policy.get('threshold_grid')}`",
                f"- Overcoverage audit: `{threshold_policy.get('overcoverage_audit_summary_path')}`",
                f"- Background false-positive foreground fraction at threshold: `{threshold_policy.get('background_false_positive_foreground_fraction')}`",
                f"- Positive/boundary recall at threshold: `{threshold_policy.get('positive_boundary_recall')}`",
                f"- Category gate status: `{category_gate_status.get('promotion_evidence_status')}`",
                f"- Category gate reasons: `{', '.join(category_gate_status.get('reasons', []))}`",
                f"- Category gates failed: `{category_gate_status.get('failed_gate_count')}` of `{category_gate_status.get('gate_count')}`",
                f"- Validation adjudication status: `{validation_adjudication.get('status')}`",
                f"- Validation adjudication path: `{validation_adjudication.get('path')}`",
                f"- Validation adjudication applied gates: `{validation_adjudication.get('applied_count')}` "
                f"`nonblocking={validation_adjudication.get('nonblocking_count')}` "
                f"`blocking={validation_adjudication.get('blocking_count')}`",
                f"- Root causes: `{', '.join(root_cause.get('root_causes', []))}`",
                f"- Remediation path: `{root_cause.get('remediation_path')}`",
                (
                    f"- Metrics: `dice={metrics.get('dice')}` `jaccard={metrics.get('jaccard')}` "
                    f"`precision={metrics.get('precision')}` `recall={metrics.get('recall')}`"
                ),
                "",
            ]
        )
    if compatibility_summary:
        lines.extend(
            [
                "## Compatibility Reference",
                "",
                f"- Available: `{compatibility_summary['available']}`",
                f"- Artifact: `{compatibility_summary['artifact_path']}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_html_report(
    *,
    output_path: Path,
    decision: Dict[str, Any],
    manifest_audit: Dict[str, Any],
    candidate_summaries: Sequence[Dict[str, Any]],
) -> None:
    manifest_categories = manifest_audit.get("categories", {})
    candidate_sections: List[str] = []
    for summary in candidate_summaries:
        metrics = summary.get("metrics", {})
        prediction_rows = summary.get("prediction_rows", [])
        threshold_policy = summary.get("threshold_policy", {})
        category_gate_status = summary.get("category_gate_status", {})
        validation_adjudication = summary.get("validation_adjudication", {})
        front_panel_path = summary.get("front_facing_validation_panel_path")
        front_panel_html = ""
        if front_panel_path:
            front_panel_rule = html.escape(str(summary.get("front_facing_validation_panel_selection_rule")))
            front_panel_indices = html.escape(str(summary.get("front_facing_validation_panel_manifest_indices")))
            front_panel_html = (
                "<figure class='front-panel'>"
                f"<img src=\"{html.escape(str(Path('review_assets') / Path(str(front_panel_path)).name))}\" "
                f"alt=\"{html.escape(summary['family'])} validation examples\" />"
                "<figcaption>Front-facing validation examples selected from the deterministic comparison manifest. "
                "The rows come from the same scored prediction table used by the promotion gates. "
                f"Selection rule: {front_panel_rule}. Manifest indices: {front_panel_indices}.</figcaption>"
                "</figure>"
            )
        gallery = []
        for row in prediction_rows:
            image_label = html.escape(str(row.get("image_name") or Path(str(row.get("image_path", ""))).name))
            subject_label = html.escape(str(row.get("subject_id", "")))
            category_label = html.escape(str(row.get("category", "")))
            crop_box = html.escape(str(row.get("crop_box", "")))
            manifest_index = int(row.get("manifest_index", 0))
            gallery.append(
                "<figure class='panel-card'>"
                f"<img src=\"{html.escape(str(Path('review_assets') / Path(row['review_panel_path']).name))}\" alt=\"{html.escape(summary['family'])} panel {manifest_index}\" />"
                "<figcaption>"
                f"<strong>Panel {manifest_index:02d}</strong> | {category_label} | {subject_label} | {image_label}<br>"
                f"crop={crop_box}<br>"
                f"dice={float(row['dice']):.3f} | jaccard={float(row['jaccard']):.3f} | "
                f"precision={float(row['precision']):.3f} | recall={float(row['recall']):.3f}<br>"
                f"truth_fg={float(row['truth_foreground_fraction']):.3f} | pred_fg={float(row['prediction_foreground_fraction']):.3f}<br>"
                f"threshold={float(row.get('threshold', threshold_policy.get('threshold') or COMPARE_PREDICTION_THRESHOLD)):.3f} | "
                f"threshold policy={html.escape(str(threshold_policy.get('threshold_policy_status')))}<br>"
                "panel order: raw | truth overlay | prediction overlay"
                "</figcaption>"
                "</figure>"
            )
        gate_reasons = summary["gate"].get("reasons", [])
        split_integrity = summary.get("split_integrity", {})
        root_cause = summary.get("root_cause", {})
        resize_sensitivity = summary.get("resize_sensitivity", {})
        provenance = summary.get("provenance", {})
        transfer_base_report = summary.get("transfer_base_report", {})
        reason_items = (
            "".join(f"<li>{html.escape(reason)}</li>" for reason in gate_reasons)
            if gate_reasons
            else "<li>none</li>"
        )
        candidate_sections.append(
            "<section class='candidate-section'>"
            f"<h2>{html.escape(summary['family'])}</h2>"
            "<div class='summary-grid'>"
            f"<div><strong>Available</strong><br>{html.escape(str(summary['available']))}</div>"
            f"<div><strong>Runtime use</strong><br>{html.escape(str(summary.get('runtime_use_status')))}</div>"
            f"<div><strong>Promotion evidence</strong><br>{html.escape(str(summary.get('promotion_evidence_status')))}</div>"
            f"<div><strong>Seed</strong><br>{html.escape(str(summary.get('seed')))}</div>"
            f"<div><strong>Blocked</strong><br>{html.escape(str(summary['gate']['blocked']))}</div>"
            f"<div><strong>Role</strong><br>{html.escape(str(summary['comparison_role']))}</div>"
            f"<div><strong>Dice</strong><br>{float(metrics.get('dice', 0.0)):.3f}</div>"
            f"<div><strong>Jaccard</strong><br>{float(metrics.get('jaccard', 0.0)):.3f}</div>"
            f"<div><strong>Precision</strong><br>{float(metrics.get('precision', 0.0)):.3f}</div>"
            f"<div><strong>Recall</strong><br>{float(metrics.get('recall', 0.0)):.3f}</div>"
            "</div>"
            f"<p class='artifact'><strong>Artifact:</strong> {html.escape(str(summary.get('artifact_path')))}</p>"
            f"<p class='artifact'><strong>Split integrity:</strong> {html.escape(str(split_integrity.get('reason')))} "
            f"train image overlap={html.escape(str(split_integrity.get('train_image_overlap_count')))} "
            f"subject overlap={html.escape(str(split_integrity.get('subject_overlap_count')))}</p>"
            f"<p class='artifact'><strong>Transfer base:</strong> {html.escape(str(transfer_base_report.get('artifact_path') or provenance.get('transfer_base_artifact_path') or provenance.get('base_model_path')))} "
            f"scope={html.escape(str(transfer_base_report.get('mitochondria_training_scope')))} "
            f"inference_claim={html.escape(str(transfer_base_report.get('mitochondria_inference_claim_status')))} "
            f"physical_train={html.escape(str(transfer_base_report.get('physical_training_image_count')))} "
            f"physical_test={html.escape(str(transfer_base_report.get('physical_testing_image_count')))} "
            f"fitted_images={html.escape(str(transfer_base_report.get('actual_fitted_image_count')))} "
            f"fitted_masks={html.escape(str(transfer_base_report.get('actual_fitted_mask_count')))} "
            f"resize={html.escape(str(transfer_base_report.get('resize_policy')))}</p>"
            f"<p class='artifact'><strong>Resize sensitivity:</strong> {html.escape(str(resize_sensitivity.get('status')))}</p>"
            f"<p class='artifact'><strong>Resize decision:</strong> {html.escape(str(resize_sensitivity.get('resize_decision_status')))} "
            f"reason={html.escape(str(resize_sensitivity.get('resize_decision_reason')))} "
            f"selected_run={html.escape(str(resize_sensitivity.get('selected_run_id')))} "
            f"selected_crop={html.escape(str(resize_sensitivity.get('selected_crop_size')))} "
            f"selected_output={html.escape(str(resize_sensitivity.get('selected_output_size')))} "
            f"summary={html.escape(str(resize_sensitivity.get('resize_screening_summary_path')))}</p>"
            f"<p class='artifact'><strong>Root causes:</strong> {html.escape(', '.join(root_cause.get('root_causes', [])))} "
            f"<strong>Remediation:</strong> {html.escape(str(root_cause.get('remediation_path')))}</p>"
            f"<p class='artifact'><strong>Threshold policy:</strong> {html.escape(str(threshold_policy.get('threshold_policy_status')))} "
            f"threshold={html.escape(str(threshold_policy.get('threshold')))} "
            f"promotion_ready={html.escape(str(threshold_policy.get('threshold_is_promotion_ready')))} "
            f"rationale={html.escape(str(threshold_policy.get('threshold_rationale')))} "
            f"rule={html.escape(str(threshold_policy.get('threshold_selection_rule')))} "
            f"audit={html.escape(str(threshold_policy.get('overcoverage_audit_summary_path')))} "
            f"background_fp={html.escape(str(threshold_policy.get('background_false_positive_foreground_fraction')))} "
            f"positive_boundary_recall={html.escape(str(threshold_policy.get('positive_boundary_recall')))}</p>"
            f"<p class='artifact'><strong>Category gates:</strong> {html.escape(str(category_gate_status.get('promotion_evidence_status')))} "
            f"failed={html.escape(str(category_gate_status.get('failed_gate_count')))} "
            f"of={html.escape(str(category_gate_status.get('gate_count')))} "
            f"reasons={html.escape(', '.join(category_gate_status.get('reasons', [])))}</p>"
            f"<p class='artifact'><strong>Validation adjudication:</strong> {html.escape(str(validation_adjudication.get('status')))} "
            f"path={html.escape(str(validation_adjudication.get('path')))} "
            f"applied={html.escape(str(validation_adjudication.get('applied_count')))} "
            f"nonblocking={html.escape(str(validation_adjudication.get('nonblocking_count')))} "
            f"blocking={html.escape(str(validation_adjudication.get('blocking_count')))}</p>"
            "<div class='gate-box'>"
            "<strong>Gate reasons</strong>"
            f"<ul>{reason_items}</ul>"
            "</div>"
            f"{front_panel_html}"
            "<div class='gallery'>"
            + "".join(gallery)
            + "</div>"
            "</section>"
        )
    html_text = (
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;line-height:1.45;color:#1f2937;background:#f8fafc;}"
        "h1,h2,h3{margin:0 0 12px 0;color:#0f172a;}"
        ".hero,.candidate-section,.manifest-box{background:#fff;border:1px solid #dbe3ec;border-radius:12px;padding:18px;margin-bottom:18px;box-shadow:0 1px 2px rgba(15,23,42,.04);}"
        ".summary-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0 14px 0;}"
        ".summary-grid div{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px;}"
        ".gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(640px,1fr));gap:16px;}"
        ".front-panel{margin:0 0 16px 0;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px;}"
        ".front-panel img{width:100%;height:auto;border-radius:8px;border:1px solid #e2e8f0;background:#fff;}"
        ".front-panel figcaption{font-size:13px;margin-top:10px;color:#334155;}"
        ".panel-card{margin:0;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px;}"
        ".panel-card img{width:100%;height:auto;border-radius:8px;border:1px solid #e2e8f0;background:#fff;}"
        ".panel-card figcaption{font-size:13px;margin-top:10px;color:#334155;}"
        ".artifact{font-size:13px;word-break:break-word;}"
        ".gate-box{background:#fff7ed;border:1px solid #fdba74;border-radius:10px;padding:12px;margin:12px 0 16px 0;}"
        ".manifest-meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:12px;}"
        ".manifest-meta div{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px;}"
        ".legend{color:#475569;font-size:14px;margin-top:8px;}"
        "@media (max-width: 1100px){.gallery{grid-template-columns:1fr;}.summary-grid,.manifest-meta{grid-template-columns:repeat(2,minmax(0,1fr));}}"
        "</style></head><body>"
        "<section class='hero'>"
        "<h1>Glomeruli Candidate Comparison Report</h1>"
        f"<p><strong>Decision:</strong> {html.escape(str(decision['decision_state']))} "
        f"<strong>Reason:</strong> {html.escape(str(decision['decision_reason']))} "
        f"<strong>Promoted family:</strong> {html.escape(str(decision.get('promoted_family')))}</p>"
        "</section>"
        "<section class='manifest-box'>"
        "<h2>Manifest Coverage</h2>"
        "<p class='legend'>This report is scored on a deterministic crop manifest. Each review panel shows <strong>raw crop | truth overlay | prediction overlay</strong>.</p>"
        "<div class='manifest-meta'>"
        f"<div><strong>Total crops</strong><br>{int(manifest_audit.get('count', 0))}</div>"
        f"<div><strong>Unique images</strong><br>{int(manifest_audit.get('unique_images', 0))}</div>"
        f"<div><strong>Unique subjects</strong><br>{int(manifest_audit.get('unique_subjects', 0))}</div>"
        f"<div><strong>Category counts</strong><br>background={int(manifest_categories.get('background', 0))} | boundary={int(manifest_categories.get('boundary', 0))} | positive={int(manifest_categories.get('positive', 0))}</div>"
        "</div>"
        "</section>"
        + "".join(candidate_sections)
        + "</body></html>"
    )
    output_path.write_text(html_text, encoding="utf-8")


def _validation_mask_paths(data_root: Path) -> List[Path]:
    image_paths = get_items_full_images(data_root)
    mask_paths: List[Path] = []
    for item in image_paths:
        mask_paths.append(training_item_mask_path(item))
    if not mask_paths:
        raise ValueError(
            f"No validation masks found for candidate comparison under {data_root}. "
            "Use a supported paired full-image root or the manifest-backed "
            "raw_data/cohorts registry root with admitted masked rows."
        )
    return mask_paths


def compare_glomeruli_candidates(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = validate_supported_segmentation_training_root(args.data_dir, stage="glomeruli")
    run_id = str(args.run_id or _generated_run_id(args.seed))
    output_root = Path(args.output_dir).expanduser() / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    overcoverage_audit = _read_overcoverage_audit(getattr(args, "overcoverage_audit_dir", None))
    resize_screening_summary_path = getattr(args, "resize_screening_summary", None)
    resize_screening_rows = _read_resize_screening_summary(resize_screening_summary_path)
    validation_adjudication = _read_validation_adjudications(getattr(args, "validation_adjudication", None))
    explicit_threshold = getattr(args, "prediction_threshold", None) is not None
    requested_threshold = float(args.prediction_threshold) if explicit_threshold else None
    threshold_policy = _threshold_policy_from_audit(
        overcoverage_audit,
        selected_threshold=requested_threshold,
        explicit_threshold=explicit_threshold,
    )
    prediction_threshold = float(threshold_policy["threshold"])
    model_root = Path(args.model_dir).expanduser()
    model_root.mkdir(parents=True, exist_ok=True)
    asset_dir = output_root / "review_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    shared_split_path = None
    if not (args.transfer_model_path and args.scratch_model_path):
        shared_split_path = _write_shared_training_split(data_root, output_root, seed=args.seed)

    if args.transfer_model_path:
        transfer_runtime = CandidateRuntime(
            family="transfer",
            role="candidate",
            model_path=Path(args.transfer_model_path).expanduser(),
            seed=None,
            command=None,
            status="available" if Path(args.transfer_model_path).expanduser().exists() else "unavailable",
            error=None if Path(args.transfer_model_path).expanduser().exists() else "transfer_model_path_not_found",
        )
    else:
        transfer_runtime = _run_training_command(
            _candidate_command(
                data_dir=data_root,
                model_dir=model_root,
                model_name=args.transfer_model_name,
                epochs=args.transfer_epochs,
                learning_rate=args.learning_rate,
                image_size=args.image_size,
                crop_size=args.crop_size,
                batch_size=args.batch_size,
                loss_name=args.loss,
                seed=args.seed,
                from_scratch=False,
                base_model=Path(args.transfer_base_model).expanduser() if args.transfer_base_model else None,
                split_manifest_path=shared_split_path,
                negative_crop_manifest_path=Path(args.negative_crop_manifest).expanduser() if args.negative_crop_manifest else None,
                negative_crop_sampler_weight=args.negative_crop_sampler_weight,
                augmentation_variant=args.augmentation_variant,
                device=args.device,
            ),
            family="transfer",
            role="candidate",
            model_root=model_root,
            seed=args.seed,
        )

    if args.scratch_model_path:
        scratch_runtime = CandidateRuntime(
            family="scratch",
            role="candidate",
            model_path=Path(args.scratch_model_path).expanduser(),
            seed=None,
            command=None,
            status="available" if Path(args.scratch_model_path).expanduser().exists() else "unavailable",
            error=None if Path(args.scratch_model_path).expanduser().exists() else "scratch_model_path_not_found",
        )
    else:
        scratch_runtime = _run_training_command(
            _candidate_command(
                data_dir=data_root,
                model_dir=model_root,
                model_name=args.scratch_model_name,
                epochs=args.scratch_epochs,
                learning_rate=args.learning_rate,
                image_size=args.image_size,
                crop_size=args.crop_size,
                batch_size=args.batch_size,
                loss_name=args.loss,
                seed=args.seed,
                from_scratch=True,
                split_manifest_path=shared_split_path,
                negative_crop_manifest_path=Path(args.negative_crop_manifest).expanduser() if args.negative_crop_manifest else None,
                negative_crop_sampler_weight=args.negative_crop_sampler_weight,
                augmentation_variant=args.augmentation_variant,
                device=args.device,
            ),
            family="scratch",
            role="candidate",
            model_root=model_root,
            seed=args.seed,
        )

    candidate_runtimes = [transfer_runtime, scratch_runtime]
    manifest, manifest_audit = _build_deterministic_manifest(
        data_root=data_root,
        runtimes=candidate_runtimes,
        crop_size=args.crop_size,
        examples_per_category=args.examples_per_category,
    )
    manifest = _annotate_manifest_with_context(manifest, data_root)
    manifest_path = output_root / "deterministic_validation_manifest.json"
    _write_manifest(manifest, manifest_path)

    candidate_summaries = [
        _evaluate_runtime(
            transfer_runtime,
            manifest,
            asset_dir,
            expected_size=args.image_size,
            prediction_threshold=prediction_threshold,
        ),
        _evaluate_runtime(
            scratch_runtime,
            manifest,
            asset_dir,
            expected_size=args.image_size,
            prediction_threshold=prediction_threshold,
        ),
    ]
    candidate_audits: Dict[str, Dict[str, Any]] = {}
    validation_adjudication_rows: list[dict[str, Any]] = []
    for summary in candidate_summaries:
        audit = _candidate_audit_rows(
            summary=summary,
            manifest=manifest,
            expected_size=args.image_size,
            prediction_threshold=prediction_threshold,
            threshold_policy=threshold_policy,
            resize_screening_rows=resize_screening_rows,
            resize_screening_summary_path=resize_screening_summary_path,
        )
        applied_adjudications = _apply_validation_adjudication_to_audit(audit, validation_adjudication)
        _refresh_root_cause_after_adjudication(audit, summary)
        validation_adjudication_rows.extend(applied_adjudications)
        candidate_audits[summary["family"]] = audit
        summary["runtime_use_status"] = audit["artifact_status"]["runtime_use_status"]
        family_adjudication_count = sum(1 for row in applied_adjudications if row.get("family") == summary["family"])
        family_nonblocking_count = sum(
            1
            for row in applied_adjudications
            if row.get("family") == summary["family"]
            and row.get("validation_adjudication_status") == VALIDATION_ADJUDICATION_NONBLOCKING
        )
        family_blocking_count = sum(
            1
            for row in applied_adjudications
            if row.get("family") == summary["family"]
            and row.get("validation_adjudication_status") == VALIDATION_ADJUDICATION_BLOCKING
        )
        summary["validation_adjudication"] = {
            "status": validation_adjudication.get("status"),
            "path": validation_adjudication.get("path"),
            "row_count": validation_adjudication.get("row_count"),
            "applied_count": family_adjudication_count,
            "nonblocking_count": family_nonblocking_count,
            "blocking_count": family_blocking_count,
        }
        audit_statuses = [
            audit["artifact_status"]["promotion_evidence_status"],
            audit["split_integrity"]["promotion_evidence_status"],
            audit["prediction_shape"]["family_status"].get(summary["family"], {}).get(
                "promotion_evidence_status",
                PROMOTION_ELIGIBLE,
            ),
            audit["preprocessing_parity"]["promotion_evidence_status"],
            audit["resize_sensitivity"]["promotion_evidence_status"],
            manifest_audit.get("promotion_evidence_status", PROMOTION_ELIGIBLE),
        ]
        if PROMOTION_AUDIT_MISSING in audit_statuses:
            summary["promotion_evidence_status"] = PROMOTION_AUDIT_MISSING
        elif PROMOTION_NOT_ELIGIBLE in audit_statuses or summary.get("gate", {}).get("blocked"):
            summary["promotion_evidence_status"] = PROMOTION_NOT_ELIGIBLE
        elif PROMOTION_INSUFFICIENT in audit_statuses:
            summary["promotion_evidence_status"] = PROMOTION_INSUFFICIENT
        else:
            summary["promotion_evidence_status"] = PROMOTION_ELIGIBLE
        summary["split_integrity"] = audit["split_integrity"]
        summary["prediction_shape_audit"] = audit["prediction_shape"]
        summary["category_gate_audit"] = audit["category_gate"]
        summary["category_gate_status"] = audit["category_gate"].get("family_status", {}).get(
            summary["family"],
            {"promotion_evidence_status": PROMOTION_ELIGIBLE, "reasons": [], "failed_gate_count": 0, "gate_count": 0},
        )
        summary["resize_policy_parity"] = audit["resize_policy_parity"]
        summary["resize_sensitivity"] = audit["resize_sensitivity"]
        summary["preprocessing_parity"] = audit["preprocessing_parity"]
        summary["transfer_base_report"] = audit["transfer_base_report"]
        summary["root_cause"] = audit["root_cause"]
        summary["threshold_policy"] = dict(threshold_policy)
        extra_gate_reasons = []
        if not threshold_policy.get("threshold_is_promotion_ready"):
            extra_gate_reasons.append(str(threshold_policy["threshold_policy_status"]))
        if audit["split_integrity"]["promotion_evidence_status"] != PROMOTION_ELIGIBLE:
            extra_gate_reasons.append(audit["split_integrity"]["reason"])
        for status in audit["prediction_shape"]["family_status"].values():
            extra_gate_reasons.extend(status.get("reasons", []))
        if audit["artifact_status"]["promotion_evidence_status"] == PROMOTION_AUDIT_MISSING:
            extra_gate_reasons.extend(audit["artifact_status"]["reasons"])
        if audit["resize_sensitivity"]["promotion_evidence_status"] != PROMOTION_ELIGIBLE:
            extra_gate_reasons.append(audit["resize_sensitivity"]["status"])
        if extra_gate_reasons:
            summary.setdefault("gate", {}).setdefault("reasons", [])
            summary["gate"]["reasons"] = sorted(set(summary["gate"]["reasons"] + extra_gate_reasons))
            summary["gate"]["blocked"] = True
            if not threshold_policy.get("threshold_is_promotion_ready"):
                summary["promotion_evidence_status"] = PROMOTION_INSUFFICIENT
    validation_adjudication["applied_count"] = len(validation_adjudication_rows)
    validation_adjudication["nonblocking_count"] = sum(
        1
        for row in validation_adjudication_rows
        if row.get("validation_adjudication_status") == VALIDATION_ADJUDICATION_NONBLOCKING
    )
    validation_adjudication["blocking_count"] = sum(
        1
        for row in validation_adjudication_rows
        if row.get("validation_adjudication_status") == VALIDATION_ADJUDICATION_BLOCKING
    )
    validation_adjudication["unused_count"] = max(
        int(validation_adjudication.get("row_count") or 0)
        - len({str(row.get("adjudication_id")) for row in validation_adjudication_rows}),
        0,
    )

    compatibility_summary = None
    if args.compat_model_path:
        compatibility_runtime = CandidateRuntime(
            family="compatibility_reference",
            role="compatibility_reference",
            model_path=Path(args.compat_model_path).expanduser(),
            seed=None,
            command=None,
            status="available" if Path(args.compat_model_path).expanduser().exists() else "unavailable",
            error=None if Path(args.compat_model_path).expanduser().exists() else "compat_model_path_not_found",
        )
        compatibility_summary = _evaluate_runtime(
            compatibility_runtime,
            manifest,
            asset_dir,
            expected_size=args.image_size,
            prediction_threshold=prediction_threshold,
        )

    decision = determine_promotion_decision(
        candidate_summaries,
        compatibility_summary=compatibility_summary,
        tie_margin=TIE_MARGIN,
    )
    decision["promotion_evidence_status"] = _decision_promotion_status(decision, candidate_summaries)

    candidate_rows = []
    for summary in candidate_summaries:
        candidate_rows.append(
            {
                "family": summary["family"],
                "comparison_role": summary["comparison_role"],
                "available": summary["available"],
                "artifact_path": summary["artifact_path"],
                "front_facing_validation_panel_path": summary.get("front_facing_validation_panel_path"),
                "front_facing_validation_panel_selection_rule": summary.get(
                    "front_facing_validation_panel_selection_rule"
                ),
                "front_facing_validation_panel_manifest_indices": "|".join(
                    str(index) for index in summary.get("front_facing_validation_panel_manifest_indices", [])
                ),
                "seed": summary.get("seed"),
                "runtime_use_status": summary.get("runtime_use_status"),
                "promotion_evidence_status": summary.get("promotion_evidence_status"),
                "blocked": summary["gate"]["blocked"],
                "reasons": "|".join(summary["gate"].get("reasons", [])),
                "root_causes": "|".join(summary.get("root_cause", {}).get("root_causes", [])),
                "remediation_path": summary.get("root_cause", {}).get("remediation_path"),
                "negative_crop_supervision_status": summary.get("provenance", {}).get("negative_crop_supervision_status"),
                "negative_crop_manifest_path": summary.get("provenance", {}).get("negative_crop_manifest_path"),
                "negative_crop_manifest_sha256": summary.get("provenance", {}).get("negative_crop_manifest_sha256"),
                "negative_crop_count": summary.get("provenance", {}).get("negative_crop_count"),
                "mask_derived_background_crop_count": summary.get("provenance", {}).get("mask_derived_background_crop_count"),
                "curated_negative_crop_count": summary.get("provenance", {}).get("curated_negative_crop_count"),
                "negative_crop_sampler_weight": summary.get("provenance", {}).get("negative_crop_sampler_weight"),
                "augmentation_policy": json.dumps(summary.get("provenance", {}).get("augmentation_policy", {}), sort_keys=True),
                "threshold_policy_status": summary.get("threshold_policy", {}).get("threshold_policy_status"),
                "threshold": summary.get("threshold_policy", {}).get("threshold"),
                "threshold_rationale": summary.get("threshold_policy", {}).get("threshold_rationale"),
                "threshold_selection_rule": summary.get("threshold_policy", {}).get("threshold_selection_rule"),
                "threshold_is_promotion_ready": summary.get("threshold_policy", {}).get("threshold_is_promotion_ready"),
                "threshold_grid": json.dumps(summary.get("threshold_policy", {}).get("threshold_grid", [])),
                "overcoverage_audit_dir": summary.get("threshold_policy", {}).get("overcoverage_audit_dir"),
                "overcoverage_audit_summary_path": summary.get("threshold_policy", {}).get("overcoverage_audit_summary_path"),
                "background_false_positive_foreground_fraction": summary.get("threshold_policy", {}).get("background_false_positive_foreground_fraction"),
                "positive_boundary_recall": summary.get("threshold_policy", {}).get("positive_boundary_recall"),
                "category_gate_status": summary.get("category_gate_status", {}).get("promotion_evidence_status"),
                "category_gate_reasons": "|".join(summary.get("category_gate_status", {}).get("reasons", [])),
                "category_gate_failed_count": summary.get("category_gate_status", {}).get("failed_gate_count"),
                "category_gate_count": summary.get("category_gate_status", {}).get("gate_count"),
                "validation_adjudication_status": summary.get("validation_adjudication", {}).get("status"),
                "validation_adjudication_path": summary.get("validation_adjudication", {}).get("path"),
                "validation_adjudication_applied_count": summary.get("validation_adjudication", {}).get("applied_count"),
                "validation_adjudication_nonblocking_count": summary.get("validation_adjudication", {}).get("nonblocking_count"),
                "validation_adjudication_blocking_count": summary.get("validation_adjudication", {}).get("blocking_count"),
                "resize_sensitivity_status": summary.get("resize_sensitivity", {}).get("status"),
                "resize_screening_summary_path": summary.get("resize_sensitivity", {}).get("resize_screening_summary_path"),
                "resize_decision_status": summary.get("resize_sensitivity", {}).get("resize_decision_status"),
                "resize_decision_reason": summary.get("resize_sensitivity", {}).get("resize_decision_reason"),
                "selected_resize_run_id": summary.get("resize_sensitivity", {}).get("selected_run_id"),
                "selected_resize_crop_size": summary.get("resize_sensitivity", {}).get("selected_crop_size"),
                "selected_resize_output_size": summary.get("resize_sensitivity", {}).get("selected_output_size"),
                "dice": summary.get("metrics", {}).get("dice"),
                "jaccard": summary.get("metrics", {}).get("jaccard"),
                "precision": summary.get("metrics", {}).get("precision"),
                "recall": summary.get("metrics", {}).get("recall"),
            }
        )

    prediction_rows = [
        row
        for summary in candidate_summaries
        for row in summary.get("prediction_rows", [])
    ]
    metric_by_category_rows = aggregate_metric_by_category(prediction_rows)
    prediction_shape_rows = [
        row
        for audit in candidate_audits.values()
        for row in audit["prediction_shape"].get("rows", [])
    ]
    category_gate_rows = [
        row
        for audit in candidate_audits.values()
        for row in audit.get("category_gate", {}).get("rows", [])
    ]
    resize_policy_rows = [
        row
        for audit in candidate_audits.values()
        for row in audit.get("resize_policy", [])
    ]
    failure_reproduction_rows = [
        row
        for audit in candidate_audits.values()
        for row in audit.get("failure_reproduction", [])
    ]
    decision_payload = {
        "decision": decision,
        "run_id": run_id,
        "output_dir": str(output_root),
        "model_dir": str(model_root),
        "manifest_path": str(manifest_path),
        "candidate_manifest": list(manifest),
        "manifest_audit": manifest_audit,
        "candidate_summaries": candidate_summaries,
        "candidate_audits": candidate_audits,
        "threshold_policy": threshold_policy,
        "overcoverage_audit": overcoverage_audit,
        "validation_adjudication": validation_adjudication,
        "validation_adjudication_applied_path": str(output_root / "validation_adjudication_applied.csv"),
        "validation_adjudication_applied_rows": validation_adjudication_rows,
        "resize_screening_summary_path": str(Path(resize_screening_summary_path).expanduser()) if resize_screening_summary_path else None,
        "resize_screening_summary_rows": resize_screening_rows,
        "metric_by_category_path": str(output_root / "metric_by_category.csv"),
        "prediction_shape_audit_path": str(output_root / "prediction_shape_audit.csv"),
        "category_gate_audit_path": str(output_root / "category_gate_audit.csv"),
        "resize_policy_audit_path": str(output_root / "resize_policy_audit.csv"),
        "failure_reproduction_audit_path": str(output_root / "failure_reproduction_audit.csv"),
        "documentation_claim_audit_path": str(output_root / "documentation_claim_audit.md"),
        "compatibility_summary": compatibility_summary,
    }
    report_json = output_root / "promotion_report.json"
    report_md = output_root / "promotion_report.md"
    report_html = output_root / "promotion_report.html"
    _write_csv(candidate_rows, output_root / "candidate_summary.csv")
    _write_csv(prediction_rows, output_root / "candidate_predictions.csv")
    write_csv_rows(metric_by_category_rows, output_root / "metric_by_category.csv")
    write_csv_rows(prediction_shape_rows, output_root / "prediction_shape_audit.csv")
    write_csv_rows(category_gate_rows, output_root / "category_gate_audit.csv")
    write_csv_rows(validation_adjudication_rows, output_root / "validation_adjudication_applied.csv")
    write_csv_rows(resize_policy_rows, output_root / "resize_policy_audit.csv")
    write_csv_rows(failure_reproduction_rows, output_root / "failure_reproduction_audit.csv")
    docs_to_audit: Dict[str, str] = {}
    for doc_path in (Path("README.md"), Path("docs/ONBOARDING_GUIDE.md")):
        if doc_path.exists():
            docs_to_audit[str(doc_path)] = doc_path.read_text(encoding="utf-8")
    doc_audit = documentation_claim_audit(
        docs_to_audit,
        cited_report_status=decision["promotion_evidence_status"],
        cited_report_path=report_json,
    )
    write_documentation_claim_audit(doc_audit, output_root / "documentation_claim_audit.md")
    decision_payload["documentation_claim_audit"] = doc_audit
    report_json.write_text(json.dumps(decision_payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_report(
        output_path=report_md,
        decision=decision,
        manifest_audit=manifest_audit,
        candidate_summaries=candidate_summaries,
        compatibility_summary=compatibility_summary,
    )
    _write_html_report(
        output_path=report_html,
        decision=decision,
        manifest_audit=manifest_audit,
        candidate_summaries=candidate_summaries,
    )
    return decision_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare glomeruli transfer and scratch candidates")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Supported glomeruli root: raw_data/cohorts, raw_data/cohorts/<cohort_id>, or an active paired raw_data project root",
    )
    parser.add_argument(
        "--output-dir",
        default=str(get_runtime_segmentation_evaluation_path("glomeruli_candidate_comparison")),
        help=(
            "Root directory for candidate-comparison runs "
            "(the run id is always appended)"
        ),
    )
    parser.add_argument("--run-id", help="Run directory name to create under --output-dir")
    parser.add_argument(
        "--model-dir",
        default=str(get_runtime_models_path() / "segmentation" / "glomeruli"),
        help="Root model-artifact directory for trained candidates; train_glomeruli writes transfer/ and scratch/ underneath",
    )
    parser.add_argument("--transfer-base-model", help="Explicit mitochondria base artifact for transfer training")
    parser.add_argument("--transfer-model-path", help="Existing transfer candidate artifact to evaluate instead of training")
    parser.add_argument("--scratch-model-path", help="Existing scratch candidate artifact to evaluate instead of training")
    parser.add_argument("--compat-model-path", help="Optional non-promoted comparison artifact path")
    parser.add_argument("--seed", type=int, default=42, help="Explicit seed for the initial one-seed-per-family workflow")
    parser.add_argument("--transfer-epochs", type=int, default=30, help="Transfer candidate epochs")
    parser.add_argument("--scratch-epochs", type=int, default=50, help="Scratch candidate epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional shared batch size override")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Shared training learning rate")
    parser.add_argument("--image-size", type=int, default=256, help="Model input size")
    parser.add_argument("--crop-size", type=int, default=512, help="Deterministic validation crop size and training crop size")
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        help="Device forwarded to fresh candidate training commands. Omit to auto-select cuda, then mps, then cpu.",
    )
    parser.add_argument("--loss", default="", help="Optional loss override forwarded to the training CLI")
    parser.add_argument("--negative-crop-manifest", help="Validated negative/background crop manifest forwarded to fresh candidate training")
    parser.add_argument("--negative-crop-sampler-weight", type=float, default=0.0, help="Deterministic negative/background crop manifest sampling weight")
    parser.add_argument("--augmentation-variant", default="fastai_default", choices=["fastai_default", "spatial_only", "no_aug"], help="Recorded augmentation policy variant for candidate provenance")
    parser.add_argument("--overcoverage-audit-dir", help="Optional glomeruli-overcoverage-audit output directory containing audit_summary.json and threshold_sweep.csv")
    parser.add_argument("--resize-screening-summary", help="Optional resize_policy_screening_summary.csv used to clear, select, or retain resize_benefit_unproven")
    parser.add_argument(
        "--validation-adjudication",
        help=(
            "Optional explicit CSV of reviewed validation gate failures. "
            "Matched rows are recorded in category_gate_audit.csv and "
            "validation_adjudication_applied.csv; unmatched rows are hard errors."
        ),
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=None,
        help=(
            "Foreground probability threshold for deterministic binary masks. "
            "When --overcoverage-audit-dir is supplied and this is omitted, "
            "the comparison derives a threshold from the audit sweep; without "
            "an audit, omission uses the legacy unverified 0.01 threshold."
        ),
    )
    parser.add_argument("--examples-per-category", type=int, default=DEFAULT_EXAMPLES_PER_CATEGORY, help="Deterministic manifest examples per category")
    parser.add_argument("--transfer-model-name", default="glomeruli_transfer_candidate", help="Transfer candidate model name prefix")
    parser.add_argument("--scratch-model-name", default="glomeruli_scratch_candidate", help="Scratch candidate model name prefix")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(verbose=True)
    if not args.run_id:
        args.run_id = _generated_run_id(args.seed)
    runtime_root = os.environ.get("EQ_RUNTIME_ROOT")
    command = [sys.executable, "-m", "eq.training.compare_glomeruli_candidates", *sys.argv[1:]]
    with direct_execution_log_context(
        surface="compare_glomeruli_candidates",
        explicit_run_id=args.run_id,
        runtime_root=runtime_root,
        dry_run=False,
        command=command,
        workflow="glomeruli_candidate_comparison",
        logger_name="eq",
    ) as log_context:
        logger.info("RUN_ID=%s", args.run_id)
        logger.info("OUTPUT_DIR=%s", Path(args.output_dir).expanduser() / args.run_id)
        logger.info("MODEL_DIR=%s", Path(args.model_dir).expanduser())
        logger.info("TRANSFER_MODEL_PATH=%s", args.transfer_model_path)
        logger.info("SCRATCH_MODEL_PATH=%s", args.scratch_model_path)
        logger.info("EXECUTION_LOG=%s", log_context.log_path)
        compare_glomeruli_candidates(args)


if __name__ == "__main__":
    main()
