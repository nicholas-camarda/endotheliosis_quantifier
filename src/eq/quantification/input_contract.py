"""Canonical quantification input contract resolution and provenance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

TARGET_DEFINITION_VERSION = "endotheliosis_reviewed_label_contract_v1"
STABLE_LABEL_OVERRIDE_PREFIX = (
    "derived_data/quantification_inputs/reviewed_label_overrides/"
    "endotheliosis_grade_model/"
)


class QuantificationInputContractError(ValueError):
    """Raised when quantification inputs cannot satisfy the label contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal_path(path: Path) -> str:
    return path.expanduser().as_posix()


def reject_generated_output_override_path(path: Path) -> None:
    normalized = _normal_path(path)
    if "/output/quantification_results/" in normalized or normalized.startswith(
        "output/quantification_results/"
    ):
        raise QuantificationInputContractError(
            "Reviewed label overrides must live under stable derived input roots, "
            f"not generated quantification outputs: {path}"
        )


def validate_committed_label_override_path(raw_path: str | Path | None) -> None:
    """Validate the committed config path shape without requiring runtime files."""
    if raw_path in (None, ""):
        return
    path = Path(str(raw_path)).expanduser()
    reject_generated_output_override_path(path)
    if not path.is_absolute() and not path.as_posix().startswith(
        STABLE_LABEL_OVERRIDE_PREFIX
    ):
        raise QuantificationInputContractError(
            "Committed inputs.label_overrides must be runtime-root relative under "
            f"{STABLE_LABEL_OVERRIDE_PREFIX}: {raw_path}"
        )


def _file_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.exists() and path.is_file():
        return sha256_file(path)
    return None


@dataclass(frozen=True)
class ResolvedQuantificationInputContract:
    data_dir: Path
    segmentation_model: Path
    output_dir: Path
    score_source: str
    annotation_source: str | None
    mapping_file: Path | None
    label_overrides_path: Path | None
    target_definition_version: str = TARGET_DEFINITION_VERSION

    def reference(self) -> dict[str, Any]:
        label_override_status = (
            "reviewed_override_file" if self.label_overrides_path else "none"
        )
        return {
            "target_definition_version": self.target_definition_version,
            "data_dir": str(self.data_dir),
            "score_source": self.score_source,
            "annotation_source": self.annotation_source or "",
            "mapping_file": str(self.mapping_file) if self.mapping_file else "",
            "mapping_file_hash": _file_hash(self.mapping_file),
            "label_overrides": label_override_status,
            "label_overrides_path": str(self.label_overrides_path)
            if self.label_overrides_path
            else "none",
            "label_overrides_hash": _file_hash(self.label_overrides_path),
            "segmentation_artifact": str(self.segmentation_model),
            "segmentation_artifact_hash": _file_hash(self.segmentation_model),
            "output_dir": str(self.output_dir),
        }


def resolve_quantification_input_contract(
    *,
    data_dir: Path,
    segmentation_model: Path,
    output_dir: Path,
    mapping_file: Path | None = None,
    annotation_source: str | Path | None = None,
    score_source: str = "auto",
    label_overrides_path: Path | None = None,
) -> ResolvedQuantificationInputContract:
    """Resolve and validate input paths before quantification labels are loaded."""
    if score_source not in {"auto", "labelstudio", "spreadsheet"}:
        raise QuantificationInputContractError(f"Unsupported score_source: {score_source}")
    data_dir = Path(data_dir).expanduser()
    segmentation_model = Path(segmentation_model).expanduser()
    output_dir = Path(output_dir).expanduser()
    mapping_file = Path(mapping_file).expanduser() if mapping_file else None
    label_overrides_path = (
        Path(label_overrides_path).expanduser() if label_overrides_path else None
    )
    if not data_dir.exists():
        raise FileNotFoundError(f"Required quantification data_dir does not exist: {data_dir}")
    if not segmentation_model.exists():
        raise FileNotFoundError(
            f"Required quantification segmentation_model does not exist: {segmentation_model}"
        )
    if mapping_file and not mapping_file.exists():
        raise FileNotFoundError(f"Quantification mapping_file does not exist: {mapping_file}")
    if label_overrides_path:
        reject_generated_output_override_path(label_overrides_path)
        if not label_overrides_path.exists():
            raise FileNotFoundError(
                f"Required reviewed label override input does not exist: {label_overrides_path}"
            )

    annotation_text = str(annotation_source) if annotation_source else None
    if annotation_text and not annotation_text.startswith("git:"):
        annotation_path = Path(annotation_text).expanduser()
        if annotation_path.suffix and not annotation_path.exists():
            raise FileNotFoundError(
                f"Quantification annotation_source does not exist: {annotation_path}"
            )

    return ResolvedQuantificationInputContract(
        data_dir=data_dir,
        segmentation_model=segmentation_model,
        output_dir=output_dir,
        score_source=score_source,
        annotation_source=annotation_text,
        mapping_file=mapping_file,
        label_overrides_path=label_overrides_path,
    )


def grouping_identity_from_scored_table(scored_table: pd.DataFrame) -> dict[str, Any]:
    required = {"subject_image_id", "subject_id"}
    missing = sorted(required.difference(scored_table.columns))
    if missing:
        raise QuantificationInputContractError(
            f"Scored table is missing required grouping identity columns: {missing}"
        )
    subject_image_ids = scored_table["subject_image_id"].astype(str)
    duplicate_ids = sorted(
        subject_image_ids[subject_image_ids.duplicated(keep=False)].unique().tolist()
    )
    if duplicate_ids:
        raise QuantificationInputContractError(
            f"Scored table has duplicate subject_image_id values: {duplicate_ids[:10]}"
        )
    subjects = scored_table["subject_id"].astype(str)
    if subjects.str.strip().eq("").any():
        raise QuantificationInputContractError("Scored table has blank subject_id values")
    return {
        "row_identity": "subject_image_id",
        "subject_key": "subject_id",
        "validation_group_key": "subject_id",
        "subject_image_id_unique": True,
        "grouping_key_derivation": "subject_id column from canonical scored table",
        "row_count": int(len(scored_table)),
        "subject_count": int(subjects.nunique()),
        "duplicate_subject_image_ids": duplicate_ids,
    }


def label_contract_reference_for_scored_table(
    contract: ResolvedQuantificationInputContract,
    scored_table: pd.DataFrame,
    *,
    base_scored_input_path: Path | None = None,
    annotation_artifact_path: Path | None = None,
) -> dict[str, Any]:
    reference = contract.reference()
    reference["grouping_identity"] = grouping_identity_from_scored_table(scored_table)
    reference["base_scored_input_path"] = (
        str(base_scored_input_path) if base_scored_input_path else ""
    )
    reference["base_scored_input_hash"] = _file_hash(base_scored_input_path)
    reference["annotation_source_hash"] = _file_hash(annotation_artifact_path)
    reference["score_override_audit_path"] = ""
    return reference
