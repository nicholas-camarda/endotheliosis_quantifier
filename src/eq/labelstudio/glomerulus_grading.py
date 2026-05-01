"""Glomerulus-instance Label Studio grading contract parsing and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from eq.quantification.labelstudio_scores import (
    _load_annotation_payload,
    _normalize_image_name,
    extract_label_studio_grade,
)

COMPLETE_GLOMERULUS_LABEL = 'complete_glomerulus'
CUTOFF_PARTIAL_GLOMERULUS_LABEL = 'cutoff_partial_glomerulus'
GRADE_FROM_NAME = 'endotheliosis_grade'
REGION_TYPES = {'brushlabels', 'polygonlabels', 'rectanglelabels'}


class LabelStudioGlomerulusContractError(ValueError):
    """Raised when Label Studio glomerulus-instance grading violates contract."""


def load_glomerulus_grading_records(annotation_source: str | Path) -> pd.DataFrame:
    """Load validated glomerulus-instance grading records from a Label Studio JSON export."""
    payload = _load_annotation_payload(annotation_source)
    rows: list[dict[str, Any]] = []
    for task in payload:
        rows.extend(_records_for_task(task, annotation_source=annotation_source))
    return pd.DataFrame(rows, columns=_record_columns())


def prepare_rollup_records(records: pd.DataFrame) -> pd.DataFrame:
    """Return complete graded glomeruli suitable for downstream rollups."""
    if records.empty:
        return records.copy()
    complete = records[records['completeness_status'].eq('complete')].copy()
    return complete.reset_index(drop=True)


def _records_for_task(
    task: dict[str, Any], *, annotation_source: str | Path
) -> list[dict[str, Any]]:
    image_name = _normalize_image_name(task)
    image_id = Path(image_name).stem
    task_id = str(task.get('id', ''))
    annotations = task.get('annotations') or []
    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        rows.extend(
            _records_for_annotation(
                task=task,
                annotation=annotation,
                image_name=image_name,
                image_id=image_id,
                task_id=task_id,
                annotation_source=annotation_source,
            )
        )
    return rows


def _records_for_annotation(
    *,
    task: dict[str, Any],
    annotation: dict[str, Any],
    image_name: str,
    image_id: str,
    task_id: str,
    annotation_source: str | Path,
) -> list[dict[str, Any]]:
    grader = _grader_provenance(annotation)
    annotation_id = str(annotation.get('id', ''))
    result = annotation.get('result') or []
    prediction_regions = _prediction_regions(task)
    task_mask_release_id = str(task.get('data', {}).get('mask_release_id') or '').strip()
    parent_prediction_id = str(annotation.get('parent_prediction') or '').strip()
    regions: dict[str, dict[str, Any]] = {}
    grades: dict[str, list[dict[str, Any]]] = {}
    unlinked_grades: list[dict[str, Any]] = []

    for item in result:
        item_id = str(item.get('id') or '')
        if not item_id:
            continue
        if _is_glomerulus_region(item):
            regions[item_id] = item
        elif _is_grade_result(item):
            grades.setdefault(item_id, []).append(item)
            if item_id not in regions:
                unlinked_grades.append(item)

    if unlinked_grades and not regions:
        raise LabelStudioGlomerulusContractError(
            'Missing grade-to-region linkage: image-level average labels are '
            'legacy baseline data, not per-glomerulus ground truth'
        )
    if unlinked_grades:
        raise LabelStudioGlomerulusContractError(
            'Missing grade-to-region linkage for Label Studio grade result'
        )

    rows: list[dict[str, Any]] = []
    for region_id, region in sorted(regions.items()):
        if region.get('type') != 'brushlabels':
            raise LabelStudioGlomerulusContractError(
                f'non-brush-region for region {region_id}: expected brushlabels'
            )
        labels = _region_labels(region)
        is_complete = COMPLETE_GLOMERULUS_LABEL in labels
        is_cutoff = CUTOFF_PARTIAL_GLOMERULUS_LABEL in labels
        if is_complete == is_cutoff:
            raise LabelStudioGlomerulusContractError(
                'Each glomerulus region must be exactly one of complete_glomerulus '
                'or cutoff_partial_glomerulus'
            )

        region_grades = grades.get(region_id, [])
        if len(region_grades) > 1:
            raise LabelStudioGlomerulusContractError(
                f'duplicate-grade for region {region_id}'
            )

        grade = _single_grade(region_grades[0]) if region_grades else None
        if is_cutoff and grade is not None:
            raise LabelStudioGlomerulusContractError(
                f'excluded-region-grade for region {region_id}'
            )
        if is_complete and grade is None:
            raise LabelStudioGlomerulusContractError(
                f'Complete glomerulus region {region_id} is missing a grade'
            )

        completeness_status = 'complete' if is_complete else 'excluded'
        exclusion_reason = CUTOFF_PARTIAL_GLOMERULUS_LABEL if is_cutoff else ''
        record_id = _record_id(image_id, region_id, annotation_id, grader['grader_user_id'])
        proposal_kind, region_edit_state, mask_release_id, mask_source = _lineage_for_region(
            region_id=region_id,
            region=region,
            prediction_regions=prediction_regions,
            task_mask_release_id=task_mask_release_id,
            parent_prediction_id=parent_prediction_id,
        )
        rows.append(
            {
                'source_glomerulus_record_id': record_id,
                'image_id': image_id,
                'image_name': image_name,
                'glomerulus_instance_id': region_id,
                'region_id': region_id,
                'region_type': str(region.get('type') or ''),
                'region_rle': (region.get('value') or {}).get('rle'),
                'region_original_width': region.get('original_width'),
                'region_original_height': region.get('original_height'),
                'roi_source': 'label_studio_human',
                'completeness_status': completeness_status,
                'exclusion_reason': exclusion_reason,
                'human_grade': grade,
                'grader_user_id': grader['grader_user_id'],
                'grader_email': grader['grader_email'],
                'grader_first_name': grader['grader_first_name'],
                'grader_last_name': grader['grader_last_name'],
                'task_id': task_id,
                'annotation_id': annotation_id,
                'task_created_at': str(task.get('created_at') or ''),
                'task_updated_at': str(task.get('updated_at') or ''),
                'annotation_created_at': str(annotation.get('created_at') or ''),
                'annotation_updated_at': str(annotation.get('updated_at') or ''),
                'lead_time': annotation.get('lead_time'),
                'annotation_source': str(annotation_source),
                'mask_release_id': mask_release_id,
                'mask_source': mask_source,
                'proposal_kind': proposal_kind,
                'region_edit_state': region_edit_state,
                'parent_prediction_id': parent_prediction_id,
            }
        )
    return rows


def _prediction_regions(task: dict[str, Any]) -> dict[str, tuple[dict[str, Any], str]]:
    predictions = task.get('predictions') or []
    mapped: dict[str, tuple[dict[str, Any], str]] = {}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        model_version = str(prediction.get('model_version') or '')
        for item in prediction.get('result') or []:
            rid = str(item.get('id') or '').strip()
            if rid:
                mapped[rid] = (item, model_version)
    return mapped


def _lineage_for_region(
    *,
    region_id: str,
    region: dict[str, Any],
    prediction_regions: dict[str, tuple[dict[str, Any], str]],
    task_mask_release_id: str,
    parent_prediction_id: str,
) -> tuple[str, str, str, str]:
    predicted = prediction_regions.get(region_id)
    if predicted is not None:
        prediction_region, model_version = predicted
        mask_release_id = task_mask_release_id or _mask_release_from_model_version(
            model_version
        )
        if not mask_release_id:
            raise LabelStudioGlomerulusContractError(
                f'contradictory-lineage for region {region_id}: '
                'proposal_kind=auto_preload requires mask_release_id'
            )
        same_geometry = _geometry_signature(region) == _geometry_signature(
            prediction_region
        )
        return (
            'auto_preload',
            'unedited_auto' if same_geometry else 'human_refined_boundary',
            mask_release_id,
            'medsam_finetuned_glomeruli',
        )
    if parent_prediction_id:
        return (
            'box_assisted_manual',
            'human_refined_boundary',
            task_mask_release_id,
            'box_assisted_medsam',
        )
    return ('human_manual', 'manual_drawn', task_mask_release_id, 'manual')


def _mask_release_from_model_version(model_version: str) -> str:
    text = str(model_version).strip()
    prefix = 'medsam-release:'
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return ''


def _geometry_signature(result: dict[str, Any]) -> str:
    value = result.get('value') or {}
    geometry = {
        'format': value.get('format'),
        'rle': value.get('rle'),
        'points': value.get('points'),
        'brushlabels': value.get('brushlabels'),
        'polygonlabels': value.get('polygonlabels'),
        'rectanglelabels': value.get('rectanglelabels'),
    }
    return str(geometry)


def _grader_provenance(annotation: dict[str, Any]) -> dict[str, str]:
    completed_by = annotation.get('completed_by')
    if completed_by in (None, ''):
        raise LabelStudioGlomerulusContractError(
            'missing-provenance: Label Studio annotation lacks completed_by'
        )
    if isinstance(completed_by, dict):
        grader_id = str(completed_by.get('id') or '').strip()
        if not grader_id:
            raise LabelStudioGlomerulusContractError(
                'missing-provenance: completed_by object lacks id'
            )
        return {
            'grader_user_id': grader_id,
            'grader_email': str(completed_by.get('email') or ''),
            'grader_first_name': str(completed_by.get('first_name') or ''),
            'grader_last_name': str(completed_by.get('last_name') or ''),
        }
    return {
        'grader_user_id': str(completed_by),
        'grader_email': '',
        'grader_first_name': '',
        'grader_last_name': '',
    }


def _is_glomerulus_region(result: dict[str, Any]) -> bool:
    return result.get('type') in REGION_TYPES and bool(
        set(_region_labels(result)).intersection(
            {COMPLETE_GLOMERULUS_LABEL, CUTOFF_PARTIAL_GLOMERULUS_LABEL}
        )
    )


def _is_grade_result(result: dict[str, Any]) -> bool:
    return result.get('type') == 'choices' and result.get('from_name') == GRADE_FROM_NAME


def _region_labels(result: dict[str, Any]) -> list[str]:
    value = result.get('value') or {}
    labels: list[str] = []
    for key in ('brushlabels', 'polygonlabels', 'rectanglelabels', 'labels'):
        raw = value.get(key) or []
        labels.extend(str(item) for item in raw)
    return labels


def _single_grade(result: dict[str, Any]) -> float | None:
    return extract_label_studio_grade({'result': [result]})


def _record_id(
    image_id: str, glomerulus_instance_id: str, annotation_id: str, grader_user_id: str
) -> str:
    return f'{image_id}::{glomerulus_instance_id}::{annotation_id}::{grader_user_id}'


def _record_columns() -> list[str]:
    return [
        'source_glomerulus_record_id',
        'image_id',
        'image_name',
        'glomerulus_instance_id',
        'region_id',
        'region_type',
        'region_rle',
        'region_original_width',
        'region_original_height',
        'roi_source',
        'completeness_status',
        'exclusion_reason',
        'human_grade',
        'grader_user_id',
        'grader_email',
        'grader_first_name',
        'grader_last_name',
        'task_id',
        'annotation_id',
        'task_created_at',
        'task_updated_at',
        'annotation_created_at',
        'annotation_updated_at',
        'lead_time',
        'annotation_source',
        'mask_release_id',
        'mask_source',
        'proposal_kind',
        'region_edit_state',
        'parent_prediction_id',
    ]
