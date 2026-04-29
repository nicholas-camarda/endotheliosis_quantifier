"""Utilities for recovering image-level endotheliosis scores from Label Studio exports."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from eq.data_management.canonical_contract import parse_subject_image_filename
from eq.quantification.burden import ALLOWED_SCORE_VALUES
from eq.quantification.migration import inventory_raw_project


class LabelStudioScoreError(ValueError):
    """Raised when a Label Studio result cannot produce one unambiguous grade."""


def _load_annotation_payload(source: str | Path) -> list[dict[str, Any]]:
    source_str = str(source)
    if source_str.startswith('git:'):
        _, revision, git_path = source_str.split(':', 2)
        raw = subprocess.check_output(
            ['git', 'show', f'{revision}:{git_path}'], text=True
        )
        return json.loads(raw)

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(
            f'Label Studio annotation source not found: {source_path}'
        )
    return json.loads(source_path.read_text(encoding='utf-8'))


def _candidate_annotation_paths(project_dir: Path) -> list[str]:
    local_candidates = [
        project_dir / 'annotations' / 'annotations.json',
        project_dir / 'annotations.json',
        project_dir / 'labelstudio_annotations.json',
    ]
    return [str(path) for path in local_candidates if path.exists()]


def discover_label_studio_annotation_source(project_dir: Path) -> Optional[str]:
    """Find the most plausible annotation export source for a raw project."""
    for candidate in _candidate_annotation_paths(Path(project_dir)):
        if Path(candidate).exists():
            return candidate
    return None


def _normalize_image_name(task: dict[str, Any]) -> str:
    file_upload = str(task.get('file_upload') or '').strip()
    if file_upload:
        return file_upload.split('-', 1)[-1] if '-' in file_upload else file_upload

    image_value = str(task.get('data', {}).get('image') or '').strip()
    if image_value:
        return Path(image_value).name
    return ''


def _coerce_supported_grade(choice: Any) -> float | None:
    text = str(choice).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    for allowed in ALLOWED_SCORE_VALUES:
        if abs(float(allowed) - value) < 1e-9:
            return float(allowed)
    return None


def extract_label_studio_grade(annotation: dict[str, Any]) -> float | None:
    """Extract one supported endotheliosis grade from one Label Studio annotation.

    The shared quantification rule is intentionally narrow: exactly one supported
    grade choice is accepted. Missing grade choices return None. Multiple
    supported grade choices are ambiguous and fail closed.
    """
    supported_grades: list[float] = []
    for result in annotation.get('result', []):
        if result.get('type') != 'choices':
            continue
        choices = result.get('value', {}).get('choices', [])
        for choice in choices:
            grade = _coerce_supported_grade(choice)
            if grade is not None:
                supported_grades.append(grade)
    if len(supported_grades) > 1:
        raise LabelStudioScoreError(
            'Ambiguous Label Studio grade result: multiple supported grade choices '
            f'{supported_grades}'
        )
    if supported_grades:
        return supported_grades[0]
    return None


def _subject_prefix_from_image_name(image_name: str) -> str:
    parsed = parse_subject_image_filename(image_name)
    if parsed is None:
        return ''
    return parsed.subject_prefix


def _build_join_maps(project_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    inventory = inventory_raw_project(Path(project_dir))
    if inventory.empty:
        images_dir = Path(project_dir) / 'images'
        masks_dir = Path(project_dir) / 'masks'
        if not images_dir.exists() or not masks_dir.exists():
            return {}, {}

        def join_stem(path_or_name: str | Path) -> str:
            stem = Path(path_or_name).stem.lower()
            return stem.removesuffix('_mask')

        image_paths = [
            path
            for path in sorted(images_dir.rglob('*'))
            if path.is_file()
            and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        ]
        masks_by_stem = {
            join_stem(path): str(path)
            for path in sorted(masks_dir.rglob('*'))
            if path.is_file()
            and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        }
        image_lookup: dict[str, str] = {}
        mask_lookup: dict[str, str] = {}
        for image_path in image_paths:
            keys = {image_path.name, image_path.name.lower(), join_stem(image_path)}
            for key in keys:
                image_lookup[key] = str(image_path)
                if join_stem(image_path) in masks_by_stem:
                    mask_lookup[key] = masks_by_stem[join_stem(image_path)]
        return image_lookup, mask_lookup

    image_lookup = {
        str(row.image_name): str(row.image_path)
        for row in inventory.itertuples(index=False)
        if isinstance(row.image_name, str)
    }
    mask_lookup = {
        str(row.image_name): str(row.mask_path)
        for row in inventory.itertuples(index=False)
        if isinstance(row.image_name, str)
        and isinstance(row.mask_path, str)
        and row.mask_path
    }
    return image_lookup, mask_lookup


def recover_label_studio_score_table(
    project_dir: Path,
    annotation_source: str | Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Recover image-level scores from a Label Studio export and join them to current raw files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_annotation_payload(annotation_source)
    image_lookup, mask_lookup = _build_join_maps(Path(project_dir))

    raw_rows: list[dict[str, Any]] = []
    for task in payload:
        image_name = _normalize_image_name(task)
        subject_prefix = _subject_prefix_from_image_name(image_name)
        annotations = task.get('annotations', [])
        if not annotations:
            raw_rows.append(
                {
                    'image_name': image_name,
                    'subject_prefix': subject_prefix,
                    'task_id': task.get('id'),
                    'task_created_at': task.get('created_at'),
                    'task_updated_at': task.get('updated_at'),
                    'annotation_updated_at': None,
                    'grade': None,
                    'annotation_status': 'missing_annotation',
                }
            )
            continue

        for annotation in annotations:
            grade = extract_label_studio_grade(annotation)
            raw_rows.append(
                {
                    'image_name': image_name,
                    'subject_prefix': subject_prefix,
                    'task_id': task.get('id'),
                    'task_created_at': task.get('created_at'),
                    'task_updated_at': task.get('updated_at'),
                    'annotation_updated_at': annotation.get('updated_at'),
                    'grade': grade,
                    'annotation_status': 'graded'
                    if grade is not None
                    else 'missing_grade',
                }
            )

    all_rows = pd.DataFrame(raw_rows)
    all_rows.to_csv(output_dir / 'labelstudio_all_annotations.csv', index=False)

    chosen_rows: list[dict[str, Any]] = []
    for image_name, image_rows in all_rows.groupby(
        'image_name', dropna=False, sort=True
    ):
        ranked = image_rows.sort_values(
            ['annotation_updated_at', 'task_updated_at', 'task_created_at'],
            na_position='first',
        )
        latest_row = ranked.iloc[-1].to_dict()
        latest_grade = latest_row.get('grade')

        resolution = 'latest_annotation'
        chosen = latest_row
        if pd.isna(latest_grade):
            resolution = 'latest_annotation_missing_grade'

        image_keys = (
            str(image_name),
            str(image_name).lower(),
            Path(str(image_name)).stem.lower(),
        )
        image_path = next(
            (image_lookup[key] for key in image_keys if key in image_lookup), ''
        )
        mask_path = next(
            (mask_lookup[key] for key in image_keys if key in mask_lookup), ''
        )
        if image_path and mask_path:
            join_status = 'ok'
        elif image_path:
            join_status = 'missing_mask'
        elif mask_path:
            join_status = 'missing_image'
        else:
            join_status = 'missing_image_and_mask'

        chosen_rows.append(
            {
                'image_name': image_name,
                'image_stem': Path(str(image_name)).stem,
                'subject_prefix': chosen.get('subject_prefix')
                or _subject_prefix_from_image_name(str(image_name)),
                'score': float(chosen['grade'])
                if not pd.isna(chosen.get('grade'))
                else None,
                'score_status': 'ok'
                if not pd.isna(chosen.get('grade'))
                else 'missing_score',
                'score_resolution': resolution,
                'annotation_source': str(annotation_source),
                'source_task_id': chosen.get('task_id'),
                'source_annotation_updated_at': chosen.get('annotation_updated_at'),
                'source_task_updated_at': chosen.get('task_updated_at'),
                'raw_image_path': image_path,
                'raw_mask_path': mask_path,
                'join_status': join_status,
            }
        )

    score_table = (
        pd.DataFrame(chosen_rows)
        .sort_values(['subject_prefix', 'image_name'])
        .reset_index(drop=True)
    )
    score_table.to_csv(output_dir / 'labelstudio_scores.csv', index=False)

    duplicate_rows = all_rows[
        all_rows.duplicated(subset=['image_name'], keep=False)
    ].copy()
    duplicate_rows.to_csv(
        output_dir / 'labelstudio_duplicate_annotations.csv', index=False
    )

    summary = {
        'annotation_source': str(annotation_source),
        'n_annotation_rows': int(len(all_rows)),
        'n_unique_images': int(score_table['image_name'].nunique())
        if not score_table.empty
        else 0,
        'n_duplicate_images': int(duplicate_rows['image_name'].nunique())
        if not duplicate_rows.empty
        else 0,
        'score_status_counts': score_table['score_status']
        .value_counts(dropna=False)
        .to_dict()
        if not score_table.empty
        else {},
        'join_status_counts': score_table['join_status']
        .value_counts(dropna=False)
        .to_dict()
        if not score_table.empty
        else {},
        'score_resolution_counts': score_table['score_resolution']
        .value_counts(dropna=False)
        .to_dict()
        if not score_table.empty
        else {},
    }
    (output_dir / 'labelstudio_score_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )

    return {
        'all_annotations': output_dir / 'labelstudio_all_annotations.csv',
        'scores': output_dir / 'labelstudio_scores.csv',
        'duplicates': output_dir / 'labelstudio_duplicate_annotations.csv',
        'summary': output_dir / 'labelstudio_score_summary.json',
    }
