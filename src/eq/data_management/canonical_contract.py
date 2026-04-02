"""Canonical filename parsing, validation, and migration planning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
CANONICAL_SUBJECT_IMAGE_RE = re.compile(r'^(?P<subject_prefix>[A-Za-z0-9]+)-(?P<image_index>[1-9]\d*)$')
LEGACY_SUBJECT_IMAGE_RE = re.compile(r'^(?P<subject_prefix>[A-Za-z0-9]+)_Image(?P<legacy_index>\d+)$')


@dataclass(frozen=True)
class ParsedSubjectImage:
    """Parsed filename identity for either canonical or legacy glomeruli files."""

    naming: str
    subject_prefix: str
    subject_image_id: str
    image_index: Optional[int] = None
    legacy_stem: Optional[str] = None


def _strip_mask_suffix(stem: str) -> str:
    return stem[:-5] if stem.endswith('_mask') else stem


def parse_subject_image_stem(stem: str, allow_mask_suffix: bool = True) -> Optional[ParsedSubjectImage]:
    """Parse a filename stem into canonical or legacy identity information."""
    normalized = _strip_mask_suffix(stem) if allow_mask_suffix else stem

    canonical_match = CANONICAL_SUBJECT_IMAGE_RE.fullmatch(normalized)
    if canonical_match:
        subject_prefix = canonical_match.group('subject_prefix')
        image_index = int(canonical_match.group('image_index'))
        return ParsedSubjectImage(
            naming='canonical',
            subject_prefix=subject_prefix,
            subject_image_id=f'{subject_prefix}-{image_index}',
            image_index=image_index,
            legacy_stem=None,
        )

    legacy_match = LEGACY_SUBJECT_IMAGE_RE.fullmatch(normalized)
    if legacy_match:
        subject_prefix = legacy_match.group('subject_prefix')
        legacy_index = int(legacy_match.group('legacy_index'))
        return ParsedSubjectImage(
            naming='legacy',
            subject_prefix=subject_prefix,
            subject_image_id='',
            image_index=None,
            legacy_stem=f'{subject_prefix}_Image{legacy_index}',
        )

    return None


def parse_subject_image_filename(filename: str) -> Optional[ParsedSubjectImage]:
    """Parse a filename with extension into canonical or legacy identity information."""
    return parse_subject_image_stem(Path(filename).stem, allow_mask_suffix=True)


def subject_prefix_from_subject_image_id(subject_image_id: str) -> str:
    """Return the subject prefix from a canonical subject-image identifier."""
    parsed = parse_subject_image_stem(subject_image_id, allow_mask_suffix=False)
    if parsed is None or parsed.naming != 'canonical':
        raise ValueError(f'Invalid canonical subject_image_id: {subject_image_id}')
    return parsed.subject_prefix


def canonical_image_filename(subject_image_id: str, suffix: str) -> str:
    """Build the canonical image filename."""
    return f'{subject_image_id}{suffix.lower()}'


def canonical_mask_filename(subject_image_id: str, suffix: str) -> str:
    """Build the canonical mask filename."""
    return f'{subject_image_id}_mask{suffix.lower()}'


def iter_project_files(project_dir: Path, kind: str) -> list[Path]:
    """Return all image or mask files in a raw project tree."""
    root = Path(project_dir) / kind
    if not root.exists():
        return []

    files: list[Path] = []
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for suffix in IMAGE_SUFFIXES:
            files.extend(sorted(subject_dir.glob(f'*{suffix}')))
    return files


def _normalize_mapping_frame(mapping_df: pd.DataFrame) -> pd.DataFrame:
    if 'subject_image_id' not in mapping_df.columns:
        raise ValueError("Mapping file must contain a 'subject_image_id' column")

    legacy_column: Optional[str] = None
    for candidate in ('legacy_stem', 'legacy_image_name', 'legacy_filename', 'legacy_path'):
        if candidate in mapping_df.columns:
            legacy_column = candidate
            break
    if legacy_column is None:
        raise ValueError(
            "Mapping file must contain one of: 'legacy_stem', 'legacy_image_name', 'legacy_filename', 'legacy_path'"
        )

    normalized = mapping_df.copy()
    normalized['legacy_stem'] = normalized[legacy_column].astype(str).map(
        lambda value: _strip_mask_suffix(Path(value).stem)
    )
    normalized['subject_image_id'] = normalized['subject_image_id'].astype(str)
    normalized = normalized[['legacy_stem', 'subject_image_id']].drop_duplicates()
    duplicates = normalized[normalized.duplicated(subset=['legacy_stem'], keep=False)]
    if not duplicates.empty:
        raise ValueError(f'Duplicate legacy mappings detected: {duplicates.to_dict("records")}')
    return normalized


def load_contract_mapping(mapping_file: Path) -> pd.DataFrame:
    """Load a migration mapping table from CSV, TSV, JSON, or Excel."""
    mapping_file = Path(mapping_file)
    if not mapping_file.exists():
        raise FileNotFoundError(f'Mapping file not found: {mapping_file}')

    suffix = mapping_file.suffix.lower()
    if suffix == '.csv':
        mapping_df = pd.read_csv(mapping_file)
    elif suffix == '.tsv':
        mapping_df = pd.read_csv(mapping_file, sep='\t')
    elif suffix in {'.xlsx', '.xls'}:
        mapping_df = pd.read_excel(mapping_file)
    elif suffix == '.json':
        mapping_df = pd.read_json(mapping_file)
    else:
        raise ValueError(f'Unsupported mapping file type: {mapping_file.suffix}')

    return _normalize_mapping_frame(mapping_df)


def build_migration_plan(
    project_dir: Path,
    metadata_df: pd.DataFrame,
    mapping_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Create a fail-closed migration plan for raw images and masks."""
    metadata_subject_ids = set(metadata_df['subject_id'].dropna().astype(str))
    mapping_by_legacy: Dict[str, str] = {}
    if mapping_file is not None:
        mapping_df = load_contract_mapping(mapping_file)
        mapping_by_legacy = dict(zip(mapping_df['legacy_stem'], mapping_df['subject_image_id']))

    rows: list[dict[str, Any]] = []
    for kind in ('images', 'masks'):
        for source_path in iter_project_files(project_dir, kind):
            parsed = parse_subject_image_filename(source_path.name)
            source_stem = _strip_mask_suffix(source_path.stem)
            is_mask = kind == 'masks'
            suffix = source_path.suffix.lower()

            row: dict[str, Any] = {
                'kind': kind,
                'source_path': str(source_path),
                'source_filename': source_path.name,
                'source_stem': source_stem,
                'source_subject_dir': source_path.parent.name,
                'suffix': suffix,
                'status': '',
                'reason': '',
                'subject_image_id': '',
                'target_path': '',
            }

            if parsed is None:
                row['status'] = 'invalid_name'
                row['reason'] = 'filename_not_parseable'
                rows.append(row)
                continue

            if parsed.naming == 'canonical':
                subject_image_id = parsed.subject_image_id
                row['subject_image_id'] = subject_image_id
                subject_dir = subject_prefix_from_subject_image_id(subject_image_id)
                filename = canonical_mask_filename(subject_image_id, suffix) if is_mask else canonical_image_filename(subject_image_id, suffix)
                target_path = Path(project_dir) / kind / subject_dir / filename
                row['target_path'] = str(target_path)
                if subject_image_id not in metadata_subject_ids:
                    row['status'] = 'canonical_missing_from_metadata'
                    row['reason'] = 'subject_image_id_not_in_metadata'
                elif target_path == source_path:
                    row['status'] = 'already_canonical'
                    row['reason'] = 'ok'
                else:
                    row['status'] = 'canonical_relocation_required'
                    row['reason'] = 'canonical_name_in_wrong_subject_dir'
                rows.append(row)
                continue

            legacy_stem = parsed.legacy_stem or source_stem
            subject_image_id = mapping_by_legacy.get(legacy_stem, '')
            row['subject_image_id'] = subject_image_id
            if not subject_image_id:
                row['status'] = 'unmapped_legacy'
                row['reason'] = 'legacy_name_has_no_mapping'
                rows.append(row)
                continue

            if subject_image_id not in metadata_subject_ids:
                row['status'] = 'mapped_id_missing_from_metadata'
                row['reason'] = 'mapping_target_not_in_metadata'
                rows.append(row)
                continue

            subject_dir = subject_prefix_from_subject_image_id(subject_image_id)
            filename = canonical_mask_filename(subject_image_id, suffix) if is_mask else canonical_image_filename(subject_image_id, suffix)
            target_path = Path(project_dir) / kind / subject_dir / filename
            row['target_path'] = str(target_path)
            row['status'] = 'rename_required'
            row['reason'] = 'mapped_from_legacy'
            rows.append(row)

    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan

    actionable = plan[plan['status'].isin({'rename_required', 'canonical_relocation_required'})]
    if not actionable.empty:
        duplicate_targets = actionable[actionable.duplicated(subset=['target_path'], keep=False)]
        if not duplicate_targets.empty:
            collisions = set(duplicate_targets['target_path'].tolist())
            collision_mask = plan['target_path'].isin(collisions)
            plan.loc[collision_mask, 'status'] = 'target_collision'
            plan.loc[collision_mask, 'reason'] = 'multiple_sources_map_to_same_target'

    return plan.sort_values(['kind', 'source_subject_dir', 'source_filename']).reset_index(drop=True)


def apply_migration_plan(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Apply a migration plan in place and return an execution report."""
    if plan_df.empty:
        return plan_df.copy()

    report = plan_df.copy()
    executable = report['status'].isin({'rename_required', 'canonical_relocation_required'})
    for index, row in report[executable].iterrows():
        source_path = Path(str(row['source_path']))
        target_path = Path(str(row['target_path']))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not source_path.exists():
            report.at[index, 'status'] = 'missing_source_at_apply_time'
            report.at[index, 'reason'] = 'source_path_did_not_exist'
            continue
        if target_path.exists():
            report.at[index, 'status'] = 'target_exists'
            report.at[index, 'reason'] = 'target_path_already_exists'
            continue
        source_path.rename(target_path)
        report.at[index, 'status'] = 'renamed'
        report.at[index, 'reason'] = 'applied'

    return report


def validate_project_contract(
    project_dir: Path,
    metadata_df: pd.DataFrame,
    require_canonical: bool = True,
) -> Dict[str, Any]:
    """Validate raw images and masks against the canonical subject-image contract."""
    project_dir = Path(project_dir)
    metadata_subject_ids = set(metadata_df['subject_id'].dropna().astype(str))

    image_files = iter_project_files(project_dir, 'images')
    mask_files = iter_project_files(project_dir, 'masks')

    errors: list[str] = []
    warnings: list[str] = []
    image_ids: set[str] = set()
    mask_ids: set[str] = set()

    stats: Dict[str, Any] = {
        'total_images': len(image_files),
        'total_masks': len(mask_files),
        'canonical_images': 0,
        'legacy_images': 0,
        'invalid_images': 0,
        'canonical_masks': 0,
        'legacy_masks': 0,
        'invalid_masks': 0,
    }

    for kind, files in (('image', image_files), ('mask', mask_files)):
        for path in files:
            parsed = parse_subject_image_filename(path.name)
            suffix_key = f'{"canonical" if parsed and parsed.naming == "canonical" else "legacy" if parsed else "invalid"}_{kind}s'
            stats[suffix_key] += 1
            if parsed is None:
                errors.append(f'Unparseable {kind} filename: {path}')
                continue
            if parsed.naming == 'legacy':
                if require_canonical:
                    errors.append(f'Legacy {kind} filename not allowed under canonical contract: {path}')
                continue
            subject_image_id = parsed.subject_image_id
            if subject_image_id not in metadata_subject_ids:
                warnings.append(f'{kind.capitalize()} has canonical ID missing from metadata: {path}')
            if kind == 'image':
                image_ids.add(subject_image_id)
            else:
                mask_ids.add(subject_image_id)

    images_without_masks = sorted(image_ids - mask_ids)
    masks_without_images = sorted(mask_ids - image_ids)
    metadata_without_images = sorted(metadata_subject_ids - image_ids)
    images_without_metadata = sorted(image_ids - metadata_subject_ids)

    for subject_image_id in images_without_masks:
        errors.append(f'No canonical mask found for {subject_image_id}')
    for subject_image_id in masks_without_images:
        errors.append(f'No canonical image found for {subject_image_id}')

    if metadata_without_images:
        warnings.append(f'Metadata subject-image IDs without canonical images: {metadata_without_images[:10]}')
    if images_without_metadata:
        warnings.append(f'Canonical images not represented in metadata: {images_without_metadata[:10]}')

    status = 'PASS' if not errors else 'FAIL'
    return {
        'overall_status': status,
        'stats': stats,
        'images_without_masks': images_without_masks,
        'masks_without_images': masks_without_images,
        'metadata_without_images': metadata_without_images,
        'images_without_metadata': images_without_metadata,
        'errors': errors,
        'warnings': warnings,
    }


def save_contract_report(report: Dict[str, Any], output_path: Path) -> Path:
    """Persist a validation report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    return output_path
