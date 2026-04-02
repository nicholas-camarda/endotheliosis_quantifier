"""Migration helpers for canonicalizing preeclampsia raw data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from eq.core.constants import IMAGE_EXTENSIONS, MASK_EXTENSIONS
from eq.data_management.canonical_naming import (
    canonical_image_name,
    canonical_mask_name,
    parse_image_path,
    parse_mask_path,
    subject_prefix_from_subject_image_id,
)


def _subject_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir()]) if root.exists() else []


def _find_matching_mask(mask_subject_dir: Path, image_stem: str) -> Optional[Path]:
    for suffix in MASK_EXTENSIONS:
        candidate = mask_subject_dir / f'{image_stem}_mask{suffix}'
        if candidate.exists():
            return candidate
    return None


def inventory_raw_project(raw_project_dir: Path) -> pd.DataFrame:
    """Create an auditable inventory of the preeclampsia raw project."""
    images_dir = raw_project_dir / 'images'
    masks_dir = raw_project_dir / 'masks'
    records: list[dict[str, object]] = []

    for subject_dir in _subject_dirs(images_dir):
        mask_subject_dir = masks_dir / subject_dir.name
        for image_path in sorted(subject_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            parsed_image = parse_image_path(image_path, allow_legacy=True)
            matching_mask = _find_matching_mask(mask_subject_dir, image_path.stem)
            parsed_mask = parse_mask_path(matching_mask, allow_legacy=True) if matching_mask is not None else None
            records.append(
                {
                    'subject_prefix': subject_dir.name,
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'image_stem': image_path.stem,
                    'image_suffix': image_path.suffix.lower(),
                    'image_naming_format': parsed_image.naming_format if parsed_image else 'invalid',
                    'mask_path': str(matching_mask) if matching_mask else None,
                    'mask_name': matching_mask.name if matching_mask else None,
                    'mask_suffix': matching_mask.suffix.lower() if matching_mask else None,
                    'mask_naming_format': parsed_mask.naming_format if parsed_mask else None,
                    'has_mask': matching_mask is not None,
                    'current_subject_image_id': parsed_image.subject_image_id if parsed_image else None,
                }
            )

    return pd.DataFrame.from_records(records)


def _read_mapping_table(mapping_file: Optional[Path]) -> pd.DataFrame:
    if mapping_file is None or not mapping_file.exists():
        return pd.DataFrame(columns=['legacy_image_stem', 'canonical_subject_image_id'])

    mapping = pd.read_csv(mapping_file)
    required = {'legacy_image_stem', 'canonical_subject_image_id'}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f'Mapping file is missing required columns: {sorted(missing)}')
    cleaned = mapping.loc[:, ['legacy_image_stem', 'canonical_subject_image_id']].copy()
    cleaned['legacy_image_stem'] = cleaned['legacy_image_stem'].astype(str).str.strip()
    cleaned['canonical_subject_image_id'] = cleaned['canonical_subject_image_id'].astype(str).str.strip()
    return cleaned[cleaned['legacy_image_stem'].ne('')]


def generate_mapping_template(raw_project_dir: Path, output_path: Path) -> Path:
    """Write a mapping template for legacy filenames that need canonical ids."""
    inventory = inventory_raw_project(raw_project_dir)
    template = inventory[inventory['image_naming_format'] == 'legacy'][
        ['subject_prefix', 'image_stem', 'image_name', 'mask_name', 'has_mask']
    ].copy()
    template = template.rename(columns={'image_stem': 'legacy_image_stem'})
    template['canonical_subject_image_id'] = ''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    return output_path


def migrate_raw_project_to_canonical(
    raw_project_dir: Path,
    mapping_file: Optional[Path],
    report_dir: Path,
    dry_run: bool = True,
) -> dict[str, Path]:
    """Rename raw image/mask pairs into canonical form using an explicit mapping file."""
    report_dir.mkdir(parents=True, exist_ok=True)
    inventory = inventory_raw_project(raw_project_dir)
    mapping = _read_mapping_table(mapping_file)
    mapping_lookup = {
        row['legacy_image_stem']: row['canonical_subject_image_id']
        for _, row in mapping.iterrows()
        if row['canonical_subject_image_id']
    }

    report_rows: list[dict[str, object]] = []
    for row in inventory.to_dict(orient='records'):
        image_path = Path(str(row['image_path']))
        image_naming_format = row['image_naming_format']
        mask_path = Path(str(row['mask_path'])) if row['mask_path'] else None
        canonical_subject_image_id = None
        status = 'unresolved'
        reason = None
        target_image_path = None
        target_mask_path = None

        if image_naming_format == 'canonical':
            canonical_subject_image_id = row['current_subject_image_id']
            target_image_path = str(image_path)
            target_mask_path = str(mask_path) if mask_path else None
            status = 'already_canonical'
        elif image_naming_format == 'legacy':
            canonical_subject_image_id = mapping_lookup.get(str(row['image_stem']))
            if not canonical_subject_image_id:
                status = 'unresolved_missing_mapping'
                reason = 'legacy filename is not present in the mapping table'
            elif mask_path is None:
                status = 'unresolved_missing_mask'
                reason = 'image has no matching raw mask'
            else:
                subject_prefix = subject_prefix_from_subject_image_id(canonical_subject_image_id)
                image_target = raw_project_dir / 'images' / subject_prefix / canonical_image_name(
                    canonical_subject_image_id, image_path.suffix
                )
                mask_target = raw_project_dir / 'masks' / subject_prefix / canonical_mask_name(
                    canonical_subject_image_id, mask_path.suffix
                )
                target_image_path = str(image_target)
                target_mask_path = str(mask_target)
                if image_target.exists() and image_target.resolve() != image_path.resolve():
                    status = 'unresolved_image_conflict'
                    reason = 'target canonical image already exists'
                elif mask_target.exists() and mask_target.resolve() != mask_path.resolve():
                    status = 'unresolved_mask_conflict'
                    reason = 'target canonical mask already exists'
                else:
                    status = 'ready' if dry_run else 'renamed'
                    if not dry_run:
                        image_target.parent.mkdir(parents=True, exist_ok=True)
                        mask_target.parent.mkdir(parents=True, exist_ok=True)
                        image_path.rename(image_target)
                        mask_path.rename(mask_target)
        else:
            status = 'invalid_filename'
            reason = 'image filename does not parse as canonical or legacy'

        report_rows.append(
            {
                **row,
                'canonical_subject_image_id': canonical_subject_image_id,
                'target_image_path': target_image_path,
                'target_mask_path': target_mask_path,
                'status': status,
                'reason': reason,
            }
        )

    report_df = pd.DataFrame.from_records(report_rows)
    report_path = report_dir / 'migration_report.csv'
    report_df.to_csv(report_path, index=False)

    summary = {
        'dry_run': dry_run,
        'raw_project_dir': str(raw_project_dir),
        'mapping_file': str(mapping_file) if mapping_file else None,
        'status_counts': report_df['status'].value_counts(dropna=False).to_dict(),
    }
    summary_path = report_dir / 'migration_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        'migration_report': report_path,
        'migration_summary': summary_path,
    }
