"""Dataset assembly for glomerulus-level quantification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eq.data_management.canonical_naming import parse_image_path
from eq.data_management.metadata_processor import MetadataProcessor


def load_standardized_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load metadata into the canonical long format used by quantification."""
    processor = MetadataProcessor()
    metadata = processor.process_glomeruli_scoring_matrix(metadata_path).copy()
    metadata = metadata.rename(columns={'subject_id': 'subject_image_id'})
    metadata['subject_image_id'] = metadata['subject_image_id'].astype(str)
    metadata['subject_prefix'] = metadata['subject_image_id'].str.split('-').str[0]
    metadata['glomerulus_id'] = metadata['glomerulus_id'].astype(int)
    metadata['score'] = metadata['score'].astype(float)
    return metadata


def _scan_canonical_image_mask_pairs(raw_project_dir: Path) -> pd.DataFrame:
    images_dir = raw_project_dir / 'images'
    records: list[dict[str, object]] = []
    for image_path in sorted(images_dir.rglob('*')):
        if not image_path.is_file():
            continue
        parsed = parse_image_path(image_path, allow_legacy=False)
        if parsed is None:
            continue
        mask_path = (
            raw_project_dir
            / 'masks'
            / parsed.subject_prefix
            / f'{parsed.subject_image_id}_mask{image_path.suffix.lower()}'
        )
        if not mask_path.exists():
            mask_path = None
        records.append(
            {
                'subject_image_id': parsed.subject_image_id,
                'subject_prefix': parsed.subject_prefix,
                'image_path': str(image_path),
                'mask_path': str(mask_path) if mask_path else None,
                'raw_pair_status': 'matched_pair' if mask_path else 'missing_mask',
            }
        )
    return pd.DataFrame.from_records(records)


def build_scored_example_table(
    metadata_path: Path, raw_project_dir: Path, output_dir: Path
) -> dict[str, Path]:
    """Create one row per scored glomerulus with canonical provenance fields."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_standardized_metadata(metadata_path)
    canonical_pairs = _scan_canonical_image_mask_pairs(raw_project_dir)

    scored = metadata.merge(
        canonical_pairs, how='left', on=['subject_image_id', 'subject_prefix']
    )
    scored['join_status'] = scored['raw_pair_status'].fillna('missing_canonical_pair')
    scored['roi_status'] = 'pending'
    scored['embedding_status'] = 'pending'
    scored['roi_image_path'] = pd.NA
    scored['roi_mask_path'] = pd.NA
    scored['roi_assignment_strategy'] = pd.NA

    scored_path = output_dir / 'scored_examples.csv'
    scored.to_csv(scored_path, index=False)

    summary = {
        'total_scored_rows': int(len(scored)),
        'join_status_counts': scored['join_status']
        .value_counts(dropna=False)
        .to_dict(),
        'subjects_with_scores': int(scored['subject_image_id'].nunique()),
    }
    summary_path = output_dir / 'scored_examples_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))

    return {'scored_examples': scored_path, 'summary': summary_path}
