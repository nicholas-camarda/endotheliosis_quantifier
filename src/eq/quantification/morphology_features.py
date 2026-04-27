"""Morphology-aware ROI features for endotheliosis quantification."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from eq.utils.logger import get_logger

MORPHOLOGY_FEATURE_COLUMNS = [
    'morph_mask_area_px',
    'morph_pale_lumen_area_fraction',
    'morph_pale_lumen_candidate_count',
    'morph_pale_lumen_mean_area_fraction',
    'morph_pale_lumen_max_area_fraction',
    'morph_pale_lumen_mean_circularity',
    'morph_pale_lumen_mean_eccentricity',
    'morph_open_space_density',
    'morph_ridge_response_mean',
    'morph_line_density',
    'morph_skeleton_length_per_mask_area',
    'morph_slit_like_object_count',
    'morph_slit_like_area_fraction',
    'morph_border_false_slit_area_fraction',
    'morph_border_false_slit_object_count',
    'morph_slit_boundary_overlap_fraction',
    'morph_ridge_to_lumen_ratio',
    'morph_rbc_like_color_burden',
    'morph_rbc_filled_round_lumen_candidate_count',
    'morph_rbc_filled_lumen_area_fraction',
    'morph_dark_filled_lumen_shape_evidence',
    'morph_nuclear_mesangial_confounder_area_fraction',
    'morph_nuclear_mesangial_confounder_count',
    'morph_slit_excluded_nuclear_overlap_fraction',
    'morph_blur_laplacian_variance',
    'morph_stain_intensity_range',
    'morph_orientation_ambiguity_score',
    'morph_lumen_detectability_score',
]


def _read_roi_arrays(image_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(image_path).convert('RGB'))
    mask = np.asarray(Image.open(mask_path).convert('L')) > 127
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return image, mask


def _component_summaries(binary: np.ndarray, min_area: int) -> list[dict[str, float]]:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    rows: list[dict[str, float]] = []
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        component = labels == label_index
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        circularity = (
            float(4.0 * np.pi * area / (perimeter * perimeter))
            if perimeter > 0
            else 0.0
        )
        yy, xx = np.where(component)
        eccentricity = 0.0
        if len(xx) >= 3:
            cov = np.cov(np.column_stack([xx, yy]), rowvar=False)
            eigenvalues = np.sort(np.linalg.eigvalsh(cov))
            if eigenvalues[-1] > 0:
                eccentricity = float(
                    np.sqrt(max(0.0, 1.0 - eigenvalues[0] / eigenvalues[-1]))
                )
        aspect = float(max(width, height) / max(1, min(width, height)))
        rows.append(
            {
                'area': float(area),
                'circularity': float(np.clip(circularity, 0.0, 1.0)),
                'eccentricity': float(np.clip(eccentricity, 0.0, 1.0)),
                'aspect_ratio': aspect,
                'width': float(width),
                'height': float(height),
            }
        )
    return rows


def _elongated_component_mask(
    binary: np.ndarray,
    *,
    min_area: int,
    min_aspect: float,
    min_eccentricity: float,
    max_circularity: float = 0.55,
) -> np.ndarray:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    elongated = np.zeros(binary.shape, dtype=bool)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label_index
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        circularity = (
            float(4.0 * np.pi * area / (perimeter * perimeter))
            if perimeter > 0
            else 0.0
        )
        yy, xx = np.where(component)
        eccentricity = 0.0
        if len(xx) >= 3:
            cov = np.cov(np.column_stack([xx, yy]), rowvar=False)
            eigenvalues = np.sort(np.linalg.eigvalsh(cov))
            if eigenvalues[-1] > 0:
                eccentricity = float(
                    np.sqrt(max(0.0, 1.0 - eigenvalues[0] / eigenvalues[-1]))
                )
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        if (
            aspect_ratio >= min_aspect or eccentricity >= min_eccentricity
        ) and circularity <= max_circularity:
            elongated |= component
    return elongated


def _compact_component_mask(
    binary: np.ndarray, *, min_area: int, max_area: float, max_aspect: float
) -> np.ndarray:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    compact = np.zeros(binary.shape, dtype=bool)
    for label_index in range(1, num_labels):
        area = float(stats[label_index, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        if aspect_ratio <= max_aspect:
            compact |= labels == label_index
    return compact


def _mask_boundary_band(mask: np.ndarray, radius_px: int = 8) -> np.ndarray:
    """Return the inner boundary band where capsule/crop-edge artifacts are common."""
    if mask.sum() == 0:
        return np.zeros(mask.shape, dtype=bool)
    kernel = np.ones((radius_px * 2 + 1, radius_px * 2 + 1), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask & ~eroded


def _masked_percentile(values: np.ndarray, percentile: float, default: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.percentile(finite, percentile))


def compute_morphology_masks(
    image: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Return deterministic visual masks used by features and review panels."""
    if mask.sum() == 0:
        zeros = np.zeros(mask.shape, dtype=bool)
        return {
            'pale_lumen': zeros,
            'ridge': zeros,
            'slit_like': zeros,
            'border_false_slit': zeros,
            'rbc_like': zeros,
            'nuclear_mesangial': zeros,
            'slit_excluded_nuclear_overlap': zeros,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    masked_gray = gray[mask]
    boundary_band = _mask_boundary_band(mask, radius_px=8)
    interior_mask = mask & ~boundary_band
    pale_threshold = max(
        _masked_percentile(masked_gray, 90, 255.0),
        float(masked_gray.mean() + 0.75 * masked_gray.std()),
    )
    pale = (gray >= pale_threshold) & mask
    pale = cv2.morphologyEx(
        pale.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    ).astype(bool)

    dark_threshold = min(
        _masked_percentile(masked_gray, 18, 0.0),
        float(masked_gray.mean() - 0.70 * masked_gray.std()),
    )
    dark = (gray <= dark_threshold) & mask
    dark = cv2.morphologyEx(
        dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    ).astype(bool)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(bool) & mask
    ridge = edges & mask

    line_min_area = max(5, int(mask.sum() * 0.00015))
    dark_line_seed = dark | (edges & (gray <= _masked_percentile(masked_gray, 58, 0.0)))
    dark_line_seed = cv2.morphologyEx(
        dark_line_seed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    dark_slits = _elongated_component_mask(
        dark_line_seed,
        min_area=line_min_area,
        min_aspect=2.0,
        min_eccentricity=0.88,
        max_circularity=0.70,
    )
    pale_slits = _elongated_component_mask(
        pale,
        min_area=line_min_area,
        min_aspect=3.0,
        min_eccentricity=0.93,
        max_circularity=0.45,
    )
    raw_slit_like = (dark_slits | pale_slits) & mask
    border_false_slit = raw_slit_like & boundary_band
    slit_like = raw_slit_like & interior_mask
    pale = pale & ~slit_like

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    red = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    blue = image[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    saturation_values = saturation[mask]
    hematoxylin_like = (
        mask
        & dark
        & (saturation > np.percentile(saturation_values, 58))
        & (blue >= green * 1.05)
        & (red >= green * 1.02)
    )
    nuclear_mesangial = _compact_component_mask(
        hematoxylin_like,
        min_area=max(5, int(mask.sum() * 0.00008)),
        max_area=max(30.0, float(mask.sum()) * 0.018),
        max_aspect=4.0,
    )

    slit_excluded_nuclear_overlap = slit_like & nuclear_mesangial
    slit_like = slit_like & ~nuclear_mesangial
    pale = pale & ~nuclear_mesangial

    rbc_like = (
        mask
        & ~pale
        & ~slit_like
        & ~nuclear_mesangial
        & (red > green * 1.10)
        & (red > blue * 1.16)
        & (saturation > np.percentile(saturation_values, 65))
        & (gray <= _masked_percentile(masked_gray, 92, 255.0))
    )
    rbc_like = cv2.morphologyEx(
        rbc_like.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    ).astype(bool)

    return {
        'pale_lumen': pale,
        'ridge': ridge,
        'slit_like': slit_like,
        'border_false_slit': border_false_slit,
        'rbc_like': rbc_like,
        'nuclear_mesangial': nuclear_mesangial,
        'slit_excluded_nuclear_overlap': slit_excluded_nuclear_overlap,
    }


def _empty_feature_row() -> dict[str, float | str]:
    row: dict[str, float | str] = {column: 0.0 for column in MORPHOLOGY_FEATURE_COLUMNS}
    row['morphology_feature_status'] = 'missing_roi'
    return row


def _extract_feature_row(image_path: str, mask_path: str) -> dict[str, float | str]:
    image_file = Path(image_path) if image_path else None
    mask_file = Path(mask_path) if mask_path else None
    if (
        image_file is None
        or mask_file is None
        or not image_file.is_file()
        or not mask_file.is_file()
    ):
        return _empty_feature_row()
    image, mask = _read_roi_arrays(image_path, mask_path)
    mask_area = int(mask.sum())
    if mask_area == 0:
        row = _empty_feature_row()
        row['morphology_feature_status'] = 'empty_mask'
        return row

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    masks = compute_morphology_masks(image, mask)
    pale = masks['pale_lumen']
    ridge = masks['ridge']
    slit_like = masks['slit_like']
    border_false_slit = masks['border_false_slit']
    rbc_like = masks['rbc_like']
    nuclear_mesangial = masks['nuclear_mesangial']
    slit_excluded_nuclear_overlap = masks['slit_excluded_nuclear_overlap']

    min_component_area = max(8, int(mask_area * 0.002))
    pale_components = _component_summaries(pale, min_component_area)
    rbc_components = _component_summaries(rbc_like, min_component_area)
    slit_components = _component_summaries(slit_like, min_component_area)
    border_slit_components = _component_summaries(border_false_slit, min_component_area)
    nuclear_components = _component_summaries(
        nuclear_mesangial, max(5, int(mask_area * 0.00008))
    )

    pale_areas = np.array([item['area'] for item in pale_components], dtype=np.float64)
    pale_circularity = np.array(
        [item['circularity'] for item in pale_components], dtype=np.float64
    )
    pale_eccentricity = np.array(
        [item['eccentricity'] for item in pale_components], dtype=np.float64
    )
    round_rbc_components = [
        item
        for item in rbc_components
        if item['circularity'] >= 0.35 and item['aspect_ratio'] <= 2.5
    ]
    dark_round_components = [
        item
        for item in _component_summaries(
            (gray < np.percentile(gray[mask], 30)) & mask, min_component_area
        )
        if item['circularity'] >= 0.25 and item['aspect_ratio'] <= 3.0
    ]
    ridge_values = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    ridge_response = np.abs(ridge_values)[mask]
    pale_fraction = float(pale.sum() / mask_area)
    rbc_fraction = float(rbc_like.sum() / mask_area)
    slit_fraction = float(slit_like.sum() / mask_area)
    border_slit_fraction = float(border_false_slit.sum() / mask_area)
    raw_slit_area = float(slit_like.sum() + border_false_slit.sum())
    slit_boundary_overlap_fraction = (
        float(border_false_slit.sum() / raw_slit_area) if raw_slit_area > 0 else 0.0
    )
    nuclear_fraction = float(nuclear_mesangial.sum() / mask_area)
    line_density = float(ridge.sum() / mask_area)
    lumen_detectability = min(1.0, pale_fraction + rbc_fraction)
    orientation_ambiguity = float(
        np.clip(1.0 - min(1.0, lumen_detectability + line_density), 0.0, 1.0)
    )

    return {
        'morphology_feature_status': 'ok',
        'morph_mask_area_px': float(mask_area),
        'morph_pale_lumen_area_fraction': pale_fraction,
        'morph_pale_lumen_candidate_count': float(len(pale_components)),
        'morph_pale_lumen_mean_area_fraction': float(pale_areas.mean() / mask_area)
        if pale_areas.size
        else 0.0,
        'morph_pale_lumen_max_area_fraction': float(pale_areas.max() / mask_area)
        if pale_areas.size
        else 0.0,
        'morph_pale_lumen_mean_circularity': float(pale_circularity.mean())
        if pale_circularity.size
        else 0.0,
        'morph_pale_lumen_mean_eccentricity': float(pale_eccentricity.mean())
        if pale_eccentricity.size
        else 0.0,
        'morph_open_space_density': pale_fraction,
        'morph_ridge_response_mean': float(np.mean(ridge_response))
        if ridge_response.size
        else 0.0,
        'morph_line_density': line_density,
        'morph_skeleton_length_per_mask_area': line_density,
        'morph_slit_like_object_count': float(len(slit_components)),
        'morph_slit_like_area_fraction': slit_fraction,
        'morph_border_false_slit_area_fraction': border_slit_fraction,
        'morph_border_false_slit_object_count': float(len(border_slit_components)),
        'morph_slit_boundary_overlap_fraction': slit_boundary_overlap_fraction,
        'morph_ridge_to_lumen_ratio': float(line_density / max(pale_fraction, 1e-6)),
        'morph_rbc_like_color_burden': rbc_fraction,
        'morph_rbc_filled_round_lumen_candidate_count': float(
            len(round_rbc_components)
        ),
        'morph_rbc_filled_lumen_area_fraction': rbc_fraction,
        'morph_dark_filled_lumen_shape_evidence': float(
            sum(item['area'] for item in dark_round_components) / mask_area
        ),
        'morph_nuclear_mesangial_confounder_area_fraction': nuclear_fraction,
        'morph_nuclear_mesangial_confounder_count': float(len(nuclear_components)),
        'morph_slit_excluded_nuclear_overlap_fraction': float(
            slit_excluded_nuclear_overlap.sum() / mask_area
        ),
        'morph_blur_laplacian_variance': float(
            cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var()
        ),
        'morph_stain_intensity_range': float(
            np.percentile(gray[mask], 95) - np.percentile(gray[mask], 5)
        ),
        'morph_orientation_ambiguity_score': orientation_ambiguity,
        'morph_lumen_detectability_score': float(lumen_detectability),
    }


def _feature_diagnostics(feature_df: pd.DataFrame) -> dict[str, Any]:
    numeric = feature_df[MORPHOLOGY_FEATURE_COLUMNS].apply(
        pd.to_numeric, errors='coerce'
    )
    variances = numeric.var(axis=0, skipna=True)
    ranges = {
        column: {
            'min': float(numeric[column].min(skipna=True)),
            'max': float(numeric[column].max(skipna=True)),
        }
        for column in numeric.columns
    }
    return {
        'row_count': int(len(feature_df)),
        'subject_count': int(feature_df['subject_id'].nunique())
        if 'subject_id' in feature_df.columns
        else 0,
        'feature_count': int(len(MORPHOLOGY_FEATURE_COLUMNS)),
        'status_counts': feature_df['morphology_feature_status']
        .value_counts()
        .to_dict(),
        'nonfinite_counts': {
            column: int(
                (~np.isfinite(numeric[column].to_numpy(dtype=np.float64))).sum()
            )
            for column in numeric.columns
        },
        'zero_variance_features': [
            str(column) for column, value in variances.items() if np.isclose(value, 0.0)
        ],
        'near_zero_variance_features': [
            str(column)
            for column, value in variances.items()
            if np.isfinite(value) and 0.0 < float(value) < 1e-8
        ],
        'missingness_counts': numeric.isna().sum().astype(int).to_dict(),
        'feature_ranges': ranges,
    }


def write_morphology_feature_tables(
    roi_table: pd.DataFrame, feature_sets_dir: Path, diagnostics_dir: Path
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Write morphology feature tables from ROI image/mask crops."""
    logger = get_logger('eq.quantification.morphology_features')
    feature_sets_dir = Path(feature_sets_dir)
    diagnostics_dir = Path(diagnostics_dir)
    feature_sets_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    source_rows = roi_table.reset_index(drop=True).to_dict(orient='records')
    total_rows = len(source_rows)
    started_at = time.monotonic()
    for row_index, source_row in enumerate(source_rows, start=1):
        feature_row = _extract_feature_row(
            str(source_row.get('roi_image_path', '')),
            str(source_row.get('roi_mask_path', '')),
        )
        for column in [
            'subject_image_id',
            'subject_id',
            'sample_id',
            'image_id',
            'cohort_id',
            'score',
            'roi_image_path',
            'roi_mask_path',
            'raw_image_path',
            'raw_mask_path',
        ]:
            if column in source_row:
                feature_row[column] = source_row.get(column)
        rows.append(feature_row)
        if row_index == 1 or row_index % 25 == 0 or row_index == total_rows:
            elapsed_seconds = max(time.monotonic() - started_at, 1e-6)
            rows_per_minute = row_index / elapsed_seconds * 60.0
            logger.info(
                'Morphology feature extraction progress: %d/%d ROI rows '
                '(elapsed=%.1fs, rate=%.1f rows/min)',
                row_index,
                total_rows,
                elapsed_seconds,
                rows_per_minute,
            )

    feature_df = pd.DataFrame(rows)
    for column in MORPHOLOGY_FEATURE_COLUMNS:
        feature_df[column] = pd.to_numeric(feature_df[column], errors='coerce').fillna(
            0.0
        )

    feature_path = feature_sets_dir / 'morphology_features.csv'
    feature_df.to_csv(feature_path, index=False)
    logger.info('Morphology feature table written -> %s', feature_path)

    subject_path = feature_sets_dir / 'subject_morphology_features.csv'
    subject_df = pd.DataFrame()
    if 'subject_id' in feature_df.columns and not feature_df.empty:
        aggregation: dict[str, Any] = {
            column: 'mean' for column in MORPHOLOGY_FEATURE_COLUMNS
        }
        for column in ['cohort_id', 'score']:
            if column in feature_df.columns:
                aggregation[column] = 'first' if column == 'cohort_id' else 'mean'
        subject_df = feature_df.groupby('subject_id', as_index=False).agg(aggregation)
        subject_df['n_images'] = feature_df.groupby('subject_id').size().to_numpy()
    subject_df.to_csv(subject_path, index=False)
    logger.info('Subject morphology feature table written -> %s', subject_path)

    metadata = {
        'feature_family': 'deterministic_morphology_roi_features',
        'feature_columns': MORPHOLOGY_FEATURE_COLUMNS,
        'review_status': 'requires_operator_adjudication_for_biological_claims',
        'rbc_confounder_policy': 'RBC-like color features are explicit confounder signals, not closed-lumen proof.',
        'nuclear_mesangial_confounder_policy': 'Compact dark purple mesangial/nuclear structures are explicit false-slit confounder signals, not collapsed-lumen proof.',
        'border_false_slit_policy': 'Slit-like candidates touching the inner glomerular boundary band are rejected from true slit features and tracked separately as border false-slit evidence.',
    }
    metadata_path = feature_sets_dir / 'morphology_feature_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info('Morphology feature metadata written -> %s', metadata_path)

    diagnostics_path = diagnostics_dir / 'morphology_feature_diagnostics.json'
    diagnostics_path.write_text(
        json.dumps(_feature_diagnostics(feature_df), indent=2), encoding='utf-8'
    )
    logger.info('Morphology feature diagnostics written -> %s', diagnostics_path)

    return feature_df, {
        'morphology_features': feature_path,
        'morphology_feature_metadata': metadata_path,
        'subject_morphology_features': subject_path,
        'morphology_feature_diagnostics': diagnostics_path,
    }
