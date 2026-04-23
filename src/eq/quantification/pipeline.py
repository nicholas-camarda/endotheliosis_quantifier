"""Contract-first quantification pipeline for endotheliosis scoring."""

from __future__ import annotations

import json
import pickle
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from eq.core.constants import DEFAULT_IMAGE_SIZE
from eq.data_management.canonical_contract import (
    apply_migration_plan,
    build_migration_plan,
    canonical_image_filename,
    canonical_mask_filename,
    save_contract_report,
    subject_prefix_from_subject_image_id,
    validate_project_contract,
)
from eq.data_management.metadata_processor import MetadataProcessor
from eq.data_management.model_loading import load_model_safely
from eq.evaluation.quantification_metrics import calculate_quantification_metrics
from eq.inference.prediction_core import create_prediction_core
from eq.quantification.labelstudio_scores import (
    discover_label_studio_annotation_source,
    recover_label_studio_score_table,
)
from eq.quantification.migration import generate_mapping_template, inventory_raw_project
from eq.quantification.ordinal import (
    NUMERICAL_INSTABILITY_PATTERNS,
    CanonicalOrdinalClassifier,
    build_grouped_ordinal_cohort_profile,
)
from eq.training.transfer_learning import _get_encoder_module
from eq.utils.logger import get_logger

ALLOWED_SCORE_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64)


class ContractPreparationError(RuntimeError):
    """Raised when the raw project contract is not ready for quantification."""


def _save_json(data: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)
    return output_path


def _score_to_class_index(score: float) -> int:
    matches = np.where(np.isclose(ALLOWED_SCORE_VALUES, float(score)))[0]
    if len(matches) != 1:
        raise ValueError(f'Unsupported endotheliosis score: {score}')
    return int(matches[0])


def _class_index_to_score(index: np.ndarray | int) -> np.ndarray | float:
    if isinstance(index, np.ndarray):
        return ALLOWED_SCORE_VALUES[index]
    return float(ALLOWED_SCORE_VALUES[int(index)])


def _score_probability_column_name(score: float) -> str:
    score_str = str(score).replace('.', '_')
    return f'prob_score_{score_str}'


def _score_label(score: float) -> str:
    return f'{score:g}'


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def _find_canonical_path(
    root: Path, subject_image_id: str, is_mask: bool
) -> Optional[Path]:
    subject_dir = root / subject_prefix_from_subject_image_id(subject_image_id)
    if not subject_dir.exists():
        return None
    for suffix in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
        candidate = subject_dir / (
            canonical_mask_filename(subject_image_id, suffix)
            if is_mask
            else canonical_image_filename(subject_image_id, suffix)
        )
        if candidate.exists():
            return candidate
    return None


def _threshold_mask(mask_array: np.ndarray) -> np.ndarray:
    return (mask_array > 127).astype(np.uint8)


def _component_angle(
    center_x: float, center_y: float, centroid_x: float, centroid_y: float
) -> float:
    dx = centroid_x - center_x
    dy = centroid_y - center_y
    return float((np.arctan2(-dx, -dy) + (2.0 * np.pi)) % (2.0 * np.pi))


def _extract_components(
    mask_array: np.ndarray, min_area: int = 64
) -> list[dict[str, Any]]:
    binary = _threshold_mask(mask_array)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    height, width = binary.shape
    center_x = width / 2.0
    center_y = height / 2.0

    components: list[dict[str, Any]] = []
    for label_index in range(1, num_labels):
        x, y, w, h, area = stats[label_index]
        if int(area) < min_area:
            continue
        centroid_x, centroid_y = centroids[label_index]
        component_mask = (labels == label_index).astype(np.uint8)
        components.append(
            {
                'component_index': label_index,
                'bbox_x': int(x),
                'bbox_y': int(y),
                'bbox_w': int(w),
                'bbox_h': int(h),
                'area': int(area),
                'centroid_x': float(centroid_x),
                'centroid_y': float(centroid_y),
                'distance_from_center': float(
                    np.hypot(centroid_x - center_x, centroid_y - center_y)
                ),
                'angle_from_top_ccw': _component_angle(
                    center_x, center_y, centroid_x, centroid_y
                ),
                'mask': component_mask,
            }
        )

    components.sort(
        key=lambda item: (item['angle_from_top_ccw'], item['distance_from_center'])
    )
    for rank, component in enumerate(components, start=1):
        component['glomerulus_id'] = rank
    return components


def _build_union_mask(
    mask_array: np.ndarray, min_component_area: int = 64
) -> dict[str, Any] | None:
    binary_mask = _threshold_mask(mask_array)
    if not binary_mask.any():
        return None

    components = _extract_components(mask_array, min_area=min_component_area)
    if components:
        union_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        for component in components:
            union_mask = np.maximum(union_mask, component['mask'].astype(np.uint8))
        component_count = len(components)
        largest_component_area = max(int(component['area']) for component in components)
        selection = 'union_mask'
    else:
        union_mask = binary_mask
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        component_count = max(0, int(num_labels) - 1)
        largest_component_area = (
            int(stats[1:, cv2.CC_STAT_AREA].max())
            if num_labels > 1
            else int(binary_mask.sum())
        )
        selection = 'union_mask_all_positive_fallback'

    ys, xs = np.where(union_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    union_area = int(union_mask.sum())
    largest_fraction = float(largest_component_area / union_area) if union_area else 0.0

    return {
        'mask': union_mask,
        'bbox_x0': x0,
        'bbox_y0': y0,
        'bbox_x1': x1,
        'bbox_y1': y1,
        'component_count': int(component_count),
        'component_selection': selection,
        'union_area': union_area,
        'largest_component_area_fraction': largest_fraction,
        'bbox_width': int(x1 - x0),
        'bbox_height': int(y1 - y0),
    }


def build_scored_example_table(
    project_dir: Path, metadata_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Join standardized metadata to canonical raw images and masks."""
    rows: list[dict[str, Any]] = []
    images_root = Path(project_dir) / 'images'
    masks_root = Path(project_dir) / 'masks'

    metadata = metadata_df.rename(columns={'subject_id': 'subject_image_id'}).copy()
    metadata['subject_image_id'] = metadata['subject_image_id'].astype(str)

    for row in metadata.itertuples(index=False):
        subject_image_id = str(row.subject_image_id)
        image_path = _find_canonical_path(images_root, subject_image_id, is_mask=False)
        mask_path = _find_canonical_path(masks_root, subject_image_id, is_mask=True)
        if image_path and mask_path:
            join_status = 'ok'
        elif image_path:
            join_status = 'missing_mask'
        elif mask_path:
            join_status = 'missing_image'
        else:
            join_status = 'missing_image_and_mask'
        rows.append(
            {
                'subject_image_id': subject_image_id,
                'subject_prefix': subject_prefix_from_subject_image_id(
                    subject_image_id
                ),
                'glomerulus_id': int(row.glomerulus_id),
                'score': float(row.score),
                'raw_image_path': str(image_path) if image_path else '',
                'raw_mask_path': str(mask_path) if mask_path else '',
                'join_status': join_status,
                'roi_status': 'pending' if join_status == 'ok' else 'join_failed',
            }
        )

    scored_table = (
        pd.DataFrame(rows)
        .sort_values(['subject_image_id', 'glomerulus_id'])
        .reset_index(drop=True)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_table.to_csv(output_dir / 'scored_examples.csv', index=False)
    return scored_table


def build_image_level_scored_example_table(
    project_dir: Path, score_table: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Create one scored example per raw image/mask pair from Label Studio-derived scores."""
    rows: list[dict[str, Any]] = []

    for row in score_table.itertuples(index=False):
        join_status = str(row.join_status)
        score_status = str(row.score_status)
        roi_status = (
            'pending' if join_status == 'ok' and score_status == 'ok' else 'join_failed'
        )
        rows.append(
            {
                'subject_image_id': str(row.image_stem),
                'image_name': str(row.image_name),
                'subject_prefix': str(row.subject_prefix),
                'glomerulus_id': 1,
                'score': float(row.score) if pd.notna(row.score) else np.nan,
                'raw_image_path': str(row.raw_image_path or ''),
                'raw_mask_path': str(row.raw_mask_path or ''),
                'join_status': join_status,
                'score_status': score_status,
                'score_resolution': str(row.score_resolution),
                'roi_status': roi_status,
            }
        )

    scored_table = (
        pd.DataFrame(rows)
        .sort_values(['subject_prefix', 'image_name'])
        .reset_index(drop=True)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_table.to_csv(output_dir / 'scored_examples.csv', index=False)
    return scored_table


def extract_roi_crops(
    scored_table: pd.DataFrame,
    output_dir: Path,
    padding: int = 32,
    min_component_area: int = 64,
) -> pd.DataFrame:
    """Extract deterministic glomerulus crops from canonical image+mask pairs."""
    output_dir = Path(output_dir)
    image_crop_dir = output_dir / 'images'
    mask_crop_dir = output_dir / 'masks'
    image_crop_dir.mkdir(parents=True, exist_ok=True)
    mask_crop_dir.mkdir(parents=True, exist_ok=True)

    result = scored_table.copy()
    result['roi_image_path'] = ''
    result['roi_mask_path'] = ''
    result['roi_bbox_x0'] = np.nan
    result['roi_bbox_y0'] = np.nan
    result['roi_bbox_x1'] = np.nan
    result['roi_bbox_y1'] = np.nan
    result['roi_area'] = np.nan
    result['roi_fill_fraction'] = np.nan
    result['roi_mean_intensity'] = np.nan
    result['roi_openness_score'] = np.nan

    for subject_image_id, subject_rows in result.groupby('subject_image_id', sort=True):
        first_row = subject_rows.iloc[0]
        if first_row['join_status'] != 'ok':
            continue

        image_path = Path(str(first_row['raw_image_path']))
        mask_path = Path(str(first_row['raw_mask_path']))
        image_array = np.array(Image.open(image_path).convert('RGB'))
        mask_array = np.array(Image.open(mask_path).convert('L'))

        components = _extract_components(mask_array, min_area=min_component_area)
        component_by_id = {
            int(component['glomerulus_id']): component for component in components
        }

        for index, scored_row in subject_rows.iterrows():
            glomerulus_id = int(scored_row['glomerulus_id'])
            component = component_by_id.get(glomerulus_id)
            if component is None:
                result.at[index, 'roi_status'] = 'component_not_found'
                continue

            x = int(component['bbox_x'])
            y = int(component['bbox_y'])
            w = int(component['bbox_w'])
            h = int(component['bbox_h'])
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(image_array.shape[1], x + w + padding)
            y1 = min(image_array.shape[0], y + h + padding)

            crop_image = image_array[y0:y1, x0:x1]
            crop_mask = (component['mask'][y0:y1, x0:x1] * 255).astype(np.uint8)

            crop_name = f'{subject_image_id}_g{glomerulus_id:03d}.png'
            image_crop_path = image_crop_dir / crop_name
            mask_crop_path = mask_crop_dir / crop_name
            Image.fromarray(crop_image).save(image_crop_path)
            Image.fromarray(crop_mask).save(mask_crop_path)

            gray_crop = np.array(Image.fromarray(crop_image).convert('L'))
            quant_metrics = calculate_quantification_metrics(crop_mask, gray_crop)
            fill_fraction = (
                float((crop_mask > 0).sum() / crop_mask.size) if crop_mask.size else 0.0
            )

            result.at[index, 'roi_status'] = 'ok'
            result.at[index, 'roi_image_path'] = str(image_crop_path)
            result.at[index, 'roi_mask_path'] = str(mask_crop_path)
            result.at[index, 'roi_bbox_x0'] = x0
            result.at[index, 'roi_bbox_y0'] = y0
            result.at[index, 'roi_bbox_x1'] = x1
            result.at[index, 'roi_bbox_y1'] = y1
            result.at[index, 'roi_area'] = int(component['area'])
            result.at[index, 'roi_fill_fraction'] = fill_fraction
            result.at[index, 'roi_mean_intensity'] = (
                float(gray_crop[crop_mask > 0].mean()) if (crop_mask > 0).any() else 0.0
            )
            result.at[index, 'roi_openness_score'] = float(quant_metrics.openness_score)

    result.to_csv(output_dir / 'roi_scored_examples.csv', index=False)
    return result


def extract_image_level_roi_crops(
    scored_table: pd.DataFrame,
    output_dir: Path,
    padding: int = 32,
    min_component_area: int = 64,
) -> pd.DataFrame:
    """Extract one union ROI crop per scored image using the full multi-component mask."""
    output_dir = Path(output_dir)
    image_crop_dir = output_dir / 'images'
    mask_crop_dir = output_dir / 'masks'
    image_crop_dir.mkdir(parents=True, exist_ok=True)
    mask_crop_dir.mkdir(parents=True, exist_ok=True)

    result = scored_table.copy()
    result['roi_image_path'] = ''
    result['roi_mask_path'] = ''
    result['roi_bbox_x0'] = np.nan
    result['roi_bbox_y0'] = np.nan
    result['roi_bbox_x1'] = np.nan
    result['roi_bbox_y1'] = np.nan
    result['roi_area'] = np.nan
    result['roi_fill_fraction'] = np.nan
    result['roi_mean_intensity'] = np.nan
    result['roi_openness_score'] = np.nan
    result['roi_component_count'] = np.nan
    result['roi_component_selection'] = ''
    result['roi_union_bbox_width'] = np.nan
    result['roi_union_bbox_height'] = np.nan
    result['roi_largest_component_area_fraction'] = np.nan

    for index, scored_row in result.iterrows():
        if str(scored_row.get('join_status', '')) != 'ok':
            continue
        if str(scored_row.get('score_status', 'ok')) != 'ok':
            result.at[index, 'roi_status'] = 'missing_score'
            continue

        image_path = Path(str(scored_row['raw_image_path']))
        mask_path = Path(str(scored_row['raw_mask_path']))
        image_array = np.array(Image.open(image_path).convert('RGB'))
        mask_array = np.array(Image.open(mask_path).convert('L'))
        union = _build_union_mask(mask_array, min_component_area=min_component_area)
        if union is None:
            result.at[index, 'roi_status'] = 'component_not_found'
            continue

        result.at[index, 'roi_component_count'] = int(union['component_count'])
        result.at[index, 'roi_component_selection'] = str(union['component_selection'])

        x0 = max(0, int(union['bbox_x0']) - padding)
        y0 = max(0, int(union['bbox_y0']) - padding)
        x1 = min(image_array.shape[1], int(union['bbox_x1']) + padding)
        y1 = min(image_array.shape[0], int(union['bbox_y1']) + padding)

        crop_image = image_array[y0:y1, x0:x1]
        crop_mask = (union['mask'][y0:y1, x0:x1] * 255).astype(np.uint8)

        crop_name = f'{scored_row["subject_image_id"]}.png'
        image_crop_path = image_crop_dir / crop_name
        mask_crop_path = mask_crop_dir / crop_name
        Image.fromarray(crop_image).save(image_crop_path)
        Image.fromarray(crop_mask).save(mask_crop_path)

        gray_crop = np.array(Image.fromarray(crop_image).convert('L'))
        quant_metrics = calculate_quantification_metrics(crop_mask, gray_crop)
        fill_fraction = (
            float((crop_mask > 0).sum() / crop_mask.size) if crop_mask.size else 0.0
        )

        result.at[index, 'roi_status'] = 'ok'
        result.at[index, 'roi_image_path'] = str(image_crop_path)
        result.at[index, 'roi_mask_path'] = str(mask_crop_path)
        result.at[index, 'roi_bbox_x0'] = x0
        result.at[index, 'roi_bbox_y0'] = y0
        result.at[index, 'roi_bbox_x1'] = x1
        result.at[index, 'roi_bbox_y1'] = y1
        result.at[index, 'roi_area'] = int(union['union_area'])
        result.at[index, 'roi_fill_fraction'] = fill_fraction
        result.at[index, 'roi_mean_intensity'] = (
            float(gray_crop[crop_mask > 0].mean()) if (crop_mask > 0).any() else 0.0
        )
        result.at[index, 'roi_openness_score'] = float(quant_metrics.openness_score)
        result.at[index, 'roi_union_bbox_width'] = int(union['bbox_width'])
        result.at[index, 'roi_union_bbox_height'] = int(union['bbox_height'])
        result.at[index, 'roi_largest_component_area_fraction'] = float(
            union['largest_component_area_fraction']
        )

    result.to_csv(output_dir / 'roi_scored_examples.csv', index=False)
    return result


def _resolve_feature_map(encoded_output: Any) -> torch.Tensor:
    if isinstance(encoded_output, torch.Tensor):
        return encoded_output
    if isinstance(encoded_output, (list, tuple)):
        for item in reversed(encoded_output):
            if isinstance(item, torch.Tensor):
                return item
        raise TypeError('Encoder output did not contain a tensor feature map')
    if isinstance(encoded_output, dict):
        for key in ('out', 'features', 'last_hidden_state'):
            value = encoded_output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f'Unsupported encoder output type: {type(encoded_output)}')


def _prepare_encoder_for_forward(encoder: torch.nn.Module) -> torch.nn.Module:
    """Wrap encoder-like containers into a callable module for feature extraction."""
    if isinstance(encoder, torch.nn.ModuleList):
        if len(encoder) == 0:
            raise ValueError('Encoder ModuleList is empty')
        return encoder[0]
    return encoder


def extract_embedding_table(
    roi_table: pd.DataFrame,
    segmentation_model_path: Path,
    output_dir: Path,
    expected_size: int = DEFAULT_IMAGE_SIZE,
) -> pd.DataFrame:
    """Extract frozen encoder embeddings for ROI crops."""
    logger = get_logger('eq.quantification.embeddings')
    learn = load_model_safely(str(segmentation_model_path), model_type='glomeruli')
    learn.model.eval()
    encoder = _get_encoder_module(learn.model)
    if encoder is None:
        raise RuntimeError('Could not resolve encoder module from segmentation learner')
    encoder = _prepare_encoder_for_forward(encoder).to(
        next(learn.model.parameters()).device
    )
    encoder.eval()

    device = next(learn.model.parameters()).device
    prediction_core = create_prediction_core(expected_size)

    valid_rows = (
        roi_table[roi_table['roi_status'] == 'ok'].copy().reset_index(drop=True)
    )
    if valid_rows.empty:
        raise ContractPreparationError(
            'No ROI crops were extracted successfully; cannot build embeddings'
        )

    embeddings: list[np.ndarray] = []
    for row in valid_rows.itertuples(index=False):
        roi_image = Image.open(str(row.roi_image_path)).convert('RGB')
        tensor = prediction_core.preprocess_image(roi_image).to(device)
        with torch.no_grad():
            feature_map = _resolve_feature_map(encoder(tensor))
            pooled = F.adaptive_avg_pool2d(feature_map, output_size=1)
        embeddings.append(pooled.flatten().detach().cpu().numpy().astype(np.float32))

    embedding_matrix = np.vstack(embeddings)
    embedding_columns = [
        f'embedding_{index:04d}' for index in range(embedding_matrix.shape[1])
    ]
    embedding_df = pd.concat(
        [
            valid_rows.reset_index(drop=True),
            pd.DataFrame(embedding_matrix, columns=embedding_columns),
        ],
        axis=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_df.to_csv(output_dir / 'roi_embeddings.csv', index=False)
    _save_json(
        {
            'segmentation_model_path': str(segmentation_model_path),
            'expected_size': int(expected_size),
            'embedding_dim': int(embedding_matrix.shape[1]),
            'pooling': 'adaptive_avg_pool2d_1x1',
            'representation': 'frozen_segmentation_encoder',
        },
        output_dir / 'embedding_metadata.json',
    )
    logger.info('Saved ROI embeddings to %s', output_dir / 'roi_embeddings.csv')
    return embedding_df


def _save_preview_image(
    image: Image.Image, output_path: Path, max_side: int = 900
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    preview.save(output_path)
    return output_path


def _render_mask_overlay(
    raw_image_path: Path, raw_mask_path: Path, bbox: tuple[int, int, int, int]
) -> Image.Image:
    image = Image.open(raw_image_path).convert('RGB')
    mask = np.array(Image.open(raw_mask_path).convert('L'))
    image_array = np.array(image).astype(np.float32)
    positive = mask > 0
    overlay_color = np.array([255.0, 64.0, 64.0], dtype=np.float32)
    image_array[positive] = 0.65 * image_array[positive] + 0.35 * overlay_color
    overlay = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(overlay)
    draw.rectangle(bbox, outline=(32, 200, 255), width=4)
    return overlay


def _build_probability_rows(row: pd.Series) -> str:
    rows: list[str] = []
    for score in ALLOWED_SCORE_VALUES:
        column = _score_probability_column_name(float(score))
        probability = float(row.get(column, 0.0))
        rows.append(
            '<tr>'
            f'<td>{escape(_score_label(float(score)))}</td>'
            f'<td><div class="prob-bar"><span style="width:{probability * 100:.1f}%"></span></div></td>'
            f'<td>{probability:.3f}</td>'
            '</tr>'
        )
    return ''.join(rows)


def _review_narrative(row: pd.Series) -> str:
    return (
        f'Prediction was {float(row["predicted_score"]):.1f} against a true grade of {float(row["score"]):.1f}. '
        f'The full mask contained {int(row["roi_component_count"])} connected component(s) with fill fraction '
        f'{float(row["roi_fill_fraction"]):.3f} and openness score {float(row["roi_openness_score"]):.3f}. '
        'These are descriptive audit signals for review, not faithful feature attribution.'
    )


def _select_review_examples(
    predictions_df: pd.DataFrame, max_examples: int = 7
) -> pd.DataFrame:
    if predictions_df.empty:
        return predictions_df.copy()

    work = predictions_df.copy()
    selected: list[str] = []
    selected_rows: list[pd.Series] = []

    def take_rows(frame: pd.DataFrame, count: int, bucket: str) -> None:
        nonlocal selected, selected_rows
        for _, row in frame.iterrows():
            key = str(row['subject_image_id'])
            if key in selected:
                continue
            picked = row.copy()
            picked['selection_bucket'] = bucket
            selected_rows.append(picked)
            selected.append(key)
            if (
                len(selected_rows) >= max_examples
                or sum(
                    1 for item in selected_rows if item['selection_bucket'] == bucket
                )
                >= count
            ):
                break

    take_rows(
        work.sort_values(
            ['absolute_error', 'entropy', 'top_two_margin'],
            ascending=[False, False, True],
        ),
        count=2,
        bucket='highest_error',
    )
    take_rows(
        work.sort_values(
            ['entropy', 'top_two_margin', 'absolute_error'],
            ascending=[False, True, False],
        ),
        count=2,
        bucket='highest_uncertainty',
    )
    confident_correct = work[
        work['predicted_class'] == work['score_class']
    ].sort_values(['top_two_margin', 'entropy'], ascending=[False, True])
    take_rows(confident_correct, count=2, bucket='confident_correct')

    midpoint = float(np.median(ALLOWED_SCORE_VALUES))
    mid_range = work.assign(
        distance_to_midpoint=np.abs(work['score'] - midpoint),
        expected_distance_to_midpoint=np.abs(work['expected_score'] - midpoint),
    ).sort_values(
        [
            'distance_to_midpoint',
            'expected_distance_to_midpoint',
            'absolute_error',
            'entropy',
        ],
        ascending=[True, True, True, True],
    )
    take_rows(mid_range, count=1, bucket='representative_mid_range')

    if len(selected_rows) < min(max_examples, len(work)):
        filler = work.sort_values(
            ['absolute_error', 'entropy'], ascending=[False, False]
        )
        take_rows(filler, count=max_examples, bucket='additional_review_case')

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return work.head(min(max_examples, len(work))).copy()
    return selected_df.head(min(max_examples, len(work))).reset_index(drop=True)


def generate_html_review_report(
    predictions_df: pd.DataFrame,
    metrics_summary: dict[str, Any],
    output_dir: Path,
    max_examples: int = 7,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    assets_dir = output_dir / 'assets'
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_review_examples(predictions_df, max_examples=max_examples)
    selected_path = output_dir / 'selected_examples.csv'
    selected.to_csv(selected_path, index=False)

    cards: list[str] = []
    for example_index, row in selected.iterrows():
        raw_image_path = Path(str(row['raw_image_path']))
        raw_mask_path = Path(str(row['raw_mask_path']))
        roi_image_path = Path(str(row['roi_image_path']))
        bbox = (
            int(row['roi_bbox_x0']),
            int(row['roi_bbox_y0']),
            int(row['roi_bbox_x1']),
            int(row['roi_bbox_y1']),
        )

        raw_preview_path = assets_dir / f'example_{example_index:02d}_raw.png'
        overlay_preview_path = assets_dir / f'example_{example_index:02d}_overlay.png'
        roi_preview_path = assets_dir / f'example_{example_index:02d}_roi.png'

        raw_image = Image.open(raw_image_path).convert('RGB')
        raw_with_bbox = raw_image.copy()
        raw_draw = ImageDraw.Draw(raw_with_bbox)
        raw_draw.rectangle(bbox, outline=(32, 200, 255), width=4)
        _save_preview_image(raw_with_bbox, raw_preview_path)
        _save_preview_image(
            _render_mask_overlay(raw_image_path, raw_mask_path, bbox),
            overlay_preview_path,
        )
        _save_preview_image(
            Image.open(roi_image_path).convert('RGB'), roi_preview_path, max_side=500
        )

        probability_rows = _build_probability_rows(row)
        cards.append(
            f"""
            <section class="example-card">
              <h2>{escape(str(row['subject_image_id']))} <span class="bucket">{escape(str(row['selection_bucket']))}</span></h2>
              <div class="image-grid">
                <figure>
                  <img src="assets/{raw_preview_path.name}" alt="Raw image with ROI bounding box">
                  <figcaption>Raw image with union ROI bounding box</figcaption>
                </figure>
                <figure>
                  <img src="assets/{overlay_preview_path.name}" alt="Full mask overlay">
                  <figcaption>Full multi-component mask overlay</figcaption>
                </figure>
                <figure>
                  <img src="assets/{roi_preview_path.name}" alt="Union ROI crop">
                  <figcaption>Union ROI crop used for embeddings</figcaption>
                </figure>
              </div>
              <div class="summary-grid">
                <div><strong>True grade</strong><span>{float(row['score']):.1f}</span></div>
                <div><strong>Predicted grade</strong><span>{float(row['predicted_score']):.1f}</span></div>
                <div><strong>Expected score</strong><span>{float(row['expected_score']):.3f}</span></div>
                <div><strong>Absolute error</strong><span>{float(row['absolute_error']):.3f}</span></div>
                <div><strong>Top-two margin</strong><span>{float(row['top_two_margin']):.3f}</span></div>
                <div><strong>Entropy</strong><span>{float(row['entropy']):.3f}</span></div>
              </div>
              <p class="narrative">{escape(_review_narrative(row))}</p>
              <div class="detail-grid">
                <div>
                  <h3>Class Probabilities</h3>
                  <table>
                    <thead><tr><th>Grade</th><th>Probability</th><th>Value</th></tr></thead>
                    <tbody>{probability_rows}</tbody>
                  </table>
                </div>
                <div>
                  <h3>Audit Features</h3>
                  <table>
                    <tbody>
                      <tr><th>Mask components</th><td>{int(row['roi_component_count'])}</td></tr>
                      <tr><th>ROI area</th><td>{int(row['roi_area'])}</td></tr>
                      <tr><th>Fill fraction</th><td>{float(row['roi_fill_fraction']):.3f}</td></tr>
                      <tr><th>Mean intensity</th><td>{float(row['roi_mean_intensity']):.3f}</td></tr>
                      <tr><th>Openness score</th><td>{float(row['roi_openness_score']):.3f}</td></tr>
                      <tr><th>Union bbox</th><td>{int(row['roi_union_bbox_width'])} x {int(row['roi_union_bbox_height'])}</td></tr>
                      <tr><th>Largest-component area fraction</th><td>{float(row['roi_largest_component_area_fraction']):.3f}</td></tr>
                      <tr><th>ROI selection</th><td>{escape(str(row['roi_component_selection']))}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
            """
        )

    overall = metrics_summary.get('overall', {})
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Endotheliosis Review Report</title>
      <style>
        body {{ font-family: "Helvetica Neue", Arial, sans-serif; margin: 2rem auto; max-width: 1200px; color: #1f2933; background: #f7fafc; }}
        h1, h2, h3 {{ color: #102a43; }}
        .overall {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem; }}
        .overall div, .summary-grid div {{ background: white; border-radius: 10px; padding: 0.85rem 1rem; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08); }}
        .overall strong, .summary-grid strong {{ display: block; font-size: 0.85rem; color: #486581; margin-bottom: 0.25rem; }}
        .overall span, .summary-grid span {{ font-size: 1.1rem; font-weight: 600; }}
        .note {{ background: #fff7e6; border-left: 4px solid #d9822b; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; }}
        .example-card {{ background: white; padding: 1.25rem; border-radius: 14px; margin-bottom: 1.25rem; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08); }}
        .image-grid, .detail-grid, .summary-grid {{ display: grid; gap: 1rem; }}
        .image-grid {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 1rem; }}
        .detail-grid {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
        img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid #d9e2ec; background: #f0f4f8; }}
        figure {{ margin: 0; }}
        figcaption {{ font-size: 0.85rem; color: #52606d; margin-top: 0.4rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 0.45rem 0.4rem; border-bottom: 1px solid #e5e7eb; font-size: 0.92rem; }}
        .prob-bar {{ width: 100%; background: #e5e7eb; border-radius: 999px; height: 10px; overflow: hidden; }}
        .prob-bar span {{ display: block; height: 100%; background: linear-gradient(90deg, #2cb1bc, #127fbf); }}
        .bucket {{ font-size: 0.85rem; color: #486581; }}
        .narrative {{ color: #334e68; line-height: 1.5; }}
      </style>
    </head>
    <body>
      <h1>Endotheliosis Example Review</h1>
      <div class="note">
        The model target is the Label Studio image-level grade attached to the full image/mask pair. Probabilities,
        top-two margin, and entropy are confidence proxies for review. The audit features shown below are descriptive
        context, not faithful feature attribution or mechanistic explanation.
      </div>
      <section class="overall">
        <div><strong>Examples reviewed</strong><span>{len(selected)}</span></div>
        <div><strong>Total examples</strong><span>{int(metrics_summary.get('n_examples', len(predictions_df)))}</span></div>
        <div><strong>MAE</strong><span>{float(overall.get('mae', np.nan)):.3f}</span></div>
        <div><strong>Accuracy</strong><span>{float(overall.get('accuracy', np.nan)):.3f}</span></div>
        <div><strong>Within-one-bin</strong><span>{float(overall.get('within_one_bin_accuracy', np.nan)):.3f}</span></div>
        <div><strong>Quadratic weighted kappa</strong><span>{float(overall.get('quadratic_weighted_kappa', np.nan)):.3f}</span></div>
      </section>
      {''.join(cards)}
    </body>
    </html>
    """

    html_path = output_dir / 'ordinal_review.html'
    html_path.write_text(html, encoding='utf-8')
    return {
        'html': html_path,
        'selected_examples': selected_path,
        'assets_dir': assets_dir,
    }


def evaluate_embedding_table(
    embedding_df: pd.DataFrame, output_dir: Path, n_splits: int = 3
) -> Dict[str, Path]:
    """Train and evaluate the first ordinal endotheliosis predictor."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_columns = [
        column for column in embedding_df.columns if column.startswith('embedding_')
    ]
    if not embedding_columns:
        raise ValueError('Embedding table does not contain embedding columns')

    work_df = embedding_df.copy().reset_index(drop=True)
    work_df['score_class'] = work_df['score'].map(_score_to_class_index)

    x = work_df[embedding_columns].to_numpy(dtype=np.float64)
    y = work_df['score_class'].to_numpy(dtype=np.int64)
    groups = work_df['subject_prefix'].astype(str).to_numpy()
    class_indices = np.arange(len(ALLOWED_SCORE_VALUES), dtype=np.int64)

    unique_groups = np.unique(groups)
    split_count = min(max(2, n_splits), len(unique_groups))
    if split_count < 2:
        raise ContractPreparationError(
            'Need at least two subject groups for grouped evaluation'
        )

    group_kfold = GroupKFold(n_splits=split_count)
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    fold_warning_messages: list[dict[str, Any]] = []
    cohort_profile = build_grouped_ordinal_cohort_profile(
        y,
        groups,
        len(embedding_columns),
        classes=class_indices,
        score_values=ALLOWED_SCORE_VALUES,
    )

    for fold_index, (train_idx, test_idx) in enumerate(
        group_kfold.split(x, y, groups=groups), start=1
    ):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]
        model = CanonicalOrdinalClassifier(classes=class_indices).fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        pred_class = probabilities.argmax(axis=1)
        pred_score = _class_index_to_score(pred_class)
        true_score = _class_index_to_score(y_test)
        expected_score = probabilities @ ALLOWED_SCORE_VALUES
        sorted_probabilities = np.sort(probabilities, axis=1)
        top_two_margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        entropy = _entropy(probabilities)

        fold_df = (
            work_df.iloc[test_idx]
            .drop(columns=['score_class', *embedding_columns], errors='ignore')
            .copy()
        )
        fold_df['fold'] = fold_index
        fold_df['score_class'] = y[test_idx]
        fold_df['predicted_score'] = pred_score
        fold_df['predicted_class'] = pred_class
        fold_df['expected_score'] = expected_score
        fold_df['top_two_margin'] = top_two_margin
        fold_df['entropy'] = entropy
        fold_df['absolute_error'] = np.abs(
            fold_df['score'].to_numpy(dtype=np.float64) - pred_score.astype(np.float64)
        )
        fold_df['prediction_error'] = pred_score.astype(np.float64) - fold_df[
            'score'
        ].to_numpy(dtype=np.float64)
        for class_index, score_value in enumerate(ALLOWED_SCORE_VALUES):
            fold_df[_score_probability_column_name(float(score_value))] = probabilities[
                :, class_index
            ]
        predictions.append(fold_df)

        fold_metrics.append(
            {
                'fold': fold_index,
                'num_examples': int(len(test_idx)),
                'mae': float(mean_absolute_error(true_score, pred_score)),
                'accuracy': float(accuracy_score(y_test, pred_class)),
                'within_one_bin_accuracy': float(
                    np.mean(np.abs(y_test - pred_class) <= 1)
                ),
                'quadratic_weighted_kappa': float(
                    cohen_kappa_score(y_test, pred_class, weights='quadratic')
                ),
            }
        )
        fold_warning_messages.append(
            {
                'fold': int(fold_index),
                'messages': model.warning_messages_,
                'train_class_counts': model.training_class_counts_,
                'test_class_counts': {
                    str(int(class_index)): int(np.sum(y_test == class_index))
                    for class_index in class_indices
                },
                'train_threshold_positive_counts': {
                    f'>{int(class_index)}': int(np.sum(y_train > class_index))
                    for class_index in class_indices[:-1]
                },
            }
        )

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_path = output_dir / 'ordinal_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)

    if 'score_class' in predictions_df.columns:
        merged_predictions = predictions_df.copy()
    else:
        merged_predictions = predictions_df.merge(
            work_df[['subject_image_id', 'glomerulus_id', 'score_class']],
            on=['subject_image_id', 'glomerulus_id'],
            how='left',
        )
    confusion = confusion_matrix(
        merged_predictions['score_class'],
        merged_predictions['predicted_class'],
        labels=list(range(len(ALLOWED_SCORE_VALUES))),
    )
    confusion_df = pd.DataFrame(
        confusion,
        index=[f'true_{score:g}' for score in ALLOWED_SCORE_VALUES],
        columns=[f'pred_{score:g}' for score in ALLOWED_SCORE_VALUES],
    )
    confusion_path = output_dir / 'ordinal_confusion_matrix.csv'
    confusion_df.to_csv(confusion_path)

    summary = {
        'n_examples': int(len(work_df)),
        'n_subject_groups': int(len(unique_groups)),
        'n_splits': int(split_count),
        'fold_metrics': fold_metrics,
        'overall': {
            'mae': float(
                mean_absolute_error(
                    predictions_df['score'], predictions_df['predicted_score']
                )
            ),
            'accuracy': float(
                accuracy_score(
                    merged_predictions['score_class'],
                    merged_predictions['predicted_class'],
                )
            ),
            'within_one_bin_accuracy': float(
                np.mean(
                    np.abs(
                        merged_predictions['score_class']
                        - merged_predictions['predicted_class']
                    )
                    <= 1
                )
            ),
            'quadratic_weighted_kappa': float(
                cohen_kappa_score(
                    merged_predictions['score_class'],
                    merged_predictions['predicted_class'],
                    weights='quadratic',
                )
            ),
        },
    }

    scaler = StandardScaler().fit(x)
    final_model = CanonicalOrdinalClassifier(classes=class_indices).fit(
        scaler.transform(x), y
    )
    combined_warning_messages = list(
        dict.fromkeys(
            [
                *[
                    message
                    for fold_entry in fold_warning_messages
                    for message in fold_entry['messages']
                ],
                *final_model.warning_messages_,
            ]
        )
    )
    full_target_class_support = all(
        int(count) > 0 for count in cohort_profile['class_counts'].values()
    )
    certification_blockers: list[str] = []
    if combined_warning_messages:
        certification_blockers.append('numerical_instability')
    if not full_target_class_support:
        certification_blockers.append('missing_target_class_support')
    summary['ordinal_model'] = final_model.metadata()
    summary['cohort_profile'] = cohort_profile
    summary['stability'] = {
        'warning_patterns': list(NUMERICAL_INSTABILITY_PATTERNS),
        'fold_warning_messages': fold_warning_messages,
        'final_model_warning_messages': final_model.warning_messages_,
        'zero_unresolved_warning_gate_passed': not combined_warning_messages,
        'full_target_class_support': full_target_class_support,
        'certification_status': (
            'supported' if not certification_blockers else 'incomplete'
        ),
        'certification_blockers': certification_blockers,
    }
    metrics_path = _save_json(summary, output_dir / 'ordinal_metrics.json')
    model_path = output_dir / 'ordinal_embedding_model.pkl'
    with model_path.open('wb') as handle:
        pickle.dump(
            {
                'allowed_scores': ALLOWED_SCORE_VALUES.tolist(),
                'embedding_columns': embedding_columns,
                'scaler': scaler,
                'model': final_model,
                'model_metadata': final_model.metadata(),
            },
            handle,
        )

    report_artifacts = generate_html_review_report(
        predictions_df=predictions_df,
        metrics_summary=summary,
        output_dir=output_dir / 'review_report',
    )

    return {
        'predictions': predictions_path,
        'confusion_matrix': confusion_path,
        'metrics': metrics_path,
        'model': model_path,
        'review_html': report_artifacts['html'],
        'review_examples': report_artifacts['selected_examples'],
        'review_assets_dir': report_artifacts['assets_dir'],
    }


def run_contract_first_quantification(
    project_dir: Path,
    segmentation_model_path: Path,
    output_dir: Path,
    mapping_file: Optional[Path] = None,
    annotation_source: Optional[str | Path] = None,
    score_source: str = 'auto',
    apply_migration: bool = False,
    stop_after: str = 'model',
) -> Dict[str, Path]:
    """Prepare the quantification contract and run the embedding-first scorer."""
    logger = get_logger('eq.quantification.pipeline')
    project_dir = Path(project_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = output_dir / 'raw_inventory.csv'
    inventory_raw_project(project_dir).to_csv(inventory_path, index=False)
    mapping_template_path = generate_mapping_template(
        project_dir, output_dir / 'legacy_to_canonical_mapping_template.csv'
    )

    if score_source not in {'auto', 'labelstudio', 'spreadsheet'}:
        raise ValueError(f'Unsupported score_source: {score_source}')

    if annotation_source is None and score_source in {'auto', 'labelstudio'}:
        annotation_source = discover_label_studio_annotation_source(project_dir)

    use_labelstudio_scores = score_source == 'labelstudio' or (
        score_source == 'auto' and annotation_source is not None
    )

    if use_labelstudio_scores:
        if annotation_source is None:
            raise FileNotFoundError(
                'No Label Studio annotation source was found. Provide --annotation-source or use --score-source spreadsheet.'
            )

        score_outputs = recover_label_studio_score_table(
            project_dir=project_dir,
            annotation_source=annotation_source,
            output_dir=output_dir / 'labelstudio_scores',
        )
        score_table = pd.read_csv(score_outputs['scores'])
        validation_summary = json.loads(
            score_outputs['summary'].read_text(encoding='utf-8')
        )
        if validation_summary.get('join_status_counts', {}).get('ok', 0) == 0:
            raise ContractPreparationError(
                'No scored image/mask pairs joined successfully from the Label Studio export'
            )

        scored_table = build_image_level_scored_example_table(
            project_dir=project_dir,
            score_table=score_table,
            output_dir=output_dir / 'scored_examples',
        )
        if stop_after == 'contract':
            return {
                'raw_inventory': inventory_path,
                'mapping_template': mapping_template_path,
                'labelstudio_scores': score_outputs['scores'],
                'labelstudio_summary': score_outputs['summary'],
                'duplicate_annotations': score_outputs['duplicates'],
                'scored_examples': output_dir
                / 'scored_examples'
                / 'scored_examples.csv',
            }

        roi_table = extract_image_level_roi_crops(
            scored_table, output_dir / 'roi_crops'
        )
        if stop_after == 'roi':
            return {
                'raw_inventory': inventory_path,
                'mapping_template': mapping_template_path,
                'labelstudio_scores': score_outputs['scores'],
                'labelstudio_summary': score_outputs['summary'],
                'duplicate_annotations': score_outputs['duplicates'],
                'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
            }

        embedding_table = extract_embedding_table(
            roi_table=roi_table,
            segmentation_model_path=Path(segmentation_model_path),
            output_dir=output_dir / 'embeddings',
        )
        if stop_after == 'embeddings':
            return {
                'raw_inventory': inventory_path,
                'mapping_template': mapping_template_path,
                'labelstudio_scores': score_outputs['scores'],
                'labelstudio_summary': score_outputs['summary'],
                'duplicate_annotations': score_outputs['duplicates'],
                'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
                'embeddings': output_dir / 'embeddings' / 'roi_embeddings.csv',
            }

        model_artifacts = evaluate_embedding_table(
            embedding_table, output_dir / 'ordinal_model'
        )
        return {
            'raw_inventory': inventory_path,
            'mapping_template': mapping_template_path,
            'labelstudio_scores': score_outputs['scores'],
            'labelstudio_summary': score_outputs['summary'],
            'duplicate_annotations': score_outputs['duplicates'],
            'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
            'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
            'embeddings': output_dir / 'embeddings' / 'roi_embeddings.csv',
            **model_artifacts,
        }

    metadata_file = project_dir / 'subject_metadata.xlsx'
    if not metadata_file.exists():
        raise FileNotFoundError(f'Metadata file not found: {metadata_file}')

    metadata_output_dir = output_dir / 'metadata'
    processor = MetadataProcessor()
    metadata_df = processor.process_glomeruli_scoring_matrix(
        metadata_file, output_path=metadata_output_dir / 'standardized_metadata.csv'
    )

    migration_plan = build_migration_plan(
        project_dir, metadata_df, mapping_file=mapping_file
    )
    migration_plan_path = output_dir / 'contract_migration_plan.csv'
    migration_plan.to_csv(migration_plan_path, index=False)

    if apply_migration:
        migration_plan = apply_migration_plan(migration_plan)
        migration_plan_path = output_dir / 'contract_migration_applied.csv'
        migration_plan.to_csv(migration_plan_path, index=False)

    validation_report = validate_project_contract(
        project_dir, metadata_df, require_canonical=True
    )
    validation_path = save_contract_report(
        validation_report, output_dir / 'canonical_contract_validation.json'
    )

    if validation_report['overall_status'] != 'PASS':
        unresolved = migration_plan[
            migration_plan['status'].isin(
                {
                    'invalid_name',
                    'unmapped_legacy',
                    'target_collision',
                    'mapped_id_missing_from_metadata',
                    'canonical_missing_from_metadata',
                }
            )
        ]
        logger.error(
            'Canonical contract validation failed; see %s and %s',
            migration_plan_path,
            validation_path,
        )
        if not unresolved.empty:
            unresolved.to_csv(
                output_dir / 'canonical_contract_unresolved.csv', index=False
            )
        raise ContractPreparationError(
            'Canonical contract validation failed. Review the migration report and unresolved rows before ROI extraction.'
        )

    scored_table = build_scored_example_table(
        project_dir, metadata_df, output_dir / 'scored_examples'
    )
    if stop_after == 'contract':
        return {
            'raw_inventory': inventory_path,
            'mapping_template': mapping_template_path,
            'metadata': metadata_output_dir / 'standardized_metadata.csv',
            'migration_plan': migration_plan_path,
            'validation': validation_path,
            'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
        }

    roi_table = extract_roi_crops(scored_table, output_dir / 'roi_crops')
    if stop_after == 'roi':
        return {
            'raw_inventory': inventory_path,
            'mapping_template': mapping_template_path,
            'metadata': metadata_output_dir / 'standardized_metadata.csv',
            'migration_plan': migration_plan_path,
            'validation': validation_path,
            'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
        }

    embedding_table = extract_embedding_table(
        roi_table=roi_table,
        segmentation_model_path=Path(segmentation_model_path),
        output_dir=output_dir / 'embeddings',
    )
    if stop_after == 'embeddings':
        return {
            'raw_inventory': inventory_path,
            'mapping_template': mapping_template_path,
            'metadata': metadata_output_dir / 'standardized_metadata.csv',
            'migration_plan': migration_plan_path,
            'validation': validation_path,
            'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
            'embeddings': output_dir / 'embeddings' / 'roi_embeddings.csv',
        }

    model_artifacts = evaluate_embedding_table(
        embedding_table, output_dir / 'ordinal_model'
    )
    return {
        'raw_inventory': inventory_path,
        'mapping_template': mapping_template_path,
        'metadata': metadata_output_dir / 'standardized_metadata.csv',
        'migration_plan': migration_plan_path,
        'validation': validation_path,
        'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
        'roi_table': output_dir / 'roi_crops' / 'roi_scored_examples.csv',
        'embeddings': output_dir / 'embeddings' / 'roi_embeddings.csv',
        **model_artifacts,
    }


def prepare_quantification_contract(
    raw_project_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    mapping_file: Optional[Path] = None,
    annotation_source: Optional[str | Path] = None,
    score_source: str = 'auto',
    migrate: bool = False,
    dry_run: bool = True,
) -> Dict[str, Path]:
    """Build the contract artifacts needed before embeddings and scoring."""
    raw_project_dir = Path(raw_project_dir)
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = output_dir / 'raw_inventory.csv'
    inventory_raw_project(raw_project_dir).to_csv(inventory_path, index=False)
    mapping_template_path = generate_mapping_template(
        raw_project_dir, output_dir / 'legacy_to_canonical_mapping_template.csv'
    )

    if score_source not in {'auto', 'labelstudio', 'spreadsheet'}:
        raise ValueError(f'Unsupported score_source: {score_source}')

    if annotation_source is None and score_source in {'auto', 'labelstudio'}:
        annotation_source = discover_label_studio_annotation_source(raw_project_dir)

    use_labelstudio_scores = score_source == 'labelstudio' or (
        score_source == 'auto' and annotation_source is not None
    )

    if use_labelstudio_scores:
        if annotation_source is None:
            raise FileNotFoundError(
                'No Label Studio annotation source was found. Provide --annotation-source or use --score-source spreadsheet.'
            )

        score_outputs = recover_label_studio_score_table(
            project_dir=raw_project_dir,
            annotation_source=annotation_source,
            output_dir=output_dir / 'labelstudio_scores',
        )
        score_table = pd.read_csv(score_outputs['scores'])
        scored_table = build_image_level_scored_example_table(
            raw_project_dir, score_table, output_dir / 'scored_examples'
        )
        return {
            'raw_inventory': inventory_path,
            'mapping_template': mapping_template_path,
            'labelstudio_scores': score_outputs['scores'],
            'labelstudio_summary': score_outputs['summary'],
            'duplicate_annotations': score_outputs['duplicates'],
            'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
        }

    metadata_output_dir = output_dir / 'metadata'
    processor = MetadataProcessor()
    metadata_df = processor.process_glomeruli_scoring_matrix(
        metadata_path, output_path=metadata_output_dir / 'standardized_metadata.csv'
    )

    migration_plan = build_migration_plan(
        raw_project_dir, metadata_df, mapping_file=mapping_file
    )
    migration_plan_path = output_dir / 'contract_migration_plan.csv'
    migration_plan.to_csv(migration_plan_path, index=False)

    if migrate and not dry_run:
        applied_plan = apply_migration_plan(migration_plan)
        migration_plan_path = output_dir / 'contract_migration_applied.csv'
        applied_plan.to_csv(migration_plan_path, index=False)

    validation_report = validate_project_contract(
        raw_project_dir, metadata_df, require_canonical=True
    )
    validation_path = save_contract_report(
        validation_report, output_dir / 'canonical_contract_validation.json'
    )
    scored_table = build_scored_example_table(
        raw_project_dir, metadata_df, output_dir / 'scored_examples'
    )

    outputs: Dict[str, Path] = {
        'raw_inventory': inventory_path,
        'mapping_template': mapping_template_path,
        'metadata': metadata_output_dir / 'standardized_metadata.csv',
        'migration_plan': migration_plan_path,
        'validation': validation_path,
        'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
    }
    unresolved = migration_plan[
        migration_plan['status'].isin(
            {
                'invalid_name',
                'unmapped_legacy',
                'target_collision',
                'mapped_id_missing_from_metadata',
                'canonical_missing_from_metadata',
            }
        )
    ]
    if not unresolved.empty:
        unresolved_path = output_dir / 'canonical_contract_unresolved.csv'
        unresolved.to_csv(unresolved_path, index=False)
        outputs['unresolved'] = unresolved_path
    if not scored_table.empty:
        outputs['scored_examples'] = (
            output_dir / 'scored_examples' / 'scored_examples.csv'
        )
    return outputs


def run_endotheliosis_scoring_pipeline(
    raw_project_dir: Path,
    metadata_path: Path,
    segmentation_model_path: Path,
    output_dir: Path,
    mapping_file: Optional[Path] = None,
    annotation_source: Optional[str | Path] = None,
    score_source: str = 'auto',
    migrate: bool = False,
    dry_run: bool = True,
) -> Dict[str, Path]:
    """Run the full contract-first scoring flow using the existing quantification pipeline."""
    output_dir = Path(output_dir)
    contract_outputs = prepare_quantification_contract(
        raw_project_dir=raw_project_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        mapping_file=mapping_file,
        annotation_source=annotation_source,
        score_source=score_source,
        migrate=migrate,
        dry_run=dry_run,
    )

    if 'validation' in contract_outputs:
        validation = json.loads(
            Path(contract_outputs['validation']).read_text(encoding='utf-8')
        )
        if validation.get('overall_status') != 'PASS':
            raise ContractPreparationError(
                'Canonical contract validation failed. Review the migration plan, validation report, and unresolved rows first.'
            )
    elif 'labelstudio_summary' in contract_outputs:
        summary = json.loads(
            Path(contract_outputs['labelstudio_summary']).read_text(encoding='utf-8')
        )
        if summary.get('join_status_counts', {}).get('ok', 0) == 0:
            raise ContractPreparationError(
                'Label Studio score recovery did not produce any joined scored image/mask pairs.'
            )
    else:
        raise ContractPreparationError(
            'Quantification contract preparation did not produce a recognizable validation artifact.'
        )

    return run_contract_first_quantification(
        project_dir=Path(raw_project_dir),
        segmentation_model_path=Path(segmentation_model_path),
        output_dir=output_dir,
        mapping_file=mapping_file,
        annotation_source=annotation_source,
        score_source=score_source,
        apply_migration=migrate and not dry_run,
        stop_after='model',
    )
