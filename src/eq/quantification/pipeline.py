"""Contract-first quantification pipeline for endotheliosis scoring."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error
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


def _find_canonical_path(root: Path, subject_image_id: str, is_mask: bool) -> Optional[Path]:
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


def _component_angle(center_x: float, center_y: float, centroid_x: float, centroid_y: float) -> float:
    dx = centroid_x - center_x
    dy = centroid_y - center_y
    return float((np.arctan2(-dx, -dy) + (2.0 * np.pi)) % (2.0 * np.pi))


def _extract_components(mask_array: np.ndarray, min_area: int = 64) -> list[dict[str, Any]]:
    binary = _threshold_mask(mask_array)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
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
                'distance_from_center': float(np.hypot(centroid_x - center_x, centroid_y - center_y)),
                'angle_from_top_ccw': _component_angle(center_x, center_y, centroid_x, centroid_y),
                'mask': component_mask,
            }
        )

    components.sort(key=lambda item: (item['angle_from_top_ccw'], item['distance_from_center']))
    for rank, component in enumerate(components, start=1):
        component['glomerulus_id'] = rank
    return components


def build_scored_example_table(project_dir: Path, metadata_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
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
                'subject_prefix': subject_prefix_from_subject_image_id(subject_image_id),
                'glomerulus_id': int(row.glomerulus_id),
                'score': float(row.score),
                'raw_image_path': str(image_path) if image_path else '',
                'raw_mask_path': str(mask_path) if mask_path else '',
                'join_status': join_status,
                'roi_status': 'pending' if join_status == 'ok' else 'join_failed',
            }
        )

    scored_table = pd.DataFrame(rows).sort_values(['subject_image_id', 'glomerulus_id']).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_table.to_csv(output_dir / 'scored_examples.csv', index=False)
    return scored_table


def build_image_level_scored_example_table(
    project_dir: Path,
    score_table: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Create one scored example per raw image/mask pair from Label Studio-derived scores."""
    rows: list[dict[str, Any]] = []

    for row in score_table.itertuples(index=False):
        join_status = str(row.join_status)
        score_status = str(row.score_status)
        roi_status = 'pending' if join_status == 'ok' and score_status == 'ok' else 'join_failed'
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

    scored_table = pd.DataFrame(rows).sort_values(['subject_prefix', 'image_name']).reset_index(drop=True)
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
        component_by_id = {int(component['glomerulus_id']): component for component in components}

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
            fill_fraction = float((crop_mask > 0).sum() / crop_mask.size) if crop_mask.size else 0.0

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
    """Extract one ROI crop per scored image using the largest connected mask component."""
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
        components = _extract_components(mask_array, min_area=min_component_area)
        result.at[index, 'roi_component_count'] = len(components)

        if not components:
            result.at[index, 'roi_status'] = 'component_not_found'
            continue

        component = max(components, key=lambda item: (int(item['area']), -float(item['distance_from_center'])))
        result.at[index, 'roi_component_selection'] = (
            'single_component' if len(components) == 1 else 'largest_component'
        )

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

        crop_name = f"{scored_row['subject_image_id']}.png"
        image_crop_path = image_crop_dir / crop_name
        mask_crop_path = mask_crop_dir / crop_name
        Image.fromarray(crop_image).save(image_crop_path)
        Image.fromarray(crop_mask).save(mask_crop_path)

        gray_crop = np.array(Image.fromarray(crop_image).convert('L'))
        quant_metrics = calculate_quantification_metrics(crop_mask, gray_crop)
        fill_fraction = float((crop_mask > 0).sum() / crop_mask.size) if crop_mask.size else 0.0

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
    encoder = _prepare_encoder_for_forward(encoder).to(next(learn.model.parameters()).device)
    encoder.eval()

    device = next(learn.model.parameters()).device
    prediction_core = create_prediction_core(expected_size)

    valid_rows = roi_table[roi_table['roi_status'] == 'ok'].copy().reset_index(drop=True)
    if valid_rows.empty:
        raise ContractPreparationError('No ROI crops were extracted successfully; cannot build embeddings')

    embeddings: list[np.ndarray] = []
    for row in valid_rows.itertuples(index=False):
        roi_image = Image.open(str(row.roi_image_path)).convert('RGB')
        tensor = prediction_core.preprocess_image(roi_image).to(device)
        with torch.no_grad():
            feature_map = _resolve_feature_map(encoder(tensor))
            pooled = F.adaptive_avg_pool2d(feature_map, output_size=1)
        embeddings.append(pooled.flatten().detach().cpu().numpy().astype(np.float32))

    embedding_matrix = np.vstack(embeddings)
    embedding_columns = [f'embedding_{index:04d}' for index in range(embedding_matrix.shape[1])]
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


class OrdinalThresholdModel:
    """Simple cumulative-threshold ordinal classifier built from binary logits."""

    def __init__(self, n_classes: int) -> None:
        if n_classes < 2:
            raise ValueError('OrdinalThresholdModel requires at least two classes')
        self.n_classes = n_classes
        self.models: list[LogisticRegression | float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'OrdinalThresholdModel':
        self.models = []
        for threshold in range(self.n_classes - 1):
            binary_target = (y > threshold).astype(int)
            if binary_target.min() == binary_target.max():
                self.models.append(float(binary_target[0]))
                continue
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            model.fit(x, binary_target)
            self.models.append(model)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        cumulative = []
        for model in self.models:
            if isinstance(model, float):
                cumulative.append(np.full(shape=(x.shape[0],), fill_value=model, dtype=np.float64))
            else:
                cumulative.append(model.predict_proba(x)[:, 1])
        cumulative_probs = np.vstack(cumulative).T if cumulative else np.empty((x.shape[0], 0), dtype=np.float64)

        if cumulative_probs.size:
            for column in range(1, cumulative_probs.shape[1]):
                cumulative_probs[:, column] = np.minimum(
                    cumulative_probs[:, column - 1], cumulative_probs[:, column]
                )

        probabilities = np.zeros((x.shape[0], self.n_classes), dtype=np.float64)
        probabilities[:, 0] = 1.0 - cumulative_probs[:, 0]
        for class_index in range(1, self.n_classes - 1):
            probabilities[:, class_index] = cumulative_probs[:, class_index - 1] - cumulative_probs[:, class_index]
        probabilities[:, -1] = cumulative_probs[:, -1]
        probabilities = np.clip(probabilities, 0.0, 1.0)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probabilities / row_sums

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)


def evaluate_embedding_table(embedding_df: pd.DataFrame, output_dir: Path, n_splits: int = 3) -> Dict[str, Path]:
    """Train and evaluate the first ordinal endotheliosis predictor."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_columns = [column for column in embedding_df.columns if column.startswith('embedding_')]
    if not embedding_columns:
        raise ValueError('Embedding table does not contain embedding columns')

    work_df = embedding_df.copy().reset_index(drop=True)
    work_df['score_class'] = work_df['score'].map(_score_to_class_index)

    x = work_df[embedding_columns].to_numpy(dtype=np.float64)
    y = work_df['score_class'].to_numpy(dtype=np.int64)
    groups = work_df['subject_prefix'].astype(str).to_numpy()

    unique_groups = np.unique(groups)
    split_count = min(max(2, n_splits), len(unique_groups))
    if split_count < 2:
        raise ContractPreparationError('Need at least two subject groups for grouped evaluation')

    group_kfold = GroupKFold(n_splits=split_count)
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold_index, (train_idx, test_idx) in enumerate(group_kfold.split(x, y, groups=groups), start=1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        model = OrdinalThresholdModel(n_classes=len(ALLOWED_SCORE_VALUES)).fit(x_train, y[train_idx])
        pred_class = model.predict(x_test)
        pred_score = _class_index_to_score(pred_class)
        true_score = _class_index_to_score(y[test_idx])
        fold_df = work_df.iloc[test_idx][['subject_image_id', 'subject_prefix', 'glomerulus_id', 'score']].copy()
        fold_df['fold'] = fold_index
        fold_df['predicted_score'] = pred_score
        fold_df['predicted_class'] = pred_class
        predictions.append(fold_df)

        fold_metrics.append(
            {
                'fold': fold_index,
                'num_examples': int(len(test_idx)),
                'mae': float(mean_absolute_error(true_score, pred_score)),
                'accuracy': float(accuracy_score(y[test_idx], pred_class)),
                'within_one_bin_accuracy': float(np.mean(np.abs(y[test_idx] - pred_class) <= 1)),
                'quadratic_weighted_kappa': float(
                    cohen_kappa_score(y[test_idx], pred_class, weights='quadratic')
                ),
            }
        )

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_path = output_dir / 'ordinal_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)

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
            'mae': float(mean_absolute_error(predictions_df['score'], predictions_df['predicted_score'])),
            'accuracy': float(
                accuracy_score(merged_predictions['score_class'], merged_predictions['predicted_class'])
            ),
            'within_one_bin_accuracy': float(
                np.mean(np.abs(merged_predictions['score_class'] - merged_predictions['predicted_class']) <= 1)
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
    metrics_path = _save_json(summary, output_dir / 'ordinal_metrics.json')

    scaler = StandardScaler().fit(x)
    final_model = OrdinalThresholdModel(n_classes=len(ALLOWED_SCORE_VALUES)).fit(scaler.transform(x), y)
    model_path = output_dir / 'ordinal_embedding_model.pkl'
    with model_path.open('wb') as handle:
        pickle.dump(
            {
                'allowed_scores': ALLOWED_SCORE_VALUES.tolist(),
                'embedding_columns': embedding_columns,
                'scaler': scaler,
                'model': final_model,
            },
            handle,
        )

    return {
        'predictions': predictions_path,
        'confusion_matrix': confusion_path,
        'metrics': metrics_path,
        'model': model_path,
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
        validation_summary = json.loads(score_outputs['summary'].read_text(encoding='utf-8'))
        if validation_summary.get('join_status_counts', {}).get('ok', 0) == 0:
            raise ContractPreparationError('No scored image/mask pairs joined successfully from the Label Studio export')

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
                'scored_examples': output_dir / 'scored_examples' / 'scored_examples.csv',
            }

        roi_table = extract_image_level_roi_crops(scored_table, output_dir / 'roi_crops')
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

        model_artifacts = evaluate_embedding_table(embedding_table, output_dir / 'ordinal_model')
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
        metadata_file,
        output_path=metadata_output_dir / 'standardized_metadata.csv',
    )

    migration_plan = build_migration_plan(project_dir, metadata_df, mapping_file=mapping_file)
    migration_plan_path = output_dir / 'contract_migration_plan.csv'
    migration_plan.to_csv(migration_plan_path, index=False)

    if apply_migration:
        migration_plan = apply_migration_plan(migration_plan)
        migration_plan_path = output_dir / 'contract_migration_applied.csv'
        migration_plan.to_csv(migration_plan_path, index=False)

    validation_report = validate_project_contract(project_dir, metadata_df, require_canonical=True)
    validation_path = save_contract_report(validation_report, output_dir / 'canonical_contract_validation.json')

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
        logger.error('Canonical contract validation failed; see %s and %s', migration_plan_path, validation_path)
        if not unresolved.empty:
            unresolved.to_csv(output_dir / 'canonical_contract_unresolved.csv', index=False)
        raise ContractPreparationError(
            'Canonical contract validation failed. Review the migration report and unresolved rows before ROI extraction.'
        )

    scored_table = build_scored_example_table(project_dir, metadata_df, output_dir / 'scored_examples')
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

    model_artifacts = evaluate_embedding_table(embedding_table, output_dir / 'ordinal_model')
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
            raw_project_dir,
            score_table,
            output_dir / 'scored_examples',
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
        metadata_path,
        output_path=metadata_output_dir / 'standardized_metadata.csv',
    )

    migration_plan = build_migration_plan(raw_project_dir, metadata_df, mapping_file=mapping_file)
    migration_plan_path = output_dir / 'contract_migration_plan.csv'
    migration_plan.to_csv(migration_plan_path, index=False)

    if migrate and not dry_run:
        applied_plan = apply_migration_plan(migration_plan)
        migration_plan_path = output_dir / 'contract_migration_applied.csv'
        applied_plan.to_csv(migration_plan_path, index=False)

    validation_report = validate_project_contract(raw_project_dir, metadata_df, require_canonical=True)
    validation_path = save_contract_report(validation_report, output_dir / 'canonical_contract_validation.json')
    scored_table = build_scored_example_table(raw_project_dir, metadata_df, output_dir / 'scored_examples')

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
        outputs['scored_examples'] = output_dir / 'scored_examples' / 'scored_examples.csv'
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
        validation = json.loads(Path(contract_outputs['validation']).read_text(encoding='utf-8'))
        if validation.get('overall_status') != 'PASS':
            raise ContractPreparationError(
                'Canonical contract validation failed. Review the migration plan, validation report, and unresolved rows first.'
            )
    elif 'labelstudio_summary' in contract_outputs:
        summary = json.loads(Path(contract_outputs['labelstudio_summary']).read_text(encoding='utf-8'))
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
