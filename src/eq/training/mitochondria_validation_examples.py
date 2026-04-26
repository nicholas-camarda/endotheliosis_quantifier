#!/usr/bin/env python3
"""Generate held-out mitochondria validation example panels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from eq.data_management.model_loading import load_model_safely
from eq.data_management.standard_getters import get_y_full
from eq.training.compare_glomeruli_candidates import (
    _draw_panel_label,
    _error_overlay,
    _front_facing_panel_font,
    _model_input_panel,
    _overlay_mask,
    _raw_context_panel,
    _resize_panel_rgb,
)
from eq.training.promotion_gates import binary_dice_jaccard, binary_precision_recall
from eq.utils.paths import (
    get_runtime_mitochondria_data_path,
    get_runtime_segmentation_evaluation_path,
)

DEFAULT_IMAGE_SIZE = 256
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_IMAGES = 40
DEFAULT_EXAMPLES = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_MIN_TRUTH_FOREGROUND = 0.02
DEFAULT_VISUAL_MIN_TRUTH_FOREGROUND = 0.04
DEFAULT_VISUAL_MAX_EDGE_CONTACT = 0.0
DEFAULT_VISUAL_MIN_CENTER_FRACTION = 0.25
DEFAULT_VISUAL_MAX_CROP_CORRELATION = 0.92
PANEL_TILE_SIZE = 320


def _load_image_and_mask(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(image_path).convert('L'))
    mask = np.asarray(Image.open(get_y_full(image_path)).convert('L'))
    return image, (mask > 0).astype(np.uint8)


def _crop_array(arr: np.ndarray, crop_box: Sequence[int]) -> np.ndarray:
    left, top, right, bottom = [int(value) for value in crop_box]
    return arr[top:bottom, left:right]


def _candidate_crop_boxes(
    mask: np.ndarray, crop_size: int
) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    if height < crop_size or width < crop_size:
        raise ValueError(
            f'Image is smaller than crop_size={crop_size}: shape={mask.shape}'
        )
    boxes: list[tuple[int, int, int, int]] = []
    step = crop_size
    for top in range(0, height - crop_size + 1, step):
        for left in range(0, width - crop_size + 1, step):
            boxes.append((left, top, left + crop_size, top + crop_size))

    ys, xs = np.where(mask.astype(bool))
    if len(xs):
        for quantile in (0.15, 0.35, 0.55, 0.75):
            center_x = int(np.quantile(xs, quantile))
            center_y = int(np.quantile(ys, quantile))
            left = max(0, min(width - crop_size, center_x - crop_size // 2))
            top = max(0, min(height - crop_size, center_y - crop_size // 2))
            boxes.append((left, top, left + crop_size, top + crop_size))

    unique: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        unique.append(box)
    return unique


def _prepare_batch(crops: Sequence[np.ndarray], image_size: int) -> torch.Tensor:
    arrays = []
    for crop in crops:
        rgb = _gray_to_rgb(_model_input_panel(crop, image_size))
        arrays.append(np.asarray(rgb, dtype=np.float32) / 255.0)
    batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


def _predict_crops(
    *,
    learn: Any,
    crops: Sequence[np.ndarray],
    image_size: int,
    threshold: float,
    batch_size: int,
) -> list[np.ndarray]:
    device = next(learn.model.parameters()).device
    predictions: list[np.ndarray] = []
    for start in range(0, len(crops), batch_size):
        batch_crops = crops[start : start + batch_size]
        tensor = _prepare_batch(batch_crops, image_size).to(device)
        with torch.no_grad():
            raw_output = learn.model(tensor)
        if raw_output.shape[1] == 2:
            probabilities = torch.softmax(raw_output, dim=1)[:, 1]
        else:
            probabilities = torch.sigmoid(raw_output).squeeze(1)
        predictions.extend(
            (probabilities.detach().cpu().numpy() > threshold).astype(np.uint8)
        )
    return predictions


def _write_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text('', encoding='utf-8')
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _edge_contact_fraction(mask: np.ndarray) -> float:
    truth = mask.astype(bool)
    total = int(truth.sum())
    if total == 0:
        return 0.0
    edge = np.concatenate([truth[0, :], truth[-1, :], truth[:, 0], truth[:, -1]])
    return float(edge.sum() / total)


def _center_truth_fraction(mask: np.ndarray) -> float:
    truth = mask.astype(bool)
    total = int(truth.sum())
    if total == 0:
        return 0.0
    height, width = truth.shape
    y0 = height // 4
    y1 = height - y0
    x0 = width // 4
    x1 = width - x0
    return float(truth[y0:y1, x0:x1].sum() / total)


def _row_is_visual_candidate(row: dict[str, Any]) -> bool:
    return (
        float(row['truth_foreground_fraction']) >= DEFAULT_VISUAL_MIN_TRUTH_FOREGROUND
        and float(row['edge_contact_fraction']) <= DEFAULT_VISUAL_MAX_EDGE_CONTACT
        and float(row['center_truth_fraction']) >= DEFAULT_VISUAL_MIN_CENTER_FRACTION
    )


def _metric_quantiles(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, float]:
    if not rows:
        return {}
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return {
        'min': float(np.min(values)),
        'p10': float(np.quantile(values, 0.10)),
        'p25': float(np.quantile(values, 0.25)),
        'median': float(np.quantile(values, 0.50)),
        'p75': float(np.quantile(values, 0.75)),
        'p90': float(np.quantile(values, 0.90)),
        'max': float(np.max(values)),
    }


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _mask_centroid(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def _shift_mask(mask: np.ndarray, *, dy: int, dx: int) -> np.ndarray:
    height, width = mask.shape
    shifted = np.zeros_like(mask)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    src_y0 = max(0, -dy)
    src_x0 = max(0, -dx)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _dice_value(truth: np.ndarray, pred: np.ndarray) -> float:
    truth_bool = truth.astype(bool)
    pred_bool = pred.astype(bool)
    denominator = int(truth_bool.sum()) + int(pred_bool.sum())
    if denominator == 0:
        return 1.0
    return float(2 * int((truth_bool & pred_bool).sum()) / denominator)


def _best_shift_check(
    truth: np.ndarray, pred: np.ndarray, *, max_pixels: int = 20
) -> dict[str, Any]:
    best_dice = -1.0
    best_shift = [0, 0]
    for dy in range(-max_pixels, max_pixels + 1):
        for dx in range(-max_pixels, max_pixels + 1):
            shifted_dice = _dice_value(truth, _shift_mask(pred, dy=dy, dx=dx))
            if shifted_dice > best_dice:
                best_dice = shifted_dice
                best_shift = [dy, dx]
    truth_centroid = _mask_centroid(truth)
    pred_centroid = _mask_centroid(pred)
    centroid_delta = None
    if truth_centroid is not None and pred_centroid is not None:
        centroid_delta = [
            pred_centroid[0] - truth_centroid[0],
            pred_centroid[1] - truth_centroid[1],
        ]
    return {
        'truth_bbox': _mask_bbox(truth),
        'prediction_bbox': _mask_bbox(pred),
        'truth_centroid_xy': truth_centroid,
        'prediction_centroid_xy': pred_centroid,
        'centroid_delta_prediction_minus_truth_xy': centroid_delta,
        'unshifted_dice': _dice_value(truth, pred),
        'best_shifted_dice': best_dice,
        'best_shift_dy_dx': best_shift,
    }


def _image_similarity(row_a: dict[str, Any], row_b: dict[str, Any]) -> float:
    crop_a = np.asarray(row_a['_image_crop'], dtype=np.float32).ravel()
    crop_b = np.asarray(row_b['_image_crop'], dtype=np.float32).ravel()
    crop_a = crop_a - float(crop_a.mean())
    crop_b = crop_b - float(crop_b.mean())
    denominator = float(np.linalg.norm(crop_a) * np.linalg.norm(crop_b))
    if denominator == 0:
        return 1.0
    return float(np.dot(crop_a, crop_b) / denominator)


def _is_diverse_from_selected(
    row: dict[str, Any],
    selected: Sequence[dict[str, Any]],
    *,
    max_correlation: float = DEFAULT_VISUAL_MAX_CROP_CORRELATION,
) -> bool:
    for selected_row in selected:
        if (
            'crop_box' in row
            and 'crop_box' in selected_row
            and str(row['crop_box']) == str(selected_row['crop_box'])
        ):
            return False
        if (
            '_image_crop' in row
            and '_image_crop' in selected_row
            and _image_similarity(row, selected_row) > max_correlation
        ):
            return False
    return True


def _select_examples(
    rows: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()

    def sort_key(item: dict[str, Any]) -> tuple[int, float, float, float, float]:
        return (
            0 if _row_is_visual_candidate(item) else 1,
            -float(item['dice']),
            -float(item['jaccard']),
            -float(item['truth_foreground_fraction']),
            float(item['edge_contact_fraction']),
        )

    sorted_rows = sorted(rows, key=sort_key)
    for row in sorted_rows:
        image_name = str(row['image_name'])
        if image_name in used_images:
            continue
        if selected and not _is_diverse_from_selected(row, selected):
            continue
        selected.append(row)
        used_images.add(image_name)
        if len(selected) == count:
            return selected

    for row in sorted_rows:
        image_name = str(row['image_name'])
        if image_name in used_images:
            continue
        selected.append(row)
        used_images.add(image_name)
        if len(selected) == count:
            return selected
    return selected


def _gray_to_rgb(arr: np.ndarray) -> np.ndarray:
    return np.repeat(arr[..., None], 3, axis=2).astype(np.uint8)


def _save_panel(
    *,
    asset_path: Path,
    model_path: Path,
    rows: Sequence[dict[str, Any]],
    image_size: int,
    threshold: float,
) -> None:
    tile = PANEL_TILE_SIZE
    gap = 22
    top = 118
    label_height = 82
    columns = (
        'Raw source',
        f'Input ({image_size}px)',
        'Ground truth',
        'Prediction',
        'Overlay',
    )
    width = (tile * len(columns)) + (gap * (len(columns) + 1))
    height = top + len(rows) * (tile + label_height + gap) + gap
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    title_font = _front_facing_panel_font(22)
    label_font = _front_facing_panel_font(14)
    small_font = _front_facing_panel_font(12)
    draw.text(
        (gap, 18),
        'Mitochondria Held-Out Validation Examples',
        fill=(20, 20, 20),
        font=title_font,
    )
    draw.text(
        (gap, 52),
        f'Source: physical testing split | threshold={threshold:.2f} | model_input={image_size}px',
        fill=(45, 45, 45),
        font=label_font,
    )
    draw.text(
        (gap, 78), f'Model: {model_path.name}', fill=(80, 80, 80), font=small_font
    )
    for column_index, column in enumerate(columns):
        x = gap + column_index * (tile + gap)
        draw.text((x, top - 28), column, fill=(20, 20, 20), font=label_font)

    for row_index, row in enumerate(rows):
        source_image = row['_source_image']
        image_crop = row['_image_crop']
        truth_crop = row['_truth_crop']
        pred_crop = row['_pred_crop']
        crop_box = row['crop_box']
        y = top + row_index * (tile + label_height + gap)
        input_rgb = _gray_to_rgb(_model_input_panel(image_crop, image_size))
        crop_rgb = _gray_to_rgb(image_crop)
        truth_overlay = _overlay_mask(crop_rgb, truth_crop, (0, 255, 0))
        pred_overlay = _overlay_mask(crop_rgb, pred_crop, (255, 0, 0))
        error_overlay = _error_overlay(crop_rgb, truth_crop, pred_crop)
        tiles = (
            _raw_context_panel(
                _gray_to_rgb(source_image),
                [int(value) for value in crop_box.split('|')],
                tile,
            ),
            _resize_panel_rgb(input_rgb, tile),
            _resize_panel_rgb(truth_overlay, tile),
            _resize_panel_rgb(pred_overlay, tile),
            _resize_panel_rgb(error_overlay, tile),
        )
        for column_index, tile_image in enumerate(tiles):
            x = gap + column_index * (tile + gap)
            canvas.paste(tile_image, (x, y))

        label_y = y + tile + 10
        _draw_panel_label(
            draw,
            (gap, label_y),
            (
                f'{row_index + 1}. {row["image_name"]}',
                (
                    f'Dice {float(row["dice"]):.3f} | Jaccard {float(row["jaccard"]):.3f} | '
                    f'Precision {float(row["precision"]):.3f} | Recall {float(row["recall"]):.3f}'
                ),
                (
                    f'truth_fg {float(row["truth_foreground_fraction"]):.3f} | '
                    f'pred_fg {float(row["prediction_foreground_fraction"]):.3f} | '
                    f'crop [{crop_box.replace("|", ", ")}] | input_resize={image_size}px'
                ),
            ),
            small_font,
        )

    draw.text(
        (gap, height - 26),
        'Raw source shows the selected crop box. Overlay colors: green=TP, red=FP, blue=FN.',
        fill=(45, 45, 45),
        font=small_font,
    )
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(asset_path)


def _write_summary_json(
    *,
    output_path: Path,
    model_path: Path,
    data_dir: Path,
    panel_path: Path,
    csv_path: Path,
    scored_rows: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
    image_size: int,
    threshold: float,
    max_images: int,
    min_truth_foreground: float,
) -> None:
    visual_rows = [row for row in scored_rows if _row_is_visual_candidate(row)]
    selected_examples = [
        {key: value for key, value in row.items() if not key.startswith('_')}
        for row in selected_rows
    ]
    summary = {
        'source_split': 'physical_testing',
        'model_path': str(model_path),
        'data_dir': str(data_dir),
        'panel_path': str(panel_path),
        'csv_path': str(csv_path),
        'threshold': threshold,
        'input_size': image_size,
        'max_images': max_images,
        'min_truth_foreground': min_truth_foreground,
        'candidate_crop_count': len(scored_rows),
        'visual_candidate_count': len(visual_rows),
        'visual_selection_rule': {
            'min_truth_foreground_fraction': DEFAULT_VISUAL_MIN_TRUTH_FOREGROUND,
            'max_edge_contact_fraction': DEFAULT_VISUAL_MAX_EDGE_CONTACT,
            'min_center_truth_fraction': DEFAULT_VISUAL_MIN_CENTER_FRACTION,
            'distinct_source_images': True,
            'max_pairwise_crop_correlation_preferred': DEFAULT_VISUAL_MAX_CROP_CORRELATION,
            'primary_sort': 'visual_candidates_first_then_descending_dice_jaccard_with_preferred_diversity',
        },
        'dice_quantiles_all_candidates': _metric_quantiles(scored_rows, 'dice'),
        'dice_quantiles_visual_candidates': _metric_quantiles(visual_rows, 'dice'),
        'truth_foreground_quantiles_all_candidates': _metric_quantiles(
            scored_rows, 'truth_foreground_fraction'
        ),
        'selected_examples': selected_examples,
        'selected_alignment_checks': [
            {
                'image_name': str(row['image_name']),
                'crop_box': str(row['crop_box']),
                **_best_shift_check(row['_truth_crop'], row['_pred_crop']),
            }
            for row in selected_rows
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


def generate_mitochondria_validation_examples(
    *,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    max_images: int = DEFAULT_MAX_IMAGES,
    examples: int = DEFAULT_EXAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_truth_foreground: float = DEFAULT_MIN_TRUTH_FOREGROUND,
) -> dict[str, Path]:
    images_dir = data_dir / 'images'
    if not images_dir.is_dir():
        raise ValueError(f'Mitochondria testing data must contain images/: {data_dir}')
    image_paths = sorted(images_dir.glob('*.tif'))[:max_images]
    if not image_paths:
        raise ValueError(f'No testing images found under {images_dir}')

    learn = load_model_safely(str(model_path), model_type='mito')
    learn.model.eval()

    candidate_rows: list[dict[str, Any]] = []
    crops: list[np.ndarray] = []
    truth_crops: list[np.ndarray] = []
    source_images: list[np.ndarray] = []
    image_crops: list[np.ndarray] = []
    for image_path in image_paths:
        source_image, truth_mask = _load_image_and_mask(image_path)
        for crop_box in _candidate_crop_boxes(truth_mask, image_size):
            truth_crop = _crop_array(truth_mask, crop_box)
            if float(truth_crop.mean()) < min_truth_foreground:
                continue
            image_crop = _crop_array(source_image, crop_box)
            candidate_rows.append(
                {
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'mask_path': str(get_y_full(image_path)),
                    'crop_box': '|'.join(str(value) for value in crop_box),
                    'threshold': threshold,
                    'input_size': image_size,
                    'truth_foreground_fraction': float(truth_crop.mean()),
                }
            )
            crops.append(image_crop)
            truth_crops.append(truth_crop)
            source_images.append(source_image)
            image_crops.append(image_crop)
    if not candidate_rows:
        raise ValueError(
            f'No foreground-containing mitochondria crops found under {data_dir}'
        )

    predictions = _predict_crops(
        learn=learn,
        crops=crops,
        image_size=image_size,
        threshold=threshold,
        batch_size=batch_size,
    )
    scored_rows: list[dict[str, Any]] = []
    for row, truth_crop, pred_crop, source_image, image_crop in zip(
        candidate_rows,
        truth_crops,
        predictions,
        source_images,
        image_crops,
        strict=True,
    ):
        metrics = binary_dice_jaccard(truth_crop, pred_crop)
        metrics.update(binary_precision_recall(truth_crop, pred_crop))
        scored = dict(row)
        scored.update(
            {
                'prediction_foreground_fraction': float(pred_crop.mean()),
                'edge_contact_fraction': _edge_contact_fraction(truth_crop),
                'center_truth_fraction': _center_truth_fraction(truth_crop),
                'dice': metrics['dice'],
                'jaccard': metrics['jaccard'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                '_source_image': source_image,
                '_image_crop': image_crop,
                '_truth_crop': truth_crop,
                '_pred_crop': pred_crop,
            }
        )
        scored_rows.append(scored)

    selected_rows = _select_examples(scored_rows, examples)
    panel_path = output_dir / 'mitochondria_validation_predictions.png'
    csv_path = output_dir / 'mitochondria_validation_predictions.csv'
    summary_path = output_dir / 'mitochondria_validation_summary.json'
    _save_panel(
        asset_path=panel_path,
        model_path=model_path,
        rows=selected_rows,
        image_size=image_size,
        threshold=threshold,
    )
    csv_rows = [
        {key: value for key, value in row.items() if not key.startswith('_')}
        for row in sorted(
            scored_rows,
            key=lambda item: (-float(item['dice']), -float(item['jaccard'])),
        )
    ]
    _write_csv(csv_rows, csv_path)
    _write_summary_json(
        output_path=summary_path,
        model_path=model_path,
        data_dir=data_dir,
        panel_path=panel_path,
        csv_path=csv_path,
        scored_rows=scored_rows,
        selected_rows=selected_rows,
        image_size=image_size,
        threshold=threshold,
        max_images=max_images,
        min_truth_foreground=min_truth_foreground,
    )
    return {
        'panel_path': panel_path,
        'csv_path': csv_path,
        'summary_path': summary_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate held-out mitochondria validation example panels'
    )
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=get_runtime_mitochondria_data_path() / 'testing',
        help='Held-out mitochondria testing root containing images/ and masks/',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=get_runtime_segmentation_evaluation_path(
            'mitochondria_validation_examples'
        ),
    )
    parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--max-images', type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument('--examples', type=int, default=DEFAULT_EXAMPLES)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        '--min-truth-foreground', type=float, default=DEFAULT_MIN_TRUTH_FOREGROUND
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    generate_mitochondria_validation_examples(
        model_path=args.model_path.expanduser(),
        data_dir=args.data_dir.expanduser(),
        output_dir=args.output_dir.expanduser(),
        image_size=args.image_size,
        threshold=args.threshold,
        max_images=args.max_images,
        examples=args.examples,
        batch_size=args.batch_size,
        min_truth_foreground=args.min_truth_foreground,
    )


if __name__ == '__main__':
    main()
