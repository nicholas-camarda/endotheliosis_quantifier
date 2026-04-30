"""Run the MedSAM/manual glomeruli comparison pilot from YAML."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

from eq.data_management.model_loading import load_model_safely
from eq.evaluation.medsam_torch_runtime import (
    medsam_subprocess_extra_env,
    resolve_medsam_torch_python,
)
from eq.inference.prediction_core import create_prediction_core
from eq.quantification.endotheliosis_grade_model import (
    _predict_tiled_segmentation_probability,
)
from eq.training.promotion_gates import binary_dice_jaccard, binary_precision_recall
from eq.utils.execution_logging import (
    direct_execution_log_context,
    runtime_root_environment,
)

LOGGER = logging.getLogger(
    'eq.evaluation.run_medsam_manual_glomeruli_comparison_workflow'
)

WORKFLOW_ID = 'medsam_manual_glomeruli_comparison'
DEFAULT_RUN_ID = 'pilot_medsam_manual_glomeruli_comparison'
DEFAULT_OUTPUT_DIR = 'output/segmentation_evaluation/medsam_manual_glomeruli_comparison'
DEFAULT_MANIFEST_PATH = 'raw_data/cohorts/manifest.csv'
DEFAULT_MEDSAM_PYTHON = '/Users/ncamarda/mambaforge/envs/medsam/bin/python'
DEFAULT_MEDSAM_REPO = '/Users/ncamarda/Projects/MedSAM'
DEFAULT_MEDSAM_CHECKPOINT = (
    '/Users/ncamarda/Projects/MedSAM/work_dir/MedSAM/medsam_vit_b.pth'
)
DEFAULT_TRANSFER_MODEL = (
    '/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/'
    'glomeruli_candidate_transfer-transfer_loss-custom_s1lr1e-3_s2lr_lrfind_e20_b12_lr1e-3_sz256/'
    'glomeruli_candidate_transfer-transfer_loss-custom_s1lr1e-3_s2lr_lrfind_e20_b12_lr1e-3_sz256.pkl'
)
DEFAULT_SCRATCH_MODEL = (
    '/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/scratch/'
    'glomeruli_candidate_no_mito_base-scratch_e25_b12_lr1e-3_sz256/'
    'glomeruli_candidate_no_mito_base-scratch_e25_b12_lr1e-3_sz256.pkl'
)
MANUAL_MASK_LANES = {'manual_mask_core', 'manual_mask_external'}
DEFAULT_METRIC_FIELDS = [
    'method',
    'candidate_artifact',
    'manifest_row_id',
    'cohort_id',
    'lane_assignment',
    'dice',
    'jaccard',
    'precision',
    'recall',
    'pixel_accuracy',
    'manual_foreground_fraction',
    'prediction_foreground_fraction',
    'area_ratio',
    'manual_component_count',
    'prediction_component_count',
    'manual_bbox_x0',
    'manual_bbox_y0',
    'manual_bbox_x1',
    'manual_bbox_y1',
    'prediction_bbox_x0',
    'prediction_bbox_y0',
    'prediction_bbox_x1',
    'prediction_bbox_y1',
]


def _emit(message: str) -> None:
    LOGGER.info('%s', message)
    print(message, flush=True)


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f'Config does not exist: {config_path}')
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Config must be a mapping: {config_path}')
    if payload.get('workflow') != WORKFLOW_ID:
        raise ValueError(f'MedSAM/manual config must use `workflow: {WORKFLOW_ID}`.')
    return payload


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f'{key} must be a mapping')
    return value


def _runtime_root(config: dict[str, Any]) -> Path:
    run_cfg = _mapping(config, 'run')
    env_name = str(run_cfg.get('runtime_root_env') or 'EQ_RUNTIME_ROOT')
    runtime_value = os.environ.get(env_name) or run_cfg.get('runtime_root_default')
    if not runtime_value:
        raise ValueError(
            f'Runtime root is not set. Export {env_name} or set run.runtime_root_default.'
        )
    return Path(str(runtime_value)).expanduser()


def _runtime_path(runtime_root: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return runtime_root / path


def _file_hash(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ''
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_evaluation_output_path(
    runtime_root: Path, raw_output_path: str | Path
) -> Path:
    output_path = _runtime_path(runtime_root, raw_output_path).resolve()
    runtime_resolved = runtime_root.resolve()
    raw_data_root = runtime_resolved / 'raw_data'
    try:
        output_path.relative_to(raw_data_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            f'Generated MedSAM comparison outputs must not be written under raw_data: {output_path}'
        )
    return output_path


def _component_count(mask: np.ndarray) -> int:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, _labels = cv2.connectedComponents(binary, connectivity=8)
    return int(max(0, count - 1))


def _bbox(mask: np.ndarray) -> tuple[int | None, int | None, int | None, int | None]:
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size == 0 or ys.size == 0:
        return None, None, None, None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def load_binary_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def select_pilot_inputs(
    manifest: pd.DataFrame,
    runtime_root: Path,
    *,
    pilot_size: int = 20,
    cohort_targets: dict[str, int] | None = None,
) -> pd.DataFrame:
    if cohort_targets is None:
        half = int(pilot_size // 2)
        cohort_targets = {
            'vegfri_dox': half,
            'lauren_preeclampsia': int(pilot_size - half),
        }

    frame = manifest.fillna('').copy()
    eligible = frame[
        (frame['admission_status'].astype(str) == 'admitted')
        & frame['lane_assignment'].astype(str).isin(MANUAL_MASK_LANES)
        & frame['image_path'].astype(str).str.strip().ne('')
        & frame['mask_path'].astype(str).str.strip().ne('')
    ].copy()
    eligible['_image_abs'] = eligible['image_path'].map(
        lambda value: runtime_root / str(value)
    )
    eligible['_mask_abs'] = eligible['mask_path'].map(
        lambda value: runtime_root / str(value)
    )
    eligible = eligible[
        eligible['_image_abs'].map(Path.exists) & eligible['_mask_abs'].map(Path.exists)
    ].copy()
    eligible = eligible.sort_values(
        ['cohort_id', 'source_sample_id', 'manifest_row_id'], kind='mergesort'
    )

    selected_frames: list[pd.DataFrame] = []
    for cohort_id, target_count in cohort_targets.items():
        cohort_rows = eligible[eligible['cohort_id'].astype(str) == cohort_id].copy()
        if cohort_rows.empty or target_count <= 0:
            continue
        cohort_rows = cohort_rows.drop_duplicates(
            subset=['source_sample_id'], keep='first'
        )
        selected_frames.append(cohort_rows.head(int(target_count)))

    selected = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else eligible.head(0).copy()
    )
    if len(selected) < pilot_size:
        already = set(selected['manifest_row_id'].astype(str))
        fill = eligible[~eligible['manifest_row_id'].astype(str).isin(already)].head(
            pilot_size - len(selected)
        )
        selected = pd.concat([selected, fill], ignore_index=True)

    selected = selected.head(pilot_size).copy().reset_index(drop=True)
    selected['selection_rank'] = np.arange(1, len(selected) + 1)
    selected['selection_reason'] = 'deterministic_cohort_subject_balanced_pilot'
    selected['image_path_resolved'] = selected['_image_abs'].map(str)
    selected['mask_path_resolved'] = selected['_mask_abs'].map(str)
    return selected.drop(columns=['_image_abs', '_mask_abs'], errors='ignore')


def derive_oracle_boxes(
    mask: np.ndarray, *, min_component_area: int = 2000, padding: int = 16
) -> tuple[list[dict[str, int]], list[dict[str, int | str]]]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    boxes: list[dict[str, int]] = []
    skipped: list[dict[str, int | str]] = []
    height, width = binary.shape
    for label_index in range(1, count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        component_index = int(label_index)
        if area < int(min_component_area):
            skipped.append(
                {
                    'component_index': component_index,
                    'component_area': area,
                    'skip_reason': 'component_area_below_minimum',
                }
            )
            continue
        x0 = int(stats[label_index, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_index, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[label_index, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(stats[label_index, cv2.CC_STAT_HEIGHT])
        boxes.append(
            {
                'component_index': component_index,
                'bbox_x0': max(0, x0 - int(padding)),
                'bbox_y0': max(0, y0 - int(padding)),
                'bbox_x1': min(width, x1 + int(padding)),
                'bbox_y1': min(height, y1 + int(padding)),
                'component_area': area,
                'padding': int(padding),
            }
        )
    return boxes, skipped


def metric_row(
    *,
    method: str,
    candidate_artifact: str,
    manifest_row_id: str,
    cohort_id: str,
    lane_assignment: str,
    manual_mask: np.ndarray,
    predicted_mask: np.ndarray,
) -> dict[str, Any]:
    manual = (np.asarray(manual_mask) > 0).astype(np.uint8)
    predicted = (np.asarray(predicted_mask) > 0).astype(np.uint8)
    if manual.shape != predicted.shape:
        raise ValueError(
            f'Manual and predicted masks must have the same shape: {manual.shape} != {predicted.shape}'
        )
    overlap = binary_dice_jaccard(manual, predicted)
    pr = binary_precision_recall(manual, predicted)
    pixel_accuracy = float((manual == predicted).mean()) if manual.size else 0.0
    manual_sum = float(manual.sum())
    predicted_sum = float(predicted.sum())
    manual_bbox = _bbox(manual)
    predicted_bbox = _bbox(predicted)
    row = {
        'method': method,
        'candidate_artifact': candidate_artifact,
        'manifest_row_id': manifest_row_id,
        'cohort_id': cohort_id,
        'lane_assignment': lane_assignment,
        'dice': overlap['dice'],
        'jaccard': overlap['jaccard'],
        'precision': pr['precision'],
        'recall': pr['recall'],
        'pixel_accuracy': pixel_accuracy,
        'manual_foreground_fraction': float(manual.mean()) if manual.size else 0.0,
        'prediction_foreground_fraction': float(predicted.mean())
        if predicted.size
        else 0.0,
        'area_ratio': 0.0 if manual_sum == 0 else float(predicted_sum / manual_sum),
        'manual_component_count': _component_count(manual),
        'prediction_component_count': _component_count(predicted),
        'manual_bbox_x0': manual_bbox[0],
        'manual_bbox_y0': manual_bbox[1],
        'manual_bbox_x1': manual_bbox[2],
        'manual_bbox_y1': manual_bbox[3],
        'prediction_bbox_x0': predicted_bbox[0],
        'prediction_bbox_y0': predicted_bbox[1],
        'prediction_bbox_x1': predicted_bbox[2],
        'prediction_bbox_y1': predicted_bbox[3],
    }
    return {field: row[field] for field in DEFAULT_METRIC_FIELDS}


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(((np.asarray(mask) > 0).astype(np.uint8) * 255)).save(path)


def _draw_overlay(
    *,
    image_path: Path,
    manual_mask: np.ndarray,
    medsam_mask: np.ndarray | None,
    current_masks: dict[str, np.ndarray],
    boxes: list[dict[str, int]],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert('RGB')
    overlay = image.copy().convert('RGBA')
    for mask, color in [(manual_mask, (0, 255, 0, 90)), (medsam_mask, (255, 0, 0, 90))]:
        if mask is None:
            continue
        rgba = Image.new('RGBA', image.size, (0, 0, 0, 0))
        alpha = (np.asarray(mask) > 0).astype(np.uint8) * color[3]
        color_arr = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
        color_arr[..., 0] = color[0]
        color_arr[..., 1] = color[1]
        color_arr[..., 2] = color[2]
        color_arr[..., 3] = alpha
        rgba = Image.fromarray(color_arr, mode='RGBA')
        overlay = Image.alpha_composite(overlay, rgba)
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        draw.rectangle(
            [box['bbox_x0'], box['bbox_y0'], box['bbox_x1'], box['bbox_y1']],
            outline=(0, 0, 255, 255),
            width=3,
        )
    y = 8
    draw.text(
        (8, y), 'manual=green medsam=red oracle_box=blue', fill=(255, 255, 255, 255)
    )
    for name in sorted(current_masks):
        y += 16
        draw.text((8, y), f'current mask available: {name}', fill=(255, 255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert('RGB').save(output_path)


def _predict_tiled_current_mask(
    *,
    model: Any,
    image_path: Path,
    threshold: float,
    tile_size: int,
    stride: int,
    expected_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = Image.open(image_path).convert('RGB')
    probability, audit = _predict_tiled_segmentation_probability(
        model=model,
        image=image,
        tile_size=tile_size,
        stride=stride,
        expected_size=expected_size,
    )
    mask = (probability >= float(threshold)).astype(np.uint8)
    audit.update({'threshold': float(threshold)})
    return mask, audit


def _medsam_batch_script() -> str:
    return r"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import sam_model_registry
from skimage import io, transform

payload = json.loads(Path("__INPUT_JSON__").read_text(encoding="utf-8"))
device_str = str(payload["device"]).strip().lower()

if device_str == "mps":
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MedSAM batch inference requested device=mps but torch.backends.mps.is_available() is False"
        )
elif device_str == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "MedSAM batch inference requested device=cuda but torch.cuda.is_available() is False"
        )
device = torch.device(device_str)
model = sam_model_registry["vit_b"](checkpoint=payload["checkpoint"])
model = model.to(device)
model.eval()
failures = []
for item in payload["items"]:
    try:
        image_np = io.imread(item["image_path"])
        if image_np.ndim == 2:
            image_3c = np.repeat(image_np[:, :, None], 3, axis=-1)
        else:
            image_3c = image_np[:, :, :3]
        height = int(item["height"])
        width = int(item["width"])
        img_1024 = transform.resize(
            image_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.image_encoder(tensor)
        union = np.zeros((height, width), dtype=np.uint8)
        for box in item["boxes"]:
            box_np = np.array([[box["bbox_x0"], box["bbox_y0"], box["bbox_x1"], box["bbox_y1"]]], dtype=np.float32)
            box_1024 = box_np / np.array([width, height, width, height], dtype=np.float32) * 1024
            box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=embedding.device)[:, None, :]
            with torch.no_grad():
                sparse, dense = model.prompt_encoder(points=None, boxes=box_torch, masks=None)
                logits, _ = model.mask_decoder(
                    image_embeddings=embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                pred = torch.sigmoid(logits)
                pred = F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=False)
            union = np.maximum(union, (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8))
        Path(item["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(union * 255).save(item["output_path"])
    except Exception as exc:
        failures.append({
            "manifest_row_id": item.get("manifest_row_id", ""),
            "image_path": item.get("image_path", ""),
            "failure_reason": str(exc),
        })
Path(payload["failures_path"]).write_text(json.dumps(failures, indent=2), encoding="utf-8")
"""


def _run_medsam_batch(
    *,
    medsam_python: Path,
    medsam_repo: Path,
    checkpoint: Path,
    device: str,
    items: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    input_json = output_dir / 'medsam_batch_input.json'
    failures_path = output_dir / 'medsam_batch_failures.json'
    payload = {
        'checkpoint': str(checkpoint),
        'device': device,
        'items': items,
        'failures_path': str(failures_path),
    }
    input_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    script = _medsam_batch_script().replace('__INPUT_JSON__', str(input_json))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(medsam_repo)
    env.update(medsam_subprocess_extra_env(device=device))
    result = subprocess.run(
        [str(medsam_python), '-c', script],
        cwd=str(medsam_repo),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (output_dir / 'medsam_stdout.txt').write_text(result.stdout, encoding='utf-8')
    (output_dir / 'medsam_stderr.txt').write_text(result.stderr, encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError(
            f'MedSAM batch inference failed with exit code {result.returncode}: {result.stderr}'
        )
    if failures_path.exists():
        return json.loads(failures_path.read_text(encoding='utf-8'))
    return []


def _preflight(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f'Missing required MedSAM/manual comparison {label}: {path}'
            )


def run_medsam_manual_glomeruli_comparison_workflow(
    config_path: Path, *, dry_run: bool = False
) -> dict[str, Path]:
    config = _load_config(config_path)
    runtime_root = _runtime_root(config)
    run_cfg = _mapping(config, 'run')
    inputs = _mapping(config, 'inputs')
    medsam = _mapping(config, 'medsam')
    current = _mapping(config, 'current_segmenter')
    pilot = _mapping(config, 'pilot')
    tiling = _mapping(config, 'tiling')
    outputs = _mapping(config, 'outputs')
    run_id = str(run_cfg.get('name') or DEFAULT_RUN_ID)
    evaluation_dir = ensure_evaluation_output_path(
        runtime_root, outputs.get('evaluation_dir', f'{DEFAULT_OUTPUT_DIR}/{run_id}')
    )
    manifest_path = _runtime_path(
        runtime_root, inputs.get('manifest_path', DEFAULT_MANIFEST_PATH)
    )
    medsam_python = resolve_medsam_torch_python(config)
    medsam_repo = Path(str(medsam.get('repo') or DEFAULT_MEDSAM_REPO)).expanduser()
    medsam_script = medsam_repo / 'MedSAM_Inference.py'
    checkpoint = Path(
        str(medsam.get('checkpoint') or DEFAULT_MEDSAM_CHECKPOINT)
    ).expanduser()
    transfer_model = Path(
        str(current.get('transfer_model_path') or DEFAULT_TRANSFER_MODEL)
    ).expanduser()
    scratch_model = Path(
        str(current.get('scratch_model_path') or DEFAULT_SCRATCH_MODEL)
    ).expanduser()
    command = [
        sys.executable,
        '-m',
        'eq.evaluation.run_medsam_manual_glomeruli_comparison_workflow',
        '--config',
        str(config_path),
    ]
    if dry_run:
        command.append('--dry-run')

    with (
        runtime_root_environment(runtime_root),
        direct_execution_log_context(
            surface=WORKFLOW_ID,
            config_run_name=run_id,
            runtime_root=runtime_root,
            dry_run=dry_run,
            config_path=config_path,
            command=command,
            workflow=WORKFLOW_ID,
            logger_name='eq',
        ) as log_context,
    ):
        _emit(f'EXECUTION_LOG={log_context.log_path}')
        _emit(f'WORKFLOW={WORKFLOW_ID}')
        _emit(f'EVALUATION_DIR={evaluation_dir}')
        _emit(f'MANIFEST={manifest_path}')

        _preflight(
            {
                'manifest_path': manifest_path,
                'medsam_python': medsam_python,
                'medsam_repo': medsam_repo,
                'medsam_script': medsam_script,
                'medsam_checkpoint': checkpoint,
                'transfer_model_path': transfer_model,
                'scratch_model_path': scratch_model,
            }
        )
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        manifest = pd.read_csv(manifest_path)
        selected = select_pilot_inputs(
            manifest, runtime_root, pilot_size=int(pilot.get('size', 20))
        )
        selected_fields = [
            'manifest_row_id',
            'cohort_id',
            'lane_assignment',
            'source_sample_id',
            'image_path',
            'mask_path',
            'image_path_resolved',
            'mask_path_resolved',
            'selection_rank',
            'selection_reason',
        ]
        selected[selected_fields].to_csv(evaluation_dir / 'inputs.csv', index=False)
        summary: dict[str, Any] = {
            'workflow': WORKFLOW_ID,
            'run_id': run_id,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'config_path': str(config_path),
            'runtime_root': str(runtime_root),
            'evaluation_dir': str(evaluation_dir),
            'manifest_path': str(manifest_path),
            'medsam_python': str(medsam_python),
            'medsam_repo': str(medsam_repo),
            'medsam_checkpoint': str(checkpoint),
            'medsam_checkpoint_hash': _file_hash(checkpoint),
            'current_segmenters': {
                'transfer': str(transfer_model),
                'scratch': str(scratch_model),
            },
            'pilot_row_count': int(len(selected)),
            'dry_run': bool(dry_run),
            'log_path': str(log_context.log_path),
            'interpretation': 'Audit-scoped oracle-prompt MedSAM/manual/current segmenter comparison; not scientific promotion.',
        }
        if dry_run:
            (evaluation_dir / 'summary.json').write_text(
                json.dumps(summary, indent=2), encoding='utf-8'
            )
            return {
                'evaluation_dir': evaluation_dir,
                'summary': evaluation_dir / 'summary.json',
            }

        medsam_mask_dir = evaluation_dir / 'medsam_masks'
        current_mask_dir = evaluation_dir / 'current_segmenter_masks'
        overlay_dir = evaluation_dir / 'overlays'
        for directory in (medsam_mask_dir, current_mask_dir, overlay_dir):
            directory.mkdir(parents=True, exist_ok=True)

        oracle_rows: list[dict[str, Any]] = []
        medsam_items: list[dict[str, Any]] = []
        manual_masks: dict[str, np.ndarray] = {}
        for row in selected.to_dict(orient='records'):
            manual_mask = load_binary_mask(Path(str(row['mask_path_resolved'])))
            manual_masks[str(row['manifest_row_id'])] = manual_mask
            boxes, skipped = derive_oracle_boxes(
                manual_mask,
                min_component_area=int(pilot.get('min_component_area', 2000)),
                padding=int(pilot.get('box_padding', 16)),
            )
            for box in boxes:
                oracle_rows.append(
                    {**{k: row[k] for k in selected_fields}, **box, 'skip_reason': ''}
                )
            for skip in skipped:
                oracle_rows.append({**{k: row[k] for k in selected_fields}, **skip})
            medsam_items.append(
                {
                    'manifest_row_id': str(row['manifest_row_id']),
                    'image_path': str(row['image_path_resolved']),
                    'height': int(manual_mask.shape[0]),
                    'width': int(manual_mask.shape[1]),
                    'boxes': boxes,
                    'output_path': str(
                        medsam_mask_dir
                        / f'{row["manifest_row_id"]}_medsam_oracle_union.png'
                    ),
                }
            )
        _write_csv(evaluation_dir / 'oracle_boxes.csv', oracle_rows)
        prompt_failures = _run_medsam_batch(
            medsam_python=medsam_python,
            medsam_repo=medsam_repo,
            checkpoint=checkpoint,
            device=str(medsam.get('device') or 'cpu'),
            items=medsam_items,
            output_dir=evaluation_dir,
        )
        _write_csv(
            evaluation_dir / 'prompt_failures.csv',
            prompt_failures,
            fieldnames=['manifest_row_id', 'image_path', 'failure_reason'],
        )

        metrics: list[dict[str, Any]] = []
        current_audits: list[dict[str, Any]] = []
        candidate_paths = {'transfer': transfer_model, 'scratch': scratch_model}
        current_models: dict[str, Any] = {}
        for family, model_path in candidate_paths.items():
            _emit(f'Loading current glomeruli candidate once: {family}={model_path}')
            learner = load_model_safely(str(model_path), model_type='glomeruli')
            learner.model.eval()
            current_models[family] = learner.model
        for row in selected.to_dict(orient='records'):
            manifest_row_id = str(row['manifest_row_id'])
            manual_mask = manual_masks[manifest_row_id]
            medsam_mask_path = (
                medsam_mask_dir / f'{manifest_row_id}_medsam_oracle_union.png'
            )
            if medsam_mask_path.exists():
                medsam_mask = load_binary_mask(medsam_mask_path)
                metrics.append(
                    metric_row(
                        method='medsam_oracle',
                        candidate_artifact=str(checkpoint),
                        manifest_row_id=manifest_row_id,
                        cohort_id=str(row['cohort_id']),
                        lane_assignment=str(row['lane_assignment']),
                        manual_mask=manual_mask,
                        predicted_mask=medsam_mask,
                    )
                )
            else:
                medsam_mask = None
            current_masks: dict[str, np.ndarray] = {}
            for family, model_path in candidate_paths.items():
                current_mask, audit = _predict_tiled_current_mask(
                    model=current_models[family],
                    image_path=Path(str(row['image_path_resolved'])),
                    threshold=float(current.get('threshold', 0.75)),
                    tile_size=int(tiling.get('tile_size', 512)),
                    stride=int(tiling.get('stride', 512)),
                    expected_size=int(tiling.get('expected_size', 256)),
                )
                current_masks[family] = current_mask
                current_path = (
                    current_mask_dir / family / f'{manifest_row_id}_{family}.png'
                )
                _write_mask(current_path, current_mask)
                current_audits.append(
                    {
                        'manifest_row_id': manifest_row_id,
                        'candidate_family': family,
                        'model_path': str(model_path),
                        **audit,
                    }
                )
                metrics.append(
                    metric_row(
                        method=f'current_segmenter_{family}',
                        candidate_artifact=str(model_path),
                        manifest_row_id=manifest_row_id,
                        cohort_id=str(row['cohort_id']),
                        lane_assignment=str(row['lane_assignment']),
                        manual_mask=manual_mask,
                        predicted_mask=current_mask,
                    )
                )
            row_boxes = [
                box
                for box in oracle_rows
                if str(box.get('manifest_row_id', '')) == manifest_row_id
                and not str(box.get('skip_reason', ''))
            ]
            _draw_overlay(
                image_path=Path(str(row['image_path_resolved'])),
                manual_mask=manual_mask,
                medsam_mask=medsam_mask,
                current_masks=current_masks,
                boxes=row_boxes,
                output_path=overlay_dir / f'{manifest_row_id}_overlay.png',
            )
        _write_csv(
            evaluation_dir / 'metrics.csv', metrics, fieldnames=DEFAULT_METRIC_FIELDS
        )
        metric_frame = pd.DataFrame(metrics)
        grouped = (
            metric_frame.groupby(
                ['method', 'candidate_artifact', 'cohort_id', 'lane_assignment'],
                dropna=False,
            )[['dice', 'jaccard', 'precision', 'recall', 'pixel_accuracy']]
            .mean()
            .reset_index()
        )
        grouped.to_csv(evaluation_dir / 'metric_by_source.csv', index=False)
        _write_csv(evaluation_dir / 'current_segmenter_audit.csv', current_audits)
        summary.update(
            {
                'dry_run': False,
                'prompt_failure_count': int(len(prompt_failures)),
                'metric_rows': int(len(metrics)),
                'outputs': {
                    'inputs': str(evaluation_dir / 'inputs.csv'),
                    'oracle_boxes': str(evaluation_dir / 'oracle_boxes.csv'),
                    'metrics': str(evaluation_dir / 'metrics.csv'),
                    'metric_by_source': str(evaluation_dir / 'metric_by_source.csv'),
                    'prompt_failures': str(evaluation_dir / 'prompt_failures.csv'),
                    'medsam_masks': str(medsam_mask_dir),
                    'current_segmenter_masks': str(current_mask_dir),
                    'overlays': str(overlay_dir),
                },
                'mean_metrics': metric_frame.groupby('method')[
                    ['dice', 'jaccard', 'precision', 'recall']
                ]
                .mean()
                .to_dict(orient='index'),
            }
        )
        (evaluation_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8'
        )
        return {
            'evaluation_dir': evaluation_dir,
            'summary': evaluation_dir / 'summary.json',
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run MedSAM/manual glomeruli comparison workflow.'
    )
    parser.add_argument(
        '--config', default='configs/medsam_manual_glomeruli_comparison.yaml'
    )
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_medsam_manual_glomeruli_comparison_workflow(
        Path(args.config), dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
