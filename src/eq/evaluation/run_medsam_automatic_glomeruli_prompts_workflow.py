"""Run MedSAM automatic-prompt glomeruli pilot from YAML."""

from __future__ import annotations

import argparse
import json
import logging
import os
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
from eq.evaluation.medsam_glomeruli_workflow import (
    DEFAULT_MEDSAM_CHECKPOINT,
    DEFAULT_MEDSAM_REPO,
    DEFAULT_METRIC_FIELDS,
    DEFAULT_SCRATCH_MODEL,
    DEFAULT_TRANSFER_MODEL,
    _file_hash,
    _preflight,
    _run_medsam_batch,
    _runtime_path,
    _runtime_root,
    _write_csv,
    _write_mask,
    ensure_evaluation_output_path,
    load_binary_mask,
    metric_row,
    select_pilot_inputs,
)
from eq.evaluation.medsam_torch_runtime import resolve_medsam_torch_python
from eq.quantification.endotheliosis_grade_model import (
    _predict_tiled_segmentation_probability,
)
from eq.utils.execution_logging import (
    direct_execution_log_context,
    runtime_root_environment,
)

LOGGER = logging.getLogger(
    'eq.evaluation.run_medsam_automatic_glomeruli_prompts_workflow'
)

WORKFLOW_ID = 'medsam_automatic_glomeruli_prompts'
DEFAULT_RUN_ID = 'pilot_medsam_automatic_glomeruli_prompts'
DEFAULT_OUTPUT_DIR = 'output/segmentation_evaluation/medsam_automatic_glomeruli_prompts'
DEFAULT_DERIVED_OUTPUT_DIR = 'output/derived_masks/medsam_automatic_glomeruli'
DEFAULT_MANIFEST_PATH = 'raw_data/cohorts/manifest.csv'
AUTOMATIC_MASK_SOURCE = 'medsam_automatic_glomeruli'
AUTOMATIC_METRIC_FIELDS = [
    'prompt_mode',
    'proposal_threshold',
    'candidate_family',
    'mask_source',
    *DEFAULT_METRIC_FIELDS,
]
PROPOSAL_BOX_FIELDS = [
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
    'candidate_family',
    'candidate_artifact',
    'threshold',
    'component_index',
    'proposal_index',
    'bbox_x0',
    'bbox_y0',
    'bbox_x1',
    'bbox_y1',
    'component_area',
    'decision',
    'decision_reason',
    'left_component_index',
    'right_component_index',
    'merge_iou',
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
        raise ValueError(f'Automatic MedSAM config must use `workflow: {WORKFLOW_ID}`.')
    return payload


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f'{key} must be a mapping')
    return value


def _bbox_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    x0 = max(int(left['bbox_x0']), int(right['bbox_x0']))
    y0 = max(int(left['bbox_y0']), int(right['bbox_y0']))
    x1 = min(int(left['bbox_x1']), int(right['bbox_x1']))
    y1 = min(int(left['bbox_y1']), int(right['bbox_y1']))
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = max(0, int(left['bbox_x1']) - int(left['bbox_x0'])) * max(
        0, int(left['bbox_y1']) - int(left['bbox_y0'])
    )
    right_area = max(0, int(right['bbox_x1']) - int(right['bbox_x0'])) * max(
        0, int(right['bbox_y1']) - int(right['bbox_y0'])
    )
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def _merge_two_boxes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        **left,
        'bbox_x0': min(int(left['bbox_x0']), int(right['bbox_x0'])),
        'bbox_y0': min(int(left['bbox_y0']), int(right['bbox_y0'])),
        'bbox_x1': max(int(left['bbox_x1']), int(right['bbox_x1'])),
        'bbox_y1': max(int(left['bbox_y1']), int(right['bbox_y1'])),
        'component_area': int(left.get('component_area', 0))
        + int(right.get('component_area', 0)),
    }


def _merge_overlapping_boxes(
    boxes: list[dict[str, Any]], *, merge_iou: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged = [dict(box) for box in boxes]
    decisions: list[dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        for left_index in range(len(merged)):
            for right_index in range(left_index + 1, len(merged)):
                iou = _bbox_iou(merged[left_index], merged[right_index])
                if iou < float(merge_iou):
                    continue
                left = merged[left_index]
                right = merged[right_index]
                merged[left_index] = _merge_two_boxes(left, right)
                decisions.append(
                    {
                        'decision': 'merged',
                        'decision_reason': 'bbox_iou_at_or_above_merge_threshold',
                        'left_component_index': left.get('component_index', ''),
                        'right_component_index': right.get('component_index', ''),
                        'merge_iou': iou,
                    }
                )
                del merged[right_index]
                changed = True
                break
            if changed:
                break
    return merged, decisions


def derive_proposal_boxes(
    probability: np.ndarray,
    *,
    threshold: float,
    min_component_area: int,
    max_component_area: int,
    padding: int,
    merge_iou: float,
    max_boxes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    binary = (np.asarray(probability) >= float(threshold)).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    height, width = binary.shape
    boxes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for label_index in range(1, count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        base = {
            'component_index': int(label_index),
            'component_area': area,
            'threshold': float(threshold),
        }
        if area < int(min_component_area):
            decisions.append(
                {
                    **base,
                    'decision': 'skipped',
                    'decision_reason': 'component_area_below_minimum',
                }
            )
            continue
        if area > int(max_component_area):
            decisions.append(
                {
                    **base,
                    'decision': 'skipped',
                    'decision_reason': 'component_area_above_maximum',
                }
            )
            continue
        x0 = int(stats[label_index, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_index, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[label_index, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(stats[label_index, cv2.CC_STAT_HEIGHT])
        boxes.append(
            {
                **base,
                'bbox_x0': max(0, x0 - int(padding)),
                'bbox_y0': max(0, y0 - int(padding)),
                'bbox_x1': min(width, x1 + int(padding)),
                'bbox_y1': min(height, y1 + int(padding)),
            }
        )
    boxes, merge_decisions = _merge_overlapping_boxes(boxes, merge_iou=merge_iou)
    decisions.extend(merge_decisions)
    boxes = sorted(
        boxes,
        key=lambda box: (
            int(box['bbox_y0']),
            int(box['bbox_x0']),
            -int(box.get('component_area', 0)),
        ),
    )
    kept: list[dict[str, Any]] = []
    for index, box in enumerate(boxes, start=1):
        if index > int(max_boxes):
            decisions.append(
                {
                    **box,
                    'proposal_index': index,
                    'decision': 'overflow',
                    'decision_reason': 'proposal_count_above_maximum',
                }
            )
            continue
        kept_box = {
            'proposal_index': len(kept) + 1,
            'bbox_x0': int(box['bbox_x0']),
            'bbox_y0': int(box['bbox_y0']),
            'bbox_x1': int(box['bbox_x1']),
            'bbox_y1': int(box['bbox_y1']),
            'component_area': int(box.get('component_area', 0)),
            'threshold': float(threshold),
            'decision': 'generated',
        }
        kept.append(kept_box)
        decisions.append({**kept_box, 'decision_reason': 'accepted'})
    return kept, decisions


def _manual_components(
    manual_mask: np.ndarray, *, min_component_area: int
) -> list[dict[str, Any]]:
    binary = (np.asarray(manual_mask) > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    components: list[dict[str, Any]] = []
    for label_index in range(1, count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        components.append(
            {
                'component_index': int(label_index),
                'area': area,
                'mask': labels == label_index,
                'bbox_x0': int(stats[label_index, cv2.CC_STAT_LEFT]),
                'bbox_y0': int(stats[label_index, cv2.CC_STAT_TOP]),
                'bbox_x1': int(stats[label_index, cv2.CC_STAT_LEFT])
                + int(stats[label_index, cv2.CC_STAT_WIDTH]),
                'bbox_y1': int(stats[label_index, cv2.CC_STAT_TOP])
                + int(stats[label_index, cv2.CC_STAT_HEIGHT]),
            }
        )
    return components


def proposal_recall_row(
    *,
    manual_mask: np.ndarray,
    proposal_boxes: list[dict[str, Any]],
    manifest_row_id: str,
    cohort_id: str,
    lane_assignment: str,
    candidate_family: str,
    candidate_artifact: str,
    threshold: float,
    min_component_area: int,
) -> dict[str, Any]:
    components = _manual_components(manual_mask, min_component_area=min_component_area)
    matched = 0
    missed_boxes: list[str] = []
    for component in components:
        is_matched = False
        for box in proposal_boxes:
            x0 = int(box['bbox_x0'])
            y0 = int(box['bbox_y0'])
            x1 = int(box['bbox_x1'])
            y1 = int(box['bbox_y1'])
            covered = component['mask'][y0:y1, x0:x1].sum()
            if float(covered) / float(component['area']) >= 0.5:
                is_matched = True
                break
        if is_matched:
            matched += 1
        else:
            missed_boxes.append(
                f'{component["component_index"]}:{component["bbox_x0"]},{component["bbox_y0"]},{component["bbox_x1"]},{component["bbox_y1"]}'
            )
    total = len(components)
    return {
        'manifest_row_id': manifest_row_id,
        'cohort_id': cohort_id,
        'lane_assignment': lane_assignment,
        'candidate_family': candidate_family,
        'candidate_artifact': candidate_artifact,
        'threshold': float(threshold),
        'manual_component_count': total,
        'matched_manual_component_count': int(matched),
        'missed_manual_component_count': int(total - matched),
        'proposal_count': int(len(proposal_boxes)),
        'overflow_count': 0,
        'proposal_recall': 1.0 if total == 0 else float(matched / total),
        'missed_component_boxes': ';'.join(missed_boxes),
    }


def classify_automatic_prompt_result(
    *,
    proposal_recall: float,
    auto_dice: float,
    prompt_failure_count: int,
    min_proposal_recall: float,
    min_auto_dice: float,
    max_prompt_failures: int,
) -> dict[str, Any]:
    if int(prompt_failure_count) > int(max_prompt_failures):
        failure_mode = 'downstream_integration'
        recommendation = 'resolve_prompt_failures_before_transition'
        transition_status = 'blocked'
        source = ''
    elif float(proposal_recall) < float(min_proposal_recall):
        failure_mode = 'proposal_localization'
        recommendation = 'improve_box_proposer_before_fine_tuning'
        transition_status = 'blocked'
        source = ''
    elif float(auto_dice) < float(min_auto_dice):
        failure_mode = 'medsam_boundary_quality'
        recommendation = 'open_medsam_sam_fine_tuning_change'
        transition_status = 'blocked'
        source = ''
    else:
        failure_mode = 'none_detected'
        recommendation = 'not_recommended_prompt_based_generation_first'
        transition_status = 'ready_for_derived_mask_generation'
        source = AUTOMATIC_MASK_SOURCE
    return {
        'gates_passed': failure_mode == 'none_detected',
        'failure_mode': failure_mode,
        'recommended_generated_mask_source': source,
        'mask_source': source,
        'primary_segmenter_transition_status': transition_status,
        'fine_tuning_recommendation': recommendation,
    }


def _automatic_metric_row(
    *,
    prompt_mode: str,
    proposal_threshold: float,
    candidate_family: str,
    mask_source: str,
    method: str,
    candidate_artifact: str,
    manifest_row_id: str,
    cohort_id: str,
    lane_assignment: str,
    manual_mask: np.ndarray,
    predicted_mask: np.ndarray,
) -> dict[str, Any]:
    base = metric_row(
        method=method,
        candidate_artifact=candidate_artifact,
        manifest_row_id=manifest_row_id,
        cohort_id=cohort_id,
        lane_assignment=lane_assignment,
        manual_mask=manual_mask,
        predicted_mask=predicted_mask,
    )
    row = {
        'prompt_mode': prompt_mode,
        'proposal_threshold': float(proposal_threshold),
        'candidate_family': candidate_family,
        'mask_source': mask_source,
        **base,
    }
    return {field: row[field] for field in AUTOMATIC_METRIC_FIELDS}


def _predict_probability(
    *, model: Any, image_path: Path, tile_size: int, stride: int, expected_size: int
) -> tuple[np.ndarray, dict[str, Any]]:
    image = Image.open(image_path).convert('RGB')
    return _predict_tiled_segmentation_probability(
        model=model,
        image=image,
        tile_size=tile_size,
        stride=stride,
        expected_size=expected_size,
    )


def _draw_automatic_overlay(
    *,
    image_path: Path,
    manual_mask: np.ndarray,
    automatic_mask: np.ndarray | None,
    proposal_boxes: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert('RGB')
    overlay = image.copy().convert('RGBA')
    for mask, color in [
        (manual_mask, (0, 255, 0, 90)),
        (automatic_mask, (255, 0, 0, 90)),
    ]:
        if mask is None:
            continue
        alpha = (np.asarray(mask) > 0).astype(np.uint8) * color[3]
        color_arr = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
        color_arr[..., 0] = color[0]
        color_arr[..., 1] = color[1]
        color_arr[..., 2] = color[2]
        color_arr[..., 3] = alpha
        overlay = Image.alpha_composite(
            overlay, Image.fromarray(color_arr, mode='RGBA')
        )
    draw = ImageDraw.Draw(overlay)
    for box in proposal_boxes:
        draw.rectangle(
            [box['bbox_x0'], box['bbox_y0'], box['bbox_x1'], box['bbox_y1']],
            outline=(0, 0, 255, 255),
            width=3,
        )
    draw.text(
        (8, 8),
        'manual=green medsam_auto=red proposal_box=blue',
        fill=(255, 255, 255, 255),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert('RGB').save(output_path)


def _candidate_paths(current: dict[str, Any]) -> dict[str, Path]:
    return {
        'transfer': Path(
            str(current.get('transfer_model_path') or DEFAULT_TRANSFER_MODEL)
        ).expanduser(),
        'scratch': Path(
            str(current.get('scratch_model_path') or DEFAULT_SCRATCH_MODEL)
        ).expanduser(),
    }


def _select_best_setting(recall_rows: list[dict[str, Any]]) -> tuple[str, float]:
    frame = pd.DataFrame(recall_rows)
    grouped = (
        frame.groupby(['candidate_family', 'threshold'], dropna=False)
        .agg(
            proposal_recall=('proposal_recall', 'mean'),
            proposal_count=('proposal_count', 'mean'),
            missed=('missed_manual_component_count', 'sum'),
        )
        .reset_index()
        .sort_values(
            ['proposal_recall', 'missed', 'proposal_count', 'threshold'],
            ascending=[False, True, True, False],
            kind='mergesort',
        )
    )
    best = grouped.iloc[0]
    return str(best['candidate_family']), float(best['threshold'])


def run_medsam_automatic_glomeruli_prompts_workflow(
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
    proposal = _mapping(config, 'proposal')
    gates = _mapping(config, 'gates')
    outputs = _mapping(config, 'outputs')
    run_id = str(run_cfg.get('name') or DEFAULT_RUN_ID)
    evaluation_dir = ensure_evaluation_output_path(
        runtime_root, outputs.get('evaluation_dir', f'{DEFAULT_OUTPUT_DIR}/{run_id}')
    )
    derived_dir = ensure_evaluation_output_path(
        runtime_root,
        outputs.get('derived_masks_dir', f'{DEFAULT_DERIVED_OUTPUT_DIR}/{run_id}'),
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
    candidate_paths = _candidate_paths(current)
    command = [
        sys.executable,
        '-m',
        'eq.evaluation.run_medsam_automatic_glomeruli_prompts_workflow',
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
                **{
                    f'{family}_model_path': path
                    for family, path in candidate_paths.items()
                },
            }
        )
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {
            'workflow': WORKFLOW_ID,
            'run_id': run_id,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'config_path': str(config_path),
            'runtime_root': str(runtime_root),
            'evaluation_dir': str(evaluation_dir),
            'derived_masks_dir': str(derived_dir),
            'manifest_path': str(manifest_path),
            'medsam_python': str(medsam_python),
            'medsam_repo': str(medsam_repo),
            'medsam_checkpoint': str(checkpoint),
            'medsam_checkpoint_hash': _file_hash(checkpoint),
            'current_segmenters': {
                family: str(path) for family, path in candidate_paths.items()
            },
            'dry_run': bool(dry_run),
            'log_path': str(log_context.log_path),
            'interpretation': 'Audit-scoped automatic-prompt MedSAM evaluation; raw/manual masks are not overwritten.',
        }
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
        summary['pilot_row_count'] = int(len(selected))
        if dry_run:
            (evaluation_dir / 'summary.json').write_text(
                json.dumps(summary, indent=2), encoding='utf-8'
            )
            return {
                'evaluation_dir': evaluation_dir,
                'summary': evaluation_dir / 'summary.json',
            }

        thresholds = [
            float(value) for value in proposal.get('thresholds', [0.2, 0.35, 0.5])
        ]
        medsam_auto_dir = evaluation_dir / 'medsam_auto_masks'
        current_mask_dir = evaluation_dir / 'current_segmenter_masks'
        overlay_dir = evaluation_dir / 'overlays'
        for directory in (medsam_auto_dir, current_mask_dir, overlay_dir):
            directory.mkdir(parents=True, exist_ok=True)

        current_models: dict[str, Any] = {}
        for family, model_path in candidate_paths.items():
            _emit(f'Loading current glomeruli candidate once: {family}={model_path}')
            learner = load_model_safely(str(model_path), model_type='glomeruli')
            learner.model.eval()
            current_models[family] = learner.model

        probability_maps: dict[tuple[str, str], np.ndarray] = {}
        proposal_by_setting: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
        proposal_rows: list[dict[str, Any]] = []
        recall_rows: list[dict[str, Any]] = []
        manual_masks: dict[str, np.ndarray] = {}
        current_audits: list[dict[str, Any]] = []
        current_metrics: list[dict[str, Any]] = []
        for row in selected.to_dict(orient='records'):
            manifest_row_id = str(row['manifest_row_id'])
            manual_mask = load_binary_mask(Path(str(row['mask_path_resolved'])))
            manual_masks[manifest_row_id] = manual_mask
            for family, model_path in candidate_paths.items():
                probability, audit = _predict_probability(
                    model=current_models[family],
                    image_path=Path(str(row['image_path_resolved'])),
                    tile_size=int(tiling.get('tile_size', 512)),
                    stride=int(tiling.get('stride', 512)),
                    expected_size=int(tiling.get('expected_size', 256)),
                )
                probability_maps[(manifest_row_id, family)] = probability
                current_mask = (
                    probability >= float(current.get('comparison_threshold', 0.75))
                ).astype(np.uint8)
                _write_mask(
                    current_mask_dir / family / f'{manifest_row_id}_{family}.png',
                    current_mask,
                )
                current_audits.append(
                    {
                        'manifest_row_id': manifest_row_id,
                        'candidate_family': family,
                        'model_path': str(model_path),
                        **audit,
                    }
                )
                current_metrics.append(
                    _automatic_metric_row(
                        prompt_mode='none_current_segmenter',
                        proposal_threshold=float(
                            current.get('comparison_threshold', 0.75)
                        ),
                        candidate_family=family,
                        mask_source=f'current_segmenter_{family}',
                        method=f'current_segmenter_{family}',
                        candidate_artifact=str(model_path),
                        manifest_row_id=manifest_row_id,
                        cohort_id=str(row['cohort_id']),
                        lane_assignment=str(row['lane_assignment']),
                        manual_mask=manual_mask,
                        predicted_mask=current_mask,
                    )
                )
                for threshold in thresholds:
                    boxes, decisions = derive_proposal_boxes(
                        probability,
                        threshold=threshold,
                        min_component_area=int(
                            proposal.get('min_component_area', 2000)
                        ),
                        max_component_area=int(
                            proposal.get('max_component_area', 750000)
                        ),
                        padding=int(proposal.get('padding', 16)),
                        merge_iou=float(proposal.get('merge_iou', 0.25)),
                        max_boxes=int(proposal.get('max_boxes_per_image', 20)),
                    )
                    proposal_by_setting[(manifest_row_id, family, threshold)] = boxes
                    overflow_count = sum(
                        1
                        for decision in decisions
                        if decision.get('decision') == 'overflow'
                    )
                    for decision in decisions:
                        proposal_rows.append(
                            {
                                **{field: row[field] for field in selected_fields},
                                'candidate_family': family,
                                'candidate_artifact': str(model_path),
                                'threshold': threshold,
                                **decision,
                            }
                        )
                    recall = proposal_recall_row(
                        manual_mask=manual_mask,
                        proposal_boxes=boxes,
                        manifest_row_id=manifest_row_id,
                        cohort_id=str(row['cohort_id']),
                        lane_assignment=str(row['lane_assignment']),
                        candidate_family=family,
                        candidate_artifact=str(model_path),
                        threshold=threshold,
                        min_component_area=int(pilot.get('min_component_area', 2000)),
                    )
                    recall['overflow_count'] = int(overflow_count)
                    recall_rows.append(recall)
        _write_csv(
            evaluation_dir / 'proposal_boxes.csv',
            proposal_rows,
            fieldnames=PROPOSAL_BOX_FIELDS,
        )
        _write_csv(evaluation_dir / 'proposal_recall.csv', recall_rows)
        _write_csv(evaluation_dir / 'current_segmenter_audit.csv', current_audits)

        best_family, best_threshold = _select_best_setting(recall_rows)
        medsam_items: list[dict[str, Any]] = []
        for row in selected.to_dict(orient='records'):
            manifest_row_id = str(row['manifest_row_id'])
            manual_mask = manual_masks[manifest_row_id]
            boxes = proposal_by_setting[(manifest_row_id, best_family, best_threshold)]
            medsam_items.append(
                {
                    'manifest_row_id': manifest_row_id,
                    'image_path': str(row['image_path_resolved']),
                    'height': int(manual_mask.shape[0]),
                    'width': int(manual_mask.shape[1]),
                    'boxes': boxes,
                    'output_path': str(
                        medsam_auto_dir
                        / best_family
                        / f'{manifest_row_id}_{best_family}_{best_threshold:g}_medsam_auto.png'
                    ),
                }
            )
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

        metrics = list(current_metrics)
        for row in selected.to_dict(orient='records'):
            manifest_row_id = str(row['manifest_row_id'])
            manual_mask = manual_masks[manifest_row_id]
            boxes = proposal_by_setting[(manifest_row_id, best_family, best_threshold)]
            auto_path = (
                medsam_auto_dir
                / best_family
                / f'{manifest_row_id}_{best_family}_{best_threshold:g}_medsam_auto.png'
            )
            automatic_mask = load_binary_mask(auto_path) if auto_path.exists() else None
            if automatic_mask is not None:
                metrics.append(
                    _automatic_metric_row(
                        prompt_mode='automatic_current_segmenter_boxes',
                        proposal_threshold=best_threshold,
                        candidate_family=best_family,
                        mask_source=AUTOMATIC_MASK_SOURCE,
                        method='medsam_automatic',
                        candidate_artifact=str(checkpoint),
                        manifest_row_id=manifest_row_id,
                        cohort_id=str(row['cohort_id']),
                        lane_assignment=str(row['lane_assignment']),
                        manual_mask=manual_mask,
                        predicted_mask=automatic_mask,
                    )
                )
            _draw_automatic_overlay(
                image_path=Path(str(row['image_path_resolved'])),
                manual_mask=manual_mask,
                automatic_mask=automatic_mask,
                proposal_boxes=boxes,
                output_path=overlay_dir
                / f'{manifest_row_id}_{best_family}_{best_threshold:g}_overlay.png',
            )
        _write_csv(
            evaluation_dir / 'metrics.csv', metrics, fieldnames=AUTOMATIC_METRIC_FIELDS
        )
        metric_frame = pd.DataFrame(metrics)
        grouped = (
            metric_frame.groupby(
                [
                    'method',
                    'prompt_mode',
                    'candidate_artifact',
                    'proposal_threshold',
                    'cohort_id',
                    'lane_assignment',
                ],
                dropna=False,
            )[['dice', 'jaccard', 'precision', 'recall', 'pixel_accuracy']]
            .mean()
            .reset_index()
        )
        grouped.to_csv(evaluation_dir / 'metric_by_source.csv', index=False)

        auto_rows = metric_frame[metric_frame['method'] == 'medsam_automatic']
        selected_recall = pd.DataFrame(recall_rows)
        selected_recall = selected_recall[
            (selected_recall['candidate_family'] == best_family)
            & (selected_recall['threshold'] == best_threshold)
        ]
        mean_recall = (
            float(selected_recall['proposal_recall'].mean())
            if not selected_recall.empty
            else 0.0
        )
        mean_dice = float(auto_rows['dice'].mean()) if not auto_rows.empty else 0.0
        gate_summary = classify_automatic_prompt_result(
            proposal_recall=mean_recall,
            auto_dice=mean_dice,
            prompt_failure_count=len(prompt_failures),
            min_proposal_recall=float(gates.get('min_proposal_recall', 0.90)),
            min_auto_dice=float(gates.get('min_auto_dice', 0.85)),
            max_prompt_failures=int(gates.get('max_prompt_failures', 0)),
        )
        derived_manifest_rows: list[dict[str, Any]] = []
        if bool(outputs.get('generate_broad_derived_masks', False)):
            if not gate_summary['gates_passed']:
                raise ValueError(
                    'Broad derived-mask generation requested, but automatic pilot gates did not pass.'
                )
            derived_dir.mkdir(parents=True, exist_ok=True)
            for item in medsam_items:
                source = Path(str(item['output_path']))
                target = derived_dir / source.name
                if source.exists():
                    target.write_bytes(source.read_bytes())
                    status = 'copied_from_pilot'
                else:
                    status = 'missing_source'
                derived_manifest_rows.append(
                    {
                        'source_image_path': item['image_path'],
                        'derived_mask_path': str(target),
                        'mask_source': AUTOMATIC_MASK_SOURCE,
                        'proposal_source': best_family,
                        'candidate_artifact': str(candidate_paths[best_family]),
                        'threshold': best_threshold,
                        'medsam_checkpoint': str(checkpoint),
                        'run_id': run_id,
                        'generation_status': status,
                    }
                )
        _write_csv(evaluation_dir / 'derived_mask_manifest.csv', derived_manifest_rows)
        summary.update(
            {
                'dry_run': False,
                'selected_proposal_source': best_family,
                'selected_proposal_threshold': best_threshold,
                'prompt_failure_count': int(len(prompt_failures)),
                'metric_rows': int(len(metrics)),
                'proposal_recall_mean': mean_recall,
                'medsam_automatic_dice_mean': mean_dice,
                'mean_metrics': metric_frame.groupby('method')[
                    ['dice', 'jaccard', 'precision', 'recall']
                ]
                .mean()
                .to_dict(orient='index'),
                'outputs': {
                    'inputs': str(evaluation_dir / 'inputs.csv'),
                    'proposal_boxes': str(evaluation_dir / 'proposal_boxes.csv'),
                    'proposal_recall': str(evaluation_dir / 'proposal_recall.csv'),
                    'metrics': str(evaluation_dir / 'metrics.csv'),
                    'metric_by_source': str(evaluation_dir / 'metric_by_source.csv'),
                    'prompt_failures': str(evaluation_dir / 'prompt_failures.csv'),
                    'medsam_auto_masks': str(medsam_auto_dir),
                    'current_segmenter_masks': str(current_mask_dir),
                    'overlays': str(overlay_dir),
                    'derived_mask_manifest': str(
                        evaluation_dir / 'derived_mask_manifest.csv'
                    ),
                },
                **gate_summary,
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
        description='Run MedSAM automatic-prompt glomeruli workflow.'
    )
    parser.add_argument(
        '--config', default='configs/medsam_automatic_glomeruli_prompts.yaml'
    )
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_medsam_automatic_glomeruli_prompts_workflow(
        Path(args.config), dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
