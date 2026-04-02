"""ROI extraction for scored glomerulus examples."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from eq.evaluation.quantification_metrics import calculate_quantification_metrics


def _connected_component_boxes(mask_array: np.ndarray, min_area: int, threshold: int) -> list[dict[str, object]]:
    binary = (mask_array >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[dict[str, object]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        component_mask = (labels == label_id).astype(np.uint8) * 255
        components.append(
            {
                'label_id': label_id,
                'area': area,
                'bbox_left': x,
                'bbox_top': y,
                'bbox_width': w,
                'bbox_height': h,
                'component_mask': component_mask[y:y + h, x:x + w],
                'sort_key': (y, x),
            }
        )
    components.sort(key=lambda item: item['sort_key'])  # Deterministic top-to-bottom, left-to-right assignment.
    return components


def extract_rois_for_scored_examples(
    scored_examples_path: Path,
    output_dir: Path,
    mask_threshold: int = 127,
    min_component_area: int = 512,
) -> dict[str, Path]:
    """Extract ROI crops for each scored example from canonical raw image/mask pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_dir = output_dir / 'roi_crops'
    roi_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(scored_examples_path)
    updated_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []

    for subject_image_id, group in scored.groupby('subject_image_id', sort=True):
        group_rows = group.to_dict(orient='records')
        first_row = group_rows[0]
        if first_row.get('join_status') != 'matched_pair':
            for row in group_rows:
                row['roi_status'] = first_row.get('join_status')
                updated_rows.append(row)
            continue

        image_path = Path(str(first_row['image_path']))
        mask_path = Path(str(first_row['mask_path']))
        image_array = np.array(Image.open(image_path).convert('RGB'))
        grayscale_array = np.array(Image.open(image_path).convert('L'))
        mask_array = np.array(Image.open(mask_path).convert('L'))
        components = _connected_component_boxes(mask_array, min_component_area, mask_threshold)
        expected_count = len(group_rows)

        for component_index, component in enumerate(components, start=1):
            x = int(component['bbox_left'])
            y = int(component['bbox_top'])
            w = int(component['bbox_width'])
            h = int(component['bbox_height'])
            crop_image = image_array[y:y + h, x:x + w]
            crop_gray = grayscale_array[y:y + h, x:x + w]
            crop_mask = component['component_mask']
            component_rows.append(
                {
                    'subject_image_id': subject_image_id,
                    'component_rank': component_index,
                    'component_area': int(component['area']),
                    'bbox_left': x,
                    'bbox_top': y,
                    'bbox_width': w,
                    'bbox_height': h,
                }
            )
            component['crop_image'] = crop_image
            component['crop_gray'] = crop_gray
            component['crop_mask'] = crop_mask

        assigned_count = min(len(components), expected_count)
        for row_index, row in enumerate(group_rows):
            if row_index >= assigned_count:
                row['roi_status'] = 'insufficient_components'
                updated_rows.append(row)
                continue

            component = components[row_index]
            roi_image_path = roi_dir / f'{subject_image_id}_glom{int(row["glomerulus_id"]):03d}.png'
            roi_mask_path = roi_dir / f'{subject_image_id}_glom{int(row["glomerulus_id"]):03d}_mask.png'
            Image.fromarray(component['crop_image']).save(roi_image_path)
            Image.fromarray(component['crop_mask']).save(roi_mask_path)
            metrics = calculate_quantification_metrics(
                (np.array(component['crop_mask']) > 0).astype(np.uint8),
                np.array(component['crop_gray']),
            )
            row['roi_status'] = (
                'matched_component_rank'
                if len(components) == expected_count
                else 'heuristic_component_rank'
            )
            row['roi_assignment_strategy'] = 'component_rank_top_to_bottom_left_to_right'
            row['roi_image_path'] = str(roi_image_path)
            row['roi_mask_path'] = str(roi_mask_path)
            row['bbox_left'] = int(component['bbox_left'])
            row['bbox_top'] = int(component['bbox_top'])
            row['bbox_width'] = int(component['bbox_width'])
            row['bbox_height'] = int(component['bbox_height'])
            row['roi_area'] = int(component['area'])
            row['roi_fill_fraction'] = float(component['area'] / (component['bbox_width'] * component['bbox_height']))
            row['roi_mean_intensity'] = float(np.array(component['crop_gray'])[np.array(component['crop_mask']) > 0].mean())
            row['openness_score'] = float(metrics.openness_score)
            updated_rows.append(row)

    updated = pd.DataFrame.from_records(updated_rows)
    updated_path = output_dir / 'scored_examples_with_rois.csv'
    updated.to_csv(updated_path, index=False)

    components_df = pd.DataFrame.from_records(component_rows)
    components_path = output_dir / 'roi_components.csv'
    components_df.to_csv(components_path, index=False)

    summary = {
        'roi_status_counts': updated['roi_status'].value_counts(dropna=False).to_dict(),
        'total_rows': int(len(updated)),
    }
    summary_path = output_dir / 'roi_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        'scored_examples_with_rois': updated_path,
        'roi_components': components_path,
        'roi_summary': summary_path,
    }
