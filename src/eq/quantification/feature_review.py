"""Visual review artifacts for morphology-aware quantification features."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from eq.quantification.morphology_features import (
    MORPHOLOGY_FEATURE_COLUMNS,
    compute_morphology_masks,
)
from eq.utils.logger import get_logger

ADJUDICATION_COLUMNS = [
    'case_id',
    'subject_id',
    'sample_id',
    'image_id',
    'score',
    'open_empty_lumen_present',
    'open_rbc_filled_lumen_present',
    'collapsed_slit_like_lumen_present',
    'mesangial_or_nuclear_false_slit_present',
    'border_false_slit_present',
    'poor_orientation_or_quality',
    'feature_detection_problem',
    'preferred_label_if_detection_wrong',
    'notes',
]


def _pick_cases(feature_df: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    if feature_df.empty:
        return feature_df.copy()
    df = feature_df.copy().reset_index(drop=True)
    df['case_id'] = [f'morphology_case_{index:03d}' for index in range(len(df))]
    categories = [
        ('high_score', 'score', False),
        ('low_score', 'score', True),
        ('high_rbc_confounder', 'morph_rbc_like_color_burden', False),
        ('high_collapsed_line', 'morph_slit_like_area_fraction', False),
        ('high_open_lumen', 'morph_pale_lumen_area_fraction', False),
        ('poor_quality_orientation', 'morph_orientation_ambiguity_score', False),
    ]
    for category, column, ascending in categories:
        if column not in df.columns:
            continue
        selected = df.sort_values(column, ascending=ascending).head(
            max(1, max_cases // len(categories))
        )
        selected = selected.copy()
        selected['review_category'] = category
        candidates.append(selected)
    selected = (
        pd.concat(candidates, ignore_index=True)
        if candidates
        else df.head(max_cases).copy()
    )
    selected = selected.drop_duplicates('case_id').head(max_cases).copy()
    return selected


def _overlay_feature_masks(
    image_path: Path, mask_path: Path, output_path: Path
) -> None:
    image = np.asarray(Image.open(image_path).convert('RGB'))
    mask = np.asarray(Image.open(mask_path).convert('L')) > 127
    masks = compute_morphology_masks(image, mask)
    overlay = image.astype(np.float32)
    colors = [
        ('ridge', np.array([80, 120, 255], dtype=np.float32), 0.25),
        ('pale_lumen', np.array([0, 200, 80], dtype=np.float32), 0.45),
        ('rbc_like', np.array([255, 170, 0], dtype=np.float32), 0.45),
        ('border_false_slit', np.array([0, 210, 255], dtype=np.float32), 0.70),
        ('nuclear_mesangial', np.array([150, 0, 220], dtype=np.float32), 0.70),
        ('slit_like', np.array([255, 0, 0], dtype=np.float32), 0.85),
    ]
    for name, color, alpha in colors:
        feature_mask = masks[name]
        overlay[feature_mask] = (1.0 - alpha) * overlay[feature_mask] + alpha * color
    output = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    output.thumbnail((600, 600))
    output.save(output_path)


def _save_roi_preview(image_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert('RGB')
    image.thumbnail((600, 600))
    image.save(output_path)


def _bool_from_cell(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y'}:
        return True
    if text in {'0', 'false', 'no', 'n'}:
        return False
    return None


def _write_adjudication_summary(
    adjudication_path: Path, cases: pd.DataFrame, output_path: Path
) -> dict[str, Any]:
    if not adjudication_path.exists():
        payload = {
            'adjudication_status': 'not_started',
            'completed_rows': 0,
            'case_rows': int(len(cases)),
            'instructions': 'Fill operator_adjudication_template.csv and rerun the same YAML.',
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return payload

    adjudication = pd.read_csv(adjudication_path)
    scored_columns = [
        'open_empty_lumen_present',
        'open_rbc_filled_lumen_present',
        'collapsed_slit_like_lumen_present',
        'mesangial_or_nuclear_false_slit_present',
        'border_false_slit_present',
        'poor_orientation_or_quality',
        'feature_detection_problem',
    ]
    scored = adjudication[scored_columns].apply(
        lambda column: column.map(_bool_from_cell)
    )
    completed = adjudication[scored.notna().any(axis=1)].copy()
    payload = {
        'adjudication_status': 'completed' if len(completed) else 'not_started',
        'completed_rows': int(len(completed)),
        'case_rows': int(len(cases)),
        'feature_detection_problem_count': int(
            completed['feature_detection_problem']
            .apply(_bool_from_cell)
            .fillna(False)
            .sum()
        )
        if len(completed)
        else 0,
    }
    if len(completed):
        for column in scored_columns:
            values = completed[column].apply(_bool_from_cell).astype('boolean')
            payload[f'{column}_yes_count'] = int(values.sum(skipna=True))
        if 'preferred_label_if_detection_wrong' in completed.columns:
            labels = (
                completed['preferred_label_if_detection_wrong']
                .fillna('')
                .astype(str)
                .str.split(',')
                .explode()
                .str.strip()
            )
            labels = labels[labels.ne('')]
            payload['preferred_label_if_detection_wrong_counts'] = (
                labels.value_counts().astype(int).to_dict()
            )
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def write_morphology_feature_review(
    feature_df: pd.DataFrame, output_dir: Path, max_cases: int = 48
) -> dict[str, Path]:
    """Write morphology feature review HTML, assets, cases, and adjudication CSV."""
    logger = get_logger('eq.quantification.feature_review')
    output_dir = Path(output_dir)
    assets_dir = output_dir / 'assets'
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    cases_path = output_dir / 'feature_review_cases.csv'
    template_path = output_dir / 'operator_adjudication_template.csv'
    if template_path.exists():
        template_cases = pd.read_csv(template_path)
        changed = False
        for column in ADJUDICATION_COLUMNS:
            if column not in template_cases.columns:
                insert_at = (
                    ADJUDICATION_COLUMNS.index(column)
                    if column in ADJUDICATION_COLUMNS
                    else len(template_cases.columns)
                )
                template_cases.insert(
                    min(insert_at, len(template_cases.columns)), column, ''
                )
                changed = True
        if changed:
            template_cases = template_cases[ADJUDICATION_COLUMNS]
            template_cases.to_csv(template_path, index=False)
        review_categories = {}
        if cases_path.exists():
            previous_cases = pd.read_csv(cases_path)
            if {'case_id', 'review_category'}.issubset(previous_cases.columns):
                review_categories = previous_cases.set_index('case_id')[
                    'review_category'
                ].to_dict()
        all_cases = feature_df.copy().reset_index(drop=True)
        all_cases['case_id'] = [
            f'morphology_case_{index:03d}' for index in range(len(all_cases))
        ]
        case_order = template_cases['case_id'].astype(str).tolist()
        cases = (
            all_cases[all_cases['case_id'].isin(case_order)]
            .set_index('case_id')
            .loc[case_order]
            .reset_index()
        )
        cases['review_category'] = (
            cases['case_id'].map(review_categories).fillna('operator_reviewed_case')
        )
    else:
        cases = _pick_cases(feature_df, max_cases=max_cases)
    cases.to_csv(cases_path, index=False)
    logger.info(
        'Morphology feature review cases written: rows=%d -> %s', len(cases), cases_path
    )

    if not template_path.exists():
        template = cases[
            [
                column
                for column in [
                    'case_id',
                    'subject_id',
                    'sample_id',
                    'image_id',
                    'score',
                ]
                if column in cases.columns
            ]
        ].copy()
        for column in ADJUDICATION_COLUMNS:
            if column not in template.columns:
                template[column] = ''
        template = template[ADJUDICATION_COLUMNS]
        template.to_csv(template_path, index=False)
        logger.info(
            'Morphology operator adjudication template written -> %s', template_path
        )

    cards: list[str] = []
    for row_index, row in cases.iterrows():
        case_id = str(row.get('case_id', f'morphology_case_{row_index:03d}'))
        image_path = Path(str(row.get('roi_image_path', '')))
        mask_path = Path(str(row.get('roi_mask_path', '')))
        raw_name = f'{case_id}_roi.png'
        overlay_name = f'{case_id}_overlay.png'
        if image_path.is_file() and mask_path.is_file():
            _save_roi_preview(image_path, assets_dir / raw_name)
            _overlay_feature_masks(image_path, mask_path, assets_dir / overlay_name)
            images = (
                f'<figure><img src="assets/{raw_name}" alt="ROI"><figcaption>ROI crop</figcaption></figure>'
                f'<figure><img src="assets/{overlay_name}" alt="Feature overlay"><figcaption>Overlay: green=open, orange=RBC-like, cyan=border false slit, purple=mesangial/nuclear confounder, red=interior slit, blue=ridge</figcaption></figure>'
            )
        else:
            images = '<p>ROI image or mask missing.</p>'
        feature_rows = ''.join(
            '<tr>'
            f'<td>{escape(column)}</td>'
            f'<td>{float(row.get(column, 0.0)):.4g}</td>'
            '</tr>'
            for column in MORPHOLOGY_FEATURE_COLUMNS
            if column in row
        )
        cards.append(
            f"""
            <section class="case-card">
              <h2>{escape(case_id)} <span>{escape(str(row.get('review_category', '')))}</span></h2>
              <div class="image-grid">{images}</div>
              <p><strong>Subject:</strong> {escape(str(row.get('subject_id', '')))} |
              <strong>Sample:</strong> {escape(str(row.get('sample_id', '')))} |
              <strong>Image:</strong> {escape(str(row.get('image_id', row.get('subject_image_id', ''))))} |
              <strong>Score:</strong> {escape(str(row.get('score', '')))}</p>
              <table><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody>{feature_rows}</tbody></table>
            </section>
            """
        )
        if (row_index + 1) % 10 == 0 or (row_index + 1) == len(cases):
            logger.info(
                'Morphology feature review asset progress: %d/%d cases',
                row_index + 1,
                len(cases),
            )

    summary_path = output_dir / 'operator_adjudication_agreement.json'
    _write_adjudication_summary(template_path, cases, summary_path)
    logger.info('Morphology operator adjudication summary written -> %s', summary_path)
    html_path = output_dir / 'feature_review.html'
    html_path.write_text(
        f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Morphology Feature Review</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 1180px; background: #f7fafc; color: #1f2933; }}
            .case-card {{ background: white; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08); }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }}
            img {{ width: 100%; border: 1px solid #d9e2ec; border-radius: 6px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ text-align: left; border-bottom: 1px solid #e5e7eb; padding: 0.35rem; }}
            span {{ color: #52606d; font-size: 0.85rem; }}
          </style>
        </head>
        <body>
          <h1>Morphology Feature Review</h1>
          <p>Open-lumen features are overlaid in green, RBC-like patent-lumen confounder signals in orange, border-adjacent false-slit candidates in cyan, mesangial/nuclear false-slit confounders in purple, accepted interior collapsed/slit-like signals in red, and ridge/line responses in blue. Fill <code>operator_adjudication_template.csv</code> and rerun the same YAML to produce the agreement summary.</p>
          {''.join(cards)}
        </body>
        </html>
        """,
        encoding='utf-8',
    )
    logger.info('Morphology feature review HTML written -> %s', html_path)
    return {
        'morphology_feature_review_html': html_path,
        'morphology_feature_review_cases': cases_path,
        'morphology_feature_review_assets_dir': assets_dir,
        'morphology_operator_adjudication_template': template_path,
        'morphology_operator_adjudication_agreement': summary_path,
    }
