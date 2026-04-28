"""Reviewer-facing learned ROI evidence artifacts."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from eq.quantification.burden import BURDEN_COLUMN


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ''
    if not np.isfinite(numeric):
        return ''
    return f'{numeric:.{digits}f}'


def _save_preview(
    path: Any, output_path: Path, *, box: tuple[int, int, int, int] | None = None
) -> bool:
    try:
        image = Image.open(Path(str(path))).convert('RGB')
    except (FileNotFoundError, OSError, ValueError):
        return False
    if box is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=(32, 200, 255), width=3)
    image.thumbnail((600, 600))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return True


def _selected_examples(
    predictions: pd.DataFrame, max_examples: int = 12
) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    work = predictions.copy()
    work['_abs_error'] = pd.to_numeric(
        work.get('stage_index_absolute_error', 0.0), errors='coerce'
    ).fillna(0.0)
    work['_set_size'] = (
        work['prediction_set_scores'].astype(str).str.split('|').map(len)
    )
    selected: list[pd.Series] = []
    buckets = [
        ('high_error', work.sort_values('_abs_error', ascending=False)),
        ('high_uncertainty', work.sort_values('_set_size', ascending=False)),
        ('representative_correct', work.sort_values('_abs_error', ascending=True)),
        ('high_burden', work.sort_values(BURDEN_COLUMN, ascending=False)),
    ]
    seen: set[str] = set()
    for bucket, bucket_df in buckets:
        for _, row in bucket_df.iterrows():
            key = str(row.get('subject_image_id', row.name))
            if key in seen:
                continue
            row = row.copy()
            row['review_bucket'] = bucket
            selected.append(row)
            seen.add(key)
            break
    if 'cohort_id' in work.columns:
        for cohort, cohort_df in work.groupby('cohort_id'):
            for _, row in cohort_df.sort_values(
                '_abs_error', ascending=False
            ).iterrows():
                key = str(row.get('subject_image_id', row.name))
                if key in seen:
                    continue
                row = row.copy()
                row['review_bucket'] = f'cohort_{cohort}'
                selected.append(row)
                seen.add(key)
                break
    for _, row in work.sort_values('_abs_error', ascending=False).iterrows():
        if len(selected) >= max_examples:
            break
        key = str(row.get('subject_image_id', row.name))
        if key in seen:
            continue
        row = row.copy()
        row['review_bucket'] = 'additional_error_review'
        selected.append(row)
        seen.add(key)
    return pd.DataFrame(selected).drop(
        columns=['_abs_error', '_set_size'], errors='ignore'
    )


def write_learned_roi_review(
    *,
    selected_predictions: pd.DataFrame,
    nearest_examples_path: Path,
    output_dir: Path,
    candidate_summary: dict[str, Any],
) -> dict[str, Path]:
    """Write HTML, example CSV, and copied assets for learned ROI review."""
    output_dir = Path(output_dir)
    assets_dir = output_dir / 'assets'
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    examples = _selected_examples(selected_predictions)
    examples_path = output_dir / 'learned_roi_review_examples.csv'
    examples.to_csv(examples_path, index=False)
    nearest = (
        pd.read_csv(nearest_examples_path)
        if Path(nearest_examples_path).exists()
        and Path(nearest_examples_path).stat().st_size > 0
        else pd.DataFrame()
    )

    cards: list[str] = []
    for index, row in examples.reset_index(drop=True).iterrows():
        subject_image_id = str(row.get('subject_image_id', index))
        raw_asset = assets_dir / f'learned_roi_example_{index:02d}_raw.png'
        roi_asset = assets_dir / f'learned_roi_example_{index:02d}_roi.png'
        box = None
        if {'roi_bbox_x0', 'roi_bbox_y0', 'roi_bbox_x1', 'roi_bbox_y1'}.issubset(
            row.index
        ):
            try:
                box = (
                    int(float(row['roi_bbox_x0'])),
                    int(float(row['roi_bbox_y0'])),
                    int(float(row['roi_bbox_x1'])),
                    int(float(row['roi_bbox_y1'])),
                )
            except (TypeError, ValueError):
                box = None
        raw_written = _save_preview(row.get('raw_image_path', ''), raw_asset, box=box)
        roi_written = _save_preview(row.get('roi_image_path', ''), roi_asset)
        image_html = ''
        if raw_written or roi_written:
            image_html = '<div class="image-grid">'
            if raw_written:
                image_html += f'<figure><img src="assets/{raw_asset.name}" alt="Raw image"><figcaption>Raw image with ROI box</figcaption></figure>'
            if roi_written:
                image_html += f'<figure><img src="assets/{roi_asset.name}" alt="ROI crop"><figcaption>ROI crop</figcaption></figure>'
            image_html += '</div>'

        neighbor_rows = ''
        if not nearest.empty and 'subject_image_id' in nearest.columns:
            subset = nearest[
                nearest['subject_image_id'].astype(str) == subject_image_id
            ].head(3)
            for _, neighbor in subset.iterrows():
                neighbor_rows += (
                    '<tr>'
                    f'<td>{escape(str(neighbor.get("neighbor_rank", "")))}</td>'
                    f'<td>{escape(str(neighbor.get("neighbor_subject_image_id", "")))}</td>'
                    f'<td>{_format_float(neighbor.get("neighbor_score"), 1)}</td>'
                    f'<td>{_format_float(neighbor.get("neighbor_distance"))}</td>'
                    '</tr>'
                )
        if not neighbor_rows:
            neighbor_rows = '<tr><td colspan="4">No same-subject-excluded nearest examples available.</td></tr>'

        cards.append(
            f"""
            <section class="example-card">
              <h2>{escape(subject_image_id)} <span>{escape(str(row.get('review_bucket', '')))}</span></h2>
              {image_html}
              <div class="summary-grid">
                <div><strong>Observed score</strong><span>{_format_float(row.get('score'), 1)}</span></div>
                <div><strong>Predicted score</strong><span>{_format_float(row.get('predicted_score'), 1)}</span></div>
                <div><strong>Burden</strong><span>{_format_float(row.get(BURDEN_COLUMN))}</span></div>
                <div><strong>Prediction set</strong><span>{escape(str(row.get('prediction_set_scores', '')))}</span></div>
                <div><strong>Candidate</strong><span>{escape(str(row.get('candidate_id', '')))}</span></div>
                <div><strong>Fold</strong><span>{escape(str(row.get('fold', '')))}</span></div>
                <div><strong>Cohort</strong><span>{escape(str(row.get('cohort_id', '')))}</span></div>
              </div>
              <p class="provenance"><strong>ROI path:</strong> {escape(str(row.get('roi_image_path', '')))}<br><strong>Raw path:</strong> {escape(str(row.get('raw_image_path', '')))}</p>
              <h3>Nearest Scored Examples</h3>
              <table><thead><tr><th>Rank</th><th>Example</th><th>Score</th><th>Distance</th></tr></thead><tbody>{neighbor_rows}</tbody></table>
            </section>
            """
        )

    summary_json = json.dumps(candidate_summary, indent=2)
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Learned ROI Quantification Review</title>
      <style>
        body {{ font-family: Helvetica, Arial, sans-serif; margin: 2rem auto; max-width: 1180px; color: #1f2933; background: #f7fafc; }}
        h1, h2, h3 {{ color: #102a43; }}
        .note {{ background: #fff7e6; border-left: 4px solid #d9822b; padding: 1rem; border-radius: 8px; }}
        .example-card {{ background: white; border-radius: 10px; padding: 1rem; margin: 1rem 0; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08); }}
        .summary-grid, .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }}
        .summary-grid div {{ background: #f0f4f8; border-radius: 8px; padding: 0.75rem; }}
        .summary-grid strong {{ display: block; color: #486581; font-size: 0.85rem; }}
        img {{ max-width: 100%; border: 1px solid #d9e2ec; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border-bottom: 1px solid #e5e7eb; padding: 0.4rem; text-align: left; }}
        pre {{ background: #102a43; color: white; padding: 1rem; overflow: auto; border-radius: 8px; }}
      </style>
    </head>
    <body>
      <h1>Learned ROI Quantification Review</h1>
      <div class="note">These artifacts are model-support evidence for predictive grade-equivalent endotheliosis burden. They are not proof of closed-lumen biology, tissue-area percent, causal effect, or mechanism.</div>
      <h2>Candidate Summary</h2>
      <pre>{escape(summary_json)}</pre>
      <h2>Review Examples</h2>
      {''.join(cards)}
    </body>
    </html>
    """
    html_path = output_dir / 'learned_roi_review.html'
    html_path.write_text(html, encoding='utf-8')
    attribution_path = output_dir / 'learned_roi_attribution_status.json'
    attribution_path.write_text(
        json.dumps(
            {
                'status': 'unavailable',
                'reason': 'phase_1 providers do not expose provider-specific saliency or attention artifacts',
                'claim_boundary': 'no attribution artifact is treated as mechanistic proof',
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    return {
        'learned_roi_review_html': html_path,
        'learned_roi_review_examples': examples_path,
        'learned_roi_review_assets_dir': assets_dir,
        'learned_roi_attribution_status': attribution_path,
    }
