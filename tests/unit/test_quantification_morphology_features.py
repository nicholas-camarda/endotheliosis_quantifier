import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from eq.quantification.feature_review import write_morphology_feature_review
from eq.quantification.morphology_features import (
    MORPHOLOGY_FEATURE_COLUMNS,
    write_morphology_feature_tables,
)


def _write_synthetic_roi(tmp_path: Path) -> tuple[Path, Path]:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:, :] = [120, 95, 115]
    yy, xx = np.ogrid[:96, :96]
    mask = ((xx - 48) ** 2 + (yy - 48) ** 2) <= 42**2
    pale = ((xx - 35) ** 2 + (yy - 38) ** 2) <= 10**2
    rbc = ((xx - 62) ** 2 + (yy - 56) ** 2) <= 9**2
    slit = (yy > 68) & (yy < 74) & (xx > 25) & (xx < 76)
    nucleus = ((xx - 66) ** 2 + (yy - 28) ** 2) <= 5**2
    image[pale] = [240, 235, 225]
    image[rbc] = [205, 92, 74]
    image[slit] = [35, 28, 35]
    image[nucleus] = [55, 35, 82]
    image_path = tmp_path / 'roi.png'
    mask_path = tmp_path / 'roi_mask.png'
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    return image_path, mask_path


def test_morphology_features_are_finite_and_include_rbc_fields(tmp_path: Path):
    image_path, mask_path = _write_synthetic_roi(tmp_path)
    roi_table = pd.DataFrame(
        [
            {
                'subject_image_id': 'vegfri_dox__2023_06_12__m1__image0',
                'subject_id': 'vegfri_dox__2023_06_12__m1',
                'sample_id': 'vegfri_dox__2023_06_12__m1__image0',
                'image_id': 'vegfri_dox__2023_06_12__m1__image0',
                'cohort_id': 'vegfri_dox',
                'score': 2.0,
                'roi_image_path': str(image_path),
                'roi_mask_path': str(mask_path),
            }
        ]
    )

    feature_df, artifacts = write_morphology_feature_tables(
        roi_table, tmp_path / 'feature_sets', tmp_path / 'diagnostics'
    )

    assert artifacts['morphology_features'].exists()
    assert artifacts['subject_morphology_features'].exists()
    assert artifacts['morphology_feature_metadata'].exists()
    assert artifacts['morphology_feature_diagnostics'].exists()
    assert set(MORPHOLOGY_FEATURE_COLUMNS).issubset(feature_df.columns)
    assert np.isfinite(feature_df[MORPHOLOGY_FEATURE_COLUMNS].to_numpy()).all()
    assert feature_df.loc[0, 'morph_pale_lumen_area_fraction'] > 0
    assert feature_df.loc[0, 'morph_rbc_like_color_burden'] > 0
    assert feature_df.loc[0, 'morph_slit_like_object_count'] > 0
    assert 'morph_border_false_slit_area_fraction' in feature_df.columns
    assert 'morph_slit_boundary_overlap_fraction' in feature_df.columns
    assert feature_df.loc[0, 'morph_slit_boundary_overlap_fraction'] < 1
    assert feature_df.loc[0, 'morph_nuclear_mesangial_confounder_count'] > 0
    assert feature_df.loc[0, 'morph_nuclear_mesangial_confounder_area_fraction'] > 0

    diagnostics = json.loads(
        artifacts['morphology_feature_diagnostics'].read_text(encoding='utf-8')
    )
    assert diagnostics['row_count'] == 1
    assert diagnostics['subject_count'] == 1


def test_morphology_feature_review_writes_template_and_agreement(tmp_path: Path):
    image_path, mask_path = _write_synthetic_roi(tmp_path)
    feature_df = pd.DataFrame(
        [
            {
                'case_id': 'case_001',
                'subject_id': 'subject_a',
                'sample_id': 'sample_a',
                'image_id': 'image_a',
                'score': 1.5,
                'roi_image_path': str(image_path),
                'roi_mask_path': str(mask_path),
                **{column: 0.1 for column in MORPHOLOGY_FEATURE_COLUMNS},
            }
        ]
    )

    artifacts = write_morphology_feature_review(feature_df, tmp_path / 'review')

    assert artifacts['morphology_feature_review_html'].exists()
    assert artifacts['morphology_feature_review_cases'].exists()
    assert artifacts['morphology_operator_adjudication_template'].exists()
    assert artifacts['morphology_operator_adjudication_agreement'].exists()
    assert any(artifacts['morphology_feature_review_assets_dir'].iterdir())
    html = artifacts['morphology_feature_review_html'].read_text(encoding='utf-8')
    assert 'green=open' in html
    assert 'purple=mesangial/nuclear confounder' in html
    assert 'cyan=border false slit' in html
    template = pd.read_csv(artifacts['morphology_operator_adjudication_template'])
    assert 'open_rbc_filled_lumen_present' in template.columns
    assert 'mesangial_or_nuclear_false_slit_present' in template.columns
    assert 'border_false_slit_present' in template.columns
