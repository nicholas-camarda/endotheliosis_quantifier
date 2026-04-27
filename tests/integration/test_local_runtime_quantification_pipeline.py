import json
from pathlib import Path

import pandas as pd
import pytest

from eq.quantification.pipeline import run_contract_first_quantification

RUNTIME_ROOT = Path('/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier')
RAW_PROJECT_DIR = RUNTIME_ROOT / 'raw_data/cohorts/lauren_preeclampsia'
ANNOTATION_SOURCE = (
    RUNTIME_ROOT
    / 'raw_data/cohorts/lauren_preeclampsia/scores/labelstudio_annotations.json'
)
SEGMENTATION_MODEL_PATH = (
    RUNTIME_ROOT
    / 'models/segmentation/glomeruli/transfer'
    / 'glom_dynamic_unet_mps-transfer_loss-dice_s1lr1e-3_s2lr_lrfind_e30_b16_lr1e-3_sz256'
    / 'glom_dynamic_unet_mps-transfer_loss-dice_s1lr1e-3_s2lr_lrfind_e30_b16_lr1e-3_sz256.pkl'
)


def _missing_runtime_assets() -> list[Path]:
    return [
        path
        for path in (RAW_PROJECT_DIR, ANNOTATION_SOURCE, SEGMENTATION_MODEL_PATH)
        if not path.exists()
    ]


@pytest.mark.skipif(
    bool(_missing_runtime_assets()),
    reason='local ProjectsRuntime quantification assets are not available',
)
def test_local_runtime_contract_first_quantification_pipeline(tmp_path: Path):
    output_dir = tmp_path / 'quantification'

    outputs = run_contract_first_quantification(
        project_dir=RAW_PROJECT_DIR,
        segmentation_model_path=SEGMENTATION_MODEL_PATH,
        output_dir=output_dir,
        annotation_source=ANNOTATION_SOURCE,
        score_source='labelstudio',
        stop_after='model',
    )

    expected_keys = {
        'raw_inventory',
        'mapping_template',
        'labelstudio_scores',
        'labelstudio_summary',
        'duplicate_annotations',
        'scored_examples',
        'roi_table',
        'embeddings',
        'burden_predictions',
        'burden_metrics',
        'burden_model',
        'ordinal_predictions',
        'ordinal_confusion_matrix',
        'ordinal_metrics',
        'ordinal_model',
        'ordinal_review_html',
        'ordinal_review_examples',
        'ordinal_review_assets_dir',
        'quantification_review_html',
        'quantification_review_examples',
        'quantification_results_summary_md',
        'quantification_results_summary_csv',
        'quantification_readme_snippet',
    }
    assert expected_keys <= outputs.keys()
    for artifact_path in outputs.values():
        assert artifact_path.exists()
        assert artifact_path.resolve().is_relative_to(tmp_path.resolve())

    raw_inventory = pd.read_csv(outputs['raw_inventory'])
    assert len(raw_inventory) > 0

    labelstudio_summary = json.loads(
        outputs['labelstudio_summary'].read_text(encoding='utf-8')
    )
    assert labelstudio_summary['n_unique_images'] == 88
    assert labelstudio_summary['join_status_counts']['ok'] == 88
    assert labelstudio_summary['score_status_counts']['ok'] == 88

    scored_examples = pd.read_csv(outputs['scored_examples'])
    assert len(scored_examples) == 88
    assert scored_examples['join_status'].eq('ok').sum() == 88
    assert scored_examples['score_status'].eq('ok').sum() == 88

    roi_table = pd.read_csv(outputs['roi_table'])
    assert len(roi_table) == len(scored_examples)
    assert roi_table['roi_status'].eq('ok').sum() > 0
    assert (
        roi_table['roi_image_path'].map(lambda value: Path(str(value)).exists()).all()
    )

    embeddings = pd.read_csv(outputs['embeddings'])
    embedding_columns = [
        column for column in embeddings.columns if column.startswith('embedding_')
    ]
    assert len(embeddings) > 0
    assert embedding_columns

    predictions = pd.read_csv(outputs['ordinal_predictions'])
    assert len(predictions) > 0
    assert {'score', 'predicted_score', 'expected_score', 'absolute_error'} <= set(
        predictions.columns
    )

    metrics = json.loads(outputs['ordinal_metrics'].read_text(encoding='utf-8'))
    assert metrics['n_examples'] == len(embeddings)
    assert metrics['n_subject_groups'] == 8
    assert metrics['grouping_key'] == 'subject_id'
    assert metrics['ordinal_model']['estimator_class'] == 'CanonicalOrdinalClassifier'
    assert metrics['cohort_profile']['n_examples'] == 88
    assert metrics['cohort_profile']['embedding_dim'] == len(embedding_columns)
    assert metrics['stability']['zero_unresolved_warning_gate_passed'] is True
    assert metrics['stability']['full_target_class_support'] is False
    assert metrics['stability']['certification_status'] == 'incomplete'
    assert (
        'missing_target_class_support' in metrics['stability']['certification_blockers']
    )
    assert metrics['stability']['final_model_warning_messages'] == []
    assert all(
        fold_entry['messages'] == []
        for fold_entry in metrics['stability']['fold_warning_messages']
    )

    burden_metrics = json.loads(outputs['burden_metrics'].read_text(encoding='utf-8'))
    assert burden_metrics['n_examples'] == len(embeddings)
    assert 'prediction_set_coverage' in burden_metrics['overall']

    review_examples = pd.read_csv(outputs['ordinal_review_examples'])
    assert len(review_examples) > 0
    assert outputs['ordinal_review_html'].stat().st_size > 0
    assert any(outputs['ordinal_review_assets_dir'].iterdir())
    assert outputs['quantification_review_html'].stat().st_size > 0
