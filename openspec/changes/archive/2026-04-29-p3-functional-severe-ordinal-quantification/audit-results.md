# P3 Audit Results

## Input Evidence

Runtime root inspected:

`/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model`

Current P3 inputs are reconstructed from read-only burden-model artifacts:

- `primary_burden_index/model/burden_predictions.csv`
- `primary_burden_index/feature_sets/morphology_features.csv`
- `learned_roi/feature_sets/learned_roi_features.csv`
- `source_aware_estimator/summary/estimator_verdict.json`
- `severe_aware_ordinal_estimator/summary/estimator_verdict.json`
- `severe_aware_ordinal_estimator/evidence/severe_false_negative_adjudications.json`

Observed support:

- Rows: `707`
- Subjects: `60`
- Cohorts: `vegfri_dox=619`, `lauren_preeclampsia=88`
- Score counts: `0=204`, `0.5=194`, `1=113`, `1.5=93`, `2=81`, `3=22`
- Original severe rows (`score >= 2`): `103`
- P2 adjudication records: `88`
- P2 adjudication summary: `adjudicated_still_severe=56`, `adjudicated_not_severe=32`
- Required identity fields are present for candidate rows: `subject_id`, `subject_image_id`, `cohort_id`, `score`, `roi_image_path`, and `roi_mask_path`

Primary severe target: adjudicated severe. Rows with P2 false-negative adjudication records use the adjudicated result; other rows use `score >= 2`.

P2 feasibility diagnostic:

- Morphology-only balanced logistic severe screen: AUROC `0.829`, average precision `0.326`
- Threshold `0.5`: recall `0.676`, precision `0.262`, false negatives `23/71`
- Threshold near `0.477`: recall `0.803`, precision `0.288`, false negatives `14/71`
- Threshold `0.3`: recall `0.944`, precision `0.211`, false negatives `4/71`

Conclusion: P2 did not prove the current data are signal-free. P3 evaluates a final current-data product ladder: MR TIFF deployable grade model first, MR TIFF severe-risk triage second, diagnostic-only third.

## Implementation Evidence

Implemented evaluator:

- `src/eq/quantification/endotheliosis_grade_model.py`
- exported from `src/eq/quantification/__init__.py`
- called from `src/eq/quantification/pipeline.py` after severe-aware P2 artifacts

P3 writes the selector subtree under:

`burden_model/endotheliosis_grade_model/`

It also writes first-class family subtrees:

- `burden_model/three_band_ordinal_model/`
- `burden_model/four_band_ordinal_model/`
- `burden_model/severe_triage_model/`
- `burden_model/aggregate_grade_model/`
- `burden_model/embedding_grade_model/`

Grouped development validation:

- deterministic subject-level folds
- no fixed internal locked test split
- metrics labeled `grouped_out_of_fold_development_estimate`
- fold file: `endotheliosis_grade_model/splits/development_folds.csv`

Candidate families evaluated:

- empirical-prior and majority baselines
- ROI/QC severe gates
- morphology severe gates
- ROI/QC plus morphology severe gates
- learned ROI severe gates
- embedding plus morphology severe gates
- aggregate-aware severe gate
- exploratory tree severe gate
- three-band ordinal candidates
- four-band ordinal candidates
- six-bin majority baseline

Candidate warning status is recorded in candidate metrics. The selected final refit also records warning status in `model/final_model_metadata.json`.

## Runtime Result

Standalone P3 runtime command:

```bash
PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd
from eq.quantification.endotheliosis_grade_model import evaluate_endotheliosis_grade_model

root = Path('/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model')
embedding_df = pd.read_csv(root / 'primary_burden_index/model/burden_predictions.csv')
artifacts = evaluate_endotheliosis_grade_model(embedding_df, root, n_splits=3)
verdict = json.loads(artifacts['endotheliosis_grade_model_verdict'].read_text())
print(json.dumps(verdict, indent=2))
PY
```

Pre-reviewed-rubric P3 verdict after the research-partner audit extension, source-truthful candidate registration, severe-focused interaction features, and bounded logistic regularization sweep:

- Overall status: `diagnostic_only_current_data_model`
- Selected candidate: `roi_qc_morphology_three_band_ordinal`
- Selected family: `three_band_ordinal_model`
- Selected output type: `three_band`
- Quantification gate passed: `false`
- Severe safety gate passed: `false`
- MR TIFF deployment gate passed: `false`
- README-facing deployment allowed: `false`
- Hard blockers: `no_candidate_passed_quantification_gate`, `severe_safety_gate_failed`, `ordinal_grade_gate_failed`

Best severe-risk grouped out-of-fold results after the bounded sweep:

- Highest recall: `roi_qc_severe_recall_0.95_c1`, recall `0.930`, precision `0.125`, false negatives `5`, false positives `463`
- Best near-gate deployable morphology/interactions result: `severe_interactions_severe_recall_0.9_c0.01`, recall `0.817`, precision `0.208`, false negatives `13`, false positives `221`
- Best morphology precision at useful recall remained below gate: `morphology_severe_recall_0.95_c1`, recall `0.803`, precision `0.247`, false negatives `14`, false positives `174`

Strongest ordinal candidate:

- Candidate: `roi_qc_morphology_three_band_ordinal`
- Accuracy: `0.560`
- Balanced accuracy: `0.503`
- Severe-band recall: `0.515`
- Non-adjacent error rate: `0.088`
- Gate status: not deployable; severe-band recall is below the `>=0.80` ordinal gate

Final artifact status:

- `burden_model/endotheliosis_grade_model/model/` contains no final deployable model artifacts after the failed final gates.
- Stale final model artifacts from the earlier intermediate run were removed.
- Learned-feature candidates were tested only when learned columns existed.
- Embedding-heavy candidates were not registered when actual `embedding_*` columns were absent from the source frame, preventing a mislabeled morphology-only candidate from being selected as an embedding candidate.
- Minimum additional-data recommendations were written into the final verdict: source-diverse grade 2/3 MR TIFF examples with accepted glomerulus masks, direct labels for multi-component aggregate ambiguity, and model-stability work sufficient to clear severe-risk gates without heavy numeric warnings.

Warning evidence:

- Candidate warning status is recorded in metrics.
- Warning class: sklearn/numpy numeric `RuntimeWarning` during logistic fitting (`divide by zero`, `overflow`, and `invalid value encountered in matmul`)
- Candidate outputs remained finite, but the warning burden remained a modeling risk in the pre-reviewed-rubric diagnostic-only verdict and remains a residual risk for the later model-ready severe-triage result.

Research-partner repo-wide audit result:

- Repeated grouped-validation, threshold-selection, finite-feature, warning-capture, JSON, and artifact-manifest mechanics were found across quantification evaluators.
- P3 now centralizes newly touched shared mechanics in `src/eq/quantification/modeling_contracts.py`.
- Older P0/P1/P2 evaluator cleanup remains a future refactor target, but P3 no longer adds another local copy of these contracts.

## Validation

Commands run:

- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m py_compile src/eq/quantification/endotheliosis_grade_model.py src/eq/quantification/pipeline.py src/eq/quantification/__init__.py`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_endotheliosis_grade_model.py`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_endotheliosis_grade_model.py tests/unit/test_quantification_pipeline.py tests/unit/test_quantification_burden_artifact_layout_contract.py tests/unit/test_quantification_severe_aware_ordinal_estimator.py tests/unit/test_quantification_learned_roi.py tests/unit/test_quantification_source_aware_estimator.py`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check src/eq/quantification/endotheliosis_grade_model.py src/eq/quantification/pipeline.py src/eq/quantification/severe_aware_ordinal_estimator.py src/eq/quantification/__init__.py tests/unit/test_quantification_endotheliosis_grade_model.py`
- `openspec validate p3-functional-severe-ordinal-quantification --strict`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python scripts/check_openspec_explicitness.py openspec/changes/p3-functional-severe-ordinal-quantification`

Final validation status:

- Full config workflow: `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml` completed successfully in `763.397` seconds and reproduced the final P3 verdict
- Focused P3 unit test: `5 passed, 4 warnings`
- Focused quantification suite: `32 passed, 4 warnings`
- `ruff check src/eq/quantification tests/unit`: passed
- `ruff format --check src/eq/quantification tests/unit`: passed
- `openspec validate p3-functional-severe-ordinal-quantification --strict`: passed
- `scripts/check_openspec_explicitness.py openspec/changes/p3-functional-severe-ordinal-quantification`: passed

## Dox Scored-No-Mask Resolution Audit

P3 now includes a pre-MR Dox scored-no-mask smoke input audit. The audit resolves Dox `scored_only` manifest rows to exact Label Studio upload images, flags duplicate source image names, conflicting source-image scores, and missing scores, then writes a conservative clean smoke manifest.

Runtime command:

```bash
PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq dox-scored-only-resolution-audit
```

Runtime outputs:

- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/vegfri_dox/metadata/dox_scored_only_resolution_audit.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/vegfri_dox/metadata/dox_scored_no_mask_smoke_manifest.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/vegfri_dox/metadata/dox_scored_only_resolution_audit.summary.json`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/vegfri_dox/scored_no_mask_smoke/images/`

Audit counts:

- Dox scored-only rows: `238`
- Resolved exactly once to Label Studio upload image: `214`
- Duplicate source-image-name rows: `24`
- Conflicting-score rows: `10`
- Missing-score rows: `3`
- Clean Dox scored-no-mask smoke rows: `212`
- Localized runtime image rows: `212`

Interpretation:

- A real Dox scored-no-mask smoke set exists and is simpler than the MR TIFF surface.
- Clean smoke images are copied into the runtime cohort tree; the master manifest stores the localized image in canonical `image_path`, keeps `mask_path` empty, and records `eligible_dox_scored_no_mask_smoke` plus Dox smoke status/audit columns.
- The clean set is not external validation; it is a pre-MR segmentation-to-quantification bridge in the familiar Dox image domain.
- Duplicate/conflicting/missing-score rows remain visible in the audit but are excluded from the clean smoke manifest.

## Dox Scored-No-Mask Deployment Smoke

P3 now runs the clean Dox scored-no-mask smoke set through the configured segmentation artifact before MR TIFF deployment can proceed. The stage uses the normal manifest workflow, loads the configured current-namespace glomeruli segmentation artifact, computes the P3 inference feature schema from accepted predicted ROIs, loads `model/final_model.joblib`, writes image-level severe-risk predictions when ROIs exist, and compares predictions against the Dox human image-level grades.

Runtime command:

```bash
env PYTHONPATH=src EQ_RUNTIME_ROOT=/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml
```

Runtime outputs:

- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/endotheliosis_grade_model/deployment/dox_scored_no_mask_smoke_manifest.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/endotheliosis_grade_model/deployment/dox_scored_no_mask_smoke_predictions.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/endotheliosis_grade_model/deployment/dox_scored_no_mask_smoke_summary.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/endotheliosis_grade_model/deployment/dox_scored_no_mask_smoke_report.html`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/endotheliosis_grade_model/deployment/dox_scored_no_mask_smoke_contract.json`

Result:

- Dox smoke status: `failed`
- Manifest rows: `212`
- Segmentation inference device: `mps`
- Tiling policy: `512` pixel tiles, `512` pixel stride, max-probability merge into whole-field coordinates
- Accepted predicted ROI rows: `210`
- Per-image `roi_status`: `ok` for `210` rows; `component_not_found` for `2` rows
- Observed severe rows among accepted ROIs: `9`
- Predicted severe rows among accepted ROIs at selected operating threshold `0.323599`: `185`
- False negative rows: `0`
- Severe recall on accepted Dox smoke rows: `1.0`
- Severe precision on accepted Dox smoke rows: `0.048649`
- Best observed Dox threshold with recall at least `0.90`: threshold `0.506129`, predicted severe rows `149`, severe precision `0.060403`, severe recall `1.0`
- Final verdict before cluster-representative review was `model_ready_pending_mr_tiff_deployment_smoke`
- Dox gate: `dox_scored_no_mask_smoke_gate_passed = false`
- Added hard blockers: `dox_scored_no_mask_smoke_not_passed` and `mr_tiff_deployment_blocked_until_dox_smoke_passes`
- Dox failure reason: `segmentation_missing_accepted_rois|dox_smoke_severe_precision_below_0.15`

Interpretation:

- The initial full-image resize smoke path was invalid for this segmentation artifact. The configured segmenter was trained and validated with `512` pixel crops resized to `256`, so P3 now tiles each Dox smoke image before inference.
- With tiled MPS inference, segmentation does work on the Dox smoke images: `210/212` images produce accepted ROIs.
- The current product still fails the Dox smoke gate because it produces too many severe flags in this scored-no-mask Dox surface: it catches all observed severe examples but at severe precision `0.048649` at the selected operating threshold, below the high-sensitivity triage floor of `0.15`.
- Threshold tuning alone does not rescue the Dox surface; even the best threshold with recall at least `0.90` still flags `149` rows and reaches precision only `0.060403`.
- MR TIFF deployment is intentionally blocked before attempting the larger MR surface because the smaller Dox smoke exposed a quantification/product-quality failure after segmentation was corrected.

## Dox Overcall Active Triage

P3 now writes a bounded overcall review surface from the Dox smoke predictions so the next manual review is representative rather than a broad relabeling pass.

Runtime outputs:

- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/endotheliosis_grade_model/deployment/dox_scored_no_mask_overcall_triage_queue.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/endotheliosis_grade_model/deployment/dox_scored_no_mask_overcall_triage_report.html`

Queue counts:

- Total rows: `51`
- Cluster representative false positives: `12`
- Highest-confidence false positives: `18`
- Threshold-boundary false positives: `10`
- Human severe references: `9`
- Segmentation misses: `2`

Interpretation:

- The queue is generated from deployment-computable prediction fields already written by Dox smoke. It does not rerun segmentation.
- Review priority is cluster representatives first, then high-confidence overcalls, then threshold-boundary overcalls, with true severe references and segmentation misses included for calibration.
- This review should answer whether overcalls are mostly rubric ambiguity, segmentation crop artifacts, nonspecific ROI/QC features, or a true label/model mismatch.

## Reviewed Dox Overcall Wrap-Up

The first twelve cluster-representative false positives were reviewed in the Dox overcall triage queue. P3 now treats that bounded review as a hard diagnostic instead of asking for more broad relabeling.

Runtime outputs:

- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/endotheliosis_grade_model/deployment/dox_scored_no_mask_overcall_review_diagnostic.json`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/endotheliosis_grade_model/deployment/dox_scored_no_mask_first12_review_interpretation.csv`

Reviewed cluster-representative counts:

- Reviewed cluster representative rows: `12`
- Clearly usable rows: `7`
- Usable-or-uncertain rows: `11`
- Unusable rows: `1`
- Reviewer severe rows: `1`
- Reviewer severe among clearly usable rows: `0`
- Reviewer non-severe among clearly usable rows: `7`
- Reviewer non-severe among usable-or-uncertain rows: `10`

Final wrap-up decision:

- `dox_overcall_confirmed = true`
- `overall_status = diagnostic_only_current_data_model`
- `quantification_gate_passed = false`
- `severe_safety_gate_passed = false`
- `selected_output_type = severe_risk_triage_diagnostic_only`
- Added hard blockers: `dox_review_confirmed_selected_candidate_overcalls_nonsevere_usable_rois` and `selected_severe_triage_candidate_rejected_after_dox_review`
- Removed stale final deployable artifacts: `model/final_model.joblib`, `model/final_model_metadata.json`, `model/inference_schema.json`, `model/deployment_smoke_predictions.csv`, and `predictions/final_model_training_predictions.csv`

Interpretation:

- The reviewed Dox cluster representatives do not support fixing this by changing many labels. Most clearly usable representative false positives remained non-severe by review.
- The current selected ROI/QC severe-risk model is too nonspecific for deployment, even as high-sensitivity triage.
- P3 should wrap as diagnostic-only for the current data/model state. The next productive work is a separate model-improvement change, not more broad manual review inside P3.

## Residual Risks

- The reviewed-rubric normal workflow initially produced a model-ready severe-risk triage candidate, but reviewed Dox overcall evidence downgraded it to diagnostic-only.
- Dox scored-no-mask smoke is implemented and tiled MPS segmentation produces accepted ROIs for `210/212` clean smoke images, but the selected severe model overcalls badly on that surface and is rejected.
- MR TIFF segmentation-to-quantification deployment has not yet been attempted after the Dox failure; the current blockers include `dox_scored_no_mask_smoke_not_passed`, `mr_tiff_deployment_blocked_until_dox_smoke_passes`, and `mr_tiff_segmentation_to_quantification_path_not_proven`.
- The selected high-recall severe candidate still has many false positives; it is a high-sensitivity review triage product, not diagnostic automation.
- Candidate fitting emits many numeric warnings, now captured in metrics. Finite output alone is not proof of model stability.
- Ordinal grade-band output did not pass the current gate.

## Reviewed Rubric Override Rerun

Reviewed rubric label overrides were promoted from isolated experiment artifacts into the normal quantification workflow as an explicit `inputs.label_overrides` config input. The configured override file is:

`/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/quantification_review/rubric_application_experiment/expanded_rubric_review/rubric_label_overrides_for_next_modeling_run.csv`

The normal workflow applied only explicit reviewed row-level overrides keyed by `subject_image_id`; no nearest-anchor projected labels were applied. The workflow wrote:

- `scored_examples/score_label_overrides_audit.csv`
- `scored_examples/score_label_overrides_summary.json`

Override summary:

- Scored rows: `707`
- Override rows applied: `169`
- Changed rows: `112`
- Severe-boundary changed rows: `51`
- Claim boundary: `explicit reviewer label overrides only; no inferred labels applied`

Normal config rerun command:

```bash
PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml
```

Normal config rerun result:

- Completed successfully in `790.522` seconds
- Log path: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/logs/run_config/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/2026-04-29_131146.log`
- P3 overall status before reviewed Dox cluster-representative overcall rejection: `model_ready_pending_mr_tiff_deployment_smoke`
- Selected candidate: `roi_qc_severe_recall_0.95_c10`
- Selected family: `severe_triage_model`
- Selected output type: `severe_risk_triage`
- Quantification gate passed: `true`
- Severe safety gate passed: `true`
- MR TIFF deployment gate passed: `false`
- README-facing deployment allowed: `false`
- Hard blocker: `mr_tiff_segmentation_to_quantification_path_not_proven`

Selected severe-triage grouped development metrics:

- Average precision: `0.202`
- AUROC: `0.646`
- Recall: `0.944`
- Precision: `0.181`
- False negatives: `6`
- False positives: `458`

Strongest ordinal candidate remains non-deployable:

- Candidate: `roi_qc_morphology_three_band_ordinal`
- Accuracy: `0.529`
- Balanced accuracy: `0.480`
- Severe-band recall: `0.477`
- Non-adjacent error rate: `0.075`
- Gate status: failed ordinal gate

Final artifact status before reviewed Dox cluster-representative overcall rejection:

- `burden_model/endotheliosis_grade_model/model/final_model.joblib` exists
- `burden_model/endotheliosis_grade_model/model/final_model_metadata.json` exists
- `burden_model/endotheliosis_grade_model/model/inference_schema.json` exists
- `burden_model/endotheliosis_grade_model/model/deployment_smoke_predictions.csv` exists

Interpretation:

- P3 briefly became model-ready severe-risk triage after reviewed rubric overrides were applied through the normal workflow.
- The subsequent Dox scored-no-mask smoke and twelve-row cluster-representative review rejected that candidate as overcalling clearly usable non-severe ROIs.
- README-facing deployment remains blocked, and MR TIFF deployment should not proceed until a less nonspecific candidate clears the Dox smoke gate.
- The high false-positive burden and numeric warning burden remain important risks; the current result is diagnostic-only, not external validation or diagnostic automation.

Additional validation after label-override integration and Dox overcall wrap-up:

- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_pipeline.py tests/unit/test_quantification_endotheliosis_grade_model.py`: `15 passed, 4 warnings`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_endotheliosis_grade_model.py`: `9 passed, 4 warnings`
- `ruff check src/eq/quantification/pipeline.py src/eq/quantification/run_endotheliosis_quantification_workflow.py src/eq/quantification/endotheliosis_grade_model.py tests/unit/test_quantification_pipeline.py tests/unit/test_quantification_endotheliosis_grade_model.py`: passed
- `ruff format --check src/eq/quantification/pipeline.py src/eq/quantification/run_endotheliosis_quantification_workflow.py src/eq/quantification/endotheliosis_grade_model.py tests/unit/test_quantification_pipeline.py tests/unit/test_quantification_endotheliosis_grade_model.py`: passed
- `openspec validate p3-functional-severe-ordinal-quantification --strict`: passed
- `scripts/check_openspec_explicitness.py openspec/changes/p3-functional-severe-ordinal-quantification`: passed
