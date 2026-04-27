# P0 Model Endotheliosis Burden Index Audit Results

## Runtime Command

Refreshed full-cohort workflow after identity, score-workbook, validation, and stability fixes:

```bash
env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src \
  /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml
```

Completion observed: `2026-04-26 23:49:18`. Output root:

`/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated`

## Data Integrity Verdict

The identity and score-source problem is fixed for the current admitted mask-paired quantification set.

- `Rand_Assign.xlsx` is the authoritative VEGFRi/Dox subject identity and treatment-group source.
- `M1` and `M1--2023-06-12` are separate subjects because they come from different assignment sheets and have different alternate sample IDs.
- The dated long score workbook `2023-11-16_all-labeled-glom-data_score-table-filtered.xlsx` is the definitive current VEGFRi/Dox score table for this workflow.
- The older undated `all-score-table-filtered.xlsx` is not the full current source: it has 596 rows and misses sample `59`.
- Current admitted VEGFRi/Dox score agreement against the dated long workbook is `619/619 matched`.
- Non-admitted score audit rows still include foreign/unresolved mismatches or missing scores; they are not admitted into the training/evaluation cohort.

Score audit artifacts:

- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/vegfri_dox/metadata/score_workbook_agreement.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/vegfri_dox/metadata/score_workbook_agreement_summary.json`

## Identity Contract

From `scored_examples/manifest_scored_examples_summary.json`:

- Scored rows: `707`
- Subjects: `60`
- Scored image replicates: `707`
- Image IDs: `707`
- Duplicate image IDs: none
- Validation group key: `subject_id`

Current cohort split:

- `lauren_preeclampsia`: `88` rows, `8` subjects
- `vegfri_dox`: `619` rows, `52` subjects

## Model Readiness Verdict

Status: `exploratory_not_ready`.

What is ready:

- Manifest identity is now subject-preserving and workbook-backed.
- Current admitted Dox scores agree with the dated definitive score workbook.
- Subject-heldout grouped validation is implemented.
- Cohort-level stability between subject-heldout validation and final full-cohort fitting passes.
- Overall threshold support passes.

What is not ready:

- Per-image uncertainty remains broad: average prediction-set size is `5.308` of 6 possible score labels.
- Prediction-set coverage is just below nominal: `0.898` vs target `0.900`.
- Backend matrix-operation warnings remain recorded, although outputs are finite.
- The current result is not README/docs-ready as an operational model claim.

## Refreshed Results

Primary burden model, from `burden_model/burden_metrics.json`:

- Examples: `707`
- Subjects used for grouped validation: `60`
- Stage-index MAE: `22.617`
- Grade-scale MAE: `0.629`
- Prediction-set coverage: `0.898`
- Average prediction-set size: `5.308`
- Burden interval empirical coverage: `0.911`
- Support gate: `passed`
- Numerical stability: `backend_warnings_outputs_finite`

Cohort-level held-out validation summaries, from `burden_model/cohort_metrics.csv`:

- `lauren_preeclampsia`: subject-weighted mean burden `20.989`, stage-index MAE `18.686`
- `vegfri_dox`: subject-weighted mean burden `30.912`, stage-index MAE `23.176`

Final full-cohort fitted summaries, from `burden_model/final_model_cohort_metrics.csv`:

- `lauren_preeclampsia`: subject-weighted mean burden `20.403`, apparent stage-index MAE `10.263`
- `vegfri_dox`: subject-weighted mean burden `32.916`, apparent stage-index MAE `12.216`

Cohort stability, from `burden_model/cohort_stability.csv`:

- `lauren_preeclampsia`: validation-vs-final difference `0.586`, gate `passed`
- `vegfri_dox`: validation-vs-final difference `2.003`, gate `passed`

Signal screen, from `burden_model/signal_comparator_metrics.csv`:

- Embedding-only ridge: stage-index MAE `34.084`
- ROI scalar-only ridge: stage-index MAE `22.803`
- Embedding plus ROI scalar ridge: stage-index MAE `33.132`

Interpretation: ROI morphology/intensity signals are at least competitive with the current embedding-derived signal for burden prediction. This supports the next implementation direction: stronger quantification signal testing should not be limited to frozen segmentation embeddings.

## Artifact Contract Inspection

Implemented artifacts present:

- `burden_model/burden_predictions.csv`
- `burden_model/final_model_predictions.csv`
- `burden_model/burden_metrics.json`
- `burden_model/threshold_metrics.csv`
- `burden_model/threshold_support.csv`
- `burden_model/calibration_bins.csv`
- `burden_model/uncertainty_calibration.json`
- `burden_model/grouping_audit.json`
- `burden_model/validation_design.json`
- `burden_model/cohort_metrics.csv`
- `burden_model/final_model_cohort_metrics.csv`
- `burden_model/cohort_stability.csv`
- `burden_model/group_summary_intervals.csv`
- `burden_model/final_model_group_summary_intervals.csv`
- `burden_model/prediction_explanations.csv`
- `burden_model/nearest_examples.csv`
- `burden_model/signal_comparator_metrics.csv`
- `burden_model/burden_model.joblib`
- `ordinal_model/ordinal_predictions.csv`
- `ordinal_model/ordinal_metrics.json`
- `ordinal_model/ordinal_confusion_matrix.csv`
- `quantification_review/quantification_review.html`
- `quantification_review/review_examples.csv`
- `quantification_review/results_summary.md`
- `quantification_review/results_summary.csv`
- `quantification_review/readme_results_snippet.md`

## Validation Commands

Focused tests:

```bash
PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest \
  tests/unit/test_quantification_burden.py \
  tests/unit/test_quantification_pipeline.py \
  tests/integration/test_local_runtime_quantification_pipeline.py \
  tests/unit/test_quantification_cohorts.py -q
```

Result: `30 passed, 1 skipped, 8 warnings`.

Full suite:

```bash
PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q
```

Result: `203 passed, 3 skipped, 8 warnings`.

OpenSpec:

```bash
openspec validate p0-model-endotheliosis-burden-index --strict
```

Result: valid.

## Next Work Required

The biggest remaining blocker is model readiness, not data identity.

Next implementation targets:

- Improve per-image precision so prediction sets are meaningfully narrower than `5.308 / 6`.
- Follow the signal screen: test ROI scalar features, embedding plus ROI features, and simpler subject-level aggregation models as first-class candidates.
- Investigate and reduce backend matrix-operation warnings; keep outputs finite checks as hard gates.
- Keep subject-heldout validation and cohort-stability artifacts as required outputs for every candidate model.
- Use final full-cohort fitted predictions only for deployment-style cohort summaries, and keep held-out validation predictions separate for performance claims.

## Overnight Precision Candidate Expansion

Window requested by operator: `22:00 2026-04-26` to `08:00 2026-04-27`.

Implementation plan recorded and applied:

- Keep `burden_model/signal_comparator_metrics.csv` as the single canonical precision-candidate screen.
- Expand the screen beyond image-level frozen embeddings to include ROI scalar features, embedding-plus-ROI features, subject-level global-mean baseline, subject-level ROI scalar features, subject-level frozen embeddings, and subject-level embedding-plus-ROI features.
- Keep image-level candidates validated with subject-heldout folds.
- Evaluate subject-level candidates on one row per `subject_id`, with target equal to the subject mean of the 0-100 stage-index labels.
- Write subject-level out-of-fold predictions to `burden_model/subject_level_candidate_predictions.csv`.
- Write the candidate-selection summary and recommendation to `burden_model/precision_candidate_summary.json`.
- Surface the precision screen in `quantification_review/quantification_review.html`, `quantification_review/results_summary.md`, and `quantification_review/results_summary.csv`.

Commands run:

```bash
/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check \
  src/eq/quantification/burden.py \
  src/eq/quantification/pipeline.py \
  tests/unit/test_quantification_burden.py \
  tests/unit/test_quantification_pipeline.py
```

Result: passed.

```bash
PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest \
  tests/unit/test_quantification_burden.py \
  tests/unit/test_quantification_pipeline.py \
  tests/integration/test_local_runtime_quantification_pipeline.py \
  tests/unit/test_quantification_cohorts.py -q
```

Result: `30 passed, 1 skipped, 8 warnings`.

```bash
PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q
```

Result: `203 passed, 3 skipped, 8 warnings`.

```bash
openspec validate p0-model-endotheliosis-burden-index --strict
/Users/ncamarda/mambaforge/envs/eq-mac/bin/python scripts/check_openspec_explicitness.py \
  openspec/changes/p0-model-endotheliosis-burden-index
```

Result: OpenSpec valid and explicitness check passed.

```bash
env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src \
  /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml
```

Result: completed successfully at `2026-04-27 00:05:29`.

Refreshed precision artifacts:

- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/signal_comparator_metrics.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/subject_level_candidate_predictions.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/precision_candidate_summary.json`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/quantification_review/quantification_review.html`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/quantification_review/results_summary.md`

Precision candidate results:

| Candidate | Target level | Validation | Feature set | Stage-index MAE | Status |
| --- | --- | --- | --- | ---: | --- |
| `image_roi_scalar_only_ridge` | image | `subject_id_groupkfold` | ROI scalar | `22.803` | `finite_with_backend_warnings` |
| `image_embedding_plus_roi_scalar_ridge` | image | `subject_id_groupkfold` | frozen embedding plus ROI scalar | `33.132` | `finite_with_backend_warnings` |
| `image_embedding_only_ridge` | image | `subject_id_groupkfold` | frozen embedding | `34.084` | `finite_with_backend_warnings` |
| `subject_roi_scalar_only_ridge` | subject | `subject_kfold` | mean ROI scalar | `13.926` | `finite_with_backend_warnings` |
| `subject_global_mean_baseline` | subject | `subject_kfold` | none | `14.152` | `valid_finite` |
| `subject_embedding_plus_roi_scalar_ridge` | subject | `subject_kfold` | mean frozen embedding plus mean ROI scalar | `14.237` | `finite_with_backend_warnings` |
| `subject_embedding_only_ridge` | subject | `subject_kfold` | mean frozen embedding | `14.307` | `finite_with_backend_warnings` |

What worked:

- ROI scalar features are the strongest image-level comparator and are much better than frozen embeddings alone.
- Subject-level aggregation sharply improves the absolute-error target for subject/cohort summaries.
- The expanded artifact contract is now generated by the full workflow and surfaced in the HTML and Markdown review outputs.
- No precision candidate emitted nonfinite predictions.

What did not work:

- The best image-level screen (`22.803` stage-index MAE) did not improve on the current primary cumulative-threshold burden model (`22.617` stage-index MAE).
- The subject-level ROI model is promising for cohort/subject burden summaries, but it does not solve the per-image prediction-set problem because it answers a different target.
- Six of seven candidates still record backend matrix warnings, even though outputs are finite.
- The primary burden model remains below nominal prediction-set coverage (`0.898` versus `0.900`) with broad average set size (`5.308 / 6`).

Current decision:

- Keep the current primary cumulative-threshold burden model as the per-image exploratory model because it is calibrated into prediction sets and remains slightly better than the best image-level ROI ridge screen by stage-index MAE.
- Treat subject-level ROI aggregation as the strongest next direction for cohort-level summaries and quantification reporting.
- Do not call the quantification model README/docs-ready yet. The current status remains `exploratory_not_ready`.
- Next work should either integrate subject-level aggregation into the reporting estimand explicitly, or reduce the per-image uncertainty breadth with a stronger calibrated model family that keeps subject-heldout validation and finite-output gates.

## Follow-Up Readiness Plan Added 2026-04-27

Plain-language distinction:

- Image-level candidates answer: "Given one image/ROI row, what is its ordinal stage burden and uncertainty?" This is the correct target for individual image scoring and future per-glomerulus prediction, but it is currently uncertain.
- Subject-level candidates answer: "Given all image/ROI rows for one `subject_id`, what is that subject's mean burden?" This is the correct target for subject, cohort, treatment, and README-style burden summaries, but it does not prove individual image predictions are precise.

Why many rows do not automatically solve the problem:

- The workflow has `707` scored image rows, but subject-heldout validation has `60` independent subject groups.
- Multiple images from the same subject are correlated; they improve within-subject averaging, but they do not behave like 707 independent animals.
- The label is a coarse six-bin ordinal score, not a continuous measurement.
- The current issue is therefore not just "not enough data." It is a target-definition, feature-signal, calibration, label-noise, and effective-sample-size issue.

Plan encoded into the spec:

1. Make `subject_burden` and `per_image_burden` explicit report tracks.
2. Promote subject-level ROI aggregation into a first-class cohort-summary model candidate with subject-heldout validation, grouped bootstrap confidence intervals, and track-specific readiness gates.
3. Continue per-image improvement separately with ROI-feature cumulative-threshold models, embedding-plus-ROI models after diagnostics, and calibrated direct stage-index models.
4. Compare conformal calibration strategies to fix broad prediction sets and slight undercoverage.
5. Diagnose backend matrix warnings with feature diagnostics before model fitting.
6. Make README/docs readiness track-specific so a cohort-summary result cannot be mistaken for per-image operational readiness.

The next apply pass should implement section 10 of `tasks.md`.
