## Why

The full-cohort quantification run now has a credible segmented ROI input surface, but the image-level grading model is still using the wrong target contract: it treats `2.5` as an expected score bin even though the current scored cohort uses `0`, `0.5`, `1`, `1.5`, `2`, and `3`. The project also needs a model output that is easier to use for quantification than a single ordinal class, while remaining honest that the available labels are image-level severity grades rather than pixel-level percent tissue burden.

## What Changes

- Correct the quantification target label space to the observed rubric values `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]`; absence of `2.5` SHALL NOT be reported as missing target-class support.
- Add a cumulative threshold severity model over the explicit rubric thresholds `score > 0`, `score > 0.5`, `score > 1`, `score > 1.5`, and `score > 2`.
- Add a derived `endotheliosis_burden_0_100` output computed from calibrated cumulative threshold probabilities and documented as a grade-equivalent severity burden index, not a true pixel-area percentage.
- Define the target estimand explicitly at two levels: the per-image target is expected ordinal stage burden, and the primary biological summary is the mean of animal-level mean burdens rather than a raw image-row mean.
- Add calibrated per-image uncertainty outputs for the burden estimate, with score prediction sets as the primary uncertainty object and lower/upper burden prediction bounds as secondary derived outputs. These are prediction intervals for individual image-level estimates, not classical confidence intervals for a population mean.
- Add cohort/animal/group summary confidence intervals using animal-level or other certified biological-group resampling so downstream treatment or cohort summaries can report uncertainty around aggregate burden estimates without pretending image rows are independent biological units.
- Add model-evidence artifacts that explain each prediction with threshold probability profiles, nearest scored examples in embedding space, and optional ROI attribution diagnostics labeled as heuristic support rather than causal or mechanistic proof.
- Extend the existing quantification review surface so a reviewer can inspect testing/fold-held-out examples, observed versus predicted grade/burden outputs, uncertainty, nearest examples, ROI provenance, and cohort/animal summaries in one human-readable report.
- Keep the existing ordinal/multiclass output as a comparator and diagnostic surface, not as the only user-facing quantification result.
- Add direct regression and current ordinal/multiclass comparators so the final report can state whether cumulative burden modeling is actually better for the current 707-row full scored cohort.
- Add grouped validation metrics for the cumulative threshold model, including animal-level threshold support gates, threshold-level Brier scores, threshold-level discrimination where estimable, stage-index MAE as the primary absolute-error metric, grade-scale MAE as secondary, calibration summaries, cohort-stratified performance, and numerical-stability status.
- Add a grouping-key leakage audit before model selection so the implementation either certifies `subject_prefix` as the biological animal identifier or uses a stronger `animal_id` grouping column for cross-validation, conformal calibration, nearest-neighbor evidence, and aggregate intervals.
- Update `configs/endotheliosis_quantification.yaml` and the `eq run-config --config configs/endotheliosis_quantification.yaml` workflow so the current full-cohort run writes burden-index artifacts under the existing quantification output root.
- Preserve the maintained quantification CLI structure: `eq run-config --config configs/endotheliosis_quantification.yaml` remains the reproducible workflow front door, `eq quant-endo` remains the direct one-shot quantification entrypoint, and `eq prepare-quant-contract` remains the contract/scored-example preparation entrypoint. Update stale CLI/help/report wording that calls the quantification model only a baseline.
- End implementation with a thorough subagent review packet covering statistical validity, implementation fidelity, reporting clarity, documentation consistency, and robustness/regression risks before recording the final OpenSpec verdict.
- Produce a public-facing results summary from the refreshed run so README/docs updates can show concrete quantification results rather than generic workflow claims.
- **BREAKING**: downstream code that assumes a seven-bin label grid including `2.5` must switch to the six-bin rubric contract or treat `2.5` as unsupported data requiring explicit adjudication.

## Capabilities

### New Capabilities
- `endotheliosis-burden-index`: Defines the cumulative threshold severity model, `endotheliosis_burden_0_100` output semantics, validation metrics, calibration artifacts, and report requirements.

### Modified Capabilities
- `ordinal-quantification-stability`: Replaces the old seven-bin target-support assumption with the current six-bin rubric contract and requires ordinal/multiclass outputs to remain numerically stable when used as comparator artifacts.

## Impact

- Affected modules:
  - `src/eq/quantification/pipeline.py`
  - `src/eq/quantification/ordinal.py`
  - added implementation module `src/eq/quantification/burden.py`
- Affected config:
  - `configs/endotheliosis_quantification.yaml`
- Affected command:
  - `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`
- Affected output root:
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated`
- New or updated output artifacts:
  - `burden_model/burden_predictions.csv`
  - `burden_model/burden_metrics.json`
  - `burden_model/threshold_metrics.csv`
  - `burden_model/calibration_bins.csv`
  - `burden_model/uncertainty_calibration.json`
  - `burden_model/prediction_explanations.csv`
  - `burden_model/nearest_examples.csv`
  - `burden_model/cohort_metrics.csv`
  - `burden_model/group_summary_intervals.csv`
  - `burden_model/burden_model.joblib`
  - `quantification_review/quantification_review.html`
  - `quantification_review/review_examples.csv`
  - `quantification_review/results_summary.md`
  - `quantification_review/results_summary.csv`
  - `quantification_review/readme_results_snippet.md`
  - updated HTML or Markdown quantification review report that leads with the burden-index interpretation, shows uncertainty intervals and model-evidence examples, and shows ordinal/multiclass comparator metrics.
- Tests:
  - unit tests for six-bin score validation, cumulative target construction, monotonic threshold probabilities, burden-index calculation, missing-class behavior, and output schema.
  - integration or focused pipeline test proving the full-cohort manifest path writes the burden artifacts and does not report missing `2.5` support.
- Scientific claim boundary:
  - The output is predictive/prognostic grade-equivalent severity modeling from manually scored image-level labels.
  - The output is not a causal estimate, not an externally validated clinical endpoint, and not a pixel-level percent endotheliosis measurement unless future annotation data support that interpretation.

## Explicit Decisions

- Change name: `p0-model-endotheliosis-burden-index`.
- Primary new public numeric output: `endotheliosis_burden_0_100`.
- Human-readable label: `Endotheliosis burden index (0-100)`.
- Required score rubric: `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]`.
- Required cumulative thresholds: `[0.0, 0.5, 1.0, 1.5, 2.0]` with probability columns `prob_score_gt_0`, `prob_score_gt_0p5`, `prob_score_gt_1`, `prob_score_gt_1p5`, and `prob_score_gt_2`.
- Burden-index formula: `100 * mean([prob_score_gt_0, prob_score_gt_0p5, prob_score_gt_1, prob_score_gt_1p5, prob_score_gt_2])`.
- Burden-index estimand: normalized expected crossed rubric stages. The exact observed-score mapping under perfect prediction is `0 -> 0`, `0.5 -> 20`, `1.0 -> 40`, `1.5 -> 60`, `2.0 -> 80`, and `3.0 -> 100`.
- Primary aggregate estimand: mean of animal-level mean burden indices, not the unweighted mean over image rows. Group comparisons, when reported, are contrasts between animal-level means.
- Required per-row uncertainty columns: `burden_interval_low_0_100`, `burden_interval_high_0_100`, `burden_interval_coverage`, `burden_interval_method`, and `prediction_set_scores`.
- Required group-summary uncertainty artifact: `burden_model/group_summary_intervals.csv`.
- Required model-evidence artifacts: `burden_model/prediction_explanations.csv` and `burden_model/nearest_examples.csv`.
- Required reviewer-facing report root: `quantification_review/`, with `quantification_review.html`, `review_examples.csv`, `results_summary.md`, `results_summary.csv`, and `readme_results_snippet.md`.
- Required grouping audit artifact: `burden_model/grouping_audit.json`.
- Required support-gate artifact: `burden_model/threshold_support.csv`.
- Required final review artifact: `audit-results.md` SHALL include subagent-review findings and the concrete quantification results selected for README/docs consideration.
- Target-model implementation module: `src/eq/quantification/burden.py`.
- Existing ordinal comparator module remains `src/eq/quantification/ordinal.py`.
- Current full-cohort workflow remains `workflow: endotheliosis_quantification` in `configs/endotheliosis_quantification.yaml`.
- CLI names remain `prepare-quant-contract` and `quant-endo`; this change updates `eq quant-endo` help text, `eq quant-endo` terminal copy, README quantification wording, onboarding quantification wording, and generated quantification report wording, but does not introduce a rename or alias.
- Current full-cohort output root remains `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated`.

## Open Questions

- [audit_first_then_decide] Whether cumulative threshold modeling, direct regularized regression, or the existing ordinal/multiclass comparator is the best operational model for current use SHALL be decided from the refreshed full-cohort validation artifacts generated by this change.

- [defer_ok] Whether to later rename `endotheliosis_burden_0_100` in collaborator-facing reports can be revisited after reviewing interpretability with domain users; the implementation contract uses that column name for this change.
