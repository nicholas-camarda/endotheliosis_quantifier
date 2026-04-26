## Context

The current full-cohort quantification workflow runs through `eq run-config --config configs/endotheliosis_quantification.yaml`, builds a 707-row scored image/mask cohort from `raw_data/cohorts/manifest.csv`, extracts frozen segmentation/encoder embeddings, and trains the image-level grade predictor in `src/eq/quantification/pipeline.py::evaluate_embedding_table()`. That function currently uses `ALLOWED_SCORE_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]`, which makes the absence of `2.5` look like a missing target-class failure even though the available scored rubric does not use `2.5`.

The current user-facing quantification output is also too class-centric. A single predicted grade or class probability table is useful for diagnostics, but downstream quantification needs a stable continuous summary that can be aggregated by animal, treatment group, cohort, or later large-field MR image. Because the labels are manually assigned image-level grades, not pixel-area annotations of diseased endothelium, the continuous output must be named and documented as a grade-equivalent burden index rather than true percent tissue involvement.

## Goals / Non-Goals

**Goals:**

- Make the six-bin score rubric `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]` the canonical quantification target contract.
- Add a cumulative threshold model in `src/eq/quantification/burden.py` that predicts ordered exceedance probabilities for `score > 0`, `score > 0.5`, `score > 1`, `score > 1.5`, and `score > 2`.
- Write `endotheliosis_burden_0_100` into a new `burden_model/burden_predictions.csv` artifact using `100 * mean(cumulative threshold probabilities)`.
- Define the statistical estimands explicitly: per-image expected ordinal stage burden, animal-level mean burden, and group contrasts of animal-level means.
- Write calibrated per-image uncertainty bounds and score prediction sets beside each burden estimate.
- Write grouped aggregate confidence intervals for animal, cohort, and other summary strata where the required grouping columns are available.
- Write model-evidence artifacts that make the prediction inspectable with threshold probability profiles, nearest scored examples, and optional ROI attribution diagnostics.
- Extend the existing quantification example-review surface so reviewers can inspect representative held-out/test-fold rows, ROI crops, observed grade, burden prediction, ordinal comparator prediction, uncertainty, and nearest-example evidence.
- Evaluate burden modeling with grouped subject-level cross-validation using the same group semantics as the current full-cohort quantification path.
- Keep the current ordinal/multiclass model as a comparator while removing the false `2.5` missing-support blocker.
- Make the final quantification report tell the operational verdict: whether the burden-index model is stable enough and empirically better than, comparable to, or worse than the direct regression and ordinal/multiclass comparators on the current cohort.

**Non-Goals:**

- This change does not retrain glomeruli segmentation models or change segmentation promotion gates.
- This change does not claim external clinical validation.
- This change does not convert image-level grade labels into true pixel-level percentage endotheliosis.
- This change does not add support for legacy FastAI pickle artifacts or old namespace shims.
- This change does not add unmasked MR large-field inference; it produces the quantification target model needed before that handoff.

## Decisions

### Decision: Use a cumulative threshold model as the primary operational output

Implement `CumulativeBurdenIndexModel` in `src/eq/quantification/burden.py`. The model trains one regularized logistic classifier per threshold:

- `score > 0`
- `score > 0.5`
- `score > 1`
- `score > 1.5`
- `score > 2`

Each threshold classifier uses standardized frozen embeddings, L2 regularization, deterministic random state, and the same certified biological-group cross-validation fold assignments used by `evaluate_embedding_table()`. If a training fold has only one class for a threshold, the threshold model records a constant-probability classifier rather than failing late or fabricating probabilities, and the threshold is marked underpowered for operational selection.

The implementation SHALL write `burden_model/threshold_support.csv` with row and biological-group support by threshold overall and by cohort. A public threshold with inadequate positive or negative biological-group support, or a threshold that is non-estimable within a cohort, SHALL prevent the burden model from being promoted as operational for that unsupported stratum. The workflow may still write exploratory artifacts, but the report must state that the model remains exploratory for the underpowered threshold or cohort.

Alternative considered: direct multiclass probability averaging from the current ordinal/multiclass model. That keeps a familiar artifact but inherits the class-sparsity and calibration problems that already made the current output hard to interpret. It remains a comparator.

Alternative considered: direct continuous regression to `score / 3 * 100`. This is simpler, but it treats the rubric as interval-scaled and compresses the skipped `2.5` step without testing whether ordered-threshold uncertainty is better. It remains a comparator.

### Decision: Define `endotheliosis_burden_0_100` from exceeded rubric stages

The burden index is:

`100 * mean([prob_score_gt_0, prob_score_gt_0p5, prob_score_gt_1, prob_score_gt_1p5, prob_score_gt_2])`

This maps a true score of `0` to a target burden of `0`, `0.5` to `20`, `1.0` to `40`, `1.5` to `60`, `2.0` to `80`, and `3.0` to `100` under perfect predictions. This is intentionally an ordinal-stage burden index, not `score / 3 * 100`.

The primary absolute-error metric SHALL be stage-index MAE against this same `0, 20, 40, 60, 80, 100` target. Raw grade-scale MAE remains secondary. Direct regularized regression may remain a comparator, but its target SHALL be the normalized stage index rather than `score / 3 * 100`; the latter implies interval spacing that the manual rubric and missing `2.5` score do not support.

### Decision: Report per-image uncertainty as prediction intervals, not classical confidence intervals

Score prediction sets are the primary uncertainty output because the observed label is a six-level ordered rubric. For each evaluated row, `burden_model/burden_predictions.csv` SHALL include:

- `burden_interval_low_0_100`
- `burden_interval_high_0_100`
- `burden_interval_coverage`
- `burden_interval_method`
- `prediction_set_scores`

The initial implementation SHALL use grouped subject-level conformal calibration or grouped out-of-fold calibration residuals so the prediction set and interval describe uncertainty for an individual image-level rubric-stage prediction. The report SHALL call these per-image outputs calibrated prediction sets, prediction intervals, or uncertainty bounds, not confidence intervals. Continuous burden intervals SHALL be derived from held-out grouped calibration and labeled as uncertainty for the unknown rubric label, not confidence about a biological truth or annotator agreement.

This distinction matters because a confidence interval normally describes uncertainty around a population parameter such as a cohort mean, whereas a single image prediction needs a predictive uncertainty interval. For group summaries, confidence intervals are appropriate when computed by resampling biological groups rather than individual image rows.

`burden_model/uncertainty_calibration.json` SHALL report empirical prediction-set coverage and average set size overall, by cohort, and by observed score stratum where estimable.

Alternative considered: use only model probability entropy or maximum class probability as a confidence proxy. That is easy to compute but is not calibrated to empirical error, so it remains a diagnostic field rather than the required uncertainty output.

### Decision: Add grouped confidence intervals for aggregate burden summaries

`burden_model/group_summary_intervals.csv` SHALL summarize burden estimates by available strata such as `cohort_id`, certified `animal_id`, treatment group, or other validated grouping columns. The primary biological summary SHALL be the mean of animal-level mean burdens, not the mean over image rows. Group comparisons, if reported, SHALL be contrasts between animal-level means.

Confidence intervals SHALL use grouped bootstrap or another grouped resampling method that resamples biological units rather than treating all image rows as independent. The artifact SHALL record the estimand, resampling unit, weighting rule, number of clusters, and non-estimable or unstable flags. Small-cluster strata SHALL be explicitly labeled; the report SHALL NOT present narrow pooled-looking intervals for strata with too few biological groups.

The report SHALL distinguish:

- per-image burden prediction intervals,
- subject/cohort/treatment aggregate confidence intervals,
- validation error metrics from held-out grouped folds.

Alternative considered: image-row bootstrap. That would understate uncertainty when multiple images come from the same subject and is therefore not acceptable as the default summary interval.

### Decision: Provide prediction evidence, not a mechanistic explanation claim

The burden report SHALL include model-evidence artifacts:

- `prediction_explanations.csv` with per-row threshold probability profiles, uncertainty interval fields, predicted score distribution or prediction set, and links back to source ROI/image rows.
- `nearest_examples.csv` with nearest scored examples in embedding space, including their scores, distances, cohort/source identifiers, and image or ROI provenance.
- optional visual attribution panels when the selected model path can produce them without adding a second inference path.

The report SHALL describe these outputs as evidence supporting the prediction, not as proof that the model causally identified a specific histologic mechanism. For the initial implementation, nearest-neighbor evidence and threshold probability profiles are required; visual attribution can be included only if it is implemented through the same frozen-embedding/ROI path and labeled as heuristic.

For out-of-fold predictions, nearest examples SHALL be selected only from the corresponding training fold and SHALL exclude the same biological animal. Neighbor artifacts SHALL carry cohort identifiers, `lane_assignment` when available, source provenance, and distance metrics. Calibration SHALL be reported per threshold, not only through pooled burden bins, so cohort-confounded failures cannot be hidden by pooled averages.

Alternative considered: Grad-CAM-first explanation. That can be visually useful, but it risks overclaiming faithfulness and may not align cleanly with the frozen embedding plus shallow model decision surface. It is optional until the implementation can validate it against the actual ROI/embedding path.

### Decision: Enforce monotonic cumulative probabilities

Threshold probabilities SHALL be monotonic non-increasing across increasing thresholds for every prediction row. The implementation can satisfy this either by training an ordinal cumulative estimator directly or by training independent threshold classifiers and applying a deterministic monotonic projection before writing artifacts. The initial implementation should prefer the independent threshold plus projection approach because it reuses scikit-learn, is easy to test, and avoids adding a new dependency.

Alternative considered: no projection. That would make impossible outputs such as `P(score > 2) > P(score > 0.5)` possible, which is not acceptable for a user-facing severity index.

### Decision: Keep model comparison inside the same full-cohort quantification run

`run_manifest_quantification()` and `evaluate_embedding_table()` SHALL produce burden-index artifacts in addition to the existing ordinal artifacts under the same output root. The new artifacts live in `burden_model/`; the existing comparator artifacts remain in `ordinal_model/`.

The repository already has a user-review surface for ordinal predictions: `src/eq/quantification/pipeline.py::generate_html_review_report()` writes `ordinal_model/review_report/ordinal_review.html`, `selected_examples.csv`, and preview assets. P0 SHALL not discard that value. It SHALL extend or replace it with a clearer combined report under `quantification_review/` so a reviewer can inspect the current primary burden model and its comparators in one place.

The workflow SHALL write comparison metadata into the quantification review report so the next decision is visible in the artifact itself. The report must distinguish:

- primary burden-index model metrics,
- per-image prediction intervals and score prediction-set behavior,
- grouped aggregate confidence intervals,
- nearest-example and threshold-profile evidence,
- direct regression comparator metrics,
- ordinal/multiclass comparator metrics,
- numerical-stability status,
- cohort-stratified results for `lauren_preeclampsia` and `vegfri_dox`,
- interpretive caveats for image-level grade labels.

The review report SHALL include at least:

- an executive verdict section stating selected operational model status, blocker status, and whether the results are README/docs-ready;
- an overall metrics panel with row count, biological-group count, score support, stage-index MAE, grade-scale MAE, prediction-set coverage, average prediction-set size, and threshold-support warnings;
- cohort and animal-level summary tables using animal-weighted means and uncertainty intervals;
- comparator panels for burden, direct stage-index regression, and ordinal/multiclass outputs;
- a reviewer example gallery using fold-held-out rows, with raw image plus ROI box, mask overlay, ROI crop, observed score, burden estimate with uncertainty, prediction set, ordinal prediction, nearest scored examples, and threshold probability profile;
- downloadable CSV/JSON artifact links and provenance back to source image, mask, ROI, fold, cohort, and biological group.

The existing ordinal review report may remain as a comparator-specific artifact, but it SHALL NOT be the only human-readable review surface after p0.

Alternative considered: create a separate CLI or YAML config just for target-model experimentation. That would repeat the earlier confusion around user-facing utilities. The current production question is the quantification target model used by the supported full-cohort workflow, so it belongs inside the existing workflow.

### Decision: Preserve the quantification CLI names and clarify their roles

The maintained quantification control surface has three layers:

- `eq run-config --config configs/endotheliosis_quantification.yaml`: reproducible YAML workflow front door for the full cohort run.
- `eq quant-endo`: direct one-shot quantification entrypoint for explicit `--data-dir`, `--segmentation-model`, and `--output-dir` inputs.
- `eq prepare-quant-contract`: direct preparation entrypoint that calls the same quantification engine with `stop_after='contract'`, producing contract/scored-example artifacts without ROI extraction, embeddings, or model fitting.

This change SHALL NOT rename these commands or add compatibility aliases. `quant-endo` is already broad enough for endotheliosis quantification, and `prepare-quant-contract` accurately names the pre-model contract stage. Instead, p0 SHALL update stale help text, README/docs references, and report language that describe the maintained quantification model only as an ordinal baseline. After this change, `quant-endo` and the YAML workflow run the burden-index primary model plus comparator artifacts; `prepare-quant-contract` remains useful when the user needs to verify score-linked image/mask contract rows before running embeddings and modeling.

### Decision: Audit and certify the biological grouping key before model selection

The workflow SHALL write `burden_model/grouping_audit.json` before fitting or selecting an operational model. The audit SHALL certify that the grouping column used for cross-validation, conformal calibration, nearest-neighbor exclusion, and grouped bootstrap corresponds to biological animals. If `subject_prefix` is not sufficient because repeat acquisitions or date-suffixed identifiers split the same animal, the workflow SHALL derive or require a stronger `animal_id` field and use that field consistently.

The report SHALL state the selected grouping key and any unresolved ambiguity. If the grouping key cannot be certified, burden artifacts may be generated for exploration, but the model SHALL NOT be reported as operationally ready for downstream quantification.

### Decision: Treat six-bin score support as the canonical current contract

`ALLOWED_SCORE_VALUES` SHALL be defined in `src/eq/quantification/burden.py` as `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]` and imported by `src/eq/quantification/pipeline.py`. The missing-support gate SHALL evaluate the six allowed scores. A literal `2.5` score in future data SHALL fail as unsupported unless a future rubric-adjudication spec explicitly adds it.

Alternative considered: keep seven bins but mark `2.5` as optional. That preserves the current confusion and would make the report suggest a nonexistent label is merely absent by chance.

### Decision: Generate README/docs-ready result summaries only from refreshed runtime evidence

The implementation SHALL write `quantification_review/results_summary.md`, `quantification_review/results_summary.csv`, and `quantification_review/readme_results_snippet.md` after the full-cohort run. These artifacts SHALL report concrete values from the refreshed runtime outputs, not anticipated or cached metrics. The summary SHALL include:

- model verdict and claim boundary;
- cohort and animal counts;
- score distribution and threshold-support status;
- primary burden stage-index MAE and uncertainty calibration behavior;
- animal-weighted burden summaries by cohort or treatment group where available;
- comparator results for direct stage-index regression and ordinal/multiclass output;
- a short interpretation block suitable for README/docs only if the run clears the report's own readiness gates.

README/docs SHOULD display the quantification results as a compact current-results table plus one or two representative review-panel thumbnails, with the burden index labeled as an ordinal stage-burden index. Docs SHALL avoid presenting the output as a clinical percent, causal treatment effect, or externally validated endpoint.

### Decision: End implementation with independent review lanes

Before p0 is marked complete, the implementation SHALL run a final review packet using specialist subagents or equivalent lanes and record the findings in `audit-results.md`. Required lanes:

- stats review: estimand, calibration, grouped validation, intervals, support gates, and model-selection logic;
- implementation audit: whether code and artifacts implement the OpenSpec contract exactly;
- documentation/reporting review: whether README/docs/HTML/Markdown surfaces match the live artifact schema and claim boundary;
- robustness test review: whether tests cover leakage, unsupported scores, underpowered thresholds, report schema, and artifact joins.

Any high-severity issue from these reviews SHALL be fixed or explicitly recorded as a blocker before the change is called complete.

## Risks / Trade-offs

- [Risk] The burden index may look like a true percentage to readers. → Mitigation: artifact schemas, report text, and docs SHALL call it `Endotheliosis burden index (0-100)` and explicitly state that it is grade-equivalent, not pixel-area percent.
- [Risk] Independent threshold models can produce non-monotonic raw probabilities. → Mitigation: write both raw threshold diagnostics if useful, but the public burden columns SHALL use monotonic corrected probabilities and tests SHALL assert monotonicity.
- [Risk] The 707-row cohort may still be too small or too cohort-imbalanced for a stable predictive model. → Mitigation: grouped validation, cohort-stratified metrics, calibration bins, warning gates, and a model-comparison verdict are required outputs rather than optional commentary.
- [Risk] Direct regression may outperform cumulative threshold modeling. → Mitigation: the spec requires reporting the comparator result and selecting the operational model from evidence, not from preference.
- [Risk] Changing allowed score values could break older generated outputs. → Mitigation: this is a deliberate target-contract correction; old seven-bin outputs remain historical runtime artifacts and current runs must regenerate six-bin quantification artifacts.
- [Risk] Numerical warnings could still occur in comparator models. → Mitigation: warning capture remains a hard stability signal in `ordinal_model/ordinal_metrics.json` and analogous burden metrics.

## Migration Plan

1. Add shared six-bin score constants and threshold constants.
2. Implement `src/eq/quantification/burden.py` with grouped cumulative threshold fitting, monotonic projection, prediction schema, metric calculation, and final model serialization.
3. Update `src/eq/quantification/pipeline.py::evaluate_embedding_table()` to call the burden evaluator and return both `burden_model/*` and `ordinal_model/*` artifacts.
4. Update HTML/Markdown quantification review generation so burden-index results lead the report and ordinal/multiclass results are clearly labeled as comparators.
5. Add README/docs-ready result summaries generated from the refreshed runtime artifacts.
6. Add unit tests and focused pipeline tests for the six-bin rubric, missing `2.5` handling, monotonic probability schema, burden calculation, grouped metrics, report schema, and artifact paths.
7. Rerun `configs/endotheliosis_quantification.yaml` against the full admitted scored mask-paired cohort.
8. Run the final specialist review packet and record the model-comparison verdict, reviewer findings, and docs-ready result summary in the OpenSpec change before treating the implementation as complete.

Rollback is not a compatibility branch. If the burden implementation is unstable or inferior, the workflow should still retain the corrected six-bin label contract and explicitly report that the burden model is not selected for operational use.

## Explicit Decisions

- New implementation module: `src/eq/quantification/burden.py`.
- Primary entry function: `evaluate_burden_index_table(embedding_df: pd.DataFrame, output_dir: Path, n_splits: int = 3) -> dict[str, Path]`.
- Existing integration function to update: `src/eq/quantification/pipeline.py::evaluate_embedding_table()`.
- Shared allowed scores: `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]`.
- Shared thresholds: `[0.0, 0.5, 1.0, 1.5, 2.0]`.
- Public burden prediction columns: `prob_score_gt_0`, `prob_score_gt_0p5`, `prob_score_gt_1`, `prob_score_gt_1p5`, `prob_score_gt_2`, and `endotheliosis_burden_0_100`.
- Public uncertainty columns: `burden_interval_low_0_100`, `burden_interval_high_0_100`, `burden_interval_coverage`, `burden_interval_method`, and `prediction_set_scores`.
- Required uncertainty artifact: `burden_model/uncertainty_calibration.json`.
- Required group CI artifact: `burden_model/group_summary_intervals.csv`.
- Required explanation artifacts: `burden_model/prediction_explanations.csv` and `burden_model/nearest_examples.csv`.
- Required combined review report directory: `quantification_review/`.
- Required review report artifacts: `quantification_review/quantification_review.html`, `quantification_review/review_examples.csv`, `quantification_review/results_summary.md`, `quantification_review/results_summary.csv`, and `quantification_review/readme_results_snippet.md`.
- Required grouping audit artifact: `burden_model/grouping_audit.json`.
- Required threshold support artifact: `burden_model/threshold_support.csv`.
- Primary error metric: stage-index MAE against the `0, 20, 40, 60, 80, 100` target.
- Primary biological summary: mean of animal-level mean burdens.
- Required burden output directory: `burden_model/` under the quantification run output root.
- Existing ordinal output directory remains `ordinal_model/`.
- Full-cohort command remains `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`.

## Open Questions

- [audit_first_then_decide] Operational model selection SHALL be decided from the refreshed artifacts at `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated`, specifically `burden_model/burden_metrics.json`, `burden_model/cohort_metrics.csv`, `ordinal_model/ordinal_metrics.json`, and the updated review report.
