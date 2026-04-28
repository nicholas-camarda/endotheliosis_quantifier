## Context

P0 implemented `burden_model/learned_roi/` and showed the workflow can produce learned ROI estimates, calibration summaries, cohort diagnostics, and review evidence. It also showed the current promotion gates reject the estimator because of broad prediction sets, score-2 undercoverage, source/cohort sensitivity, and candidate numerical warnings.

The practical target is not to prove a universal biological tissue-percent measurement from the current labels. The target is to apply the MR TIFF quantification workflow to existing images, identify usable glomerular/ROI tissue through the current mask/ROI contract, and estimate grade-calibrated endotheliosis burden with uncertainty and reliability labels. The data are fixed for now; P1 must work within the current `707` scored rows and `60` subject groups rather than requiring manual feature labels or new cohorts.

## Goals / Non-Goals

**Goals:**

- Add a contained source-aware estimator under `src/eq/quantification/source_aware_estimator.py`.
- Integrate the estimator through `eq run-config --config configs/endotheliosis_quantification.yaml`.
- Use existing ROI/mask/scored-example, morphology, embedding, burden, and learned ROI outputs as inputs where present.
- Audit upstream MR TIFF-to-ROI adequacy before interpreting estimator outputs.
- Produce image-level and subject-level grade-equivalent burden estimates with uncertainty intervals and reliability labels.
- Report metrics separately for training/apparent fit, subject-heldout validation, and true held-out testing when a test partition exists.
- Report pooled, within-source, source-adjusted, and leave-source-out behavior using `cohort_id` as the primary source/context field.
- Generate a capped summary figure set linked from the estimator index, verdict, and combined review.
- Treat score-2 ambiguity, broad intervals, source sensitivity, and small strata as scope limiters unless they reveal broken implementation or misleading claims.
- Separate image-level, subject-level, and aggregate current-data reportability.
- Reduce artifact confusion with one indexed output subtree: `burden_model/source_aware_estimator/`.
- Keep experimental internals isolated from stable burden artifacts and avoid duplicate flat aliases.

**Non-Goals:**

- No new manual pixel/feature-labeling workflow.
- No new external cohort, external validation claim, or biological tissue-percent claim.
- No compatibility shims for legacy flat outputs or unsupported model artifacts.
- No expansion to new foundation/backbone providers in this change.
- No claim that source-aware estimation generalizes outside the current source context.
- No README/paper-style promotion unless the P1 estimator verdict explicitly marks the relevant scope reportable.

## Decisions

### Decision: Use one bounded orchestrator module

Implement `evaluate_source_aware_endotheliosis_estimator` in `src/eq/quantification/source_aware_estimator.py` and call it from `src/eq/quantification/pipeline.py` after the existing burden and learned ROI artifacts are available.

Rationale: P0 spread only a modest amount of integration into `pipeline.py`; P1 must not increase codebase confusion. A single orchestrator keeps experimental candidate logic, output paths, reliability labels, and source diagnostics in one owned module.

Alternative considered: extend `learned_roi.py` directly. Rejected because P1 has a different contract: practical source-aware estimator and artifact ergonomics, not just learned ROI provider evaluation.

### Decision: Keep the single YAML workflow entrypoint

The workflow remains `eq run-config --config configs/endotheliosis_quantification.yaml`. P1 does not add a separate CLI command.

Rationale: the repository already treats YAML workflows as the runnable control surface. Adding an ad hoc script or separate command would make the real workflow harder to reproduce.

Alternative considered: add `eq source-aware-estimator`. Rejected for now because this is still part of the quantification pipeline and should run with the same manifest, embedding, ROI, and report context.

### Decision: Use source-aware reliability labels instead of binary promotion gates

P1 SHALL emit hard blockers, scope limiters, and reliability labels separately. Hard blockers stop or prevent any estimator claim. Scope limiters narrow the claim and appear in the verdict and per-row predictions.

Rationale: score-2 undercoverage and source sensitivity may be invariant features of the current data. Treating them as automatic global failure blocks useful estimation. Treating them as reliability labels lets the estimator remain useful while preventing overclaiming.

Alternative considered: keep P0's readiness gates unchanged. Rejected because they answer a promotion question, not the practical fixed-data estimator question.

### Decision: Use `cohort_id` as the primary source/context field

P1 source-aware candidates use `cohort_id` for source adjustment, within-source summaries, and leave-source-out diagnostics. `lane_assignment` is reported when present but is not the primary adjustment target.

Rationale: `cohort_id` is the current data-source distinction tied to Lauren preeclampsia versus VEGFRi/Dox data. `lane_assignment` encodes workflow lane/mask context and should not replace the biological/source grouping.

Alternative considered: adjust on both `cohort_id` and `lane_assignment`. Deferred because the current sample size is small and the first source-aware pass should avoid overparameterized nuisance modeling.

### Decision: Unknown source rows get explicit reliability labeling

Rows with missing or non-training `cohort_id` SHALL receive `unknown_source` reliability labeling. They can receive predictions only if the selected model can compute finite outputs without pretending the row belongs to a known source. They SHALL NOT be counted as standard known-source estimates.

Rationale: the eventual MR TIFF application target will encounter images whose source context may not match the current scored data. The estimator must make this visible rather than silently mapping unknown images onto Lauren or VEGFRi/Dox behavior.

Alternative considered: fail every unknown-source row. Deferred because aggregate application may still benefit from a finite estimate with explicit `unknown_source` labeling, but this remains a non-standard reliability condition.

### Decision: Candidate set remains small and explicit

Initial P1 candidate IDs are:

- `pooled_roi_qc`
- `pooled_learned_roi`
- `pooled_hybrid`
- `source_adjusted_roi_qc`
- `source_adjusted_hybrid`
- `within_source_calibrated_hybrid`
- `subject_source_adjusted_hybrid`

Rationale: the next step is not model shopping. These candidates test whether source-aware calibration/adjustment improves practical estimation using the already available features.

Alternative considered: add torchvision/timm/foundation features now. Rejected because P0 showed artifact and interpretation complexity already need containment before expanding feature providers.

### Decision: Separate training, validation, and testing metrics

P1 SHALL write `summary/metrics_by_split.csv` and `summary/metrics_by_split.json`. Metrics must use explicit split labels:

- `training_apparent` for full-data or in-fold training performance;
- `validation_subject_heldout` for grouped out-of-fold validation where `subject_id` is held out;
- `testing_explicit_heldout` only when a predeclared test partition exists and is not used for model selection or calibration;
- `testing_not_available_current_data_sensitivity` when no independent test partition exists and the available substitute is leave-source-out, within-source, or other current-data sensitivity.

Rationale: the user needs training/validation/testing metrics after the full run, but the current scored quantification data do not automatically provide a true independent test set. Apparent full-dataset metrics are useful diagnostics but must not be mislabeled as testing.

Alternative considered: report only validation metrics. Rejected because it hides overfit/apparent performance and does not answer the requested train/validation/test view.

Alternative considered: call leave-source-out "testing." Rejected because it is a current-data sensitivity check, not external or fully independent testing.

### Decision: Generate only a capped first-read figure set

P1 SHALL write six summary PNG figures under `summary/figures/`:

- `metrics_by_split.png`
- `predicted_vs_observed.png`
- `calibration_by_score.png`
- `source_performance.png`
- `uncertainty_width_distribution.png`
- `reliability_label_counts.png`

These figures SHALL be listed in `summary/artifact_manifest.json` and linked from `INDEX.md`, `summary/estimator_verdict.md`, and `quantification_review/quantification_review.html`.

Rationale: visual review is necessary to understand model behavior, calibration, source sensitivity, and uncertainty. The figure set is capped to prevent artifact sprawl.

Alternative considered: generate plots for every candidate/source/split combination. Rejected because exhaustive plots would make the output tree harder to understand and should remain internal only if a later change needs it.

### Decision: Use one indexed output subtree

All P1 artifacts live under `burden_model/source_aware_estimator/` with these role folders:

- `summary/`
- `predictions/`
- `diagnostics/`
- `evidence/`
- `internal/`

The first file to open is `burden_model/source_aware_estimator/INDEX.md`.

Rationale: P0 produced audit-complete artifacts but too many entrypoints. P1 must make the output understandable within 30 seconds.

Alternative considered: keep adding artifacts under `burden_model/learned_roi/`. Rejected because P1 needs a cleaner reader-facing structure and a different estimator contract.

### Decision: Cap the artifact manifest

P1 SHALL write `summary/artifact_manifest.json` and `INDEX.md`; every top-level source-aware estimator artifact must be listed with role, relative path, consumer, and reportability. First-pass generated artifacts are limited to the verdict files, metrics-by-split files, summary figures, artifact manifest, image/subject predictions, upstream ROI adequacy, source sensitivity, reliability labels, evidence review, candidate metrics, and candidate summary unless the implementation records a named consumer in the manifest.

Rationale: artifact explosion is a known failure mode in this repo. A manifest makes extra outputs auditable and prevents unindexed side products from becoming the de facto interface.

Alternative considered: rely on folder names alone. Rejected because folders do not prevent proliferation or explain which files matter.

### Decision: Upstream ROI adequacy is part of the estimator verdict

P1 SHALL write `diagnostics/upstream_roi_adequacy.json` and summarize it in `estimator_verdict.json`. This includes ROI status counts, usable ROI count, failed ROI count, manual-mask versus model-derived mask context where present, segmentation artifact provenance where present, and whether the estimator has enough usable ROI rows for each reportable scope.

Rationale: the estimator is downstream of MR TIFF segmentation/ROI extraction. A good statistical model on bad or sparse ROI inputs would not satisfy the practical goal.

Alternative considered: leave ROI adequacy to existing upstream workflow artifacts. Rejected because the first-read estimator verdict must answer whether its own inputs were adequate.

### Decision: Combined reports show verdict first, internals second

`quantification_review/quantification_review.html`, `results_summary.csv`, and `results_summary.md` SHALL include the P1 estimator verdict, hard blockers, scope limiters, and reliability summary. `readme_results_snippet.md` SHALL include P1 results only if `estimator_verdict.json` explicitly marks a reportable scope.

Rationale: users and collaborators should not have to parse candidate tables to know what happened.

Alternative considered: expose all candidate metrics in the top-level report. Rejected because candidate internals are useful for audit but are not the first thing a reader should see.

## Risks / Trade-offs

- Fixed-data source adjustment can overfit source structure → Mitigation: report pooled, within-source, source-adjusted, and leave-source-out behavior side by side, and label source-dependent outputs explicitly.
- Score-2 ambiguity can make single-image estimates unreliable → Mitigation: emit per-row reliability labels such as `transitional_score_region`, broaden intervals, and discourage single-image overinterpretation in the verdict.
- Artifact count can continue to grow → Mitigation: hard-code the P1 output roles, require `INDEX.md`, and keep exhaustive candidate files under `internal/`.
- Figure generation can create plot sprawl → Mitigation: cap first-pass figures to six manifest-listed PNGs under `summary/figures/`.
- Upstream ROI failures can be hidden by later modeling → Mitigation: require `upstream_roi_adequacy.json` and include upstream adequacy status in the estimator verdict.
- Unknown future MR TIFF sources can be overinterpreted → Mitigation: require `unknown_source` reliability labels and prevent known-source reportability for unknown-source rows.
- Source-aware modeling can be misread as external validation → Mitigation: every verdict and report SHALL state that estimates are calibrated to the current scored MR TIFF/ROI data only.
- Apparent full-dataset metrics can be misread as testing → Mitigation: require split labels and use `testing_not_available_current_data_sensitivity` when no independent test partition exists.
- Integration into the stable pipeline can create code bloat → Mitigation: pipeline integration is limited to one function call and report merge; candidate logic belongs in `source_aware_estimator.py`.
- Numerical warnings may persist without nonfinite predictions → Mitigation: classify them as scope limiters unless they produce nonfinite selected outputs or invalidate model fitting.

## Migration Plan

1. Keep existing P0 artifacts and canonical specs as evidence.
2. Add `source_aware_estimator.py` and focused tests without changing the public CLI.
3. Wire the estimator into `evaluate_embedding_table` through one bounded call after learned ROI artifacts are available.
4. Generate the indexed P1 output subtree under the existing quantification runtime root.
5. Update combined quantification review surfaces to link to the P1 index and show only the verdict-level summary.
6. Validate with focused tests, full `pytest`, `openspec validate`, explicitness check, and a full `eq run-config --config configs/endotheliosis_quantification.yaml` run.

Rollback is straightforward: remove the single pipeline call and P1 module/test additions. P0 learned ROI and baseline burden outputs remain intact.

## Explicit Decisions

- Orchestrator: `src/eq/quantification/source_aware_estimator.py`.
- Function: `evaluate_source_aware_endotheliosis_estimator`.
- Output root: `burden_model/source_aware_estimator/`.
- Top-level index: `burden_model/source_aware_estimator/INDEX.md`.
- Verdict JSON: `burden_model/source_aware_estimator/summary/estimator_verdict.json`.
- Verdict Markdown: `burden_model/source_aware_estimator/summary/estimator_verdict.md`.
- Artifact manifest: `burden_model/source_aware_estimator/summary/artifact_manifest.json`.
- Metrics by split CSV: `burden_model/source_aware_estimator/summary/metrics_by_split.csv`.
- Metrics by split JSON: `burden_model/source_aware_estimator/summary/metrics_by_split.json`.
- Summary figures: `burden_model/source_aware_estimator/summary/figures/*.png` with exactly the six required first-pass figure filenames.
- Upstream adequacy diagnostics: `burden_model/source_aware_estimator/diagnostics/upstream_roi_adequacy.json`.
- Primary source/context field: `cohort_id`.
- Required grouping key: `subject_id`.
- Required score rubric: `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]`.
- Main command: `eq run-config --config configs/endotheliosis_quantification.yaml`.
- Reportable scopes: `image_level`, `subject_level`, and `aggregate_current_data`.

## Open Questions

- [audit_first_then_decide] Decide whether source-adjusted candidates include explicit `cohort_id` indicators, cohort-specific calibration, or both after inspecting P0 candidate features, score support, and source-specific residuals in `burden_model/learned_roi/`.
- [audit_first_then_decide] Decide whether any P1 result is eligible for `readme_results_snippet.md` after the full runtime verdict is available in `estimator_verdict.json`.
