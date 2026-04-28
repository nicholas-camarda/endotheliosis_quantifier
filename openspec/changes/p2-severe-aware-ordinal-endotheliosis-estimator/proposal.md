## Why

P1 produced a usable, contained current-data estimator surface, but the final runtime results show a specific scientific failure mode: the selected `pooled_roi_qc` image estimator is stable for low-to-mid grades but compresses severe score `2` and `3` endotheliosis downward. The next change should target that failure directly with severe-threshold and ordinal evidence rather than adding more high-dimensional learned features that already showed subject-heldout overfitting.

Current P1 evidence to carry forward:

- `pooled_roi_qc` was selected for image-level estimation because its training/apparent grade MAE `0.547` and subject-heldout validation grade MAE `0.568` were similar.
- Learned embeddings and hybrid features overfit: learned training/apparent grade MAE `0.250` worsened to validation grade MAE `0.929`; hybrid training/apparent grade MAE `0.229` worsened to validation grade MAE `0.900`.
- Subject-level aggregation performed better than single-image estimation: selected subject candidate `subject_source_adjusted_hybrid` had subject-level grade MAE `0.324`.
- Severe scores were the main failure mode: score `2` image-level stage-index MAE was `42.15`; score `3` image-level stage-index MAE was `62.68`; score `3` mean predicted stage index was only about `37.3` on the `0-100` scale.
- Severe support is source-confounded: Lauren/preeclampsia has scores only up to `1.5`, while VEGFRi/Dox contains all observed score `2` and `3` examples.
- P1 had no hard blockers, but retained scope limiters: broad uncertainty, current-data source-sensitivity-only testing, and nonfatal numerical warnings.

The practical goal remains unchanged: apply the current MR TIFF/ROI workflow to segment and quantify endotheliosis with reasonable, explicitly scoped results. P2 should determine whether the honest output should be a scalar burden estimate, a severe-risk label, an ordinal prediction set, a subject-level aggregate, or a combination of these.

## What Changes

- Add a severe-aware ordinal estimator track under the existing `endotheliosis_quantification` workflow.
- Audit whether severe cases (`score >= 2`) are separable using current ROI/QC, morphology, learned ROI, and embedding-derived features before fitting more complex candidates.
- Evaluate explicit severe-threshold models for `score >= 1.5`, `score >= 2`, and where support permits `score >= 3`.
- Evaluate ordinal/cumulative-threshold candidates that preserve the score ordering `[0, 0.5, 1, 1.5, 2, 3]` instead of treating the target as unconstrained scalar regression.
- Evaluate a two-stage estimator: first severe-risk detection, then grade-equivalent burden calibration within severe-risk strata.
- Preserve subject-heldout validation as the primary current-data model-selection split.
- Report training/apparent, validation, and testing availability with the same split-label discipline established in P1.
- Report source-stratified severe-threshold behavior without calling source sensitivity external validation.
- Add severe false-negative review artifacts so high-grade underprediction is visible and not hidden by overall MAE.
- Keep artifacts contained and indexed under one new subtree: `burden_model/severe_aware_ordinal_estimator/`.
- Keep first-pass outputs capped by `summary/artifact_manifest.json`.
- Participate in the repo-wide execution logging contract established by `p1-repo-wide-execution-logging-contract`: the P2 evaluator emits useful function-level logger events, the existing `endotheliosis_quantification` workflow remains the durable capture surface, and P2 does not create an independent log-root or file-handler system.
- Do not expand learned ROI provider extraction in this change. Existing learned ROI features may be evaluated as inputs, but new fitted foundation/backbone providers require a separate OpenSpec decision.
- Do not weaken P1 claim boundaries: outputs remain predictive grade-equivalent or severe-risk evidence for current scored MR TIFF/ROI data, not tissue percent, closed-capillary percent, causal evidence, or external validation.
- Add an explicit future decision gate for manual patch/mask annotation and upstream segmentation-backbone escalation. P2 SHALL first decide whether severe failure is caused by ROI extraction/segmentation limits, feature/model limits, or the current grading target itself.
- Include a documented MedSAM/SAM-style promptable segmentation audit path as a possible upstream comparator, with oracle-prompt and automatic-prompt results separated so prompt leakage is not mistaken for deployable performance.
- Include a documented feasibility inventory for nnU-Net, DeepLab, Mask2Former-style, and promptable MedSAM/SAM segmentation baselines because these models are not currently installed in `eq-mac` and must not be treated as available runtime dependencies until proven.

## Capabilities

### New Capabilities

- `severe-aware-ordinal-endotheliosis-estimator`: Defines the severe-threshold, ordinal, and two-stage estimator track, its validation semantics, severe false-negative review, capped artifact tree, verdict, and claim boundary.

### Modified Capabilities

- `endotheliosis-burden-index`: Adds combined-report integration for severe-aware ordinal estimator verdicts, summary rows, testing-availability rows, and severe-risk/ordinal figures.
- `source-aware-endotheliosis-estimator`: Adds a handoff requirement that P2 consumes P1 source-aware verdicts and preserves P1 scope limiters rather than treating source-aware results as promoted truth.

## Impact

Affected code surfaces:

- `src/eq/utils/execution_logging.py`: P2 depends on the repo-wide logging contract implemented by `p1-repo-wide-execution-logging-contract`; P2 should not add separate durable logging helpers.
- `src/eq/quantification/source_aware_estimator.py`: evidence source for P1 verdict and selected current-data candidates.
- `src/eq/quantification/burden.py`: existing cumulative threshold and burden-index comparator logic that P2 should audit before adding a parallel implementation.
- `src/eq/quantification/pipeline.py`: likely integration point after burden, learned ROI, and source-aware artifacts are available.
- `src/eq/quantification/__init__.py`: likely export surface if a new evaluator is added.
- `configs/endotheliosis_quantification.yaml`: existing runnable control surface; P2 should run from this workflow unless an audit proves a separate config is necessary.

Affected artifact roots:

- Existing runtime root: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated`
- Existing P1 source-aware root: `burden_model/source_aware_estimator/`
- New P2 root: `burden_model/severe_aware_ordinal_estimator/`
- Combined review root: `quantification_review/`

Affected tests:

- New unit tests for severe-threshold support, ordinal split labels, two-stage candidate behavior, severe false-negative review, artifact manifest completeness, and hard-blocker behavior.
- Existing quantification pipeline tests for combined review integration and README-snippet eligibility.
- Regression tests proving high-score underprediction is reported as a failure mode rather than hidden by overall metrics.
- Logging-contract validation proving the new evaluator emits function-level events, inherits durable capture from `eq run-config --config configs/endotheliosis_quantification.yaml`, and does not create duplicate file handlers or a separate log root.

Compatibility and scientific interpretation:

- Existing P1 source-aware outputs remain valid runtime evidence but are archived as a completed change and synced into main specs.
- P2 should not remove or reinterpret P1 outputs; it should add a more targeted estimator track focused on severe-end failure.
- No external validation will be created by this change. Current-data sensitivity, source-stratified validation, and leave-source-out diagnostics must stay labeled as internal/current-data evidence.

## Explicit Decisions

- Change ID: `p2-severe-aware-ordinal-endotheliosis-estimator`.
- New capability spec: `openspec/changes/p2-severe-aware-ordinal-endotheliosis-estimator/specs/severe-aware-ordinal-endotheliosis-estimator/spec.md`.
- Modified capability specs:
  - `openspec/changes/p2-severe-aware-ordinal-endotheliosis-estimator/specs/endotheliosis-burden-index/spec.md`
  - `openspec/changes/p2-severe-aware-ordinal-endotheliosis-estimator/specs/source-aware-endotheliosis-estimator/spec.md`
- Workflow ID remains `endotheliosis_quantification`.
- Runnable config remains `configs/endotheliosis_quantification.yaml`.
- Proposed module path: `src/eq/quantification/severe_aware_ordinal_estimator.py`.
- Proposed evaluator function: `evaluate_severe_aware_ordinal_endotheliosis_estimator`.
- Logging participation: `evaluate_severe_aware_ordinal_endotheliosis_estimator` is a high-level function-events-only surface; durable capture is owned by the existing `endotheliosis_quantification` workflow through the repo-wide logging contract.
- Proposed output root: `burden_model/severe_aware_ordinal_estimator/`.
- Primary validation split label remains `validation_subject_heldout`.
- Apparent full-data metrics must use `training_apparent`.
- No independent testing may be reported unless an explicit held-out partition exists; otherwise use `testing_not_available_current_data_sensitivity`.
- Severe threshold labels to evaluate: `score >= 1.5`, `score >= 2`, and `score >= 3` only if support permits.
- Primary severe failure metric: severe false-negative behavior for `score >= 2`, including count, rate, and example review.
- Claim boundary: predictive grade-equivalent burden and severe-risk/ordinal evidence for current scored MR TIFF/ROI data only.

## Open Questions

- [audit_first_then_decide] Should P2 implement ordinal candidates by reusing existing cumulative-threshold logic in `src/eq/quantification/burden.py`, by adding a new contained estimator module, or by sharing lower-level threshold helpers? Decide after auditing current burden threshold code, warning sources, and artifact schemas.
- [audit_first_then_decide] Is `score >= 3` estimable as a separate severe threshold with only `22` image rows and fewer independent subjects, or should it be reported only as a non-estimable/exploratory tail stratum? Decide from threshold support by subject and source.
- [audit_first_then_decide] Which feature family should be allowed into the first severe-aware candidate set: ROI/QC only, morphology only, ROI/QC plus morphology, learned ROI only as an audit comparator, or a selected low-dimensional embedding summary? Decide from the severe separability audit and feature-warning diagnostics before fitting final candidates.
- [audit_first_then_decide] Is upstream glomerulus segmentation quality limiting severe-endotheliosis estimation on MR TIFFs enough to justify a separate segmentation-backbone comparison or new Label Studio patch/mask annotation? Decide from severe false-negative review, ROI adequacy diagnostics, MR TIFF ROI extraction evidence, and whether severe errors localize to bad/missing ROIs rather than non-separable grading signal.
- [audit_first_then_decide] Is MedSAM or another promptable segmenter worth elevating into a separate upstream segmentation comparison? Decide from oracle-box performance on existing masked images, realistic automatic-prompt performance using current glomerulus proposals, downstream ROI-feature stability, and whether any improvement affects severe false negatives.
- [audit_first_then_decide] Which upstream segmentation baseline families are feasible on the current machine and should be compared in a later segmentation-change spec? Decide from a dependency inventory covering local availability, install path, hardware support, expected runtime, training/inference API, dataset conversion burden, and whether the family can consume existing masks without new annotation.
- [defer_ok] Whether P2 results should ever become README-snippet eligible can be deferred until the final P2 verdict exists; default eligibility is false.
