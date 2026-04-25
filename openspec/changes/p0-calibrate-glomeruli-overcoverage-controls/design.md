## Context

The negative-background supervision implementation changed the data contract from "background supervision is absent" to "validated negative supervision is present and recorded." The 5-epoch MPS smoke evidence still showed background category Dice/Jaccard of `0.0`, background median predicted foreground fractions of about `0.16` for transfer and `0.10` for scratch, and near-saturated foreground fractions on many positive/boundary crops. That is an overcoverage problem, not proof that negative supervision is missing.

The current candidate-comparison path also uses a very permissive binary threshold. `src/eq/training/compare_glomeruli_candidates.py` defines `COMPARE_PREDICTION_THRESHOLD = 0.01`, and existing spec text describes underconfident-model threshold semantics instead of a calibrated threshold policy. A model with low but nonzero foreground probabilities can therefore look like a broad foreground segmenter even if its probability map is separable. Conversely, if background probabilities are genuinely high, threshold tuning will not solve the underlying training-signal problem.

Current training risk surfaces are spread across several modules:

- `src/eq/data_management/datablock_loader.py` applies dynamic crop sampling, `positive_focus_p`, `ResizeMethod.Squish`, FastAI `aug_transforms`, ImageNet normalization, and mask preprocessing.
- `src/eq/training/train_glomeruli.py` threads negative-crop manifests and `positive_focus_p`, but scratch training still needs verified loss propagation because the current code path contains a `TODO` around `loss_name`.
- `src/eq/training/transfer_learning.py` accepts custom loss configuration and negative-crop provenance for transfer candidates.
- `src/eq/training/losses.py` exposes Dice/BCE-Dice/Tversky-style losses, but the default Tversky parameters are not currently an explicit false-positive-penalizing setting.
- `src/eq/training/segmentation_validation_audit.py` already records resize policy and root-cause classes, but it does not yet produce a first-class threshold/probability calibration report for overcoverage.

This design makes the next step an audit and decision workflow before another full production training run. Short 5-epoch runs are acceptable only as screening experiments. Full retraining should happen only after the audit identifies which threshold, sampler, loss, resize, and augmentation choices are justified.

## Goals / Non-Goals

**Goals:**

- Add a deterministic audit workflow that can run against existing current-namespace glomeruli candidates without retraining.
- Produce probability-level evidence before binary-mask interpretation.
- Quantify category-level metrics across a fixed threshold grid.
- Decide whether `0.01` is an evaluation artifact, a useful threshold, or an invalid default for promotion.
- Verify training-signal controls before full retraining: negative sampler weight, `positive_focus_p`, loss propagation for scratch and transfer, and false-positive-penalizing loss behavior.
- Audit resize policy before full retraining, especially the current `crop_size=512` to `output_size=256` downsampling.
- Keep augmentation explicit and named; do not silently change augmentation while testing negative/background or threshold behavior.
- Record results inside `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/audit-results.md` in addition to runtime CSV/JSON artifacts.

**Non-Goals:**

- Do not promote a glomeruli model as part of the audit itself.
- Do not replace README/front-facing assets with quick-run artifacts.
- Do not add compatibility shims for historical FastAI pickle artifacts.
- Do not use unreviewed MR/TIFF crops as training or validation evidence.
- Do not make a new static patch training root.
- Do not tune a threshold on the same evidence and then present that same score as independent external validation.
- Do not collapse transfer and scratch into a single candidate family; both remain separately reported.

## Explicit Decisions

- New CLI command: `eq glomeruli-overcoverage-audit`.
- Implementation module: `src/eq/training/glomeruli_overcoverage_audit.py`.
- The audit reuses deterministic review-manifest construction from `src/eq/training/compare_glomeruli_candidates.py` where possible, but the overcoverage audit owns probability summaries, threshold sweeps, resize comparison summaries, and root-cause classification.
- The audit uses the runtime path helpers in `src/eq/utils/paths.py`; default output is `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/`.
- The audit writes:
  - `audit_summary.json`
  - `candidate_inputs.json`
  - `probability_quantiles.csv`
  - `threshold_sweep.csv`
  - `background_false_positive_curve.csv`
  - `resize_policy_comparison.csv`
  - `training_signal_ablation_summary.csv`
  - `review_panels/index.html`
- The audit updates `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/audit-results.md` during implementation with command lines, artifact paths, findings, and decisions.
- Required threshold grid: `0.01`, `0.05`, `0.10`, `0.25`, `0.50`.
- Required review categories: `background`, `boundary`, `positive`.
- Required candidate family labels: `transfer` and `scratch`.
- Probability maps must be summarized before thresholding. The audit stores foreground-probability quantiles by candidate, category, source image, and crop.
- Binary masks for threshold sweeps are generated by resizing probabilities back to truth shape before thresholding unless an explicitly named resize-order ablation is being run.
- Root-cause labels are restricted to:
  - `threshold_policy_artifact`
  - `training_signal_insufficient`
  - `resize_policy_artifact`
  - `augmentation_policy_artifact`
  - `insufficient_current_namespace_artifacts`
  - `inconclusive_short_run_only`
- A fresh full production training run is not part of the first audit step. The first implementation step is no-training evaluation of existing artifacts that load under `eq-mac`.

## Decisions

### 1. Audit probability distributions before changing training

The audit first loads existing current-namespace candidate artifacts and computes foreground-probability maps on deterministic category-stratified crops. It writes per-crop probability quantiles and category summaries before any thresholded mask is considered.

Rationale: A hardcoded low threshold can inflate foreground masks. Probability distributions distinguish threshold-policy failure from true model confusion.

Alternative considered: Immediately retrain with stronger negatives. Rejected because that can waste a full run if the dominant failure is threshold calibration or report semantics.

### 2. Treat threshold policy as report provenance, not an implicit constant

Candidate comparison keeps binary review artifacts, but reports must include the threshold grid and selected-threshold rationale. If a selected threshold is chosen, the report must say whether it was fixed a priori, selected from validation evidence, or only exploratory.

Rationale: Promotion claims need auditable threshold semantics. A fixed `0.01` threshold may be useful for underconfident compatibility artifacts, but it is not automatically appropriate for promotion.

Alternative considered: Switch directly to `0.50`. Rejected because the existing models may be underconfident, and a blind switch could create false all-background failures.

### 3. Separate calibration from training-signal failure

The audit classifies each candidate into one of two primary branches:

- Threshold issue: background probabilities are low or separable, but the low threshold creates foreground masks.
- Training issue: background probabilities are high or overlap heavily with positive/boundary probabilities.

Rationale: These branches require different fixes. Calibration changes the evaluation/report contract; training failure requires sampler/loss/resize/augmentation changes.

Alternative considered: Combine all fixes into one new production run. Rejected because it would confound the cause and make later scientific interpretation weaker.

### 4. Verify scratch and transfer loss behavior before loss ablations

The implementation must add tests proving both `train_glomeruli.py` scratch/no-base training and `transfer_learning.py` transfer training honor requested loss settings and record the resolved loss in provenance. False-positive-penalizing loss settings must be explicit rather than implied by a generic `tversky` name.

Rationale: A loss ablation is meaningless if one candidate family silently ignores the requested loss.

Alternative considered: Only tune transfer loss because transfer currently wires `make_loss`. Rejected because scratch is one of the two canonical candidate families.

### 5. Screen resize policy before another full run

The audit must compare current `crop_size=512` / `output_size=256` evidence against at least one less-downsampled policy when local MPS memory allows. Acceptable screening includes short controlled runs or deterministic inference/report comparisons that explicitly state what was and was not retrained.

Rationale: Glomeruli boundaries may be degraded by downsampling. If resizing is the problem, more negative examples alone may not fix broad masks.

Alternative considered: Keep `256` until promotion. Rejected because the current evidence already identifies `resize_policy_artifact` as a plausible root cause.

### 6. Keep augmentation as an explicit secondary axis

The audit records the active FastAI augmentation policy and supports named variants only after threshold/probability evidence exists. Named variants are `fastai_default`, `spatial_only`, `no_aug`, and `fastai_default_plus_noise` only if the latter is actually implemented and recorded.

Rationale: Augmentation can affect overcoverage, but changing it at the same time as threshold, negative weighting, or resize would obscure causality.

Alternative considered: Add Gaussian noise immediately. Rejected because current artifacts explicitly say Gaussian noise is not active, and claiming it without implementation would recreate documentation drift.

## Data and Artifact Flow

1. Inputs:
   - current-namespace transfer model path
   - current-namespace scratch model path
   - deterministic validation manifest or candidate-comparison review categories
   - optional negative crop manifest provenance
   - runtime path root from `src/eq/utils/paths.py`
2. No-training audit:
   - load candidate artifacts with the current certified `eq-mac` environment
   - reconstruct deterministic crops by category
   - run learner-consistent preprocessing or record any intentional deterministic equivalent
   - save probability maps summaries and thresholded metrics
3. Decision record:
   - write runtime CSV/JSON/HTML artifacts
   - update `audit-results.md` with the executed commands and conclusions
4. Screening ablations:
   - run short controlled configs only after no-training evidence identifies likely levers
   - record sampler, loss, resize, and augmentation settings in provenance
5. Full retraining:
   - allowed only after `audit-results.md` selects a policy and records why

## Risks / Trade-offs

- [Risk] Threshold tuning can overfit the deterministic review manifest. Mitigation: reports must label selected thresholds as exploratory unless later evaluated on independent held-out evidence.
- [Risk] Existing full-run artifacts may not load in the current certified environment. Mitigation: the audit records `insufficient_current_namespace_artifacts`; it does not add compatibility shims.
- [Risk] Stronger background weighting can suppress true positive recall. Mitigation: threshold sweeps and training-signal ablations must report positive and boundary recall alongside background false-positive fraction.
- [Risk] Larger output sizes can exceed MPS memory. Mitigation: resize ablations may use short screening runs with smaller batches, but failure must be recorded rather than skipped.
- [Risk] Augmentation variants can confound sampler or loss changes. Mitigation: only one axis changes per screening ablation unless the config explicitly labels the run as combined and non-diagnostic.
- [Risk] OpenSpec could again be marked complete without recording audit conclusions. Mitigation: `audit-results.md` is a required task deliverable and strict validation must run before the change is treated as apply-ready or complete.

## Open Questions

- [audit_first_then_decide] Which threshold policy should become the candidate-comparison default? Audit target: `threshold_sweep.csv`, `background_false_positive_curve.csv`, and positive/boundary recall by threshold from the no-training audit.
- [audit_first_then_decide] Does current overcoverage primarily reflect model probabilities or binary thresholding? Audit target: `probability_quantiles.csv` and review probability maps for background, boundary, and positive crops.
- [audit_first_then_decide] Which training-signal lever should be used in the next full run? Audit target: short controlled screening entries in `training_signal_ablation_summary.csv`.
- [audit_first_then_decide] Is `crop_size=512` / `output_size=256` acceptable? Audit target: `resize_policy_comparison.csv` plus any short resize-ablation training reports that complete under MPS.
- [defer_ok] Should a later independent held-out glomeruli set be created for final threshold validation? This is scientifically valuable but not required before implementing the audit workflow.
