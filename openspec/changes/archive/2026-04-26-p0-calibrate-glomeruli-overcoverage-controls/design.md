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
- Resolve the remaining `resize_benefit_unproven` gate with a controlled resize screen before treating P0 as complete.

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
- The resize-screening config is `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/quicktest_resize_screening.yaml`.
- Resize-screening run IDs are fixed:
  - `p0_resize_screen_current_512to256`
  - `p0_resize_screen_512to512`
  - `p0_resize_screen_512to384`
- Resize-screening output is `resize_policy_screening_summary.csv` under each candidate-comparison resize-screening output directory, plus the durable interpretation in `audit-results.md`.
- The audit updates `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/audit-results.md` during implementation with command lines, artifact paths, findings, and decisions.
- Required threshold grid: `0.01`, `0.05`, `0.10`, `0.25`, `0.50`.
- Required review categories: `background`, `boundary`, `positive`.
- Required candidate family labels: `transfer` and `scratch`.
- Probability maps must be summarized before thresholding. The audit stores foreground-probability quantiles by candidate, category, source image, and crop.
- Binary masks for threshold sweeps are generated by resizing probabilities back to truth shape before thresholding unless an explicitly named resize-order ablation is being run.
- Candidate comparison threshold policy labels are:
  - `threshold_policy_unverified`: no audit evidence and no explicit threshold; legacy `0.01` is review-only and not promotion-ready
  - `fixed_review_threshold`: explicit threshold without audit evidence; review-only and not promotion-ready
  - `audit_backed_fixed_threshold`: explicit threshold with attached overcoverage audit sweep
  - `validation_derived_threshold`: threshold selected from attached overcoverage audit sweep by the repository rule
- When an overcoverage audit is attached and no `--prediction-threshold` is supplied, candidate comparison derives one shared threshold from `threshold_sweep.csv`: keep thresholds where every candidate family's mean background false-positive foreground fraction is `<= 0.02`, maximize mean positive/boundary recall, then break ties by lower background foreground fraction and lower threshold.
- Only `audit_backed_fixed_threshold` and `validation_derived_threshold` can clear the threshold-policy promotion gate.
- Category-gate evaluation must not use the same pass/fail metric for empty-background crops and foreground-containing crops. Background crops are governed by background false-positive foreground fraction and pixel-level background correctness; boundary and positive crops are governed by Dice/Jaccard plus precision/recall.
- Candidate comparison must emit a category-gate audit surface before another training run is treated as necessary for `category_metric_failure`.
- Resize benefit remains an evidence gate, not a permanent name-based block. A model trained/evaluated with `crop_size=512` and `output_size=256` needs either a recorded less-downsampled/no-downsample comparison or a recorded MPS/runtime infeasibility reason.
- The current P0 category-gate evidence cleared `category_metric_failure`; therefore `resize_benefit_unproven` is the remaining P0 gate and must be resolved inside this change rather than deferred to a new change.
- Resize screening changes only the output image size axis: the reference is `crop_size=512` / `image_size=256`, the first comparator is `crop_size=512` / `image_size=512`, and the fallback comparator is `crop_size=512` / `image_size=384` if `512to512` fails under MPS.
- Resize screening keeps the deterministic split, seed, negative-crop manifest, candidate families, threshold-selection rule, loss, sampler policy, `positive_focus_p`, and augmentation variant fixed.
- Batch size may be reduced for the less-downsampled attempts to fit MPS memory, but the executed batch size and any failure must be recorded as execution evidence.
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

### 2b. Derive the default threshold from audit evidence when available

When candidate comparison receives an overcoverage audit directory and the operator does not supply `--prediction-threshold`, the comparison selects the threshold from the audit sweep. The rule first enforces background control for every canonical candidate family, then preserves positive/boundary recall. A manually supplied threshold remains supported, but it is labeled `audit_backed_fixed_threshold` only when the audit sweep is attached and `fixed_review_threshold` when no audit is attached.

Rationale: The P0 evidence showed that `0.5` fixes the quick-run background panels, but making `0.5` a global constant would be another implicit threshold policy. The repository should choose from recorded evidence and expose the rule in every report.

Alternative considered: Make `0.5` the new hardcoded comparison default. Rejected because that would answer one quick audit by creating a new hidden assumption.

### 3. Separate calibration from training-signal failure

The audit classifies each candidate into one of two primary branches:

- Threshold issue: background probabilities are low or separable, but the low threshold creates foreground masks.
- Training issue: background probabilities are high or overlap heavily with positive/boundary probabilities.

Rationale: These branches require different fixes. Calibration changes the evaluation/report contract; training failure requires sampler/loss/resize/augmentation changes.

Alternative considered: Combine all fixes into one new production run. Rejected because it would confound the cause and make later scientific interpretation weaker.

### 3b. Audit category-gate semantics before retraining

The threshold-derived quick comparison still reports `category_metric_failure`, but that label mixes different metric semantics. On true-background crops, Dice and Jaccard are not stable measures of practical background control: an empty ground-truth crop with a tiny nonzero predicted foreground fraction can receive Dice/Jaccard of `0`. P0 therefore adds a category-gate audit step before any full retraining decision. The audit must report, per family and category, the observed metric, required limit, pass/fail state, and gate rationale.

For background crops, the gate is based on background false-positive foreground fraction and pixel-level background correctness. For boundary and positive crops, the gate remains overlap/recall oriented. If the background-only failure disappears under the false-positive fraction gate, the correct conclusion is "promotion gate semantics needed correction," not "the model needs retraining." If boundary or positive crops fail under the explicit gates, then training signal, resize, or data coverage remains a real blocker.

Alternative considered: Immediately launch another training run because `category_metric_failure` remains in the report. Rejected because the current evidence shows near-zero background foreground fraction at the derived threshold, so retraining before auditing the gate would risk solving a reporting metric artifact rather than a model problem.

### 4. Verify scratch and transfer loss behavior before loss ablations

The implementation must add tests proving both `train_glomeruli.py` scratch/no-base training and `transfer_learning.py` transfer training honor requested loss settings and record the resolved loss in provenance. False-positive-penalizing loss settings must be explicit rather than implied by a generic `tversky` name.

Rationale: A loss ablation is meaningless if one candidate family silently ignores the requested loss.

Alternative considered: Only tune transfer loss because transfer currently wires `make_loss`. Rejected because scratch is one of the two canonical candidate families.

### 5. Screen resize policy before another full run

The audit must compare current `crop_size=512` / `output_size=256` evidence against at least one less-downsampled policy when local MPS memory allows. Acceptable screening includes short controlled runs or deterministic inference/report comparisons that explicitly state what was and was not retrained.

Rationale: Glomeruli boundaries may be degraded by downsampling. If resizing is the problem, more negative examples alone may not fix broad masks.

Alternative considered: Keep `256` until promotion. Rejected because the current evidence already identifies `resize_policy_artifact` as a plausible root cause.

The next resize step must be staged. First, the category-gate audit determines whether the remaining failure is a background metric artifact or a true boundary/positive segmentation failure. Second, if boundary/positive performance or prediction-to-truth foreground ratio remains concerning, run a short controlled resize screen with one changed axis, such as less-downsampled output size and adjusted batch size. Full production retraining is only justified after this screening evidence identifies a resize policy worth scaling.

### 5b. Resolve the resize evidence gate inside P0

The latest P0 evidence changed the state of the problem: `category_metric_failure` is cleared under category-appropriate gates, while `resize_benefit_unproven` remains the only current candidate gate reason. The correct next step is not another broad production run and not a new spec. It is a controlled resize screen inside P0.

The resize screen has a reference run and one primary comparator:

- Reference: `p0_resize_screen_current_512to256`, `crop_size=512`, `image_size=256`.
- Primary comparator: `p0_resize_screen_512to512`, `crop_size=512`, `image_size=512`.
- Fallback comparator: `p0_resize_screen_512to384`, `crop_size=512`, `image_size=384`, only after the `512to512` attempt records an MPS memory, unsupported-operation, or runtime failure.

Every run uses the same deterministic split, seed, negative-crop manifest, threshold-selection rule, candidate families, loss, sampler policy, `positive_focus_p`, and augmentation variant. Batch size can change only as a recorded hardware constraint. A failure is not a skip: it is a row in `resize_policy_screening_summary.csv` and an entry in `audit-results.md` with the exact command, log path, and error summary.

Decision rule:

- If `512to512` or `512to384` materially improves positive/boundary Dice, recall, or prediction-to-truth foreground ratio without increasing background false-positive foreground fraction beyond the accepted gate, P0 selects that resize policy for the next production candidate recipe.
- If the comparator is similar or worse while the current `512to256` policy passes category gates, P0 clears `resize_benefit_unproven` for the current policy with recorded evidence.
- If `512to512` fails and `512to384` also fails, P0 keeps resize unresolved but records a hardware/runtime infeasibility state rather than marking the gate complete.

Alternative considered: defer resize to a new change. Rejected because `resize_benefit_unproven` is already the remaining P0 promotion gate and marking P0 complete without the resize decision would repeat the earlier incomplete-completion failure.

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
   - for resize screening, write `resize_policy_screening_summary.csv` with reference/comparator run IDs, crop size, image size, batch size, device, threshold policy, category-gate metrics, runtime status, failure reason, and selected/cleared decision
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

- [defer_ok] Should final model promotion use a later independent held-out threshold calibration set? P0 now derives thresholds from attached audit evidence, but final promotion still needs category-complete held-out calibration evidence.
- [audit_first_then_decide] Does `category_metric_failure` persist after category-appropriate gates are applied? Audit target: `candidate_predictions.csv`, `metric_by_category.csv`, `threshold_sweep_by_crop.csv`, and a category-gate audit table.
- [audit_first_then_decide] Which resize policy, if any, should be screened next? Audit target: category-gate results plus `resize_policy_audit.csv` and `resize_policy_comparison.csv`; MPS memory/runtime failures must be recorded if a less-downsampled screen cannot run.
- [audit_first_then_decide] Which resize policy clears the P0 resize gate? Audit target: `resize_policy_screening_summary.csv` for `p0_resize_screen_current_512to256`, `p0_resize_screen_512to512`, and, if needed, `p0_resize_screen_512to384`.
- [audit_first_then_decide] Does current overcoverage primarily reflect model probabilities or binary thresholding? Audit target: `probability_quantiles.csv` and review probability maps for background, boundary, and positive crops.
- [audit_first_then_decide] Which training-signal lever should be used in the next full run? Audit target: short controlled screening entries in `training_signal_ablation_summary.csv`.
- [audit_first_then_decide] Is `crop_size=512` / `output_size=256` acceptable? Audit target: `resize_policy_comparison.csv` plus any short resize-ablation training reports that complete under MPS.
- [defer_ok] Should a later independent held-out glomeruli set be created for final threshold validation? This is scientifically valuable but not required before implementing the audit workflow.
