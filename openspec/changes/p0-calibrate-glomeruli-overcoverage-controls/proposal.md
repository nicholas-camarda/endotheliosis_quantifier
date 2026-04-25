## Why

The negative-background supervision change proved that true negative crop supervision can be generated, validated, threaded into training, and recorded in candidate-comparison artifacts, but the latest quick MPS evidence still shows foreground overcoverage on true background crops. The next blocker is no longer missing negative supervision; it is determining whether overcoverage is caused by the hardcoded low inference threshold, insufficient false-positive penalty in training, positive-biased crop sampling, lossy resize policy, or augmentation behavior.

This change prevents another expensive full training run from repeating the same failure mode. It adds a contract-first overcoverage audit that records probability distributions, threshold sweeps, resize-policy evidence, and training-signal ablations inside the OpenSpec change and runtime report artifacts before any candidate is treated as ready for full retraining or promotion review.

## What Changes

- Add a deterministic glomeruli overcoverage audit workflow for existing transfer and no-base/scratch candidate artifacts.
- Replace the single fixed `0.01` comparison threshold assumption with explicit threshold-sweep evidence over category-stratified validation crops.
- Require foreground-probability distribution artifacts before interpreting binary masks as true model behavior.
- Extend candidate-comparison reports to record threshold policy, selected threshold rationale, probability quantiles, category-level threshold curves, and background false-positive foreground fractions.
- Add a training-signal ablation contract covering negative-crop sampler weight, `positive_focus_p`, false-positive-penalizing loss behavior, and whether scratch and transfer both honor the requested loss.
- Add a resize-policy ablation contract covering the current `crop_size=512` / `output_size=256` policy against less-downsampled alternatives before full production retraining.
- Add an augmentation audit contract that records the active FastAI augmentation policy and only allows augmentation ablations as explicit, named variants.
- Require an OpenSpec-local results artifact, `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/audit-results.md`, to capture the executed audit commands, artifacts, findings, decisions, and remaining blockers.
- Add tests that fail if overcoverage audits omit background crops, threshold sweeps, probability summaries, threshold policy provenance, loss-propagation evidence, resize-policy evidence, or OpenSpec-local results updates.
- **BREAKING**: Glomeruli candidate promotion and README-facing performance claims may no longer rely on a single hardcoded `0.01` binary threshold without threshold-sweep and background-overcoverage evidence.

## Explicit Decisions

- Change ID: `p0-calibrate-glomeruli-overcoverage-controls`.
- Canonical audit command name: `eq glomeruli-overcoverage-audit`.
- Canonical implementation module: `src/eq/training/glomeruli_overcoverage_audit.py`.
- Canonical tests:
  - `tests/test_glomeruli_overcoverage_audit.py`
  - `tests/test_glomeruli_candidate_comparison.py`
  - `tests/test_segmentation_training_contract.py`
  - `tests/test_loss_contract.py`
- Canonical runtime output root: `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/`.
- Canonical OpenSpec results file: `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/audit-results.md`.
- Required default audit thresholds: `0.01`, `0.05`, `0.10`, `0.25`, and `0.50`.
- Required default validation categories: `background`, `boundary`, and `positive`.
- Required candidate families: `transfer` and `scratch`, where `scratch` means the no-mitochondria-base ImageNet-pretrained ResNet34 glomeruli comparator used by the current candidate-comparison workflow.
- Required model families to audit first:
  - the latest negative-background quick transfer and scratch artifacts from `p0_negative_background_quick_5epoch`
  - the latest non-quick real transfer and scratch artifacts available under the runtime glomeruli model root, if they load under `eq-mac`
- Required quick screening config name: `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/quicktest_overcoverage_controls_5epoch.yaml`.
- Required full-run config name after audit decisions: `configs/glomeruli_candidate_comparison.yaml`, updated only after audit evidence selects the threshold/training/resize policy.
- Runtime artifacts remain outside Git under `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/...`.
- OpenSpec artifacts record conclusions and artifact paths; they do not embed trained models, generated images, logs, or bulky runtime outputs.
- The initial implementation must not add compatibility shims for old model formats. If a model artifact does not load in the current certified environment, the audit records that artifact as unavailable and continues only for explicitly supplied current-namespace artifacts.
- The audit must distinguish four interpretations:
  - `threshold_policy_artifact`: probabilities are low or separable, but `0.01` makes masks overlarge
  - `training_signal_insufficient`: background probabilities are genuinely high or inseparable from positives
  - `resize_policy_artifact`: less-downsampled evidence materially reduces overcoverage while preserving positive recall
  - `augmentation_policy_artifact`: a named augmentation variant materially worsens overcoverage relative to a controlled comparator

## Open Questions

- [audit_first_then_decide] What threshold policy should candidate comparison use after the audit? Evidence source: `threshold_sweep.csv`, `probability_quantiles.csv`, background false-positive foreground fractions, and positive/boundary recall across thresholds.
- [audit_first_then_decide] Should the next production training recipe change negative-crop sampler weight, `positive_focus_p`, loss function, resize policy, or augmentation? Evidence source: `audit-results.md` plus runtime ablation artifacts under `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/`.
- [audit_first_then_decide] Is the current `crop_size=512` / `output_size=256` resize policy acceptable for glomeruli boundaries? Evidence source: resize-policy ablation comparing current downsampled inference/training evidence against at least one less-downsampled policy when local MPS memory permits.
- [audit_first_then_decide] Are existing full-run assets good enough to evaluate calibration, or must calibration wait for a fresh controlled quick run? Evidence source: current-namespace load checks and `eq-mac` audit execution against the model artifacts.

## Capabilities

### New Capabilities

- `glomeruli-overcoverage-audit`: deterministic overcoverage audit workflow for glomeruli segmentation candidates, including probability summaries, threshold sweeps, resize evidence, training-signal ablation records, and OpenSpec-local conclusions.

### Modified Capabilities

- `glomeruli-candidate-comparison`: candidate comparison must record threshold policy evidence and may not treat the hardcoded `0.01` threshold as promotion-ready without audit support.
- `segmentation-training-contract`: training provenance and quick ablations must prove scratch and transfer loss handling, negative sampler weighting, positive-focus behavior, resize policy, and augmentation policy before full production retraining.

## Impact

- Affected code:
  - `src/eq/training/glomeruli_overcoverage_audit.py`
  - `src/eq/training/compare_glomeruli_candidates.py`
  - `src/eq/training/segmentation_validation_audit.py`
  - `src/eq/training/train_glomeruli.py`
  - `src/eq/training/transfer_learning.py`
  - `src/eq/training/losses.py`
  - `src/eq/data_management/datablock_loader.py`
  - `src/eq/__main__.py` and the `eq` CLI registration surface
- Affected configs:
  - `configs/glomeruli_candidate_comparison.yaml`
  - `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/quicktest_overcoverage_controls_5epoch.yaml`
- Affected runtime artifacts:
  - `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/audit_summary.json`
  - `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/threshold_sweep.csv`
  - `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/probability_quantiles.csv`
  - `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/background_false_positive_curve.csv`
  - `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/resize_policy_comparison.csv`
  - `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/training_signal_ablation_summary.csv`
  - `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/review_panels/`
- Affected OpenSpec artifacts:
  - `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/audit-results.md`
- Compatibility risks:
  - Existing reports that only contain a single thresholded mask at `0.01` become incomplete for promotion decisions.
  - Older current-namespace model artifacts can still be audited if they load, but they cannot be promoted without the new threshold and background-overcoverage evidence.
  - Historical FastAI pickle artifacts remain historical unless a separate compatibility change explicitly supports them.
