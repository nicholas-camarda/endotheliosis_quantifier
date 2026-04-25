## 1. Audit Workflow Surface

- [ ] 1.1 Create `src/eq/training/glomeruli_overcoverage_audit.py` with pure functions for candidate input recording, deterministic crop evaluation, probability quantiles, threshold sweeps, root-cause classification, and artifact writing.
- [ ] 1.2 Register the `eq glomeruli-overcoverage-audit` CLI command with arguments for `--run-id`, `--transfer-model-path`, `--scratch-model-path`, `--data-dir`, `--output-dir`, `--thresholds`, `--image-size`, `--crop-size`, `--examples-per-category`, `--device`, and `--negative-crop-manifest`.
- [ ] 1.3 Reuse existing runtime path helpers so the default output is `output/segmentation_evaluation/glomeruli_overcoverage_audit/<run_id>/` under the active runtime root.
- [ ] 1.4 Implement fail-closed candidate path handling so missing transfer or scratch paths are explicit errors and no substitute model is selected.
- [ ] 1.5 Implement current-namespace load handling that records load failures as unavailable audit evidence without adding legacy pickle compatibility shims.

## 2. Deterministic Evidence Generation

- [ ] 2.1 Reuse or extract deterministic category selection from `src/eq/training/compare_glomeruli_candidates.py` so audit crops are assigned exactly one of `background`, `boundary`, or `positive`.
- [ ] 2.2 Write `candidate_inputs.json` with model paths, hashes when readable, data root, device, threshold grid, crop/image sizes, deterministic manifest source, negative crop manifest path, and package/code provenance when available.
- [ ] 2.3 Generate foreground-probability maps using learner-consistent preprocessing or a recorded deterministic equivalent.
- [ ] 2.4 Write `probability_quantiles.csv` with per-candidate, per-category, per-crop foreground probability quantiles and summary statistics.
- [ ] 2.5 Write `threshold_sweep.csv` for thresholds `0.01`, `0.05`, `0.10`, `0.25`, and `0.50` unless the caller supplies an explicit threshold list.
- [ ] 2.6 Write `background_false_positive_curve.csv` with background predicted foreground fraction summaries by candidate and threshold.
- [ ] 2.7 Generate `review_panels/index.html` showing raw crop, ground truth, foreground probability heatmap, thresholded masks, and overlays with candidate/category/crop/threshold labels.
- [ ] 2.8 Write `audit_summary.json` with root-cause labels restricted to the spec-defined set.

## 3. Candidate Comparison Integration

- [ ] 3.1 Extend `src/eq/training/compare_glomeruli_candidates.py` to accept an overcoverage audit artifact directory or selected threshold-policy artifact.
- [ ] 3.2 Update candidate summary/report output to include threshold policy, threshold grid, selected-threshold rationale, background false-positive foreground fraction, and positive/boundary recall at selected threshold.
- [ ] 3.3 Ensure reports without threshold-sweep evidence can mark candidates as runtime-compatible research artifacts but cannot mark them scientifically promoted.
- [ ] 3.4 Update root-cause reporting so present negative supervision with continued background overcoverage is classified as threshold/training/resize/augmentation overcoverage rather than missing negative supervision.
- [ ] 3.5 Update HTML review labels to include threshold and threshold-policy provenance.

## 4. Training-Signal Controls

- [ ] 4.1 Verify and, if needed, implement scratch/no-base loss propagation in `src/eq/training/train_glomeruli.py` so requested loss settings are not silently ignored.
- [ ] 4.2 Verify transfer loss propagation in `src/eq/training/transfer_learning.py` and ensure requested/resolved loss settings are recorded consistently.
- [ ] 4.3 Extend loss configuration in `src/eq/training/losses.py` so false-positive-penalizing Tversky-style settings are explicit and recorded with their parameters.
- [ ] 4.4 Update training provenance for transfer and scratch candidates to include negative crop sampler weight, `positive_focus_p`, requested loss, resolved loss class, false-positive penalty parameters when applicable, crop size, output size, augmentation variant, and run intent.
- [ ] 4.5 Implement `training_signal_ablation_summary.csv` writing for short screening runs, including changed axis, candidate family, run id, epochs, model path, comparison artifact path, and category-level overcoverage outcome.

## 5. Resize and Augmentation Audit Controls

- [ ] 5.1 Add resize-policy comparison artifact generation to write `resize_policy_comparison.csv` with current and screened crop/output sizes, resize ratio, interpolation, threshold/resize order, device, batch size, and memory or runtime failures.
- [ ] 5.2 Add config/CLI support for explicit augmentation variants `fastai_default`, `spatial_only`, and `no_aug`, rejecting unsupported names.
- [ ] 5.3 Ensure Gaussian noise can only be recorded as active if the training transform path actually includes the noise transform.
- [ ] 5.4 Create `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/quicktest_overcoverage_controls_5epoch.yaml` as the short screening config after the no-training audit identifies which axis to screen.
- [ ] 5.5 Do not update `configs/glomeruli_candidate_comparison.yaml` production defaults until `audit-results.md` records the selected threshold/training/resize policy and rationale.

## 6. OpenSpec Results Record

- [ ] 6.1 Create `openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/audit-results.md` with sections for no-training audit, screening ablations, resize-policy evidence, augmentation evidence, final decision, and remaining blockers.
- [ ] 6.2 Run the no-training audit on the `p0_negative_background_quick_5epoch` transfer and scratch artifacts if they load under `eq-mac`, and record the exact command plus artifact paths in `audit-results.md`.
- [ ] 6.3 Attempt the no-training audit on the latest non-quick real transfer and scratch artifacts that load under `eq-mac`, and record load failures or audit outputs in `audit-results.md`.
- [ ] 6.4 Based on the no-training audit, explicitly classify the next action as threshold/report correction, training-signal screening, resize screening, augmentation screening, or insufficient current-namespace evidence.
- [ ] 6.5 Record every failed command, MPS/runtime blocker, and skipped evidence source in `audit-results.md`; do not mark a task complete by silently skipping evidence.

## 7. Tests

- [ ] 7.1 Add `tests/test_glomeruli_overcoverage_audit.py` for threshold-grid defaults, probability quantile schema, background false-positive curve schema, root-cause classification, missing-model failures, and runtime output path resolution.
- [ ] 7.2 Update `tests/test_glomeruli_candidate_comparison.py` for threshold-policy provenance, no-promotion-without-threshold-sweep behavior, and present-negative-supervision overcoverage classification.
- [ ] 7.3 Update `tests/test_segmentation_training_contract.py` for training provenance fields covering sampler weight, `positive_focus_p`, crop/output size, augmentation variant, and run intent.
- [ ] 7.4 Add or update `tests/test_loss_contract.py` for scratch and transfer loss propagation plus explicit false-positive-penalizing loss parameter recording.
- [ ] 7.5 Add tests that fail if Gaussian noise is recorded as active without being present in the transform pipeline.
- [ ] 7.6 Add tests that fail if `audit-results.md` is absent or lacks a recorded audit attempt before this change is marked complete.

## 8. Validation

- [ ] 8.1 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/test_glomeruli_overcoverage_audit.py -q`.
- [ ] 8.2 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/test_glomeruli_candidate_comparison.py tests/test_segmentation_training_contract.py tests/test_loss_contract.py -q`.
- [ ] 8.3 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [ ] 8.4 Run `openspec validate p0-calibrate-glomeruli-overcoverage-controls --strict`.
- [ ] 8.5 If `scripts/check_openspec_explicitness.py` exists, run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python scripts/check_openspec_explicitness.py openspec/changes/p0-calibrate-glomeruli-overcoverage-controls`.
- [ ] 8.6 Run `openspec status --change p0-calibrate-glomeruli-overcoverage-controls` and confirm the change is apply-ready only after artifacts validate.
