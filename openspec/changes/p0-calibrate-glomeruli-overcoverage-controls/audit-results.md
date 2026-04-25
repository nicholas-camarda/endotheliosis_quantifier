# Audit Results

## No-Training Audit

Status: run on the P0 quick negative-background candidates and attempted on the latest non-quick E20/E25 candidates.

### P0 quick negative-background candidates

Command:

```bash
env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq glomeruli-overcoverage-audit --run-id p0_negative_background_quick_5epoch_no_training_audit --transfer-model-path /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/p0_quick_glomeruli_transfer-transfer_s1lr1e-3_s2lr_lrfind_e5_b12_lr1e-3_sz256/p0_quick_glomeruli_transfer-transfer_s1lr1e-3_s2lr_lrfind_e5_b12_lr1e-3_sz256.pkl --scratch-model-path /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/scratch/p0_quick_glomeruli_no_mito_base-scratch_e5_b12_lr1e-3_sz256/p0_quick_glomeruli_no_mito_base-scratch_e5_b12_lr1e-3_sz256.pkl --data-dir /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts --image-size 256 --crop-size 512 --examples-per-category 3 --device mps --negative-crop-manifest /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/glomeruli_negative_crops/manifests/p0_quick_mask_background.csv
```

Runtime output:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_overcoverage_audit/p0_negative_background_quick_5epoch_no_training_audit
```

Key artifacts:

- `audit_summary.json`
- `candidate_inputs.json`
- `probability_quantiles.csv`
- `threshold_sweep.csv`
- `threshold_sweep_by_crop.csv`
- `background_false_positive_curve.csv`
- `resize_policy_comparison.csv`
- `training_signal_ablation_summary.csv`
- `review_panels/index.html`

Result:

- Both candidate artifacts loaded under `eq-mac`; no load failures were recorded.
- The deterministic manifest used the shared candidate validation split and included 9 crops: 3 background, 3 boundary, and 3 positive.
- Root cause: `threshold_policy_artifact`.
- Evidence: `background_false_positive_excess_at_0.01_improves_or_probabilities_are_separable`.

Threshold findings:

- At threshold `0.01`, mean background predicted foreground fraction was `0.2722` for transfer and `0.2224` for scratch.
- At threshold `0.05`, mean background predicted foreground fraction fell to `0.1049` for transfer and `0.1055` for scratch.
- At threshold `0.10`, mean background predicted foreground fraction fell further to `0.0377` for transfer and `0.0336` for scratch.
- At threshold `0.25`, mean background predicted foreground fraction was essentially zero: `0.000342` for transfer and `0.000179` for scratch.
- At threshold `0.50`, background predicted foreground fraction was `0.0` for both candidates.

Positive/boundary findings:

- Transfer boundary Dice improved from `0.8207` at `0.01` to `0.9333` at `0.50`.
- Transfer positive Dice improved from `0.7607` at `0.01` to `0.8765` at `0.25`, then decreased to `0.8452` at `0.50`.
- Scratch boundary Dice improved from `0.8279` at `0.01` to `0.9167` at `0.50`.
- Scratch positive Dice improved from `0.7883` at `0.01` to `0.8746` at `0.50`.

Interpretation:

- The quick-run overcoverage visible in prior binary review panels is substantially driven by the `0.01` threshold.
- This audit does not prove the model is promotion-ready because it is a 5-epoch quick run, but it does show that immediately retraining everything is not the correct next step.
- The next implementation decision should be threshold/report correction and candidate-comparison recalculation using audit-backed threshold policy before training-signal or resize retraining is prioritized.

### Latest non-quick E20/E25 candidates

Command:

```bash
env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq glomeruli-overcoverage-audit --run-id latest_nonquick_e20_e25_no_training_audit --transfer-model-path /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/fixedloader_full_glomeruli_transfer-transfer_s1lr1e-3_s2lr_lrfind_e20_b12_lr1e-3_sz256/fixedloader_full_glomeruli_transfer-transfer_s1lr1e-3_s2lr_lrfind_e20_b12_lr1e-3_sz256.pkl --scratch-model-path /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/scratch/fixedloader_full_glomeruli_no_mito_base-scratch_e25_b12_lr1e-3_sz256/fixedloader_full_glomeruli_no_mito_base-scratch_e25_b12_lr1e-3_sz256.pkl --data-dir /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts --image-size 256 --crop-size 512 --examples-per-category 3 --device mps
```

Runtime output:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_overcoverage_audit/latest_nonquick_e20_e25_no_training_audit
```

Result:

- Both candidate artifacts loaded under `eq-mac`; no load failures were recorded.
- The audit could not construct a category-complete deterministic manifest from the shared candidate validation split.
- Manifest status: `insufficient_evidence_for_promotion`.
- Decision reason: `insufficient_heldout_category_support`.
- No probability rows were generated for these latest non-quick artifacts because the held-out manifest was empty.

Interpretation:

- The latest non-quick E20/E25 artifacts are loadable, but they still do not provide usable category-complete held-out calibration evidence.
- They cannot answer the threshold-calibration question without either a compatible held-out manifest or an explicitly labeled non-promotion calibration manifest.
- This is not a training failure conclusion; it is an evidence-surface failure for those artifacts.

## Screening Ablations

Status: threshold-policy screening run completed.

Current decision: do not run short training ablations yet. The P0 quick no-training audit points first to threshold/report correction, not immediate sampler/loss/resize retraining.

### Threshold-policy comparison rerun

Command:

```bash
env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq.training.compare_glomeruli_candidates --data-dir /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts --output-dir /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison --model-dir /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli --run-id p0_negative_background_quick_5epoch_threshold05_audit --transfer-model-path /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/p0_quick_glomeruli_transfer-transfer_s1lr1e-3_s2lr_lrfind_e5_b12_lr1e-3_sz256/p0_quick_glomeruli_transfer-transfer_s1lr1e-3_s2lr_lrfind_e5_b12_lr1e-3_sz256.pkl --scratch-model-path /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/scratch/p0_quick_glomeruli_no_mito_base-scratch_e5_b12_lr1e-3_sz256/p0_quick_glomeruli_no_mito_base-scratch_e5_b12_lr1e-3_sz256.pkl --seed 42 --image-size 256 --crop-size 512 --examples-per-category 3 --device mps --negative-crop-manifest /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/glomeruli_negative_crops/manifests/p0_quick_mask_background.csv --negative-crop-sampler-weight 0.5 --augmentation-variant fastai_default --overcoverage-audit-dir /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_overcoverage_audit/p0_negative_background_quick_5epoch_no_training_audit --prediction-threshold 0.5
```

Runtime output:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/p0_negative_background_quick_5epoch_threshold05_audit
```

Result:

- Candidate comparison now records `threshold_policy_status=overcoverage_audit_available`.
- Candidate comparison used `threshold=0.5`.
- Background false-positive foreground fraction at selected threshold was `0.0`.
- Positive/boundary recall from the audit at selected threshold was `0.896244281242143`.
- Transfer aggregate Dice/Jaccard/precision/recall: `0.899045` / `0.816605` / `0.915149` / `0.883498`.
- Scratch aggregate Dice/Jaccard/precision/recall: `0.899711` / `0.817704` / `0.877372` / `0.923217`.
- Both transfer and scratch background category Dice/Jaccard became `1.0` / `1.0` with median predicted foreground fraction `0.0`.
- Transfer boundary Dice was `0.933287`; transfer positive Dice was `0.845195`.
- Scratch boundary Dice was `0.916653`; scratch positive Dice was `0.874593`.

Interpretation:

- The threshold-policy correction eliminates the visually severe background overprediction in the quick evidence.
- The quick candidates remain `insufficient_evidence_for_promotion`, not because background overcoverage persists, but because this is still a quick run and resize benefit remains unproven.
- This supports recalibrating/reporting threshold policy before investing in another full training run.

### Validation-derived threshold-policy rerun

Command:

```bash
PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/quicktest_overcoverage_controls_5epoch.yaml
```

Runtime log:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/logs/run_config/p0_negative_background_quick_5epoch_validation_threshold_audit/2026-04-25_135544.log
```

Runtime output:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/p0_negative_background_quick_5epoch_validation_threshold_audit
```

Result:

- The amended YAML did not pass `--prediction-threshold`; candidate comparison derived the threshold from the attached overcoverage audit sweep.
- Candidate comparison recorded `threshold_policy_status=validation_derived_threshold`.
- Selected threshold: `0.25`.
- Selection rule: keep thresholds where every candidate family's mean background false-positive foreground fraction is `<= 0.02`, maximize mean positive/boundary recall, then break ties by lower background foreground fraction and lower threshold.
- Mean background false-positive foreground fraction at selected threshold: `0.00026067097981770837`.
- Mean positive/boundary recall at selected threshold: `0.9600984187634508`.
- Transfer aggregate Dice/Jaccard/precision/recall: `0.892412` / `0.805725` / `0.843754` / `0.947024`.
- Scratch aggregate Dice/Jaccard/precision/recall: `0.861223` / `0.756270` / `0.767331` / `0.981296`.
- Both candidates remained `not_promotion_eligible` because gate reasons still include `category_metric_failure` and `resize_benefit_unproven`.

Interpretation:

- `0.5` should not become a hidden global constant from this quick audit. The implemented policy now derives the threshold from audit evidence and selected `0.25` for the current quick artifacts.
- The threshold-policy gate is no longer the reason these quick candidates are blocked. The remaining blockers are category-level metric behavior and unresolved resize benefit.
- This remains quick-run evidence only and should not replace README-facing assets or serve as final promotion evidence.

### Category-gate rerun after gate-semantics correction

Command:

```bash
PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/quicktest_overcoverage_controls_5epoch.yaml
```

Runtime log:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/logs/run_config/p0_negative_background_quick_5epoch_validation_threshold_audit/2026-04-25_142527.log
```

Runtime output:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/p0_negative_background_quick_5epoch_validation_threshold_audit
```

Result:

- Candidate comparison now writes `category_gate_audit.csv`.
- The report records category-gate fields in `candidate_summary.csv`, `promotion_report.json`, `promotion_report.md`, and `promotion_report.html`.
- Category gates evaluated 42 gate rows per family and 84 rows total.
- Transfer category gate status: `promotion_eligible`; failed gates: `0/42`.
- Scratch category gate status: `promotion_eligible`; failed gates: `0/42`.
- Background crops passed using background false-positive foreground fraction and pixel-level background correctness rather than empty-mask Dice/Jaccard.
- Boundary and positive crops passed Dice, Jaccard, precision, recall, and prediction-to-truth foreground-ratio gates.
- Candidate gate reasons now contain only `resize_benefit_unproven`.
- Candidate promotion evidence status is `insufficient_evidence_for_promotion`, not `not_promotion_eligible`, because resize comparison evidence is still missing.

Interpretation:

- The prior `category_metric_failure` was a gate-semantics artifact for these quick artifacts.
- The corrected category gates do not justify sampler, loss, augmentation, or full production retraining.
- The remaining blocker is resize evidence: current artifacts use `crop_size=512` and `output_size=256`, and no less-downsampled/no-downsample comparator or recorded infeasibility exists.

## Resize-Policy Evidence

Status: resize screen completed for the P0 quick artifacts; current `512->256` policy is cleared for the quick screening evidence because the `512->512` comparator failed category gates.

Current evidence:

- The quick no-training audit recorded current `crop_size=512` and `output_size=256` in `resize_policy_comparison.csv`.
- Because threshold correction substantially improves both background false positives and positive/boundary metrics in the quick audit, resize-policy training ablation is not the immediate next step.
- Resize remains unresolved for production optimization and should come after category-gate semantics are audited.
- The next resize decision must not be inferred from `resize_benefit_unproven` alone. It requires either category-gate evidence showing positive/boundary limitations under the derived threshold or a recorded less-downsampled/no-downsample screen.
- Category gates now pass for the current `512->256` quick evidence, so resize screening was run as a production-decision gate rather than as a fix for current category failure.

### Resize-screening plan

Reason for adding this plan to P0:

- Threshold-policy evidence is now audit-backed.
- Category gates now pass under category-appropriate metrics.
- Candidate gate reasons now contain only `resize_benefit_unproven`.
- Therefore the resize screen is the remaining P0 evidence gate, not a separate future topic.

Required config:

```text
openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/quicktest_resize_screening.yaml
```

Required output:

```text
resize_policy_screening_summary.csv
```

Required screen:

- Reference: `p0_resize_screen_current_512to256`, `crop_size=512`, `image_size=256`.
- Primary comparator: `p0_resize_screen_512to512`, `crop_size=512`, `image_size=512`.
- Fallback comparator: `p0_resize_screen_512to384`, `crop_size=512`, `image_size=384`, only if `p0_resize_screen_512to512` records an MPS memory, unsupported-operation, or runtime failure.

Controls that must remain fixed:

- Deterministic split and seed.
- Negative-crop manifest.
- Transfer and scratch candidate families.
- Validation-derived threshold-selection rule.
- Loss configuration.
- Negative-crop sampler policy.
- `positive_focus_p`.
- Augmentation variant.

Allowed execution adjustment:

- Batch size may be reduced for MPS memory, but the executed value must be recorded and not interpreted as the biological comparison axis.

Decision rule:

- If `512to512` or `512to384` materially improves positive/boundary Dice, recall, or prediction-to-truth foreground ratio without increasing background false-positive foreground fraction beyond the category gate, select that resize policy for the next production candidate recipe.
- If less/no-downsample evidence is similar or worse while current `512to256` passes category gates, clear `resize_benefit_unproven` for the current policy with this evidence recorded.
- If `512to512` and `512to384` both fail to run, keep resize unresolved and record the hardware/runtime infeasibility instead of marking the gate complete.

### Resize-screening execution

Command:

```bash
PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config openspec/changes/p0-calibrate-glomeruli-overcoverage-controls/quicktest_resize_screening.yaml
```

Runtime log:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/logs/run_config/p0_resize_screening_controlled/2026-04-25_145421.log
```

Runtime outputs:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/p0_resize_screen_current_512to256
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/p0_resize_screen_512to512
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/p0_resize_screening_summary/resize_policy_screening_summary.csv
```

Result:

- Reference `p0_resize_screen_current_512to256` completed with `crop_size=512`, `image_size=256`, `batch_size=12`, device `mps`.
- Primary comparator `p0_resize_screen_512to512` completed with `crop_size=512`, `image_size=512`, `batch_size=2`, device `mps`.
- Fallback `p0_resize_screen_512to384` did not run because the no-downsample `512to512` attempt completed; this was the configured fallback condition, not a skipped failure.
- The summary decision is `current_policy_cleared`.
- Decision reason: `less_downsampled_policy_failed_category_gates`.

Reference `512->256` evidence:

- Transfer category gate status: `promotion_eligible`, failed gates `0`.
- Scratch category gate status: `promotion_eligible`, failed gates `0`.
- Transfer positive/boundary Dice and recall: `0.8887253850023118` and `0.9409138254946651`.
- Scratch positive/boundary Dice and recall: `0.8606442218067022` and `0.9792830120322366`.

No-downsample `512->512` evidence:

- Transfer category gate status: `not_promotion_eligible`, failed gates `2`.
- Scratch category gate status: `promotion_eligible`, failed gates `0`.
- Transfer positive/boundary Dice and recall: `0.9153329348227318` and `0.9405250359934643`.
- Scratch positive/boundary Dice and recall: `0.9037359061313378` and `0.9652344179958833`.
- Transfer failed background gates at `512->512`: background predicted foreground fraction `0.02036285400390625` exceeded the `0.02` limit, and pixel accuracy `0.9796371459960938` missed the `0.98` limit.

Interpretation:

- The no-downsample comparator improved positive/boundary Dice, but it introduced a transfer-family background gate failure.
- Because the resize decision rule requires preserving category-gate eligibility, `512->512` is not selected as the next production resize policy from this screen.
- The current `512->256` policy is cleared for the P0 quick screening evidence.
- This does not make the quick 5-epoch artifacts promotion-ready or README-facing assets; it only closes the resize gate for this controlled quick evidence.

### Resize philosophy and next decision

Why resize matters:

- Segmentation models do not see the full-resolution biological image directly. They see a fixed-size tensor, here usually `image_size=256`.
- The current glomeruli workflow samples a `crop_size=512` field of view and resizes it to `image_size=256`, so the model sees the same tissue context at half the linear resolution.
- Downsampling can blur thin borders, capillary walls, nuclei, and edge cues. That can reduce boundary precision or make masks look too smooth.
- Avoiding downsampling with `image_size=512` preserves detail, but it also changes the model's input distribution. Background texture, stain variation, and small structures become more visible. That can increase false-positive foreground on background-looking tissue.
- Larger input tensors also reduce feasible batch size and can change optimization behavior during training. Therefore resize is not a purely cosmetic setting; it changes both biological signal and training dynamics.

What this screen says:

- `512->512` is not globally better in this quick evidence. It improved positive/boundary Dice, but transfer failed background gates on one true-background crop.
- The failure was narrow but real: one transfer background crop had predicted foreground fraction `0.02036285400390625` against the `<=0.02` gate, and pixel accuracy `0.9796371459960938` against the `>=0.98` gate.
- The current `512->256` policy passed category gates for both transfer and scratch and is therefore acceptable for the next production candidate run.

Current verdict:

- Do not pursue resize as the primary global issue right now.
- Keep `crop_size=512` and `image_size=256` for the next real glomeruli production candidate run.
- Do not switch production training to `image_size=512` based on this screen.
- Do not spend another cycle on `image_size=384` unless a full/non-quick production run under the current policy shows persistent foreground-quality failure that cannot be explained by threshold, category gates, or training signal.

Practical next path:

- Train or evaluate the next production glomeruli candidate under the now-audited contract: current `512->256` resize policy, negative/background supervision, explicit loss provenance, validation-derived threshold selection, and category-specific gates.
- Select whichever of transfer or scratch clears the production evidence gates. The goal is not to prove transfer or scratch is philosophically superior; the goal is to obtain one usable, audited model for quantification.
- After a usable glomeruli model is selected, move to quantification validation on ordinary cohort images before MR TIFF inference.
- Treat large MR TIFF inference as a later transport/generalization problem. It should be evaluated with tiling, scale/field-of-view checks, and MR-specific concordance artifacts rather than solved by changing the global glomeruli training resize policy now.

## Augmentation Evidence

Status: implementation support added; no augmentation ablation run yet.

Current evidence:

- Supported variants are now explicit: `fastai_default`, `spatial_only`, and `no_aug`.
- Unsupported variants fail closed.
- Gaussian noise is not recorded as active unless it is actually implemented in the transform path.
- No augmentation variant comparison has been run.

## Final Decision

Status: threshold/report correction, category-gate correction, and resize-screening correction have been implemented for the P0 quick artifacts; production promotion remains unresolved.

Decision:

- Do not launch full production retraining yet.
- Candidate comparison now uses explicit threshold-policy labels: `threshold_policy_unverified`, `fixed_review_threshold`, `audit_backed_fixed_threshold`, and `validation_derived_threshold`.
- Treat `0.01` as unverified for promotion-facing reports unless an overcoverage audit is attached and the threshold-policy gate is promotion-ready.
- The P0 quick screening config now omits `prediction_threshold` so the comparison derives the threshold from the attached audit sweep. The derived threshold for the current quick artifacts is `0.25`, not hardcoded `0.5`.
- Category-gate auditing has been completed for the P0 quick artifacts and cleared `category_metric_failure` under category-appropriate gates.
- Do not prioritize sampler/loss/augmentation retraining until category-appropriate gates show a real remaining segmentation failure.
- Resize benefit is cleared for the P0 quick `512->256` evidence by the controlled screen above.
- The current reference report now consumes `resize_policy_screening_summary.csv`; transfer and scratch have `resize_sensitivity_status=resize_benefit_cleared_current_policy` and no candidate gate reasons in `p0_resize_screen_current_512to256`.
- The quick artifacts still remain insufficient evidence for promotion because they are quick screening artifacts and do not replace a final production/held-out evidence contract.

## Remaining Blockers

Status:

- Candidate-comparison recalculation has been run for the P0 quick artifacts with the overcoverage audit attached.
- A production threshold-selection rule has been implemented for audit-attached candidate comparison.
- Category-gate semantics are now audited for the P0 quick artifacts. The corrected gates cleared `category_metric_failure`.
- Resize benefit has been screened for the P0 quick artifacts. The current `512->256` policy is cleared in this controlled quick screen; `512->512` is not selected because transfer fails background category gates.
- The latest non-quick E20/E25 artifacts still lack category-complete shared held-out calibration evidence.
- No sampler/loss/augmentation training ablation has been run because the evidence cleared threshold, category, and resize gates for the quick artifacts first.
