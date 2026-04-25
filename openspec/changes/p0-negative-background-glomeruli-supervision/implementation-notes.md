# Implementation Notes

## Augmentation Audit

- Current candidate-comparison training still uses the FastAI `aug_transforms` path from `src/eq/data_management/datablock_loader.py`.
- The older `configs/glomeruli_finetuning_config.yaml` augmentation fields are not the active candidate-comparison control surface.
- Gaussian noise is not active in the implemented candidate-comparison workflow.
- No augmentation ablation was completed in this change; augmentation is now recorded as provenance via `augmentation_policy` so future comparisons can audit it explicitly.

## Negative/Background Supervision Implementation

- Implemented validated negative crop manifests and mask-derived zero-foreground background crop generation.
- Implemented MR/TIFF review-batch generation as review-only proposal assets. Unreviewed proposal rows are rejected by the training manifest validator.
- Threaded validated negative crop manifests through dynamic patching, transfer training, scratch training, candidate comparison, and YAML workflow execution.
- Negative crops are training-only all-zero mask supervision. They are not added to validation splits or treated as promotion evidence.

## Validation Commands

- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/test_negative_glomeruli_crops.py -q`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/integration/test_cli.py -q`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/test_glomeruli_candidate_comparison.py tests/test_segmentation_training_contract.py -q`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/test_glomeruli_candidate_comparison.py tests/test_segmentation_validation_audit.py -q`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`

Final full-suite result after the comparison inference fix: `167 passed, 3 skipped, 8 warnings in 38.71s`.

## Short MPS Smoke Run

Initial command:

```bash
env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config openspec/changes/p0-negative-background-glomeruli-supervision/quicktest_negative_background_5epoch.yaml
```

Run log:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/logs/run_config/p0_negative_background_quick_5epoch/2026-04-24_235659.log
```

The one-command smoke run completed mitochondria pretraining plus both glomeruli candidates, then failed during candidate comparison inference with:

```text
TypeError: typing.Any cannot be used with isinstance()
```

Root cause: `compare_glomeruli_candidates` used `learn.dls.test_dl([pil_image])`, which failed against a learner exported from the negative-crop DataBlock because FastAI attempted `isinstance(..., typing.Any)`.

Fix applied: candidate-comparison inference now uses deterministic resize-to-network-size plus ImageNet normalization and runs the loaded model directly. DataBlock negative-crop getters also no longer expose `typing.Any` return annotations.

After the fix, the failed comparison stage was rerun against the same two models trained by the smoke command:

```bash
env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq.training.compare_glomeruli_candidates --data-dir /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts --output-dir /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison --model-dir /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli --run-id p0_negative_background_quick_5epoch --transfer-model-path /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/p0_quick_glomeruli_transfer-transfer_s1lr1e-3_s2lr_lrfind_e5_b12_lr1e-3_sz256/p0_quick_glomeruli_transfer-transfer_s1lr1e-3_s2lr_lrfind_e5_b12_lr1e-3_sz256.pkl --scratch-model-path /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/scratch/p0_quick_glomeruli_no_mito_base-scratch_e5_b12_lr1e-3_sz256/p0_quick_glomeruli_no_mito_base-scratch_e5_b12_lr1e-3_sz256.pkl --seed 42 --image-size 256 --crop-size 512 --examples-per-category 3 --device mps --negative-crop-manifest /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/glomeruli_negative_crops/manifests/p0_quick_mask_background.csv --negative-crop-sampler-weight 0.5 --augmentation-variant fastai_default
```

Final comparison artifacts:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/p0_negative_background_quick_5epoch
```

Generated background manifest:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/glomeruli_negative_crops/manifests/p0_quick_mask_background.csv
```

Manifest audit:

- `negative_crop_count=1414`
- `mask_derived_background_crop_count=1414`
- `curated_negative_crop_count=0`
- `source_image_count=707`
- `rejected_overlap_count=89`
- `sampler_weight=0.5`
- `manifest_sha256=23385079d7477437271d3976982dfb93dfce1bd329969c1f396ba173e9fdbd0c`

## Smoke Results

Promotion status:

- Transfer: `not_promotion_eligible`
- Scratch: `not_promotion_eligible`

Candidate summary:

- Transfer aggregate: Dice `0.731014`, Jaccard `0.576062`, precision `0.576070`, recall `0.999976`.
- Scratch aggregate: Dice `0.756324`, Jaccard `0.608136`, precision `0.608136`, recall `1.000000`.

Background category:

- Transfer background median prediction foreground fraction: `0.161236`.
- Scratch background median prediction foreground fraction: `0.102631`.
- Both background category Dice/Jaccard remained `0.0`.

Interpretation:

- Negative/background supervision is now present and correctly recorded in the artifacts.
- The quick 5-epoch run improved the artifact contract but did not solve the biological segmentation behavior. Both candidates still overpredict foreground on true background crops and overcover boundary/positive crops.
- The report no longer classifies the failure as `negative_background_supervision_missing`; remaining root causes are `resize_policy_artifact` and `training_signal_insufficient`.

