## Implementation Notes

Implementation date: 2026-04-24

## Mixed Candidate-Comparison Surface Inventory

Stale mixed workflow references were found in:

- `src/eq/run_config.py`
- `src/eq/training/run_full_segmentation_retrain.py`
- `configs/full_segmentation_retrain.yaml`
- `README.md`
- `docs/ONBOARDING_GUIDE.md`
- `docs/TECHNICAL_LAB_NOTEBOOK.md`
- `tests/integration/test_cli.py`
- `tests/unit/test_config_paths.py`

The active runner and config were renamed to:

- `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`
- `run_glomeruli_candidate_comparison_workflow(...)`
- `configs/glomeruli_candidate_comparison.yaml`
- `workflow: glomeruli_candidate_comparison`

The retired workflow IDs `full_segmentation_retrain` and `segmentation_fixedloader_full_retrain` are rejected by `eq run-config`.

## Transport And High-Resolution Inventory

Reusable standard transport and MR/high-resolution helper logic existed in `src/eq/quantification/cohorts.py`:

- `write_segmentation_transport_audit(...)`
- `validate_segmentation_transport_inputs(...)`
- `write_mr_inference_contract(...)`
- `build_mr_concordance_workflow(...)`
- `mr_concordance_metrics(...)`

These helpers are now exposed through dedicated workflow runners:

- `src/eq/evaluation/run_glomeruli_transport_audit_workflow.py`
- `src/eq/evaluation/run_highres_glomeruli_concordance_workflow.py`

Both workflows require explicit upstream segmentation/prediction inputs and fail closed when those references are missing.

## Quantification Surface Decision

The audit-first quantification question was resolved by inspecting:

- `eq quant-endo`
- `eq prepare-quant-contract`
- `prepare_quantification_contract(...)`
- `run_endotheliosis_scoring_pipeline(...)`
- `run_contract_first_quantification(...)`

Decision: `run_contract_first_quantification(...)` remains the canonical quantification engine. The new workflow runner is `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, which exposes config-driven execution and a direct `run_endotheliosis_quantification_inputs(...)` function. `quant-endo` and `prepare-quant-contract` are retained only as thin compatibility callers into that workflow module. Dead training-style CLI arguments `--batch-size` and `--epochs` were removed from `quant-endo`.

## Validation

Focused validation run:

```bash
/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q \
  tests/integration/test_cli.py \
  tests/unit/test_config_paths.py \
  tests/unit/test_quantification_cohorts.py \
  tests/integration/test_local_runtime_quantification_pipeline.py \
  tests/test_training_entrypoint_contract.py \
  tests/test_segmentation_training_contract.py
```

Result: 72 passed, 1 skipped, 8 warnings.
