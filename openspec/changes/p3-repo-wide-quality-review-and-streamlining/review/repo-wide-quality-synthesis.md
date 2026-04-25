# Repo-Wide Quality Synthesis

## Bottom Line

The repo is coherent enough to continue, with one major caveat: workflow boundaries are now clean, but scientific improvement still depends on the follow-on negative-crop curation/sampler implementation. P3 should not be used to claim model promotion.

## Ranked Findings

1. **Negative/background supervision remains the main scientific blocker.**
   Evidence: `p1-add-negative-glomeruli-crop-supervision/source-provenance-inventory.md`; p0 review artifacts showing false-positive/over-foreground behavior.

2. **Workflow boundaries are now explicit and should stay that way.**
   Evidence: `src/eq/run_config.py`, split configs under `configs/`, p2 implementation notes, config dry-runs.

3. **Downstream workflows must require explicit upstream artifacts.**
   Evidence: `configs/glomeruli_transport_audit.yaml`, `configs/highres_glomeruli_concordance.yaml`, `configs/endotheliosis_quantification.yaml`, and p2 tests.

4. **The workspace layout is acceptable; no repo/runtime/cloud move is needed.**
   Evidence: `workspace-governor-report.md`, `analysis_registry.yaml`, runtime root listing.

5. **Some historical docs contain stale commands and should be clearly labeled historical rather than corrected into current workflows.**
   Evidence: documentation-wizard report, especially `docs/PIPELINE_INTEGRATION_PLAN.md:155`.

## Cross-Lane Conflicts

- Documentation-wizard reports `README.md:54` as a stale `--help` flag, but direct inspection shows this is the command `python -m eq --help`, not a stale flag for another CLI. Decision: reject that specific finding as a false positive.
- Research-partner inventory reports apparent missing paths such as `/eq` and `/MPS`; direct inspection shows those come from backtick command/doc tokens being over-interpreted as filesystem paths. Decision: reject as parser noise, not repo path drift.

## Action Groups

- Documentation: label stale historical planning docs and keep README/docs aligned with split workflow configs.
- CLI/workflow: keep p2 split surfaces and retired workflow rejection.
- Runtime validation: run standard dry-runs, full pytest, and the requested 5-epoch quick candidate-comparison workflow.
- Scientific follow-on: defer negative-crop curation/sampler implementation to a dedicated method-development change.

## Residual Risk

- The 5-epoch quick run validates execution only; it is not model-quality evidence.
- True-negative crop supervision is not implemented yet.
- Runtime contains legacy `output/segmentation_results/`; it is not a source-repo problem but remains a runtime cleanup candidate.

## Final Validation Results

- `openspec validate p3-repo-wide-quality-review-and-streamlining --strict`: passed.
- `python3 scripts/check_openspec_explicitness.py openspec/changes/p3-repo-wide-quality-review-and-streamlining`: passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`: passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --help`: passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/mito_pretraining_config.yaml --dry-run`: passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/glomeruli_finetuning_config.yaml --dry-run`: passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run`: passed.
- `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`: passed, `163 passed, 3 skipped, 8 warnings`.
- 5-epoch live MPS quick workflow: passed.
  - Config: `openspec/changes/p3-repo-wide-quality-review-and-streamlining/review/quicktest_glomeruli_candidate_comparison_5epoch.yaml`
  - Log: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/logs/run_config/p3_quick_5epoch/2026-04-24_224615.log`
  - Comparison output: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/p3_quick_5epoch`
  - Mito artifact: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/mitochondria/p3_quick_mito_base-pretrain_e5_b24_lr1e-3_sz256/p3_quick_mito_base-pretrain_e5_b24_lr1e-3_sz256.pkl`
  - Transfer artifact: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/p3_quick_glomeruli_transfer-transfer_s1lr1e-3_s2lr_lrfind_e5_b12_lr1e-3_sz256/p3_quick_glomeruli_transfer-transfer_s1lr1e-3_s2lr_lrfind_e5_b12_lr1e-3_sz256.pkl`
  - Scratch artifact: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/scratch/p3_quick_glomeruli_no_mito_base-scratch_e5_b12_lr1e-3_sz256/p3_quick_glomeruli_no_mito_base-scratch_e5_b12_lr1e-3_sz256.pkl`

## Quick Run Interpretation

The quick run proves that the split YAML workflow can build a cohort manifest, train the mitochondria base, train transfer and scratch glomeruli candidates, export model artifacts, load both candidates, and write candidate-comparison review artifacts under the expected runtime roots.

It does not prove promotion readiness. The quick run report explicitly marks both candidates `not_promotion_eligible`:

- transfer: Dice `0.8630`, precision `0.7593`, recall `0.9995`, blocked for `background_false_positive_foreground_excess`, `category_metric_failure`, and `resize_benefit_unproven`.
- scratch: Dice `0.7504`, precision `0.6005`, recall `1.0000`, blocked for `background_false_positive_foreground_excess`, `category_metric_failure`, `positive_or_boundary_overcoverage`, and `resize_benefit_unproven`.

The category breakdown is the important result: both candidates have background Dice `0.0`; transfer predicts foreground on background crops with median foreground fraction `0.0513`, and scratch predicts much more foreground on background crops with median foreground fraction `0.4889`. This supports the p1/p3 conclusion that negative/background supervision is the next method-development blocker.
