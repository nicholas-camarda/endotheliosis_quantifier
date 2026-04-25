## 1. Rename and split the current mixed candidate-comparison surface

- [ ] 1.1 Inventory every code path, config, test, and doc reference to `full_segmentation_retrain`, `run_full_segmentation_retrain.py`, and `run_full_segmentation_retrain(...)`.
- [ ] 1.2 Rename the mixed candidate-comparison runner to the exact module `src/eq/training/run_glomeruli_candidate_comparison_workflow.py` with the exact entrypoint `run_glomeruli_candidate_comparison_workflow(...)`, keeping its responsibility limited to manifest refresh plus candidate-comparison orchestration.
- [ ] 1.3 Replace the retired workflow identifier with the exact dedicated candidate-comparison workflow ID `glomeruli_candidate_comparison` in `src/eq/run_config.py` and any related dispatch or help text.
- [ ] 1.4 Replace `configs/segmentation_fixedloader_full_retrain.yaml` with the exact config `configs/glomeruli_candidate_comparison.yaml`, whose comments and fields describe promotion evidence only.

## 2. Add explicit transport-audit workflow surfaces

- [ ] 2.1 Inventory the current inference, evaluation, and scored-only cohort code paths that should feed a dedicated transport-audit workflow rather than candidate comparison or quantification.
- [ ] 2.2 Add the exact workflow family `glomeruli_transport_audit` to the run-config dispatcher with explicit required inputs for the segmentation artifact and cohort or manifest slice.
- [ ] 2.3 Implement or refactor the exact workflow runner module `src/eq/evaluation/run_glomeruli_transport_audit_workflow.py` so it writes transport-audit artifacts under `output/segmentation_evaluation/` and reusable prediction assets under `output/predictions/` without launching quantification.
- [ ] 2.4 Add the committed standard transport-audit YAML `configs/glomeruli_transport_audit.yaml` for non-large-field cohort regimes.

## 3. Add a separate high-resolution concordance workflow family

- [ ] 3.1 Inventory the current MR and large-image code paths, helper utilities, and input assumptions that differ materially from standard transport audit.
- [ ] 3.2 Add the exact high-resolution concordance workflow family `highres_glomeruli_concordance` to the run-config dispatcher for large-field microscope images with explicit tiling or preprocessing inputs.
- [ ] 3.3 Implement or refactor the exact workflow runner module `src/eq/evaluation/run_highres_glomeruli_concordance_workflow.py` so it records concordance and transport-evaluation artifacts for MR-like image regimes without pretending those runs are standard transport audits or downstream quantification.
- [ ] 3.4 Add the committed high-resolution concordance YAML `configs/highres_glomeruli_concordance.yaml`, whose comments describe the large-image preprocessing contract explicitly.

## 4. Audit and normalize the quantification workflow surface

- [ ] 4.1 Audit the existing `quant-endo`, `prepare-quant-contract`, `prepare_quantification_contract`, `run_endotheliosis_scoring_pipeline`, and `run_contract_first_quantification` surfaces for duplication, stale arguments, and conflicting orchestration responsibilities.
- [ ] 4.2 Apply the explicit retention criteria: preserve `quant-endo` only if it can become a thin compatibility caller that keeps only `--data-dir`, `--segmentation-model`, `--score-source`, `--annotation-source`, `--mapping-file`, `--output-dir`, `--apply-migration`, and `--stop-after`; otherwise retire it. Apply the same rule to `prepare-quant-contract`, preserving it only if it becomes a thin caller with `stop_after=contract`.
- [ ] 4.3 Add the exact workflow family `endotheliosis_quantification` to the run-config dispatcher with explicit upstream segmentation artifact or accepted predicted-ROI inputs.
- [ ] 4.4 Implement or refactor the exact canonical quantification workflow runner module `src/eq/quantification/run_endotheliosis_quantification_workflow.py` so it executes the contract-first quantification path without retraining segmentation candidates or rerunning transport audit implicitly.
- [ ] 4.5 Add the committed quantification YAML `configs/endotheliosis_quantification.yaml`, whose comments describe required inputs, runtime output roots, and ordinal-stage scope without mixing in segmentation-promotion language.

## 5. Remove stale mixed names and preserve explicit handoffs

- [ ] 5.1 Make downstream workflows fail closed when required upstream artifact references are missing instead of auto-discovering latest artifacts or launching upstream stages automatically.
- [ ] 5.2 Remove stale mixed workflow names from committed code, config comments, CLI help, and tests rather than leaving long-lived aliases, and reject the retired workflow ID `segmentation_fixedloader_full_retrain` explicitly in `eq run-config`.
- [ ] 5.3 Ensure workflow logs and provenance identify candidate comparison, standard transport audit, high-resolution concordance, or quantification explicitly and record the exact upstream artifacts each run consumed.

## 6. Validate the split end to end

- [ ] 6.1 Update or add tests that cover run-config dispatch for the exact workflow IDs `glomeruli_candidate_comparison`, `glomeruli_transport_audit`, `highres_glomeruli_concordance`, and `endotheliosis_quantification`, plus retired-workflow rejection and explicit missing-input failures for downstream workflows.
- [ ] 6.2 Run targeted CLI and test validation for the changed workflow families, including `python scripts/check_openspec_explicitness.py openspec/changes/p1-split-segmentation-and-quantification-workflows`, `python -m eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run`, `python -m eq run-config --config configs/glomeruli_transport_audit.yaml --dry-run`, `python -m eq run-config --config configs/highres_glomeruli_concordance.yaml --dry-run`, and `python -m eq run-config --config configs/endotheliosis_quantification.yaml --dry-run`, plus relevant unit or integration tests.
- [ ] 6.3 Run `env OPENSPEC_TELEMETRY=0 openspec validate p1-split-segmentation-and-quantification-workflows --strict` and resolve any spec or artifact failures.
