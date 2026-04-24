## 1. Rename and split the current mixed candidate-comparison surface

- [ ] 1.1 Inventory every code path, config, test, and doc reference to `segmentation_fixedloader_full_retrain`, `run_segmentation_fixedloader_full.py`, and `run_fixedloader_full`.
- [ ] 1.2 Rename the mixed candidate-comparison runner module and function to a candidate-comparison-specific name and keep its responsibility limited to manifest refresh plus candidate-comparison orchestration.
- [ ] 1.3 Replace the retired workflow identifier with a dedicated candidate-comparison workflow ID in `src/eq/run_config.py` and any related dispatch or help text.
- [ ] 1.4 Replace `configs/segmentation_fixedloader_full_retrain.yaml` with a dedicated candidate-comparison config whose comments and fields describe promotion evidence only.

## 2. Add explicit transport-audit workflow surfaces

- [ ] 2.1 Inventory the current inference, evaluation, and scored-only cohort code paths that should feed a dedicated transport-audit workflow rather than candidate comparison or quantification.
- [ ] 2.2 Add a `segmentation_transport_audit` workflow family to the run-config dispatcher with explicit required inputs for the segmentation artifact and cohort or manifest slice.
- [ ] 2.3 Implement or refactor a workflow runner that writes transport-audit artifacts under `output/segmentation_evaluation/` and reusable prediction assets under `output/predictions/` without launching quantification.
- [ ] 2.4 Add at least one committed transport-audit YAML example, including an MR-oriented config or mode that records the high-resolution tiling or preprocessing contract explicitly.

## 3. Add explicit quantification workflow surfaces

- [ ] 3.1 Inventory the existing `quant-endo`, `prepare-quant-contract`, and `run_contract_first_quantification` orchestration surfaces and decide which code path becomes the single workflow-runner implementation.
- [ ] 3.2 Add an `endotheliosis_quantification` workflow family to the run-config dispatcher with explicit upstream segmentation artifact or accepted predicted-ROI inputs.
- [ ] 3.3 Implement or refactor the quantification workflow runner so it executes the contract-first quantification path without retraining segmentation candidates or rerunning transport audit implicitly.
- [ ] 3.4 Add a committed quantification YAML example whose comments describe required inputs, runtime output roots, and ordinal-stage scope without mixing in segmentation-promotion language.

## 4. Remove stale mixed names and preserve explicit handoffs

- [ ] 4.1 Make downstream workflows fail closed when required upstream artifact references are missing instead of auto-discovering latest artifacts or launching upstream stages automatically.
- [ ] 4.2 Remove stale mixed workflow names from committed code, config comments, CLI help, and tests rather than leaving long-lived aliases.
- [ ] 4.3 Ensure workflow logs and provenance identify candidate comparison, transport audit, or quantification explicitly and record the exact upstream artifacts each run consumed.

## 5. Validate the split end to end

- [ ] 5.1 Update or add tests that cover run-config dispatch, committed YAML dry runs, retired-workflow rejection, and explicit missing-input failures for downstream workflows.
- [ ] 5.2 Run targeted CLI and test validation for the changed workflow families, including `python -m eq run-config --config <yaml> --dry-run` for each committed workflow config and relevant unit or integration tests.
- [ ] 5.3 Run `env OPENSPEC_TELEMETRY=0 openspec validate split-segmentation-and-quantification-workflows --strict` and resolve any spec or artifact failures.
