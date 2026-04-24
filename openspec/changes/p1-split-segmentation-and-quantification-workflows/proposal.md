## Why

The repository currently mixes three different responsibilities across stale names and partially overlapping control surfaces: glomeruli candidate promotion, external-cohort segmentation transport or prediction, and downstream endotheliosis quantification. That makes the supported workflow hard to understand, makes provenance harder to audit, and blurs the boundary between segmentation validation, predictive ROI generation, and ordinal grading.

The split now matters because the current OpenSpec contracts already treat these as distinct stages with different gates. Candidate comparison determines whether a glomeruli artifact is promotion-worthy, transport audit determines whether that artifact is usable on external cohorts such as MR, and quantification determines image-level endotheliosis outputs once the input ROI surface is accepted. One mixed workflow name such as `run_segmentation_fixedloader_full.py` obscures that separation.

## What Changes

- **BREAKING** retire the stale `segmentation_fixedloader_full_retrain` naming and replace it with the exact candidate-comparison workflow surface `workflow: glomeruli_candidate_comparison`, the runner module `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`, and the config `configs/glomeruli_candidate_comparison.yaml`.
- Add one dedicated workflow surface for glomeruli candidate comparison and promotion evidence generation only.
- Add one dedicated workflow surface for external-cohort standard transport audit at `workflow: glomeruli_transport_audit` with config `configs/glomeruli_transport_audit.yaml` and runner module `src/eq/evaluation/run_glomeruli_transport_audit_workflow.py`.
- Add one separate high-resolution concordance workflow family for large-field microscope images such as the MR phase-1 lane at `workflow: highres_glomeruli_concordance` with config `configs/highres_glomeruli_concordance.yaml` and runner module `src/eq/evaluation/run_highres_glomeruli_concordance_workflow.py`.
- Add one dedicated workflow surface for endotheliosis quantification at `workflow: endotheliosis_quantification` with config `configs/endotheliosis_quantification.yaml` and runner module `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, consuming accepted ROI inputs and running the canonical ordinal workflow without embedding segmentation-promotion or transport-audit behavior into the same config.
- Define one explicit sequencing contract across those workflow families so downstream workflows consume named outputs from upstream workflows rather than silently retraining or inferring extra stages.
- Update CLI, YAML, docs, and tests so the supported entrypoints and artifact locations reflect the split directly.

## Explicit Decisions

- The exact candidate-comparison workflow surface is `workflow: glomeruli_candidate_comparison`.
- The exact standard transport-audit workflow surface is `workflow: glomeruli_transport_audit`.
- The exact high-resolution concordance workflow surface is `workflow: highres_glomeruli_concordance`.
- The exact quantification workflow surface is `workflow: endotheliosis_quantification`.
- The exact target runner modules are:
  - `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`
  - `src/eq/evaluation/run_glomeruli_transport_audit_workflow.py`
  - `src/eq/evaluation/run_highres_glomeruli_concordance_workflow.py`
  - `src/eq/quantification/run_endotheliosis_quantification_workflow.py`
- The exact target committed workflow configs are:
  - `configs/glomeruli_candidate_comparison.yaml`
  - `configs/glomeruli_transport_audit.yaml`
  - `configs/highres_glomeruli_concordance.yaml`
  - `configs/endotheliosis_quantification.yaml`

## Capabilities

### New Capabilities
- `workflow-config-entrypoints`: Define the supported repository workflow-config families, naming contract, separation of concerns, and orchestration boundaries for candidate comparison, standard transport audit, high-resolution concordance, and quantification.

### Modified Capabilities
- `glomeruli-candidate-comparison`: Change the control-surface requirements so candidate comparison uses its own explicit workflow config and runner naming rather than a stale mixed-purpose surface.
- `scored-only-quantification-cohort`: Tighten the requirement that external cohort transport audit and MR concordance stay separate from downstream predicted-ROI grading and ordinal quantification entrypoints.
- `segmentation-training-contract`: Update the promotion-workflow control-surface contract so supported glomeruli promotion uses the renamed candidate-comparison workflow surface and does not imply one broader segmentation-plus-quantification run.

## Impact

- Affected code: `src/eq/run_config.py`, `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`, `src/eq/evaluation/run_glomeruli_transport_audit_workflow.py`, `src/eq/evaluation/run_highres_glomeruli_concordance_workflow.py`, `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, relevant quantification orchestration helpers, and workflow-path or provenance utilities.
- Affected configs: existing `configs/segmentation_fixedloader_full_retrain.yaml`, plus the new split YAMLs `configs/glomeruli_candidate_comparison.yaml`, `configs/glomeruli_transport_audit.yaml`, `configs/highres_glomeruli_concordance.yaml`, and `configs/endotheliosis_quantification.yaml`.
- Affected CLI surface: `eq run-config --config ...` workflow identifiers, help text, and any direct module entrypoints or wrappers that still expose stale names.
- Affected tests: workflow dispatch tests, config contract tests, provenance or artifact-location assertions, and any end-to-end checks that currently assume the mixed runner name.
- Affected artifacts: segmentation promotion evidence stays under `output/segmentation_evaluation/`, model-generated prediction assets stay under `output/predictions/`, and quantification outputs stay under `output/quantification_results/` without one workflow pretending to own all three at once.
