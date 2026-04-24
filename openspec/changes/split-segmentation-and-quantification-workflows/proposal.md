## Why

The repository currently mixes three different responsibilities across stale names and partially overlapping control surfaces: glomeruli candidate promotion, external-cohort segmentation transport or prediction, and downstream endotheliosis quantification. That makes the supported workflow hard to understand, makes provenance harder to audit, and blurs the boundary between segmentation validation, predictive ROI generation, and ordinal grading.

The split now matters because the current OpenSpec contracts already treat these as distinct stages with different gates. Candidate comparison determines whether a glomeruli artifact is promotion-worthy, transport audit determines whether that artifact is usable on external cohorts such as MR, and quantification determines image-level endotheliosis outputs once the input ROI surface is accepted. One mixed workflow name such as `run_segmentation_fixedloader_full.py` obscures that separation.

## What Changes

- **BREAKING** retire the stale `segmentation_fixedloader_full_retrain` naming and replace it with a candidate-comparison-specific workflow name, runner module, and config surface.
- Add one dedicated workflow surface for glomeruli candidate comparison and promotion evidence generation only.
- Add one dedicated workflow surface for external-cohort segmentation transport audit and explicit prediction or concordance generation, including the MR phase-1 inference lane.
- Add one dedicated workflow surface for endotheliosis quantification that consumes accepted ROI inputs and runs the canonical ordinal workflow without embedding segmentation-promotion or transport-audit behavior into the same config.
- Define one explicit sequencing contract across those workflow families so downstream workflows consume named outputs from upstream workflows rather than silently retraining or inferring extra stages.
- Update CLI, YAML, docs, and tests so the supported entrypoints and artifact locations reflect the split directly.

## Capabilities

### New Capabilities
- `workflow-config-entrypoints`: Define the supported repository workflow-config families, naming contract, separation of concerns, and orchestration boundaries for candidate comparison, transport audit or prediction, and quantification.

### Modified Capabilities
- `glomeruli-candidate-comparison`: Change the control-surface requirements so candidate comparison uses its own explicit workflow config and runner naming rather than a stale mixed-purpose surface.
- `scored-only-quantification-cohort`: Tighten the requirement that external cohort transport audit and MR concordance stay separate from downstream predicted-ROI grading and ordinal quantification entrypoints.
- `segmentation-training-contract`: Update the promotion-workflow control-surface contract so supported glomeruli promotion uses the renamed candidate-comparison workflow surface and does not imply one broader segmentation-plus-quantification run.

## Impact

- Affected code: `src/eq/run_config.py`, `src/eq/training/run_segmentation_fixedloader_full.py` or its replacement, new workflow-runner modules for transport audit and quantification where needed, relevant quantification orchestration helpers, and workflow-path or provenance utilities.
- Affected configs: existing `configs/segmentation_fixedloader_full_retrain.yaml`, plus new split YAMLs for candidate comparison, transport audit or MR concordance, and quantification.
- Affected CLI surface: `eq run-config --config ...` workflow identifiers, help text, and any direct module entrypoints or wrappers that still expose stale names.
- Affected tests: workflow dispatch tests, config contract tests, provenance or artifact-location assertions, and any end-to-end checks that currently assume the mixed runner name.
- Affected artifacts: segmentation promotion evidence stays under `output/segmentation_evaluation/`, model-generated prediction assets stay under `output/predictions/`, and quantification outputs stay under `output/quantification_results/` without one workflow pretending to own all three at once.
