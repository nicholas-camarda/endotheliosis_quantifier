## Why

Oracle findings 4 and 5 show that quantification currently has two user-facing target-definition paths: the YAML workflow can apply reviewed label overrides, while direct `quant-endo` and `prepare-quant-contract` commands bypass that contract. The reviewed override CSV also lives under a prior output tree, so a cleanup or rerun can delete a required modeling input.

## What Changes

- **BREAKING** Define one canonical quantification input contract for scored cohort data, reviewed label overrides, score source selection, mapping files, segmentation artifacts, and output roots.
- **BREAKING** Move reviewed label override inputs out of previous `output/quantification_results/...` trees and into a stable runtime-derived input location.
- **BREAKING** Require `eq quant-endo`, `eq prepare-quant-contract`, and `eq run-config --config configs/endotheliosis_quantification.yaml` to resolve quantification inputs through the same contract owner before any labels are consumed.
- **BREAKING** Fail closed when a direct CLI invocation would use a different target definition than the YAML workflow; do not silently ignore reviewed overrides.
- Add explicit provenance for the resolved label contract, including the override file path, content hash, base scored input hash, annotation/mapping hashes when file-backed, segmentation artifact reference/hash, matched row count, unmatched row count, duplicate handling, accepted score set, grouping identity, and effective target-definition version.
- Update tests so YAML and direct CLI invocations either produce the same resolved label contract or direct CLI fails with an actionable contract error.
- Update docs/config comments so `eq run-config --config configs/endotheliosis_quantification.yaml` remains the primary front door and direct commands are described only as thin wrappers over the same contract.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `workflow-config-entrypoints`: Quantification config entrypoints must route through the same input resolver as direct quantification CLI commands and must not hide target-definition differences.
- `scored-only-quantification-cohort`: Reviewed label overrides become a stable scored-cohort input contract with explicit provenance and fail-closed validation.
- `endotheliosis-grade-model`: P3 grade-model fitting must consume only the resolved canonical label contract and record that contract in final model and verdict artifacts.

## Explicit Decisions

- Change ID: `oracle-canonical-quantification-input-contract`.
- Oracle findings covered: finding 4, direct CLI bypasses reviewed label overrides; finding 5, override input lives under a prior output tree.
- The stable reviewed-label input path is runtime-root relative: `derived_data/quantification_inputs/reviewed_label_overrides/endotheliosis_grade_model/rubric_label_overrides.csv`.
- `configs/endotheliosis_quantification.yaml` must reference that stable runtime-derived input path instead of `output/quantification_results/.../rubric_label_overrides_for_next_modeling_run.csv`.
- The canonical resolver owner must live under `src/eq/quantification/` and reuse existing quantification path, score, cohort, and label-override helpers before adding new surfaces.
- The resolved contract must include biological grouping identity: `subject_id`, row identity, `subject_image_id` uniqueness, grouping-key derivation, row count, and subject count.
- Target-defining content hashes must include the base scored table or manifest, annotation source when file-backed, mapping file when present, reviewed override file when present, and segmentation artifact metadata reference.
- Direct CLI parser updates, if retained, must expose `--label-overrides` and pass it into the same resolver used by YAML execution. If a direct command cannot satisfy the full contract, it must fail before modeling starts and point to the YAML workflow.
- The implementation must not create a second score-loading path, a second label-override parser, or a command-specific override merge.

## Open Questions

- [audit_first_then_decide] Which existing helper should own the resolved input dataclass: `src/eq/quantification/cohorts.py`, `src/eq/quantification/labelstudio_scores.py`, or a minimal shared contract owner under `src/eq/quantification/`? Deciding audit target: current call graph in `src/eq/quantification/pipeline.py`, `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, and `src/eq/__main__.py`.
- [audit_first_then_decide] Should the existing reviewed override file be copied into the stable runtime-derived path by a task in this change or regenerated from the rubric review source? Deciding evidence source: current runtime file hash, override audit artifacts, and review queue provenance.

## logging-contract

This change does not add a new durable logging root or subprocess teeing behavior. Existing `eq run-config` run logging remains the durable command-capture surface; this change adds label-contract provenance to quantification artifacts and requires direct CLI failures to be visible before modeling starts.

## docs-impact

README/docs/config comments must describe the current YAML-first quantification input contract, the stable reviewed-override input path, and the direct CLI equivalence-or-fail-closed rule. Documentation must avoid historical migration framing and must not present prior output-tree override paths as supported inputs.

## Impact

- Affected code: `src/eq/__main__.py`, `src/eq/run_config.py`, `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, `src/eq/quantification/pipeline.py`, `src/eq/quantification/cohorts.py`, `src/eq/quantification/labelstudio_scores.py`, and `src/eq/quantification/endotheliosis_grade_model.py`.
- Affected config: `configs/endotheliosis_quantification.yaml`.
- Affected runtime inputs: reviewed label override CSVs under the runtime root.
- Affected outputs: `scored_examples/score_label_overrides_audit.csv`, `scored_examples/score_label_overrides_summary.json`, target-definition provenance artifacts, P3 `summary/final_product_verdict.json`, final model metadata, and quantification workflow run logs.
- Existing output-tree override paths become invalid as supported modeling inputs after this change, though old output directories remain historical artifacts.
