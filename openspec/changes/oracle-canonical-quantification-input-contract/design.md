## Context

The supported quantification workflow has converged on `eq run-config --config configs/endotheliosis_quantification.yaml`, but older direct CLI paths still exist. Oracle finding 4 shows that direct `quant-endo` and `prepare-quant-contract` can run without the reviewed `inputs.label_overrides` contract that the YAML workflow uses. Oracle finding 5 shows that the configured override file currently points into a previous `output/quantification_results/...` subtree, which makes a required model input depend on a deletable generated output.

This is a target-definition problem, not a convenience-option problem. If two entrypoints can fit or prepare quantification artifacts with different effective labels, then model metrics, P3 selection, and deployment verdicts are not comparable.

## Goals / Non-Goals

**Goals:**

- Make reviewed label overrides a stable runtime-derived input, not a prior-output artifact.
- Make YAML and direct CLI quantification paths resolve the same scored cohort, score source, mapping, reviewed override, segmentation artifact, and output contract.
- Record enough label-contract provenance to make reruns auditable.
- Reuse existing quantification score/cohort helpers and path resolution rather than adding a second input stack.
- Preserve `eq run-config --config configs/endotheliosis_quantification.yaml` as the primary documented path.

**Non-Goals:**

- Do not change the reviewed rubric itself.
- Do not infer labels for unreviewed rows.
- Do not add a compatibility fallback to the prior output-tree override path.
- Do not introduce a second direct-CLI-only override parser.
- Do not claim external validation or scientific promotion from this contract cleanup.

## Decisions

### Decision: Use One Resolved Quantification Input Contract

The implementation will introduce or strengthen a single resolved input contract under `src/eq/quantification/`. The exact owner must be chosen after auditing existing call sites, but the resolver must be consumed by:

- `src/eq/quantification/run_endotheliosis_quantification_workflow.py`
- `src/eq/quantification/pipeline.py`
- direct CLI handlers in `src/eq/__main__.py`
- P3 grade-model fitting in `src/eq/quantification/endotheliosis_grade_model.py`

Alternative considered: add `--label-overrides` to direct CLI and merge overrides locally there. That would fix the symptom but preserve duplicate target-definition logic. The accepted design centralizes resolution so future score-source or override-contract changes happen once.

### Decision: Store Reviewed Overrides Under Derived Input Contracts

The stable path for reviewed overrides is runtime-root relative:

`derived_data/quantification_inputs/reviewed_label_overrides/endotheliosis_grade_model/rubric_label_overrides.csv`

This path is an input contract because it is derived from human review and consumed by modeling. It is not a model output. `configs/endotheliosis_quantification.yaml` must reference this path, not an `output/quantification_results/...` path.

Alternative considered: keep the generated output path and document not to delete it. That makes cleanup unsafe and keeps required inputs under generated artifact trees, which conflicts with the repo path contract.

### Decision: Direct CLI Is A Thin Wrapper Or A Fail-Closed Command

`eq quant-endo` and `eq prepare-quant-contract` may remain only if they resolve the same input contract as YAML. They must accept reviewed override input when they can fit or prepare label-dependent artifacts. If a direct path cannot satisfy the full contract, it must fail before label loading and point to the YAML workflow.

Alternative considered: deprecate the direct commands immediately. That may be appropriate later, but this spec requires behavior equivalence or fail-closed behavior first so users cannot accidentally produce divergent targets.

### Decision: Label Contract Provenance Is Required Runtime Evidence

Every quantification run that consumes scores must write label-contract provenance alongside existing score override audits. Required fields include:

- resolved scored cohort root
- score source
- annotation source and mapping file if used
- label override path or explicit `none`
- label override content hash
- matched override rows
- unmatched override rows
- duplicate override status
- accepted score set
- effective target-definition version

P3 final model metadata and final verdict artifacts must reference this provenance.

## Explicit Decisions

- Reuse checked surfaces: `src/eq/quantification/cohorts.py`, `src/eq/quantification/labelstudio_scores.py`, `src/eq/quantification/pipeline.py`, `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, `src/eq/run_config.py`, `src/eq/__main__.py`, `configs/endotheliosis_quantification.yaml`, and existing override audit outputs under `scored_examples/`.
- No new standalone script is planned. Tests should extend existing unit tests under `tests/unit/`.
- New helper surfaces, if needed after audit, must be under `src/eq/quantification/` and must replace duplicated local logic in touched callers.
- Previous output-tree paths are historical references only; they are not supported modeling inputs after implementation.

## Risks / Trade-offs

- [Risk] A direct CLI user may see a new hard failure instead of a run with implicit labels. -> Mitigation: fail before modeling and include the exact YAML command or missing contract field.
- [Risk] Moving the override file changes local runtime setup. -> Mitigation: verify the copied or regenerated file by hash and row-count parity against the existing reviewed artifact before updating config.
- [Risk] A new resolver could become another layer of indirection. -> Mitigation: the resolver must consume existing score/cohort/path helpers and replace local caller-specific logic.
- [Risk] Current tests may assert command-specific behavior. -> Mitigation: update tests to assert resolved-contract equivalence or fail-closed behavior.

## Migration Plan

1. Audit current quantification input resolution in YAML, direct CLI, and P3 code paths.
2. Choose the existing quantification owner or minimal new `src/eq/quantification/` helper needed for the resolved input contract.
3. Move or regenerate the reviewed override CSV under the stable runtime-derived input path and verify hash/row-count parity.
4. Update `configs/endotheliosis_quantification.yaml`.
5. Update direct CLI handlers to use the shared resolver or fail closed.
6. Thread label-contract provenance into score audits, P3 metadata, and verdict artifacts.
7. Add tests for YAML/direct equivalence, direct fail-closed behavior, stable override paths, and output-tree path rejection.

## Open Questions

- [audit_first_then_decide] Existing owner for the resolver: decide after reading `src/eq/quantification/pipeline.py`, `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, `src/eq/quantification/cohorts.py`, and `src/eq/quantification/labelstudio_scores.py`.
- [audit_first_then_decide] Whether to copy the current override artifact or regenerate it: decide from runtime hash, row count, and review provenance evidence.
