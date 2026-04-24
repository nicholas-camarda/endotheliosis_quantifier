## Context

`endotheliosis_quantifier` has accumulated several legitimate but overlapping surfaces: `eq` CLI commands, `eq run-config` YAML workflows, direct training module entrypoints, OpenSpec changes, public docs, internal docs, runtime roots, cloud publish roots, and model-promotion evidence. Recent cleanup work retired static patch inputs and unsupported surfaces, but a whole-repo alignment pass still needs to verify that current objectives, workflows, claims, tests, and artifact boundaries agree.

The review must be data-first and evidence-backed. It must inspect code, tests, configs, runtime artifacts, and published or publish-preview artifacts before deciding what to streamline. The implementation must not add fallback code, duplicate path systems, compatibility shims, or silent degradation paths for stale behavior.

## Goals / Non-Goals

**Goals:**

- Build a durable review dossier under `openspec/changes/p3-repo-wide-quality-review-and-streamlining/review/`.
- Run three named review families as first-class lanes: `documentation-wizard`, `workspace-governor`, and `research-partner`.
- Synthesize all lane findings into `review/action-register.tsv` with explicit action, evidence, owner surface, risk, validation, and status columns.
- Use the action register to make only evidence-backed cleanup and streamlining edits.
- Align public docs, internal docs, OpenSpec contracts, committed YAML configs, CLI help, test expectations, and runtime/cloud path contracts.
- Strengthen reproducibility gates around config execution, path helpers, model artifact provenance, quantification joins, segmentation-promotion evidence, and final validation.

**Non-Goals:**

- Do not promote any segmentation model scientifically as part of this repo-quality change.
- Do not run new long segmentation training as part of this change unless a review finding shows that a short smoke command is insufficient to validate a modified code path.
- Do not repair legacy FastAI pickle compatibility unless a separate compatibility change explicitly scopes that work.
- Do not migrate or reorganize original OneDrive/source data in place.
- Do not rewrite the repository around a new framework, runner, or path abstraction when existing `src/eq` helpers can support the current contract.

## Decisions

### Decision: Use a review dossier as the implementation gate

Implementation starts by creating `review/preflight-path-and-artifact-map.md`, `review/documentation-wizard-report.md`, `review/workspace-governor-report.md`, `review/research-partner-report.md`, `review/repo-wide-quality-synthesis.md`, and `review/action-register.tsv`. No code, docs, or workspace cleanup edits should begin until the synthesis and action register exist.

Rationale: A broad cleanup request can otherwise turn into subjective pruning. The dossier makes every retirement, rewrite, or streamlining edit traceable to a concrete finding.

Alternative considered: edit obvious clutter immediately, then audit. Rejected because this repo has active OpenSpec changes and runtime-path constraints where premature cleanup can break real workflows.

### Decision: Treat the three plugin review families as separate lanes with merged synthesis

`documentation-wizard` owns docs-vs-code drift, public/private doc hygiene, CLI/config interface extraction, and doc regression checks. `workspace-governor` owns canonical repo/runtime/cloud layout, generated artifact boundaries, publish-preview scope, and clutter classification. `research-partner` owns repo-wide scientific, statistical, implementation, robustness, and literature-support review.

Rationale: The requested work spans documentation truth, workspace path truth, and scientific/methodological validity. Keeping lanes separate avoids flattening all findings into a generic cleanup list.

Alternative considered: one general review document. Rejected because it would lose the source-of-truth ordering and evidence requirements from the named review systems.

### Decision: Use `review/action-register.tsv` as the only implementation backlog for this change

The action register columns are:

`action_id`, `lane`, `surface`, `finding`, `evidence`, `decision`, `implementation_target`, `validation`, `risk_level`, `status`.

Each code, docs, config, test, or workspace action must have one row before implementation starts. Rows may be `accepted`, `deferred`, or `rejected`, but accepted rows require an implementation target and validation command.

Rationale: A TSV backlog is easy to diff and keeps review conclusions distinct from implementation decisions.

Alternative considered: prose-only synthesis. Rejected because prose makes it hard to audit whether every accepted finding was handled and validated.

### Decision: Streamline around `eq` and YAML configs unless audit evidence proves another user-facing surface is needed

The default supported command surfaces remain `eq --help`, `eq run-config --config <yaml> [--dry-run]`, `eq quant-endo`, `eq prepare-quant-contract`, `eq cohort-manifest`, `eq capabilities`, and `eq mode`. Direct training module commands may remain implementation-level entrypoints, but they should not be the primary documentation surface if committed YAML configs cover the workflow.

Rationale: This matches the repo's current YAML-first direction while preserving direct module commands only when they are still tested and useful.

Alternative considered: remove all direct module entrypoints during this change. Rejected pending audit because training modules may still be valid internal execution surfaces for `eq run-config`.

### Decision: Workspace cleanup is classification-first and backup-first

Generated files, caches, `.history`, `.agent-os`, stale docs, retired artifacts, and runtime outputs must be classified before any move, deletion, local exclude, or doc rewrite. Mutating workspace moves require backup-first behavior and verification if they affect runtime or cloud-adjacent assets.

Rationale: The repo has split source/runtime/cloud boundaries and prior local-history artifacts. Classification avoids deleting or moving evidence that still explains the project state.

Alternative considered: broad deletion of cache/history clutter. Rejected because some history may be evidence for migration decisions, and shared ignore rules should not be expanded blindly.

### Decision: Final validation must cover both governance and behavior

Final validation includes:

- `openspec validate p3-repo-wide-quality-review-and-streamlining --strict`
- `python3 scripts/check_openspec_explicitness.py openspec/changes/p3-repo-wide-quality-review-and-streamlining`
- `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/mito_pretraining_config.yaml --dry-run`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/glomeruli_finetuning_config.yaml --dry-run`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/segmentation_fixedloader_full_retrain.yaml --dry-run`

Rationale: This change can affect process, docs, CLI, configs, and tests. A single unit-test run is not enough to prove the repo remains coherent.

Alternative considered: only run OpenSpec validation. Rejected because this change is specifically about repository quality and workflow integrity.

## Risks / Trade-offs

- Broad audit scope may produce too many findings to implement safely in one change. Mitigation: action-register rows can be `deferred` when they are evidence-backed but too large or independent for this change.
- Review reports can become stale if generated before code changes and not updated. Mitigation: final tasks require updating the synthesis after implementation and marking accepted rows done.
- CLI streamlining can break undocumented local workflows. Mitigation: retire only unsupported or untested surfaces, preserve tested internal module entrypoints when `eq run-config` depends on them, and document current behavior directly.
- Workspace cleanup can accidentally move source-of-truth assets. Mitigation: run `workspace-governor assess` first, keep moves backup-first, and do not reorganize original source data in place.
- Full `pytest -q` or eq-mac validation may hit environment-specific ABI or MPS issues. Mitigation: distinguish sandbox/environment failures from repo failures, and rerun Mac MPS-sensitive checks unsandboxed with `eq-mac` when Metal access matters.

## Migration Plan

1. Create the review dossier directory and preflight path/artifact map.
2. Run the documentation, workspace, and research review lanes.
3. Synthesize findings into `review/repo-wide-quality-synthesis.md` and `review/action-register.tsv`.
4. Implement accepted action-register rows in small grouped patches: docs, CLI/workflow, path/artifact boundaries, tests, and OpenSpec cleanup.
5. Re-run focused validations after each group.
6. Run final OpenSpec, explicitness, CLI, config dry-run, and test validation.
7. Update the synthesis and action register so every accepted row is marked done or explicitly deferred with rationale.

Rollback is ordinary git rollback for tracked repo edits. Runtime or cloud-adjacent moves, if any are accepted, must retain backup manifests and verification evidence before source cleanup.

## Explicit Decisions

- Review outputs live under `openspec/changes/p3-repo-wide-quality-review-and-streamlining/review/`.
- `review/action-register.tsv` is the implementation backlog for this change.
- The three required lane reports are `review/documentation-wizard-report.md`, `review/workspace-governor-report.md`, and `review/research-partner-report.md`.
- The synthesis report is `review/repo-wide-quality-synthesis.md`.
- The preflight map is `review/preflight-path-and-artifact-map.md`.
- The supported YAML configs to validate are `configs/mito_pretraining_config.yaml`, `configs/glomeruli_finetuning_config.yaml`, and `configs/segmentation_fixedloader_full_retrain.yaml`.
- The canonical Mac interpreter for project validation is `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python`.

## Open Questions

- [audit_first_then_decide] Which direct module commands under `src/eq/training/` remain user-facing versus internal implementation entrypoints? Decide after comparing `eq --help`, `eq run-config` dry-runs, docs examples, tests, and documentation-wizard interface extraction.
- [audit_first_then_decide] Which clutter classes should be tracked, locally excluded, moved to runtime `_retired/`, moved to docs archive, or left untouched? Decide from `workspace-governor assess`, git status, path ownership, and evidence value.
- [audit_first_then_decide] Which methodological concerns require immediate tests or docs in this change versus separate model/data changes? Decide from the research-partner synthesis and whether the concern affects current supported commands or claims.
