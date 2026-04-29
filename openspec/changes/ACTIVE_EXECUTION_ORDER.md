# Active OpenSpec Execution Order

## Purpose

This file is the current execution plan for active OpenSpec changes. It exists so implementation agents can apply the changes in a dependency-safe order without running parallel edits across shared source, config, docs, or artifact-contract surfaces.

## Active Changes

Archived changes under `openspec/changes/archive/` are historical records. They are not execution candidates unless a future OpenSpec change explicitly reopens their scope.

| Order | Change | Status | Primary Surfaces | Collision Rule |
| --- | --- | --- | --- | --- |
| 0 | worktree-preflight | required before any change | `git status --short`, archived `p3-functional-severe-ordinal-quantification`, uncommitted source/tests/config edits | Do not start implementation until the current dirty worktree is intentionally preserved, committed, or assigned to the active change owner. |
| 1 | `oracle-current-docs-quarantine` | active | `README.md`, `docs/README.md`, `docs/INTEGRATION_GUIDE.md`, `docs/PIPELINE_INTEGRATION_PLAN.md`, `docs/SEGMENTATION_ENGINEERING_GUIDE.md`, `docs/ONBOARDING_GUIDE.md`, `docs/TECHNICAL_LAB_NOTEBOOK.md`, `docs/HISTORICAL_NOTES.md`, `docs/archive/`, docs checks | Run alone. It owns active-doc historical fallback quarantine before broader FastAI documentation cleanup. |
| 2 | `p1-align-fastai-practices-and-archive-history` | active | `src/eq/data_management/datablock_loader.py`, `src/eq/training/train_glomeruli.py`, `src/eq/training/train_mitochondria.py`, `src/eq/training/transfer_learning.py`, `src/eq/utils/run_io.py`, FastAI docs quarantine checks | Run after docs quarantine. Reconcile already-quarantined docs as satisfied or no-op tasks instead of rewriting them again. |
| 3 | `oracle-canonical-quantification-input-contract` | active | `configs/endotheliosis_quantification.yaml`, `src/eq/__main__.py`, `src/eq/run_config.py`, `src/eq/quantification/cohorts.py`, `src/eq/quantification/labelstudio_scores.py`, `src/eq/quantification/pipeline.py`, `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, `src/eq/quantification/endotheliosis_grade_model.py` | Run before fail-closed Oracle hardening so all downstream quantification work consumes one label/input contract. |
| 4 | `oracle-harden-audit-contracts` | active | Label Studio score loading, manifest training pairing, preprocessing, ROI extraction, candidate comparison, exact artifact handoffs, shared quantification modeling contracts, CLI startup/MPS scope | Run after canonical input contract and FastAI hygiene. It may touch the same modules, so it must integrate their completed central contracts rather than duplicate them. |
| 5 | `label-free-roi-embedding-atlas` | active | `configs/label_free_roi_embedding_atlas.yaml`, `src/eq/run_config.py`, `src/eq/__main__.py`, `src/eq/quantification/embedding_atlas.py`, `src/eq/quantification/embeddings.py`, `src/eq/quantification/learned_roi.py`, `src/eq/quantification/modeling_contracts.py`, docs/tests | Run last. It is additive, but depends on stable quantification inputs, fail-closed ROI/embedding artifacts, and shared modeling/finite-matrix contracts. |

## Dependency Graph

```text
worktree-preflight
        |
        v
oracle-current-docs-quarantine
        |
        v
p1-align-fastai-practices-and-archive-history
        |
        v
oracle-canonical-quantification-input-contract
        |
        v
oracle-harden-audit-contracts
        |
        v
label-free-roi-embedding-atlas
```

## Collision Matrix

| Shared Surface | Changes Touching It | Required Handling |
| --- | --- | --- |
| Active docs and archive docs | `oracle-current-docs-quarantine`, `p1-align-fastai-practices-and-archive-history`, `label-free-roi-embedding-atlas` | Docs quarantine runs first. Later changes add only current-state docs for their own behavior and must not reintroduce historical operational guidance. |
| `src/eq/__main__.py` | `oracle-canonical-quantification-input-contract`, `oracle-harden-audit-contracts`, `label-free-roi-embedding-atlas` | Canonical input contract owns quantification direct-CLI equivalence. Oracle hardening owns startup fail-closed behavior. Atlas may add a direct alias only after those contracts are stable; otherwise it stays `eq run-config` only. |
| `src/eq/run_config.py` | `oracle-canonical-quantification-input-contract`, `oracle-harden-audit-contracts`, `label-free-roi-embedding-atlas` | Canonical input contract updates `endotheliosis_quantification`; Oracle hardening removes latest-glob handoffs; atlas adds one new workflow dispatcher after both are complete. |
| `configs/endotheliosis_quantification.yaml` | `oracle-canonical-quantification-input-contract`, `oracle-harden-audit-contracts`, `label-free-roi-embedding-atlas` | Canonical input contract owns stable reviewed-label override paths. Oracle hardening owns exact upstream artifact references. Atlas must not change supervised workflow behavior. |
| `src/eq/data_management/datablock_loader.py` and `src/eq/utils/run_io.py` | `p1-align-fastai-practices-and-archive-history`, `oracle-harden-audit-contracts` | FastAI hygiene removes broad fallback and warning-only required artifacts first. Oracle hardening extends the same fail-closed contract for manifest training and provenance. |
| `src/eq/data_management/standard_getters.py`, `src/eq/data_management/canonical_naming.py`, and `src/eq/data_management/canonical_contract.py` | `oracle-harden-audit-contracts` | Oracle hardening owns exact mask pairing and canonical-name full-match parsing. Downstream changes must consume the hardened data contract rather than adding local filename parsing. |
| `src/eq/inference/prediction_core.py`, `src/eq/inference/gpu_inference.py`, and `src/eq/quantification/embeddings.py` | `oracle-harden-audit-contracts`, `label-free-roi-embedding-atlas` | Oracle hardening owns shared ImageNet-normalized inference preprocessing and threshold provenance. Atlas must reject stale or provenance-incomplete embedding inputs rather than re-normalizing or accepting older artifacts. |
| `src/eq/training/compare_glomeruli_candidates.py`, `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`, `src/eq/training/train_glomeruli.py`, and `configs/glomeruli_candidate_comparison.yaml` | `p1-align-fastai-practices-and-archive-history`, `oracle-harden-audit-contracts` | FastAI hygiene owns explicit imports, trusted loading, and required artifact handling. Oracle hardening owns requested-training failures, no auto-base discovery, exact handoffs, and resize-fallback rejection. |
| `src/eq/__init__.py` and CLI startup behavior | `oracle-harden-audit-contracts` | Oracle hardening owns visible startup failures and Mac MPS fallback scope. Later CLI additions must preserve that behavior. |
| `src/eq/quantification/modeling_contracts.py` | `oracle-harden-audit-contracts`, `label-free-roi-embedding-atlas` | Oracle hardening owns shared estimability, hard-blocker, finite-matrix, and serialization primitives. Atlas reuses them and may add only atlas-specific wrappers when reuse is insufficient. |
| `src/eq/quantification/endotheliosis_grade_model.py` | `oracle-canonical-quantification-input-contract`, `oracle-harden-audit-contracts` | Canonical input contract threads resolved label provenance. Oracle hardening updates estimability and serialization after that provenance exists. |
| `src/eq/quantification/cohorts.py` and `src/eq/quantification/labelstudio_scores.py` | `oracle-canonical-quantification-input-contract`, `oracle-harden-audit-contracts` | Canonical input owns resolved override/input provenance, grouping identity, and target-definition hashes. Oracle hardening owns current Label Studio extraction and historical backfill rejection, and must not change the canonical target-definition version without updating provenance and tests. |
| `src/eq/utils/paths.py`, `src/eq/utils/execution_logging.py`, and quantification review renderers | `label-free-roi-embedding-atlas` | Atlas must reuse existing path/logging/review surfaces. New atlas-specific rendering is allowed only after the reuse audit records why existing review helpers cannot own the output. |
| `openspec/specs/*` capability deltas | all active changes except independent new atlas capability | Archive/sync can change the base spec surface for remaining active changes. After each archive commit, revalidate and re-read all remaining active changes against the updated main specs before implementing the next change. |

## Autonomous Execution Rules

- Execute one active OpenSpec change at a time in the order above.
- Once a change starts, carry it through implementation, validation, postflight, implementation commit, archive/sync, archive validation, and archive commit without stopping for intermediate permission unless a true blocker appears or the user explicitly redirects.
- Do not run separate implementation agents on different active changes unless their write sets are proven disjoint after `git status --short` and `rg` inspection.
- Before starting each change, run `openspec validate <change> --strict` and `python3 scripts/check_openspec_explicitness.py <change>`.
- After each change, run its listed focused tests, `openspec validate <change> --strict`, the explicitness checker, and any changed global checks required by its tasks.
- Do not mark a downstream task complete by reimplementing an upstream contract locally. If a downstream change needs a behavior already owned by an earlier change, reuse that owner or stop and amend the upstream change.
- Preserve current-state documentation language. Historical material belongs only under `docs/archive/` and `docs/HISTORICAL_NOTES.md`.
- Treat missing required inputs, ambiguous labels, unsupported artifacts, invalid ROI geometry, and unestimable model folds as hard blockers or explicit insufficiency verdicts, not as warnings or fallback paths.

## Per-Change Git And Archive Lifecycle

Each active OpenSpec change should move through the same lifecycle so the repo keeps an auditable linear history and each change can be reverted or edited independently.

1. Preflight the dirty worktree before starting the change.
   - Run `git status --short`.
   - Confirm unrelated edits are committed, stashed, ignored, or explicitly assigned to this change.
   - Do not begin implementation with ambiguous unowned edits in files the change will touch.
2. Validate the active change before implementation.
   - Run `openspec validate <change> --strict`.
   - Run `python3 scripts/check_openspec_explicitness.py <change>`.
3. Implement only that change.
   - Keep task status and any implementation notes inside the active change directory current.
   - Run focused tests as tasks are completed.
   - Run the final validation commands listed in that change's `tasks.md`.
4. Postflight the implemented spec before committing.
   - Re-read the implemented change's `proposal.md`, `design.md`, `tasks.md`, and spec deltas against the actual code, docs, configs, tests, and artifact-contract edits.
   - Verify every completed task has direct evidence in the worktree or a recorded reason it is no longer applicable.
   - Verify no undocumented behavior, fallback path, duplicate owner, path-system drift, docs-currentness violation, or generated-artifact-in-Git regression was introduced.
   - Run `git diff --check`, inspect `git diff --stat`, and review the final diff for unrelated edits.
   - Record postflight findings, validation commands, residual risks, and any intentionally deferred items in the active change directory before the implementation commit.
5. Commit the implementation as its own commit.
   - Include code, docs, config, tests, runtime-contract references, and the active change task updates.
   - Use a message shaped like `implement <change>`.
6. Archive/sync the completed OpenSpec change.
   - Run `openspec archive <change> --yes` for the completed change unless the change is explicitly documented as `--skip-specs` infrastructure-only.
   - Run `openspec validate --specs --strict` after archiving.
   - Run any repo explicitness or docs checks affected by the archive/sync.
7. Commit the archive/sync as its own commit.
   - Include only OpenSpec archive/sync and directly required bookkeeping changes.
   - Use a message shaped like `archive <change>`.
8. Refresh remaining active changes before moving on.
   - Run `openspec validate <remaining-change> --strict` for every still-active change.
   - Run `python3 scripts/check_openspec_explicitness.py <remaining-change>` for every still-active change.
   - Re-read the next change's `proposal.md`, `design.md`, `tasks.md`, and spec deltas against the newly archived specs and changed source surfaces.
   - Update this execution-order file if the archive changed write surfaces, order, no-op tasks, or downstream assumptions.
9. Move to the next change only after both commits exist and the remaining-change refresh is clean, unless the user explicitly chooses a different checkpoint.

Use one integration branch with this linear commit stack unless the user explicitly asks for parallel PR branches or speculative alternatives. These active changes share enough source, config, and documentation surfaces that separate long-lived branches are likely to create avoidable merge conflicts.
