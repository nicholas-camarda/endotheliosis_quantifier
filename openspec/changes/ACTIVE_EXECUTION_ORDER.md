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
| `src/eq/quantification/modeling_contracts.py` | `oracle-harden-audit-contracts`, `label-free-roi-embedding-atlas` | Oracle hardening owns shared estimability, hard-blocker, finite-matrix, and serialization primitives. Atlas reuses them and may add only atlas-specific wrappers when reuse is insufficient. |
| `src/eq/quantification/endotheliosis_grade_model.py` | `oracle-canonical-quantification-input-contract`, `oracle-harden-audit-contracts` | Canonical input contract threads resolved label provenance. Oracle hardening updates estimability and serialization after that provenance exists. |

## Autonomous Execution Rules

- Execute one active OpenSpec change at a time in the order above.
- Do not run separate implementation agents on different active changes unless their write sets are proven disjoint after `git status --short` and `rg` inspection.
- Before starting each change, run `openspec validate <change> --strict` and `python3 scripts/check_openspec_explicitness.py <change>`.
- After each change, run its listed focused tests, `openspec validate <change> --strict`, the explicitness checker, and any changed global checks required by its tasks.
- Do not mark a downstream task complete by reimplementing an upstream contract locally. If a downstream change needs a behavior already owned by an earlier change, reuse that owner or stop and amend the upstream change.
- Preserve current-state documentation language. Historical material belongs only under `docs/archive/` and `docs/HISTORICAL_NOTES.md`.
- Treat missing required inputs, ambiguous labels, unsupported artifacts, invalid ROI geometry, and unestimable model folds as hard blockers or explicit insufficiency verdicts, not as warnings or fallback paths.
