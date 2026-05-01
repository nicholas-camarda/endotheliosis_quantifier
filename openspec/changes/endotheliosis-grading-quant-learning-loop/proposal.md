## Why

Operators need an explicit **quantification-side contract** for how **human endotheliosis grading** (especially per-glomerulus Label Studio workflows) feeds **repeatable refresh cycles** of scored cohorts and downstream burden modeling. Today the pipeline already *can* refit when labels change, but the **closed-loop cadence** (grade → ingest → cohort → quant → review → grade again) is not stated as a first-class obligation, which makes collaboration and governance ambiguous.

## What Changes

- Add normative requirements for an **authoritative grading → cohort → quantification** iterative cadence with **provenance sufficient to detect stale scores**.
- Clarify boundaries: this loop is **offline / workflow-level** (exports and reruns). It **does not** require embedding burden-model predictions into **Stage 1 primary Label Studio grading UX** (still governed separately by **`label-studio-medsam-hybrid-grading`** non-goals unless a future change revises that contract).
- Extend **`scored-only-quantification-cohort`** with lineage and cadence scenarios; lightly extend **`endotheliosis-grade-model`** so P3 outputs acknowledge grading-input generations where feasible.

## Capabilities

### New Capabilities

- _(none — deltas only to keep governance localized to quant cohort + grade model specs)_

### Modified Capabilities

- `scored-only-quantification-cohort`: ADD requirements for iterative authoritative grading refresh cycles and grading-export lineage consumed by quantification.
- `endotheliosis-grade-model`: ADD requirement tying P3 / burden-model reruns to documented grading-input provenance fields when scores originate from Label Studio or hybrid exports.

## Impact

- Documentation (`README.md`, cohort/quant operator docs) describing the cadence in plain language.
- Potential additions to cohort-building outputs (audit JSON fields) — implementation discovers exact insertion points during `/opsx:apply`.
- Optional future work (**not required here**): active-learning queues, disagreement dashboards — tag `[defer_ok]` in design.

## Explicit Decisions

- **Learning loop v1** means **workflow iteration with authoritative labels**, not automatic online learning inside Label Studio’s grading UI.
- **Stale-score prevention** is provenance-first: reruns MUST be able to determine whether grading inputs changed vs a prior quant tree.

## Open Questions

- `[audit_first_then_decide]` Which existing cohort audit artifacts (`manifest.csv` sidecars, score recovery summaries, export digests) already carry enough lineage versus needing new columns — audit current `src/eq/quantification` writers during implementation.
- `[defer_ok]` Whether active prioritization (“what to grade next”) becomes a separate capability once uncertainty artifacts exist.
