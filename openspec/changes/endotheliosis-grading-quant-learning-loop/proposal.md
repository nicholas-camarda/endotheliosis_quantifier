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
- **Per-glom forward path:** New authoritative grading is expected to produce **per-glomerulus** rows where **`label-studio-glomerulus-grading`** / hybrid exports apply; legacy **image-level** scored inputs remain a distinct cohort era.
- **Migration, not decomposition:** Moving old image-level-scored material to per-glom masks and grades requires **explicit labeling or enrichment** (for example a hybrid Label Studio project); the repository does not treat inferred per-glom scores from a single legacy image score as scientifically authoritative.

## Open Questions

- [audit_first_then_decide] Which existing cohort audit artifacts (`manifest.csv` sidecars, score recovery summaries, export digests) already carry enough lineage versus needing new columns — audit current `src/eq/quantification` writers during implementation.
- [defer_ok] Whether active prioritization (“what to grade next”) becomes a separate capability once uncertainty artifacts exist.

## logging-contract

V1 is **spec-first**; durable execution logging for `eq run-config` and quantification workflows remains the **existing** `eq` CLI and workflow log streams. Any implementation that adds lineage sidecars or audit JSON under output roots MUST follow the repo-wide execution logging contract already governing those runners—no new Git-tracked log root and no silent log path forks unless a follow-up change extends `repo-wide-execution-logging`.

## docs-impact

Update quantification / cohort onboarding documentation (`README.md` and operator-facing docs referenced in **Impact**) to describe the authoritative grading → cohort → quantification loop, dual-era scoring units (image-level legacy vs per-glom), and stale-score prevention via lineage. If an apply pass is documentation-only initially, confine edits to those docs and spec deltas.
