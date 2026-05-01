## Context

Human glomerulus-instance grading feeds scored cohort manifests and then **`eq run-config --config configs/endotheliosis_quantification.yaml`** (and related paths). **`endotheliosis-grade-model`** already describes grouped development validation and refit mechanics once labeled rows exist. **`label-studio-glomerulus-grading`** defines authoritative export shape. **`label-studio-medsam-hybrid-grading`** intentionally excludes Stage 2 predictors from Stage 1 UX.

Operators still lack a **named quant contract** that says: **when graders produce new authoritative rows, downstream quant artifacts MUST be regenerated under controlled lineage**, not reused tacitly.

## Goals / Non-Goals

**Goals:**

- Specify a **minimum closed loop**: authoritative grading export → cohort admission/rebuild rules → quantification rerun → archived outputs keyed to grading-input identity.
- Require **staleness-detectable lineage** so two quant trees can be compared against **different grading snapshots**.
- Keep scope **offline** (batch workflows), compatible with hybrid labeling exports.

**Non-Goals:**

- Building active-learning prioritization UI (**defer_ok** future change).
- Injecting burden-model predictions into Stage 1 LS grading (**explicitly excluded** unless a separate grading-contract change reverses hybrid non-goals).
- **Automated decomposition** of legacy **image-level aggregate scores** into per-glomerulus grades or masks without new human annotation or a separately scoped vision model (ill-posed; not part of this change).

## Decisions

### Decision: Cadence is workflow-documented plus lineage-enforced

**Choice:** V1 binds operators through **spec + docs + audit fields**, not through new daemons.

**Alternative:** Continuous sync service → rejected as out of scope.

### Decision: Source of truth remains authoritative human grades

**Choice:** Quant refits consume **latest authoritative** grades per existing Label Studio loader rules; loop emphasizes **repeatability**, not automated label rewriting.

### Decision: Separate future capability for “what to label next”

**Choice:** Uncertainty-driven grading queues stay **`[defer_ok]`** pending predictability artifacts.

## Risks / Trade-offs

- **[Risk]** Extra lineage columns disturb legacy manifests → **Mitigation**: additive optional fields + audits; fail-closed only when ambiguity would silently reuse wrong scores (align with existing reconciliation posture).

## Migration Plan

1. Land spec deltas + docs (may be documentation-only initially).
2. Implement lineage writers where audits prove gaps.
3. Validate with fixture cohort containing two synthetic grading snapshots.

## Explicit Decisions

- **Learning loop v1** is **offline iteration**, not LS-embedded model hints.
- **Forward-looking scoring unit** for new labeling and hybrid exports is **per glomerulus** (authoritative instance rows), not a single coerced score per multi-glom image.
- **Dual-era flexibility:** Historical cohorts that were admitted under **image-level** scoring rules remain valid for reproducibility; migrating those images to **per-glom** masks and grades is a **re-annotation / enrichment** workflow (for example hybrid Label Studio passes), not an automatic numeric split of a legacy aggregate score.

## Open Questions

- [audit_first_then_decide] Minimal sufficient lineage tuple (export digest vs `(project_id, export_timestamp)` vs manifest row hashes) — audit `src/eq/quantification` lineage writers during implementation.
