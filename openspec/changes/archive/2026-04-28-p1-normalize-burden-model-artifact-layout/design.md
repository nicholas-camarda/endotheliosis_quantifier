## Context

The original burden-index evaluator writes grouped artifacts directly under `burden_model/`: `primary_model/`, `validation/`, `calibration/`, `summaries/`, `evidence/`, `candidates/`, `diagnostics/`, and `feature_sets/`. Newer estimator modules write contained subtrees under `burden_model/<estimator_slug>/` with `INDEX.md`, `summary/`, `predictions/`, `diagnostics/`, `evidence/`, and `internal/`.

The mixed convention is a readability problem rather than a statistical change. The cleanup should preserve artifact meaning while making the tree navigable.

## Explicit Decisions

- `summary/` singular means first-read, human-facing verdict/manifest material for one estimator subtree.
- `summaries/` plural means aggregate tables such as cohort metrics or interval summaries.
- The primary burden-index evaluator becomes a contained subtree at `burden_model/primary_burden_index/`.
- The primary burden-index serialized model moves from `primary_model/` semantics to `primary_burden_index/model/` semantics.
- The root `burden_model/INDEX.md` lists every model/estimator subtree and the correct first file to open.
- Tests must fail if newly generated primary burden-index artifacts reappear in old top-level folders.
- No long-lived old-path aliases are written.
- docs-impact: docs and generated quantification review links are updated to the new paths.
- logging-contract: no execution-surface behavior changes.

## Open Questions

- [defer_ok] Historical runtime trees may retain old folders until explicitly cleaned or regenerated.
