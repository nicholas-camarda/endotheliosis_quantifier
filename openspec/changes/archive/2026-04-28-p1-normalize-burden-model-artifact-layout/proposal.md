## Why

`burden_model/` currently mixes the original burden-index model's top-level grouped folders with newer self-contained estimator subtrees. The result is confusing: `burden_model/summaries/` and `burden_model/<estimator>/summary/` are both valid but mean different things, and the primary burden-index artifacts lack a clear first-read entrypoint.

## What Changes

- Move the primary burden-index output contract into `burden_model/primary_burden_index/` with typed subfolders.
- Add `INDEX.md` files for `burden_model/` and `burden_model/primary_burden_index/` so operators have an obvious first-read path.
- Keep estimator-contained subtrees such as `source_aware_estimator/` and `severe_aware_ordinal_estimator/` as the future pattern.
- Add a learned-ROI `INDEX.md` and `summary/` first-read bundle while keeping typed learned-ROI outputs in their existing folders.
- Update docs, review links, and tests so the new layout is explicit and old flat/top-level primary folders are no longer the current contract.

## Explicit Decisions

- Change name: `p1-normalize-burden-model-artifact-layout`.
- Primary burden-index root: `burden_model/primary_burden_index/`.
- Primary burden-index model artifact dir: `burden_model/primary_burden_index/model/`.
- Primary burden-index summary tables: `burden_model/primary_burden_index/summaries/`.
- Burden root first-read file: `burden_model/INDEX.md`.
- Primary burden-index first-read file: `burden_model/primary_burden_index/INDEX.md`.
- Learned ROI first-read files: `burden_model/learned_roi/INDEX.md` and `burden_model/learned_roi/summary/`.
- Existing estimator subtrees keep `summary/` singular for first-read verdict bundles.
- No compatibility aliases or duplicate writers are added for old top-level primary folders.
- docs-impact: update `docs/OUTPUT_STRUCTURE.md`, `README.md`, and onboarding/lab docs where they describe burden-model artifact paths.
- logging-contract: no execution surface changes; existing logging tests remain sufficient.

## Open Questions

- [defer_ok] Whether to migrate or delete historical runtime folders already written under the old layout is an operator cleanup decision; this change only changes newly generated artifacts and docs.
