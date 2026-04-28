## 1. Output Contract

- [x] 1.1 Move primary burden-index generated artifacts under `burden_model/primary_burden_index/`.
- [x] 1.2 Add `burden_model/INDEX.md` and `burden_model/primary_burden_index/INDEX.md` generation.
- [x] 1.3 Add learned-ROI `INDEX.md` and `summary/` first-read artifact generation.
- [x] 1.4 Update generated quantification review and summary links to the new primary burden-index paths.

## 2. Docs And Specs

- [x] 2.1 Update `docs/OUTPUT_STRUCTURE.md` with the normalized tree.
- [x] 2.2 Update README/onboarding/lab docs references to primary burden-index paths.
- [x] 2.3 Add a spec delta describing the burden artifact layout contract.

## 3. Tests

- [x] 3.1 Update burden-model unit tests for `primary_burden_index/` paths.
- [x] 3.2 Add tests that reject old top-level primary folders in fresh output.
- [x] 3.3 Update quantification review tests for new links.

## 4. Validation

- [x] 4.1 Run focused burden, learned ROI, and quantification pipeline tests.
- [x] 4.2 Run `python -m ruff check .`.
- [x] 4.3 Run `python -m pytest -q`.
- [x] 4.4 Run `openspec validate p1-normalize-burden-model-artifact-layout --strict`.
- [x] 4.5 Run `openspec validate --specs --strict`.
