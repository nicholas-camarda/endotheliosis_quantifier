## Context

The current FastAI segmentation path is operational under `eq-mac`: `eq --help` imports, the full test suite passes, OpenSpec validation passes for the active training-calibration change, and the local interpreter reports FastAI `2.8.7` with MPS available. The remaining problem is quality and contract hygiene:

- `ruff check .` currently fails with FastAI wildcard-import undefined-name findings in active training modules and broader style/lint findings across the repo.
- `src/eq/data_management/datablock_loader.py` still contains fallback mask pairing behavior in the active training path.
- `src/eq/utils/run_io.py` catches and warns for some failures that are required evidence artifacts for supported segmentation model exports.
- Historical FastAI integration/planning docs remain in `docs/` with commands and fallback concepts that can be confused with current workflow guidance.
- Recent cleanup work has added several specialized checks and docs surfaces; this change should reuse existing helpers and consolidate overlapping surfaces instead of adding new parallel scripts or duplicate documentation paths.

The change therefore has two coordinated lanes:

```text
                 p1-align-fastai-practices-and-archive-history
                                      |
              +-----------------------+-----------------------+
              |                                               |
   Active FastAI training hygiene              Historical docs + repo streamlining
              |                                               |
 explicit imports, fail-closed pairing,        docs/HISTORICAL_NOTES.md + docs/archive/
 required artifact evidence, trusted           current README/guides describe only the
 current-namespace learner loading             maintained workflow
```

## Goals / Non-Goals

**Goals:**

- Make active FastAI segmentation training code lint-visible and `ruff check .` clean.
- Remove fallback image/mask pairing from the supported DataBlock training path.
- Make required training/export evidence artifacts fail hard when missing or unwritable.
- Keep trusted current-namespace `load_learner` usage, while documenting that FastAI learner pickle loading is not a legacy-artifact compatibility mechanism.
- Move historical FastAI integration and planning content into archive/reference docs that remain accessible.
- Keep current docs current-state only and make historical material discoverable without being operational guidance.
- Apply the current-state/no-shim rule across active code and docs, not only the README.
- Keep the repo clean and streamlined by reusing existing tools, tests, helpers, and docs surfaces before adding new ones.
- Add focused tests or validation checks for fail-closed pairing, artifact completeness, lint cleanliness, and historical-doc quarantine.

**Non-Goals:**

- Do not retrain or promote any segmentation model.
- Do not add compatibility shims for legacy FastAI pickle artifacts.
- Do not change the current YAML workflow IDs or runtime output roots.
- Do not move raw data, derived data, model artifacts, logs, or output artifacts into Git.
- Do not rewrite scientific interpretation or quantification methods beyond removing historical FastAI guidance from current docs and active code paths.
- Do not create new standalone scripts, validators, docs indexes, or helper modules when an existing repo surface can be extended cleanly.

## Explicit Decisions

- `src/eq/training/train_glomeruli.py`, `src/eq/training/train_mitochondria.py`, and `src/eq/training/transfer_learning.py` will replace `from fastai.vision.all import *` and `from fastai.callback.all import *` with exact imports for the symbols used.
- `src/eq/data_management/datablock_loader.py::get_items_full_images` will rely on `eq.data_management.standard_getters.get_y_full` for active direct-root pairing and will remove the secondary fallback candidate search.
- `src/eq/utils/run_io.py` will split artifact writes into required and optional categories:
  - required: split manifest, training history, exported learner, run metadata, package/version provenance, data-root provenance, training-mode provenance
  - optional: plotting-only artifacts such as loss curves, LR schedule plots, and validation prediction PNGs generated outside promotion gates
- `src/eq/data_management/model_loading.py` will keep direct `load_learner` calls only for trusted current-namespace artifacts and will not add legacy namespace imports.
- Historical docs will be reorganized as:
  - `docs/HISTORICAL_NOTES.md`: the front-door index for historical material
  - `docs/archive/fastai_legacy_integration.md`: retained content from `docs/INTEGRATION_GUIDE.md`
  - `docs/archive/fastai_pipeline_integration_plan.md`: retained content from `docs/PIPELINE_INTEGRATION_PLAN.md`
  - `docs/archive/fastai_historical_implementation_analysis.md`: retained content from `docs/HISTORICAL_IMPLEMENTATION_ANALYSIS.md`
- Current active docs will remain or become current-state guidance; historical content is extracted before rewriting. `docs/INTEGRATION_GUIDE.md` remains an active current implementation guide if retained.
- The current docs that must not contain active historical/fallback instructions include:
  - `README.md`
  - `docs/README.md`
  - `docs/INTEGRATION_GUIDE.md`
  - `docs/ONBOARDING_GUIDE.md`
  - `docs/SEGMENTATION_ENGINEERING_GUIDE.md`
  - `docs/TECHNICAL_LAB_NOTEBOOK.md`
- The active workflow IDs remain unchanged:
  - `segmentation_mitochondria_pretraining`
  - `segmentation_glomeruli_transfer`
  - `segmentation_fixedloader_full_retrain`
  - committed config entrypoints `configs/mito_pretraining_config.yaml`, `configs/glomeruli_finetuning_config.yaml`, and `configs/glomeruli_candidate_comparison.yaml`
- Reuse-first implementation targets:
  - extend existing pytest files under `tests/` rather than creating multiple one-off validators when pytest coverage can express the rule
  - reuse `scripts/check_openspec_explicitness.py` for OpenSpec proposal/design explicitness rather than adding another OpenSpec checker
  - reuse `src/eq/utils/paths.py` for path policy rather than adding local path constants
  - reuse `src/eq/run_config.py` and `src/eq/__main__.py` for workflow/CLI validation rather than adding competing entrypoints
  - reuse `docs/HISTORICAL_NOTES.md` and `docs/README.md` as archive discovery surfaces rather than adding multiple new doc indexes

## Decisions

### 1. Treat lint as a runtime-quality gate, not cosmetic cleanup

`ruff check .` is already the repo command for lint and format validation. FastAI wildcard imports currently hide names from static analysis and create avoidable ambiguity around which FastAI APIs are actually used. The implementation should start with active training modules, then address remaining repo-wide lint findings required to make the final command green.

Alternative considered: ignore lint findings because pytest passes. Rejected because the request is explicitly about best practices and the repo already declares `ruff check .` as a command gate.

### 2. Remove active fallback pairing rather than documenting it

The current supported training contract is a full-image root with `images/` and `masks/`, or a manifest-backed cohort registry. If an image cannot be resolved through the canonical getter or explicit manifest row, the trainer should fail before model construction. Curation or migration can handle alternate historical naming, but the active trainer should not infer secondary matches.

Alternative considered: keep fallback search but log it. Rejected because this repo's operating rule is to avoid fallback/patchwork logic and because silent pairing changes can invalidate training masks.

### 3. Classify artifacts by evidence role

Not every post-training output has the same role. Split manifests, training history, metadata, and exported model files are required for supported runtime status. Plot PNGs are useful review aids but can be regenerated or produced by validation workflows. The implementation should make this distinction explicit in code and tests.

Alternative considered: fail hard on every plot. Rejected because a plotting backend issue should not modify trained weights or data splits, and promotion-facing visual evidence is already handled by validation/comparison gates.

### 4. Archive historical docs instead of deleting them

Historical FastAI material is useful for reconstructing why legacy artifacts are unsupported, but it should not sit beside current workflow docs as if it were a runnable path. The implementation should move retained historical content under `docs/archive/`, make `docs/HISTORICAL_NOTES.md` the index, and keep active documents such as `docs/INTEGRATION_GUIDE.md` current if they remain in the docs tree.

Alternative considered: delete historical docs. Rejected because the user asked for the historical material to remain documented.

### 5. Keep current docs and active code present-state only

Current public/current workflow docs should state what the repo supports now: `eq-mac`, YAML-first workflows, current-namespace artifacts, full-image dynamic patching, trusted FastAI learner loading, and promotion gates. They should not explain migrations, old failures, or fallback rescue paths in active workflow sections. Active code should follow the same rule: no historical shims, workaround branches, silent fallbacks, or compatibility rescue paths for unsupported legacy artifacts.

Alternative considered: keep historical notes inline with stronger warnings. Rejected because inline historical commands still compete with the current workflow.

### 6. Reuse existing repo surfaces before adding new scripts

This change should strengthen the repo without growing parallel tooling. Before adding any new script, checker, helper module, or docs index, the implementation should inspect existing repo surfaces and either extend one of them or document why reuse is not appropriate. A new script is acceptable only when the job is materially distinct from existing pytest coverage, `scripts/check_openspec_explicitness.py`, CLI dry-runs, or docs/link checks that can be expressed in existing tests.

Alternative considered: add separate purpose-built scripts for documentation quarantine, code-hygiene scanning, and link checks. Rejected because that would create the exact duplicate-tooling problem this cleanup is meant to remove.

## Risks / Trade-offs

- [Risk] Removing fallback pairing may expose raw roots that previously appeared to work.
  - Mitigation: fail with image count and examples; direct users to curation/migration rather than inferring masks inside training.

- [Risk] Making required artifact writes fail hard may break long-running training at the end if a path is unwritable.
  - Mitigation: preflight required output directories and keep runtime roots under `~/ProjectsRuntime/endotheliosis_quantifier`; failing is correct if provenance cannot be written.

- [Risk] `ruff check .` may surface broad existing style issues beyond the FastAI modules.
  - Mitigation: keep the implementation mechanical and scoped to lint-cleaning required by the repo command, without unrelated refactors.

- [Risk] Moving historical docs can break links.
  - Mitigation: update `docs/README.md`, `docs/HISTORICAL_NOTES.md`, and any current-doc references; run a targeted link/path grep for moved filenames and historical helper names.

- [Risk] Historical archive docs may still contain current-looking commands.
  - Mitigation: prepend explicit historical/reference-only headers and ensure quarantine tests allow those terms only under `docs/archive/` or `docs/HISTORICAL_NOTES.md`.

- [Risk] The cleanup grows new one-off scripts for each check.
  - Mitigation: perform a reuse-first inventory, extend existing pytest or helper surfaces where possible, and require explicit justification for any new script.

## Migration Plan

1. Inventory existing helpers, tests, scripts, CLI surfaces, and docs indexes that can carry the required checks.
2. Fix active FastAI imports and lint failures in training modules.
3. Remove fallback pairing from `get_items_full_images(...)` and add regression coverage for unpaired roots.
4. Refactor required versus optional artifact handling in `src/eq/utils/run_io.py` and add focused tests for required artifact failures.
5. Move historical docs into `docs/archive/` and update `docs/HISTORICAL_NOTES.md` as the index.
6. Rewrite current docs, including `docs/INTEGRATION_GUIDE.md` if retained, to remove active historical/fallback guidance while preserving current FastAI trusted-artifact warnings.
7. Add documentation/code quarantine validation by extending existing tests or helpers unless a new script is explicitly justified.
8. Run final validation:
   - `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .`
   - `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`
   - `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`
   - `python3 scripts/check_openspec_explicitness.py openspec/changes/p1-align-fastai-practices-and-archive-history`
   - `OPENSPEC_TELEMETRY=0 openspec validate p1-align-fastai-practices-and-archive-history --strict`

Rollback is ordinary Git rollback of this change. No runtime data, model files, or cloud artifacts should be moved by the implementation.
