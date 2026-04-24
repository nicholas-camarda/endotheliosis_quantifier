> Status note (2026-04-23): checked items below reflect decisions or runtime-prep work that are already complete. Unchecked items still require repository implementation, regression coverage, integration into the unified manifest workflow, or user-facing documentation updates.

> Delegation note: implementation and review for this change may use any number of specialist subagents to improve fidelity, but their work must converge back into the one canonical runtime manifest, one cohort tree, and one shared path contract defined by this change.

## 1. Cohort Inventory

- [x] 1.1 Add a scored-only cohort inventory utility that records runtime-local asset paths, filename and sample-ID join fields, treatment groups, score coverage, and unmapped or foreign rows.
- [x] 1.2 Add a CLI entrypoint or subcommand that writes deterministic cohort inventories and the unified `raw_data/cohorts/manifest.csv` under the active runtime root without mutating raw source data in place and without moving source assets out of their original locations.
- [x] 1.3 Add tests that verify mixed-export cohorts are reported with explicit unmapped or foreign rows rather than silently merged.
- [x] 1.4 Inventory and register the user's currently relevant accessible scored quantification cohorts, including the existing Lauren preeclampsia cohort and the currently identified VEGFRi Dox and VEGFRi MR cohorts.
- [x] 1.5 Keep Lucchi and other segmentation-install datasets out of the cohort inventory path for this change, with regression coverage so they are not pulled into runtime `raw_data/cohorts/` accidentally.
- [x] 1.6 Define the authoritative starting source for each current cohort, including latest-export precedence for Dox and workbook-first precedence for MR.
- [x] 1.7 For Dox, prefer direct authoritative Label Studio mask export when available from the user's setup and fall back to embedded brushlabel recovery otherwise.
- [x] 1.8 Treat the existing runtime-local Dox brushlabel recovery surface at `raw_data/cohorts/vegfri_dox/{images,masks,metadata/decoded_brushlabel_masks.csv}` as the current starting point on this machine rather than building a second competing active Dox mask tree.

## 2. Canonical Manifest

- [x] 2.1 Define the canonical scored-only `raw_data/cohorts/manifest.csv` schema as a runtime-local contract covering row identity, `cohort_id`, localized asset paths, score linkage, optional mask fields, verification state, and admission state.
- [x] 2.2 Split the schema into minimal required input columns, optional input columns, and pipeline-generated enrichment/state columns.
- [x] 2.3 Remove original source paths from the canonical runtime manifest and keep any optional source-location audit in separate ingest artifacts if it is retained at all.
- [x] 2.4 Update touched pipeline stages to consume the unified manifest as the canonical linking surface instead of re-deriving joins from cohort-specific source structures.
- [x] 2.5 Add tests that validate minimal-input unified-manifest inputs, generated-column enrichment, uniqueness constraints after enrichment, nullable mask handling, admitted-row completeness, and fail-closed admission-state behavior.
- [x] 2.6 Define and test `harmonized_id` generation so it uses the minimal cohort-specific uniqueness set, including batch/date only for cohorts that truly need those discriminators.
- [x] 2.7 Define and test image-level row semantics for cohorts such as MR where one image carries multiple glomerular replicate grades.
- [x] 2.8 Finalize and test the explicit enriched manifest schema, including human-required columns, human-optional columns, pipeline-generated columns, and file-hash fields for admitted rows.

## 3. Data Harmonization

- [x] 3.1 Add a harmonization step that consolidates inventoried scored-only cohorts into the flat runtime `raw_data/cohorts/<cohort_id>/` layout as localized working datasets by copying needed source assets without mutating original source folders in place.
- [x] 3.2 Emit deterministic row-level runtime linkage tying each copied asset to its harmonized repo-style identifier and any runtime-visible identifiers needed for score/image matching.
- [x] 3.3 Add tests that block downstream audit or grading builds when rows cannot be deterministically harmonized into the repo contract.
- [x] 3.4 Materialize runtime `raw_data/cohorts/<cohort_id>/` directories and populate `raw_data/cohorts/manifest.csv` for the current accessible cohorts rather than only adding reusable harmonization code.
- [x] 3.5 Copy localized cohort-owned `images/`, `masks/` when present, `scores/` when needed, and `metadata/` surfaces so future users can operate from `raw_data/cohorts/<cohort_id>/` without chasing distributed source roots.
- [x] 3.6 Keep original PhD / cloud source trees in place and add regression coverage so the harmonization workflow does not act as an in-place migration or move source assets into the runtime tree.
- [x] 3.7 Update path-handling code and tests so the cohort workflow targets the active runtime root rather than the repo checkout's empty placeholder `data/` directories on this machine.
- [x] 3.8 Build `lauren_preeclampsia` from Lauren's active preeclampsia runtime, `vegfri_dox` from the Dox Label Studio exports, and `vegfri_mr` from the MR scoring workbook plus supporting logs.
- [x] 3.9 Build MR cohort assets from the external-drive giant TIFF batches with explicit cohort metadata noting the high-resolution whole-field acquisition regime.

## 4. Mapping Verification

- [x] 4.1 Add a fail-closed mapping-verification step that requires one-to-one score-to-image joins for every harmonized staged item before admission.
- [x] 4.2 Record row-level verification evidence including harmonized identifier, runtime-local image path, runtime-local score linkage, and any required integrity fields.
- [x] 4.3 Add tests that reject ambiguous joins, conflicting duplicate scores, and cross-field identifier contradictions rather than auto-resolving them.
- [x] 4.4 Add iterative discovery/reconciliation logic so failed first-pass joins trigger additional accessible-source searches before unresolved or excluded status is finalized.
- [x] 4.5 Add tests that verify recoverable rows remain pending additional discovery after an initial failed join and only become unresolved or excluded after the configured discovery coverage is exhausted.
- [x] 4.6 For Dox, recover and classify brushlabel masks separately from image-level score choices and foreign mixed rows.
- [x] 4.6 For Dox, recover and classify brushlabel masks separately from image-level score choices and foreign mixed rows, and partition reconciled Dox rows into masked-external versus scored-only lanes.
- [x] 4.11 Record Dox recovered-mask provenance while treating verified Dox masks as first-class manual-mask glomeruli training labels.
- [x] 4.7 For MR, implement and test the within-image replicate-to-median reduction with explicit provenance fields in the manifest.
- [x] 4.8 For MR, preserve the raw within-image replicate vector in a sidecar ingest artifact and test that the canonical manifest row remains image-level.
- [x] 4.9 Require and test the full admission-check bundle: uniqueness, single image path, single score assignment, no conflicting duplicates, required locator fields, readable assets, file hashes, and sampled manual mapping review.
- [x] 4.10 Encode and test the explicit duplicate/conflict policy for same-image duplicates, same-sample multiple images, ambiguous score-to-image mappings, and multi-export Dox duplication.

## 5. Localized Dataset Build

- [x] 5.1 Add localized dataset-build logic so in-scope cohorts are consolidated into self-contained runtime directories with stable local asset paths.
- [x] 5.2 Add tests that verify unified-manifest rows resolve cleanly to localized runtime assets under `raw_data/cohorts/<cohort_id>/` without depending on original source paths.
- [x] 5.3 Add tests that verify downstream stages can operate from localized cohort directories without requiring ad hoc traversal of the original PhD / cloud roots.
- [x] 5.4 Add tests that verify cohort consolidation is copy-based, not move-based, and that original source assets remain present after localized dataset build.
- [x] 5.5 Archive overlapping pre-existing runtime quantification-input directories after the new cohort tree is verified, and add tests/docs so those archived trees are never reused as active inputs.

## 6. Segmentation Admission Gate

- [x] 6.1 Add a cohort-specific segmentation transport audit workflow that records the segmentation artifact provenance, reviewed cohort slice, and cohort admission or exclusion status.
- [x] 6.2 Add review artifact generation for transport audits so cohort and treatment-group failure modes are visible before grading use.
- [x] 6.3 Add tests that block cohort admission when segmentation outputs are missing, degenerate, or grading-invalid.
- [x] 6.4 Add MR-specific transport-audit coverage for giant TIFF preprocessing or tiling so the cohort cannot be admitted through a Dox-scale inference path by accident.
- [x] 6.5 Add MR-specific gating so `vegfri_mr` is held out from training admission in phase 1 even if harmonization and transport audit succeed.
- [x] 6.6 Make `vegfri_dox` the first scored-only training-expansion candidate once it clears the full admission bundle.
- [x] 6.7 Make verified masked `vegfri_dox` rows co-equal with Lauren manual-mask rows for glomeruli segmentation training once harmonization and verification pass.

## 7. Predicted-ROI Grading Inputs

- [x] 7.1 Add a predicted-ROI grading dataset builder for admitted scored-only cohorts that preserves image, score, harmonization-manifest, mapping-verification artifact, cohort-manifest, and segmentation provenance.
- [x] 7.2 Keep predicted-ROI grading artifacts separate from canonical manual-mask artifacts in file outputs and provenance fields.
- [x] 7.3 Add tests that ensure excluded cohorts are omitted from grading dataset builds and admitted cohorts are marked as predicted-ROI inputs rather than manual-mask supervision.
- [x] 7.4 Add an MR concordance workflow that aggregates inferred glomerulus grades to an image-level median and compares that inferred median against the human image-level median from the workbook.
- [x] 7.5 Add tests that confirm MR phase 1 outputs are concordance/evaluation artifacts and not training-set expansion artifacts.
- [x] 7.6 Define and test the explicit MR inference path: TIFF tiling, segmentation, accepted inferred ROI extraction, ROI grading, image-level median aggregation, and concordance metric reporting.
- [x] 7.7 Define and test repo-aligned MR inferred-ROI acceptance rules, including component-area filtering, image non-evaluable handling when zero accepted ROIs remain, and accepted ROI count reporting.
- [x] 7.8 Emit fixed MR concordance metrics including MAE, Spearman correlation, exact agreement, and within-one-step agreement, stratified by batch and treatment group when available.

## 8. Regression And Validation

- [x] 8.1 Add regression tests covering `src/eq/quantification/dataset.py`, `src/eq/quantification/pipeline.py`, any new CLI surface, and any touched ROI / grading builders so scored-only changes do not break the existing Lauren manual-mask path.
- [x] 8.1 Add regression tests covering `src/eq/quantification/dataset.py`, `src/eq/quantification/pipeline.py`, any new CLI surface, and any touched ROI / grading builders so external-cohort changes do not break the existing Lauren manual-mask path.
- [x] 8.2 Run targeted validation for the changed CLI, manifest generation, harmonization, verification, localized dataset-building, and regression paths plus relevant unit tests.
- [x] 8.3 Run `openspec validate expand-scored-only-quantification-cohort --strict` and resolve any remaining artifact issues.
- [x] 8.4 Refactor the shared repo path helpers to resolve the active runtime root, unified cohort manifest, cohort input tree, segmentation result tree, and quantification result tree consistently across the repository, with regression coverage.

## 9. Documentation

- [x] 9.1 Update user-facing documentation to describe the flat cohort layout, the scored-only manifest contract, harmonization rules, mapping-verification requirements, and admission limits in current-state terms.
- [x] 9.2 Update any CLI help text or workflow docs touched by the new cohort-admission surfaces so runtime `raw_data/cohorts/manifest.csv`, runtime `raw_data/cohorts/<cohort_id>/`, runtime `output/segmentation_evaluation/...`, runtime `output/predictions/...`, runtime `output/quantification_results/...`, and failure states are obvious.
- [x] 9.3 Document which of the user's current cohorts were populated under the new layout, which remain unresolved or excluded, and why.
- [x] 9.4 Document the iterative discovery policy so unresolved or excluded rows are clearly understood as "searched and still not recoverable" rather than "failed one pass."
- [x] 9.5 Document that PhD / cloud roots remain untouched provenance sources while `raw_data/cohorts/<cohort_id>/` becomes the localized working dataset future users should operate from, with cohort assets copied rather than moved into runtime and with `raw_data/cohorts/manifest.csv` treated as the runtime-local canonical table.
- [x] 9.6 Document the actual repo/runtime split so users do not confuse the repo checkout's placeholder `data/` tree with the active `$EQ_RUNTIME_ROOT` working tree.
- [x] 9.7 Document the exact current-cohort rollout outcome, including per-cohort row counts by masked-scored, scored-only, unresolved, excluded, and foreign classifications where applicable.
- [x] 9.8 Document MR separately as a giant whole-field TIFF cohort with image-level rows, within-image median reduction, cohort-specific preprocessing or tiling expectations, and phase 1 concordance-only use.
- [x] 9.9 Document the explicit admission-check bundle and duplicate/conflict policy so future users do not have to infer it from code.
- [x] 9.10 Document archived runtime quantification-input directories as retired reference surfaces and state that only the new cohort tree is supported as active input.

## 10. All-Available Masked Segmentation Training

- [x] 10.1 Make the manifest-backed `raw_data/cohorts` registry root a supported glomeruli training input that enumerates every admitted masked row across the current cohorts.
- [x] 10.2 Keep Lauren's active preeclampsia cohort root as the direct `raw_data/cohorts/lauren_preeclampsia` paired `images/` and `masks/` root rather than requiring a nested `training_pairs` tree.
- [x] 10.3 Filter all-data glomeruli training to admitted `manual_mask_core` and `manual_mask_external` manifest lanes so scored-only, foreign, unresolved, and MR concordance-only rows do not enter segmentation supervision.
- [x] 10.4 Update training CLI help, docs, config overlays, and regression tests so the current all-data training command points at `raw_data/cohorts`.
- [x] 10.5 Make requested transfer training fail closed when the base artifact cannot load or supplies zero compatible weights, so it cannot silently become the no-mitochondria-base `--from-scratch` comparator.
