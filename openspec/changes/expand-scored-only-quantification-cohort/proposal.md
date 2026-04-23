## Why

The repository now has large external kidney-image cohorts from the VEGFRi+doxazosin+lisinopril and VEGFRi+MR / spironolactone experiments, but they are outside the current quantification contract because they do not carry repo-native canonical organization in the active project layout. Some of those cohorts are genuinely scored-only, while the Dox lane now also has recovered brushlabel-derived masks. If the glomeruli segmenter becomes scientifically usable on these cohorts, the project should be able to use verified masked Dox rows to improve segmentation evidence and use predicted glomerulus ROIs to expand endotheliosis grading on truly scored-only rows without pretending those two lanes are the same thing.

## What Changes

- Define an external cohort expansion workflow that inventories external cohorts, records cohort provenance, and stages them as either a masked segmentation-improvement lane or a scored-only downstream grading lane depending on what assets survive verification.
- Make a row-level manifest the canonical runtime linking surface for scored-only cohort admission, verification, and downstream grading inputs.
- Keep the hand-authored manifest burden small by requiring only the runtime-local fields needed to link copied cohort assets and scores, while letting the pipeline generate derived identifiers and admission-state columns.
- Generate `harmonized_id` from the smallest discriminator set that makes a row unique within its cohort, so cohorts with reused sample IDs can incorporate batch/date provenance while cleaner future cohorts do not need those extra fields in the runtime key.
- Add a data harmonization step that organizes every cohort into one cohort-first directory shape plus one dataset-level manifest with predictable asset paths before any cohort is audited or admitted.
- Require multiple high-fidelity mapping checks that verify every admitted staged image links unambiguously to the intended score row and source identifier before downstream use.
- Require the full admission check set for every admitted row: unique runtime identity, exactly one runtime image path, exactly one score assignment, no conflicting duplicate scores, required cohort-specific locator fields, readable runtime assets, file hashes, and a sampled manual mapping-review panel.
- Require iterative discovery and reconciliation passes across the accessible source surfaces before any row or cohort is left unresolved or excluded.
- Keep PhD/cloud data untouched as provenance-bearing source inputs, but build a localized runtime cohort dataset by copying needed assets into runtime cohort directories so future users can work from one organized surface without chasing distributed source roots.
- Copy localized cohort-owned images, masks when present, score files when needed, and metadata into the runtime cohort directories as part of cohort build-out rather than leaving the runtime layout as a partial source-linked view.
- Require a segmentation transport audit before any scored-only cohort is admitted into downstream predicted-ROI use, including explicit visual review and failure-mode reporting by cohort and treatment group.
- Treat MR as a distinct high-resolution cohort class: image-level rows come from giant whole-field TIFFs, raw workbook replicates are reduced into one image-level median score, MR transport review must explicitly cover preprocessing or tiling behavior, and MR phase 1 use is limited to external inference/concordance rather than training admission.
- Treat Dox recovered masks as a distinct masked external lane that can improve segmentation training/evaluation once harmonization, verification, and mask-quality gates pass; direct Label Studio mask export is preferred when the authoritative project is available, with embedded brushlabel recovery remaining the fallback.
- Treat scored-only Dox rows that still lack verified masks as the first predicted-ROI grading-expansion candidate once harmonization, verification, and transport gates pass.
- Define MR concordance explicitly: tile the copied giant TIFFs, run segmentation, extract accepted glomerulus ROIs using repo-aligned component/union semantics, grade those accepted ROIs, aggregate inferred ROI grades to an image-level median, and compare that inferred median against the human image-level median with fixed concordance metrics.
- Add separate contracts for:
  - admitting verified masked external rows into segmentation improvement/evaluation
  - generating predicted glomerulus ROIs and score-linked grading tables from scored-only images without introducing them as segmentation ground truth
- Persist cohort-admission and cohort-exclusion artifacts so unsupported scored-only images are not silently mixed into the canonical grading dataset.
- Keep original source-path tracking out of `manifest.csv`; if source-location audit logs are retained, they belong in separate ingest artifacts rather than in the canonical runtime manifest.
- Store cohort-owned runtime inputs and manifest state under the active runtime root, not under the mostly empty repo-local `data/` placeholders on this machine. For the current environment, that means `~/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/<cohort_id>/` for localized cohort datasets, `~/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/manifest.csv` for the canonical dataset-wide manifest, and `~/ProjectsRuntime/endotheliosis_quantifier/output/cohorts/<cohort_id>/` for model/run artifacts.
- Archive overlapping old runtime quantification-input directories once the new cohort tree is populated and verified, and make the new cohort tree the only supported active input surface.
- Refactor the shared repo path helpers so the whole repository resolves the active runtime root and new cohort tree consistently rather than adding a one-off path system for this feature.
- Allow implementation and review work for this change to use any number of delegated specialist subagents when that improves fidelity, as long as the resulting artifacts still converge back into the single canonical manifest/runtime contract rather than creating parallel truths.
- Build out the cohort directories and unified manifest for the current local/project-accessible data holdings rather than stopping at reusable infrastructure alone.
- Keep downloaded segmentation datasets such as Lucchi outside this manifest contract; they remain separate segmentation data with their own organizer and do not require unified-manifest rows unless a future change explicitly brings them into quantification.
- Execute a cohort-specific rollout:
  - `masked_core`: translate the existing active preeclampsia runtime into the new cohort layout without breaking the current masked-core path
  - `vegfri_dox`: treat the latest Dox Label Studio export as the starting surface, recover both score choices and brush masks, filter mixed foreign rows explicitly, and split the reconciled subset into verified masked rows for segmentation improvement versus scored-only rows for predicted-ROI grading expansion
  - `vegfri_mr`: derive image-level cohort rows from the MR scoring workbook plus experiment logs and giant TIFF image batches, define a reproducible within-image replicate-to-image median reduction, preserve the raw replicate vector in sidecar ingest artifacts, require MR-specific preprocessing or tiling review before segmentation transport is considered acceptable, and restrict phase 1 outputs to external inference/concordance artifacts rather than training-set expansion
- Define success as populated localized runtime cohort directories, one populated unified manifest, explicit unresolved or excluded rows when needed, documented counts, and no regression of the existing masked-core quantification path.
- Keep manual-mask segmentation supervision, scored-only grading expansion, and statistical model stabilization as separate change scopes.

## Capabilities

### New Capabilities
- `scored-only-quantification-cohort`: Defines how external cohorts are inventoried, partitioned into verified masked versus scored-only rows, audited for segmentation usability, and then either admitted into segmentation improvement, converted into predicted-ROI grading inputs, or excluded.

### Modified Capabilities

## Impact

- Affected code: `src/eq/quantification/pipeline.py`, `src/eq/quantification/dataset.py`, cohort-ingestion and harmonization utilities, ROI/crop generation helpers, review-report generation, and any CLI surface added for cohort inventory or admission.
- Affected tests: cohort-inventory tests, manifest-schema tests, repo-style harmonization tests, high-fidelity mapping verification tests, mask-quality and masked-external admission tests, segmentation-admission gate tests, predicted-ROI dataset-building tests, and regression coverage that prevents external-cohort logic from breaking the existing masked-core quantification path.
- Affected artifacts: one unified runtime manifest, localized cohort directories containing organized runtime images/masks/scores/metadata, mapping-verification fields, segmentation transport audit reports, predicted ROI images/masks, score-linked grading tables, and admission/exclusion states under the active runtime root. In the current environment, that is `~/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/manifest.csv`, `~/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/<cohort_id>/`, and `~/ProjectsRuntime/endotheliosis_quantifier/output/cohorts/<cohort_id>/`.
- Compatibility risks: external scored cohorts currently use heterogeneous filenames and experiment-specific metadata; admitting them without explicit harmonization, mapping verification, unified-manifest rows, or transport audits could contaminate grading data, confuse provenance, and overstate model validity on unsupported segmentation outputs.
