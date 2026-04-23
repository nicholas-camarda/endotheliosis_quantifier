## Context

The repository currently has two distinct needs that must not be conflated. First, the active glomeruli change must determine whether the segmenter is scientifically usable under the supported training contract. Second, the active ordinal change must stabilize the grading model before further cohort expansion changes are treated as reliable evidence. External scored kidney-image cohorts already exist in the PhD project roots, especially under the VEGFRi+doxazosin and VEGFRi+MR folders, but those cohorts are not in the active quantification contract because they do not provide segmentation masks in the canonical project layout.

Those external cohorts are still attractive for downstream grading because they contain many scored images across treatment groups. The Dox cohort already contains explicit `vehicle`, `sorafenib`, `sor + dox`, and `sor + lis` groups, while the MR lane contains a populated raw kidney-scoring workbook plus BL6 spironolactone summary outputs. The central constraint is that truly scored-only images cannot be treated as segmentation supervision. However, Dox is no longer purely scored-only here: recovered brushlabel-derived masks now exist in the runtime cohort tree. That means the change must explicitly separate two Dox sublanes:
- verified masked Dox rows, which can become external masked evidence for segmentation improvement/evaluation if their recovered masks are good enough
- Dox rows without verified masks, which remain predicted-ROI grading candidates only

MR remains different. It should not enter training in phase 1. Its first use should be external concordance: run inference on the giant TIFF images, aggregate inferred glomerulus grades to an image-level median, and compare that inferred median against the human image-level median derived from the workbook.

MR now needs to be treated as a materially different acquisition regime, not as a minor variant of Dox. On the external image drive, the Dox image folders contain many `2448 x 2048` RGB JPGs per sample folder, while the MR kidney image batches contain giant whole-field RGB TIFFs around `25,000 x 9,000` pixels. The raw MR workbook is also image-associated but not image-flat: each sample/image column carries repeated glomerular replicate grades. So MR needs an image-level manifest row contract, an explicit within-image replicate-to-score reduction, preserved raw replicate vectors in sidecar ingest artifacts, and cohort-specific preprocessing or tiling review before segmentation transport can be trusted.

This change therefore needs to define an external cohort-admission contract, not just a scored-only grading change. It should inventory external cohorts, harmonize each one into a single localized cohort-owned directory, register all rows in one unified manifest, partition rows into verified masked versus scored-only sublanes, audit segmentation transport on a cohort slice, admit verified masked rows into segmentation improvement/evaluation when justified, produce predicted-ROI grading inputs for scored-only rows when the audit passes, and preserve explicit exclusion artifacts when any lane fails.

For this repository, the change should not stop at defining the contract in the abstract. It should build the cohort-first layout and unified manifest for the currently relevant data holdings the user already has access to, so the runtime ends the change with real cohort directories rather than an unused framework.

This does not include Lucchi or similar downloaded segmentation datasets. In the current repo, Lucchi is handled by a dedicated organizer and supports mitochondria segmentation data layout, not scored quantification cohort linkage. Pulling it into the unified manifest contract here would conflate segmentation-install data with scored quantification data.

The localized cohort layout should be treated as the working runtime input surface for this repository, while PhD / cloud roots remain the immutable provenance-bearing source surfaces. Future users should be able to operate from the localized cohort directories and one unified `manifest.csv` without needing to rediscover the original distributed source layout. The localized dataset should be built by copying, never moving, needed source assets into the runtime cohort directories. The canonical manifest should be runtime-native and should not require original source paths once assets have been consolidated locally.

The concrete path split matters here. In the current checkout, repo-local `data/raw_data` and `data/derived_data` are effectively empty placeholders, while the active raw projects, derived datasets, models, logs, and outputs already live under `~/ProjectsRuntime/endotheliosis_quantifier`. The new cohort layout should therefore be specified against the active runtime root, not against the git checkout's local `data/` tree.

There is now direct runtime evidence that Dox brushlabel mask recovery is feasible under that contract. On this machine, decoded Dox brushlabel masks have already been materialized under `~/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/vegfri_dox/` with runtime-local `images/`, `masks/`, and `metadata/decoded_brushlabel_masks.csv`. The current decoded pass recovered `626` Dox rows, skipped `228` foreign Lauren-style rows, and skipped `10` ambiguous multi-brush tasks pending explicit adjudication. That runtime cohort surface should be treated as the current starting point for Dox mask-aware reconciliation rather than as a hypothetical future option.

## Goals / Non-Goals

**Goals:**
- Define how external cohorts are inventoried and staged as either candidate segmentation-improvement inputs or candidate downstream grading inputs.
- Make one unified manifest the canonical row-level linking surface for external cohort identity, provenance, score linkage, asset paths, mask availability, admission state, and downstream use.
- Keep the unified `manifest.csv` practical to author or repair by hand by separating minimal required input columns from pipeline-generated enrichment and state columns.
- Make `harmonized_id` a cohort-scoped runtime key derived from the minimal uniqueness-preserving discriminator set, rather than a direct copy of any one source field.
- Define one cohort-first runtime layout that keeps localized cohort-owned inputs and manifest state under the active runtime root's `raw_data/cohorts/<cohort_id>/` and run outputs under the active runtime root's `output/cohorts/<cohort_id>/`.
- Materialize that layout and manifest contract for the currently relevant accessible cohorts instead of leaving cohort population as future manual work.
- Preserve PhD / cloud sources in place and avoid treating this change as an in-place source migration.
- Build localized cohort-owned runtime assets as a required part of cohort consolidation so the runtime dataset is self-contained for future use.
- Require multiple high-fidelity checks that prove each admitted score row maps to the intended staged image and source cohort identifier.
- Require the full admission-check set for admitted rows, including asset readability, file hashing, and sampled manual review rather than join logic alone.
- Require iterative discovery passes over the accessible source files, workbooks, metadata logs, and naming clues before declaring a row or cohort unresolved or excluded.
- Require cohort-specific segmentation transport evidence before scored-only images are converted into predicted-ROI grading inputs.
- Require a separate mask-quality and contract check before recovered masked external rows are treated as segmentation-improvement evidence.
- Require MR-specific transport evidence that explicitly handles giant whole-field TIFF preprocessing or tiling rather than assuming the Dox-scale image path generalizes.
- Restrict MR phase 1 use to inference/concordance artifacts rather than training-set expansion.
- Make Dox the first scored-only cohort eligible for training-set expansion once it passes harmonization, verification, mask recovery, and transport gates.
- Archive pre-existing overlapping runtime quantification input surfaces after migration and treat the new cohort tree as the only supported active input surface.
- Refactor shared repo path helpers rather than introducing cohort-specific path exceptions.
- Produce one explicit dataset-wide manifest that records provenance, admission status, treatment-group membership, and `cohort_id` for every row.
- Define predicted-ROI grading artifacts that remain distinct from canonical manual-mask quantification artifacts.
- Prevent scored-only rows from being silently treated as segmentation labels, and prevent recovered masked rows from being silently treated as contract-equivalent to the current masked-core cohort before explicit admission.

**Non-Goals:**
- Improving the glomeruli segmenter itself in this change.
- Declaring the current segmenter transportable before the active promotion change completes.
- Stabilizing the ordinal estimator or changing grouped ordinal modeling behavior in this change.
- Reconstructing missing segmentation masks for external cohorts.
- Reorganizing Lucchi or other downloaded segmentation-only datasets into unified-manifest rows.
- Treating external scored cohorts as scientific evidence of generalization without explicit cohort-admission artifacts and review.

## Decisions

1. **Create a cohort-admission workflow that is downstream of both active changes.**
   - Rationale: the current active specs already separate scored-only expansion from segmentation promotion and from ordinal stabilization. This change should respect that sequencing and define the later onboarding workflow without pretending the prerequisites are already met.
   - Alternatives considered:
     - Fold cohort expansion into the glomeruli promotion change. Rejected because the broader cohort-onboarding problem now includes both segmentation-relevant masked rows and grading-only scored rows and would blur the current promotion scope.
     - Fold cohort expansion into the ordinal stabilization change. Rejected because the point of that change is to stabilize the existing estimator path before changing cohort composition.

2. **Require a segmentation transport audit before cohort admission.**
   - Rationale: predicted ROIs are only valid grading inputs if the segmenter remains non-degenerate on the new cohorts. A grading workflow built on failed segmentation would encode segmentation error, not endotheliosis severity.
   - Alternatives considered:
     - Admit all scored-only images once a segmenter loads and runs. Rejected because runtime success is compatibility evidence only.
     - Admit cohorts based on a small number of aggregate mask metrics alone. Rejected because visual degeneracy and treatment-specific failure can hide behind summary metrics.

3. **Persist one unified manifest with explicit row-level admission and exclusion states.**
   - Rationale: external cohorts have heterogeneous filenames, experiment-specific sample IDs, and potentially mixed exports. The workflow needs one explicit manifest layer that records what was seen, what joined cleanly, what treatment groups were found, and why any images or cohorts were excluded, without forcing users to inspect multiple CSVs.
   - Alternatives considered:
     - Join cohorts directly into the existing quantification tables. Rejected because silent joining would erase provenance and make later audit impossible.

4. **Use a canonical row-level manifest as the scored-only runtime contract surface.**
   - Rationale: the existing quantification workflow already relies on deterministic scored tables and explicit join statuses rather than implicit path traversal. A manifest-first design scales better than embedding cohort-specific parsing logic in every pipeline stage and gives one place to prove whether a row is admitted, excluded, verified, or still unresolved. Once data have been copied into the runtime cohort tree, the manifest should describe that runtime dataset directly rather than carrying original source paths.
   - Alternatives considered:
     - Let each stage re-derive score joins from workbooks, filenames, and logs. Rejected because it would duplicate fragile cohort-specific logic and make regression testing weaker.
     - Treat staged filenames as the primary source of truth. Rejected because filenames alone are not strong enough provenance for score linkage.

5. **Harmonize external cohort inputs into one localized cohort-first directory shape backed by one unified manifest before any audit or admission decision.**
   - Rationale: repeated project-name nesting and separate contract/output trees are harder to reason about than a single cohort-owned shape plus one dataset-level manifest. Each cohort should have one obvious home for its localized images, masks if any, score files, and metadata, while all rows live in one canonical table keyed by `cohort_id`.
   - Alternatives considered:
     - Audit and join directly from the original cloud-export filenames. Rejected because heterogeneous experiment naming would leak one-off parsing logic into every downstream step.
     - Introduce extra intermediate trees such as `quantification_contract`, `scored_only_cohorts`, or repeated project-name nesting. Rejected because they add abstraction without clarifying ownership.
     - Put the new cohort layout under the repo checkout's local `data/` directories on this machine. Rejected because the current active data and outputs already live in `~/ProjectsRuntime/endotheliosis_quantifier`, while the repo-local `data/` directories are not the operational source of truth.

6. **Require fail-closed, high-fidelity score-to-image verification before cohort admission.**
   - Rationale: these cohorts come from heterogeneous exports where filenames, workbook sample IDs, treatment assignments, and score rows can drift apart. Admission needs more than a successful parse; it needs explicit one-to-one mapping checks, duplicate/contradiction detection, and provenance strong enough to audit each admitted item back to the source row and staged image file.
   - Alternatives considered:
     - Trust the harmonized identifier alone after rename. Rejected because deterministic renaming does not prove the underlying score row was the right one.
     - Allow ambiguous joins when all candidates share the same score. Rejected because identical current values can still conceal a wrong biological unit and break future audits.

6c. **Require the full admission-check bundle rather than just deterministic joins.**
   - Rationale: the user wants a guarantee that the localized dataset is trustworthy for future operators. Admission therefore needs filesystem and review evidence in addition to identifier logic.
   - Required checks:
     - `harmonized_id` is unique within `cohort_id`
     - exactly one runtime-local `image_path` is linked to the row
     - exactly one score assignment survives reconciliation
     - no conflicting duplicate scores exist for the same runtime image
     - required cohort-specific locator fields are present
     - runtime assets are readable
     - runtime asset hashes are recorded
     - a sampled manual mapping-review panel passes for the cohort

6d. **Make duplicate/conflict policy explicit.**
   - Rationale: multiple images per biological sample and multiple within-image glomerular grades are expected in the user’s real data; those must not be conflated with true duplicates or conflicts.
   - Policy:
     - same runtime image plus identical score: deduplicate and retain duplicate evidence
     - same runtime image plus conflicting score: block admission
     - same biological sample plus multiple images: allow multiple image-level rows
     - one score row plus multiple plausible images: unresolved unless later evidence resolves it
     - same Dox task across multiple dated exports: latest export wins; older exports only recover missing fields

6f. **Use recovered masked Dox rows to improve segmentation only after explicit mask-quality admission.**
   - Rationale: now that Dox brushlabels have been decoded into real runtime masks, those rows can provide external segmentation evidence. But recovered brushlabel masks are not automatically equivalent to the canonical masked-core contract. They need explicit quality screening, harmonization, and provenance so segmentation improvement does not quietly ingest weak masks.
   - Policy:
     - verified masked Dox rows can enter segmentation evaluation and, if they clear quality gates, segmentation training augmentation
     - Dox rows without verified masks remain in the scored-only/predicted-ROI grading lane
     - the manifest must distinguish these lanes explicitly

6a. **Treat the manifest row unit as image-level for cohorts where one image carries multiple graded glomeruli.**
   - Rationale: MR raw scoring is repeated within-image glomerular grading, not one score row per independent image file. The canonical manifest should therefore use one row per image and represent within-image replicate grades through an explicit reduction plus sidecar raw replicate retention.
   - Alternatives considered:
     - Collapse MR directly to one row per biological sample regardless of image structure. Rejected because it destroys image-level linkage needed for segmentation and ROI generation.
     - Promote each raw MR replicate to its own manifest row. Rejected because the current data do not support clean replicate-to-subimage localization.

6b. **Use MR as an external concordance cohort before any training admission.**
   - Rationale: MR differs from masked-core and Dox on both image regime and label structure. Using it for training immediately would blur segmentation transport, grading performance, and within-image aggregation behavior. Phase 1 should therefore ask the narrower question: does the inferred image-level median agree with the human image-level median on giant TIFF images?
   - Alternatives considered:
     - Admit MR directly into grading training once images and scores are harmonized. Rejected because the cohort is too different and the labels are already image-level aggregates over multiple glomeruli.
     - Exclude MR entirely until a future change. Rejected because MR is still valuable as an external concordance and transport-evaluation surface.

6e. **Use repo-aligned ROI acceptance semantics for MR inference.**
   - Rationale: the repository already has two relevant precedents: promotion review blocks non-degenerate or non-whole-glomerulus predictions, and the quantification pipeline uses thresholded connected components plus full multi-component union ROI semantics. MR inference should extend those same principles rather than invent a second ROI concept.
   - Policy:
     - MR transport review must first show the promoted segmenter is non-degenerate on sampled MR tiles
     - accepted inferred glomerulus candidates must come from thresholded connected components that satisfy the repo’s minimum component-area gate
     - accepted ROI crops must use the same component/union extraction semantics and metadata style already used by the quantification pipeline
     - MR images with zero accepted inferred ROIs are non-evaluable
     - MR concordance reports must include accepted ROI counts and flag low-count images explicitly rather than silently treating them as equivalent to well-covered images

7. **Make one dataset-level `manifest.csv` the single explicit contract file for all localized cohorts.**
   - Rationale: one manifest file is easier to inspect, test, and reason about than a bundle of per-cohort tables. Verification, admission, and exclusion should live as fields on each row, with `cohort_id` partitioning the data.
   - Alternatives considered:
     - Split verification into separate mapping tables by default. Rejected because it increases ambiguity about which file is authoritative.

8. **Require only a minimal hand-authored input manifest and let the pipeline enrich it.**
   - Rationale: many cohorts will need manual curation or hand fixes. Requiring users to author internal IDs, staged paths, and pipeline-state columns would make the manifest fragile and laborious. The stable source-truth fields should be entered by hand when needed, and the pipeline should append deterministic IDs, harmonized paths, verification results, and admission states.
   - Alternatives considered:
     - Require the fully enriched manifest schema as manual input. Rejected because it would be cumbersome to build and easy to corrupt by hand.

8a. **Define the enriched manifest schema explicitly, while keeping hand-authored input minimal.**
   - Rationale: the user wants no ambiguity about what belongs in the canonical table, but they also do not want to hand-author machine-state columns. The contract should therefore enumerate the full enriched schema and still distinguish human-input fields from generated fields.
   - Enriched manifest columns:
     - human-required: `cohort_id`, `image_path`, `score`, and at least one of `source_sample_id` or `source_score_row`
     - human-optional: `mask_path`, `treatment_group`, `source_score_sheet`, `score_path`, `source_image_name`
     - pipeline-generated: `manifest_row_id`, `harmonized_id`, `join_status`, `verification_status`, `admission_status`, `exclusion_reason`, `image_sha256`, `mask_sha256`

9. **Derive `harmonized_id` from the smallest cohort-specific uniqueness set.**
   - Rationale: some legacy cohorts reuse sample IDs across dates or batches, while future cohorts may have clean unique image or sample identifiers. The runtime key should therefore use the smallest set of provenance discriminators needed to make a row unique within its cohort, instead of hard-coding batch/date into every cohort or pretending one source field is always sufficient.
   - Alternatives considered:
     - Set `harmonized_id = source_sample_id`. Rejected because reused sample IDs make that unsafe in current Dox data.
     - Always require batch and date in `harmonized_id` for every cohort. Rejected because that would encode one legacy failure mode into the global contract and make cleaner future cohorts noisier than necessary.

10. **Anchor the new cohort layout to the active runtime root, not the repo checkout.**
   - Rationale: the existing endotheliosis working state already lives in `~/ProjectsRuntime/endotheliosis_quantifier` with active `raw_data/`, `derived_data/`, `models/`, `logs/`, and `output/` trees. Putting the new unified manifest under repo-local `data/` would create a second competing runtime topology.
   - Alternatives considered:
     - Treat repo-local `data/raw_data` and `data/derived_data` as the new active surface. Rejected because that would not match the current machine's actual runtime state.

11. **Keep source data in place while making the localized cohort layout the working runtime dataset.**
   - Rationale: the user's PhD / cloud folders remain the primary evidence surface. Reorganizing those source trees in place would be risky and unnecessary. The localized cohort directories should therefore be built by copying needed assets from those sources without mutating or moving them. The runtime manifest should then describe the copied working dataset directly rather than retaining original path fields.
   - Alternatives considered:
     - Physically migrate all source data into the local cohort tree up front. Rejected because it would duplicate or mutate source holdings before the linkage contract is proven.

12. **Copy localized cohort-owned files as part of cohort build-out.**
   - Rationale: the user's main problem is that relevant images, scores, masks, and metadata are currently spread across many source directories and formats. Future operators should not need to traverse those scattered sources during normal use. Each in-scope cohort should therefore end this change with a self-contained localized working directory.
   - Alternatives considered:
     - Leave the cohort as a manifest-only view over distributed source paths. Rejected because it preserves the operational complexity the user wants to eliminate.
     - Move source assets into the runtime cohort tree. Rejected because it would alter the original source holdings and break the provenance-preserving contract.

13. **Write model-derived cohort outputs under the runtime root's `output/cohorts/<cohort_id>/`.**
   - Rationale: transport audits, predicted ROI assets, embeddings, and grader outputs are rebuildable run products and should stay separate from cohort-owned input state.
   - Alternatives considered:
      - Write model outputs next to the manifest in the data tree. Rejected because it mixes durable dataset state with disposable run artifacts.

14. **Represent scored-only grading inputs as predicted-ROI artifacts, not canonical manual-mask artifacts.**
   - Rationale: the existing quantification contract is based on images and masks with union ROI semantics. Scored-only cohorts need a parallel but clearly marked artifact family that records predicted ROI provenance and the segmentation artifact used to generate it.
   - Alternatives considered:
     - Reuse the same raw-mask fields while pointing them at predicted masks. Rejected because it would collapse manual and predicted provenance and make scientific interpretation ambiguous.

15. **Gate grading use on cohort-specific review outputs, not just on inventory counts.**
   - Rationale: the external cohorts differ by treatment, date, naming style, and likely staining / imaging context. The workflow should emit review panels and failure summaries stratified by cohort and treatment group so transport errors are visible before model training.
   - Alternatives considered:
     - Use inventory counts and sample IDs alone to approve admission. Rejected because those counts do not test whether glomeruli are actually segmented well enough for grading.

16. **Build the unified manifest and cohort directories for the current accessible datasets during this change.**
   - Rationale: the user wants this built out for their data, not just specified for some future operator. The change should therefore populate the new layout for the cohorts already identified in the PhD and active project roots, subject to the same fail-closed rules when a cohort cannot yet be admitted.
   - Alternatives considered:
      - Ship only generic ingestion infrastructure and leave current cohort population as follow-up manual work. Rejected because it would leave the main operational value unrealized.
      - Pull Lucchi and other segmentation-install datasets into the same cohort population step. Rejected because those datasets do not serve the scored quantification linkage problem this change is solving.

13. **Do not mark rows or cohorts unresolved after only one discovery pass.**
   - Rationale: the user expects the implementation to find information that exists but may be distributed across filenames, workbooks, assignment logs, or alternate exports. The workflow should therefore iterate through the accessible reconciliation clues before concluding that linkage failed.
   - Alternatives considered:
     - Treat the first failed join attempt as enough evidence for exclusion. Rejected because it would overstate missingness and leave recoverable data unused.

14. **Archive old runtime quantification-input trees once the new cohort tree is authoritative.**
   - Rationale: the user wants one clean active input surface. Leaving older overlapping runtime trees active would preserve the confusion this change is supposed to remove.
   - Alternatives considered:
     - Keep old and new runtime trees both active. Rejected because that preserves multiple operational truths.
     - Delete older trees immediately. Rejected because archival value may remain even though they should never again be used as active inputs.

15. **Refactor shared path helpers repo-wide for the new runtime contract.**
   - Rationale: this should not become another feature with its own path exceptions. The shared path layer should resolve the active runtime root, cohort manifest, cohort input tree, and cohort output tree for the whole repo.
   - Alternatives considered:
     - Add a scored-only-only path resolver. Rejected because that would create another competing path contract.

16. **Allow delegated specialist implementation and review lanes when they improve fidelity.**
   - Rationale: this change spans cohort discovery, mask recovery, manifest design, path-contract cleanup, segmentation-quality review, and grading-lane partitioning. Parallel specialist work can improve audit fidelity and implementation speed as long as all delegated work merges back into the same canonical runtime-manifest contract.
   - Policy:
     - any number of specialist subagents may be used during implementation or review
     - delegated work must converge back into the single unified manifest/runtime contract
     - delegation must not create alternate active data layouts, duplicate contract files, or conflicting admission logic

## Risks / Trade-offs

- [Risk] A segmenter that is acceptable on the promoted validation manifest may still fail on Dox or MR cohorts due to domain shift. → Mitigation: require cohort-specific segmentation transport audits with visual review and explicit exclusion states.
- [Risk] MR giant whole-field TIFFs may be too large for the default Dox-scale inference path and may require different tiling or preprocessing to avoid memory or context failures. → Mitigation: make MR transport review explicitly test preprocessing or tiling behavior before admission and treat MR as its own acquisition class in docs and tests.
- [Risk] External score exports may mix multiple experiments or legacy cohorts in the same file. → Mitigation: require the unified manifest to report unmapped or foreign sample IDs with explicit `cohort_id` and status fields before any cohort is admitted.
- [Risk] Harmonization could accidentally relabel or relocate files in a way that breaks the copied runtime dataset. → Mitigation: keep original source folders immutable, require deterministic localized asset paths plus harmonized identifiers in the runtime manifest, and keep any optional source-location audit outside the canonical manifest.
- [Risk] The change could accidentally move or rename original source assets while building the runtime dataset. → Mitigation: require copy-only cohort consolidation and add regression coverage that blocks move-based behavior.
- [Risk] A staged image could still be linked to the wrong score row even after harmonization if workbook IDs, filenames, or treatment logs disagree. → Mitigation: require multiple independent mapping checks, contradiction detection, file-level provenance, and explicit exclusion for any non one-to-one mapping.
- [Risk] Encoding legacy batch/date collisions into every runtime key would make future cleaner cohorts harder to use. → Mitigation: derive `harmonized_id` from the minimal cohort-specific uniqueness set and keep batch/date as provenance fields unless they are needed for uniqueness.
- [Risk] The pipeline could mark real recoverable rows unresolved too early because information is spread across multiple source surfaces. → Mitigation: require iterative discovery and reconciliation attempts with logged search coverage before unresolved or excluded status is finalized.
- [Risk] Predicted ROIs could make the grading model appear to improve while actually learning segmentation artifacts. → Mitigation: persist predicted-ROI provenance, stratify review by treatment group, and keep masked-core evaluation separate from scored-only expansion evaluation.
- [Risk] Heterogeneous filenames and sample metadata could silently join the wrong scores to the wrong images. → Mitigation: require deterministic join reports and explicit row-level exclusion states in the unified manifest rather than permissive fallback joins.
- [Risk] New scored-only logic could regress the current masked-core quantification path. → Mitigation: require stage-level regression tests for the existing canonical pipeline and keep scored-only admission surfaces additive rather than replacing masked-core joins.
- [Risk] The spec could create a second competing runtime layout inside the repo checkout. → Mitigation: anchor the cohort contract to the active runtime root and document that repo-local `data/` is not the operational data home on this machine.
- [Risk] Users may over-interpret successful pipeline execution as evidence that the expanded grading model is scientifically valid across cohorts. → Mitigation: state external-validity limits directly in admission artifacts and require review outputs to distinguish compatibility from methodological support.
- [Risk] Old runtime input trees could remain in circulation after the new cohort tree is built. → Mitigation: archive them, document them as retired, and route shared path helpers only to the new active surfaces.
- [Risk] Recovered Dox brushlabel masks could be treated as segmentation truth before their quality is assessed. → Mitigation: require explicit mask-quality admission and keep masked-external Dox rows separated from both masked-core and scored-only Dox rows in the manifest.
- [Risk] Large multi-surface implementation work could drift into inconsistent partial solutions. → Mitigation: allow delegated specialist lanes for fidelity, but require all outputs to collapse back into one canonical manifest/runtime contract and one shared path layer.

## Concrete Rollout

1. **`masked_core` cohort**
   - Build `~/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/manifest.csv` entries for the existing active preeclampsia runtime under `cohort_id=masked_core`.
   - Reconcile current masked-core image, mask, annotation, and mapping surfaces into the new manifest without changing the supported masked-core quantification behavior.
   - Success for this cohort: every currently supported masked-core row is represented in the manifest with explicit provenance and no regression of the current quantification path.

2. **`vegfri_dox` cohort**
   - Start from the latest dated Dox Label Studio export as the primary export surface, then reconcile against older exports and the Dox Label Studio export directory only as needed to recover missing rows or provenance.
   - Prefer direct authoritative Label Studio mask export when available from the user’s setup; otherwise recover masks from embedded `brushlabels` exports.
   - Use the existing decoded runtime Dox brushlabel surface at `~/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/vegfri_dox/` as the current starting point for mask-aware reconciliation on this machine, with `metadata/decoded_brushlabel_masks.csv` as the runtime-local recovery ledger.
   - Recover both image-level score choices and `brushlabels` glomerulus masks where present.
   - Partition Dox rows into:
     - verified masked rows that can feed segmentation improvement/evaluation once mask quality is accepted
     - scored-only rows that remain predicted-ROI grading candidates
   - Filter out foreign or mixed rows explicitly rather than silently folding them into the Dox cohort.
   - Use batch/date discriminators in `harmonized_id` only where reused IDs make them necessary.
   - If mask-quality and segmentation gates pass, allow verified masked Dox rows to become the first external segmentation-improvement cohort.
   - If transport passes, allow the remaining scored-only Dox rows to become the first predicted-ROI grading-expansion cohort.
   - Success for this cohort: a populated manifest with rows classified as masked-scored, scored-only, unresolved, excluded, or foreign, documented counts for each class, a segmentation-improvement lane for verified masked rows only if mask-quality and verification gates pass, and a predicted-ROI lane for scored-only rows only if transport and verification gates pass.

3. **`vegfri_mr` cohort**
   - Start from `VEGFRi and MR kidney scoring.xlsx` as the primary score source, using experiment logs and score-result workbooks only as supporting reconciliation surfaces.
   - Treat the canonical row unit as image-level, matching the giant TIFF image files on the external drive rather than collapsing immediately to biological-sample-level rows.
   - Define the canonical human image-level score as the median of the within-image replicate grades for workbook columns that carry repeated glomerular grades.
   - Preserve the raw within-image replicate vector in a sidecar ingest artifact rather than forcing it into the canonical manifest columns.
   - Require MR-specific preprocessing or tiling review during segmentation transport because the source images are roughly `25,000 x 9,000` whole-field TIFFs rather than Dox-scale JPGs.
   - Restrict phase 1 MR outputs to transport and concordance evaluation artifacts rather than training-set admission.
   - Run phase 1 concordance by tiling whole-field TIFFs, segmenting candidate glomeruli, extracting accepted ROI crops under repo-aligned component/union semantics, grading accepted ROIs, aggregating inferred ROI grades to an image-level median, and comparing that inferred median against the human image-level median.
   - Report concordance with fixed metrics including MAE, Spearman correlation, exact agreement, and within-one-step agreement, stratified by batch and treatment group when those splits are available.
   - Success for this cohort: a populated manifest with explicit image-level linkage, median-reduction provenance, raw replicate sidecar retention, unresolved/excluded states where image linkage cannot yet be proven, a documented MR preprocessing path, and a concordance report between human and inferred image-level medians.

## Success Criteria

This change is successful only if all of the following are true:

- Localized runtime cohort directories exist for `masked_core`, `vegfri_dox`, and `vegfri_mr` under `~/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/`.
- The unified `~/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/manifest.csv` is populated, not just an empty scaffold.
- Each cohort directory contains the localized working assets needed for future use rather than only a manifest pointing back into scattered source roots.
- Unified manifest rows for Dox explicitly separate recovered masked rows, recovered scored-only rows, foreign mixed rows, unresolved rows, and excluded rows.
- The runtime Dox cohort directory is populated with decoded brushlabel-derived mask PNGs and paired copied images under `raw_data/cohorts/vegfri_dox/` rather than leaving mask recovery as a speculative step.
- Verified masked Dox rows are eligible for segmentation improvement only after explicit mask-quality admission; they are not silently collapsed into masked-core or scored-only lanes.
- Unified manifest rows for MR explicitly document the replicate aggregation rule and preserve the runtime linkage fields needed to reproduce each score assignment.
- Unified manifest rows for MR explicitly use image-level row semantics, while separate ingest artifacts preserve the raw replicate vectors used to compute each image-level score.
- MR transport review explicitly addresses giant whole-field TIFF preprocessing or tiling rather than assuming Dox-scale inference behavior.
- MR phase 1 artifacts stop at external concordance and transport evaluation; MR is not silently added to training-set expansion.
- Dox is the only current external cohort eligible for phase 1 training-set expansion, but in two distinct ways: verified masked rows for segmentation improvement and scored-only rows for predicted-ROI grading expansion, each behind its own gates.
- The implementation writes a cohort-level summary of row counts by status so the user can see exactly what was recovered and what was not.
- Old overlapping runtime quantification-input directories are archived and documented as retired, and the new cohort tree becomes the only supported active input surface.
- Existing masked-core quantification behavior still passes regression checks and does not start depending on the new cohort workflow implicitly.
- Documentation states exactly what was done for each current cohort and what remains unresolved.

## Migration Plan

1. Finish the active glomeruli promotion and ordinal stabilization changes first.
2. Inventory each external scored cohort into one canonical row-level manifest with sample IDs, treatment groups, image counts, score coverage, and join anomalies.
3. Build runtime cohort directories for the currently relevant accessible scored quantification cohorts, including at minimum the existing masked-core cohort and the currently identified VEGFRi Dox and VEGFRi MR scored-only cohorts, under the active runtime root's `raw_data/cohorts/<cohort_id>/`.
4. Harmonize `masked_core` from the active preeclampsia runtime into the new manifest contract without regressing the current supported path.
5. Harmonize `vegfri_dox` from the latest dated export, recover brush masks and score choices where present, classify mixed foreign rows explicitly, and partition the reconciled rows into masked-versus-scored-only sublanes.
6. Harmonize `vegfri_mr` from the scoring workbook plus supporting logs and external TIFF batches, including a documented image-level median reduction and raw replicate sidecar retention.
7. Run iterative discovery and reconciliation passes across the accessible source surfaces until linkage either passes verification or is explicitly documented as still unresolved.
8. Materialize localized runtime cohort assets so each in-scope cohort has a self-contained working directory under `raw_data/cohorts/<cohort_id>/`.
9. Run the full high-fidelity mapping-verification bundle and exclude only the rows or cohort slices that still do not pass it after those discovery passes.
10. Run a segmentation transport audit on a stratified harmonized cohort slice and emit review artifacts under the active runtime root's `output/cohorts/<cohort_id>/`.
11. Admit verified masked external rows into segmentation improvement/evaluation only after mask-quality and verification gates pass, and admit scored-only rows into predicted-ROI grading inputs only after transport and verification gates pass.
12. For MR, generate only inference/concordance artifacts in phase 1 by comparing human image-level median against inferred image-level median on the copied giant TIFF images.
13. Archive older overlapping runtime quantification-input directories and mark them retired once the new cohort tree is verified.
14. Refactor shared path helpers so the repo resolves the new active runtime cohort surfaces consistently.
15. Keep existing masked-core grading artifacts unchanged; add scored-only predicted-ROI artifacts as a separate downstream input family.

## Open Questions

- Should harmonized scored-only cohorts reuse canonical `subject_image_id` semantics directly, or should they use a cohort-prefixed identifier that preserves external experiment provenance while matching repo naming style?
- Which source fields are mandatory for a row to be admitted: source workbook sheet and row, JSON task ID, treatment-assignment row, original filename, file hash, or some subset of those?
- What is the minimum cohort-specific transport evidence required for admission: fixed reviewed image count, reviewed subject count, or both?
- After phase 1 MR concordance is in place, what evidence would be sufficient to justify any later MR training admission, if any?
