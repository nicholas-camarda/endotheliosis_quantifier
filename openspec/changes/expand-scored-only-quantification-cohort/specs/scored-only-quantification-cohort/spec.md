## ADDED Requirements

### Requirement: Scored-only cohort inventory
The system SHALL inventory each scored-but-unmasked external cohort into a deterministic manifest before the cohort is used for downstream grading. The manifest SHALL record cohort-local runtime asset paths, experiment identifiers, filename and sample-ID join fields, treatment-group labels when available, score coverage, and any unmapped or foreign rows discovered during ingestion.

#### Scenario: Cohort manifest captures join anomalies
- **WHEN** a scored-only cohort export contains rows whose sample IDs or image names do not belong to the intended experiment
- **THEN** the manifest SHALL record those rows as unmapped or foreign and SHALL NOT silently mix them into the admitted cohort count

#### Scenario: Cohort manifest records treatment structure
- **WHEN** the cohort inventory finds experiment assignment metadata for treatment groups
- **THEN** the manifest SHALL record the discovered treatment groups and sample counts in concrete cohort terms

### Requirement: One unified manifest is the canonical linking surface
The system SHALL use one row-level manifest as the canonical linking surface for all localized external cohorts. The manifest SHALL be the only supported downstream source for score linkage, staged image paths, optional mask paths, treatment-group membership, provenance, verification state, lane assignment, and admission or exclusion status.

#### Scenario: Downstream grading consumes manifest rows
- **WHEN** any scored-only cohort rows are used for verification, transport audit, predicted-ROI generation, embeddings, or grading
- **THEN** those stages SHALL consume the canonical manifest rows rather than re-deriving joins from source workbooks, logs, or filenames

#### Scenario: Manifest supports both masked and unmasked cohorts
- **WHEN** an external manifest row has or does not have a verified mask
- **THEN** the manifest SHALL preserve a nullable mask field and explicit lane/admission fields rather than forcing the row into either the canonical masked-core contract or the scored-only predicted-ROI lane by default

#### Scenario: Manifest records admission states explicitly
- **WHEN** a scored-only manifest row has not passed harmonization, mapping verification, or cohort admission
- **THEN** the manifest SHALL mark that row as unresolved or excluded and SHALL NOT surface it as a grading-ready row

#### Scenario: Manifest stays runtime-native after copy
- **WHEN** a cohort has been consolidated into localized runtime assets
- **THEN** `manifest.csv` SHALL use runtime-local asset paths and SHALL NOT require original source paths as manifest fields

### Requirement: Unified manifest schema is explicit and unambiguous
The system SHALL define one explicit `manifest.csv` schema for the localized dataset under `raw_data/cohorts/manifest.csv`. The schema SHALL distinguish required input columns, optional input columns, and pipeline-generated columns, with documented nullability and uniqueness rules.

#### Scenario: Minimal required input columns are sufficient
- **WHEN** a user authors or repairs the unified manifest by hand
- **THEN** the required input columns SHALL be limited to `cohort_id`, `image_path`, `score`, and at least one score locator field from `source_score_row` or `source_sample_id`

#### Scenario: Optional input columns remain optional
- **WHEN** the unified manifest is authored from heterogeneous source files
- **THEN** columns such as `source_image_name`, `source_score_sheet`, `source_sample_id`, `source_score_row`, `score_path`, `treatment_group`, and `mask_path` SHALL remain optional unless the cohort-specific source structure requires them for deterministic joining

#### Scenario: Pipeline appends generated columns
- **WHEN** the ingestion and harmonization pipeline processes a valid input manifest
- **THEN** it SHALL append generated columns including `manifest_row_id`, `harmonized_id`, `join_status`, `verification_status`, `lane_assignment`, `admission_status`, `exclusion_reason`, `image_sha256`, and `mask_sha256`

#### Scenario: Enriched manifest schema is explicit
- **WHEN** the repository documents or validates the canonical enriched manifest
- **THEN** it SHALL treat the enriched column set as:
  - human-required: `cohort_id`, `image_path`, `score`, and at least one of `source_sample_id` or `source_score_row`
  - human-optional: `mask_path`, `treatment_group`, `source_score_sheet`, `score_path`, `source_image_name`
  - pipeline-generated: `manifest_row_id`, `harmonized_id`, `join_status`, `verification_status`, `lane_assignment`, `admission_status`, `exclusion_reason`, `image_sha256`, `mask_sha256`

#### Scenario: Image-level rows remain the canonical unit
- **WHEN** a cohort source associates multiple glomerular grades with one image file
- **THEN** the canonical manifest SHALL still use one row per image and SHALL represent any within-image replicate grading through explicit reduction metadata rather than creating one canonical row per replicate

#### Scenario: `harmonized_id` uses the minimal uniqueness set
- **WHEN** the pipeline derives `harmonized_id` for a cohort
- **THEN** it SHALL use the smallest discriminator set that makes the row unique within `cohort_id`, such as image-level identity alone for clean cohorts or source sample ID plus date or batch for cohorts with reused IDs

#### Scenario: Batch/date are included only when needed
- **WHEN** a cohort has reused sample identifiers across dated exports or scoring batches
- **THEN** the pipeline SHALL include the necessary batch/date discriminator fields in `harmonized_id` generation for that cohort while preserving those same values as explicit provenance columns

#### Scenario: Clean future cohorts are not forced to carry legacy discriminators
- **WHEN** a cohort's source image or sample identifiers are already unique within that cohort
- **THEN** the pipeline SHALL NOT require batch or date in `harmonized_id` generation merely because older cohorts needed them

#### Scenario: Manifest identity fields are unique
- **WHEN** the pipeline has produced an enriched unified manifest with multiple rows
- **THEN** `manifest_row_id` SHALL be globally unique within the file and `harmonized_id` SHALL be unique within `cohort_id`

#### Scenario: Admitted rows satisfy minimum completeness
- **WHEN** a manifest row has `admission_status=admitted`
- **THEN** that row SHALL have non-null `score`, non-null `image_path`, `verification_status=passed`, and an empty `exclusion_reason`

#### Scenario: Excluded or unresolved rows remain inspectable
- **WHEN** a manifest row is unresolved or excluded
- **THEN** the manifest SHALL preserve the best available runtime linkage fields and SHALL record the blocking `join_status`, `verification_status`, or `exclusion_reason` rather than dropping the row

#### Scenario: Manual mask provenance stays optional
- **WHEN** a cohort has no manual masks for a manifest row
- **THEN** `mask_path` SHALL remain null and the row SHALL NOT be rejected for that reason alone

#### Scenario: Missing generated columns are tolerated before ingestion
- **WHEN** a manifest has not yet been processed by the pipeline
- **THEN** the absence of generated columns such as `manifest_row_id`, `harmonized_id`, `image_path`, `join_status`, `verification_status`, `admission_status`, and `exclusion_reason` SHALL NOT make the input manifest invalid

### Requirement: Unresolved or excluded status requires iterative discovery
The system SHALL treat unresolved or excluded status as a conclusion reached after iterative discovery and reconciliation across the accessible source surfaces, not as the result of a single failed join attempt. The workflow SHALL record what source surfaces were checked before finalizing unresolved or excluded status.

#### Scenario: Recoverable linkage gets another pass
- **WHEN** a row fails an initial score-to-image join attempt but the cohort still has additional accessible source surfaces such as alternate score exports, workbook sheets, assignment logs, or filename clues
- **THEN** the system SHALL continue iterative discovery and reconciliation attempts before finalizing unresolved or excluded status

#### Scenario: Final unresolved status records search coverage
- **WHEN** a row or cohort remains unresolved after iterative discovery
- **THEN** the manifest or associated cohort artifact SHALL record which accessible source surfaces were checked and why linkage still failed

#### Scenario: Exclusion does not imply missingness on first pass
- **WHEN** a first-pass join attempt fails for a row that may still be recoverable from accessible source data
- **THEN** the system SHALL mark that row as pending additional discovery rather than immediately excluded

### Requirement: Cohort directory layout is flat, deterministic, localized, and backed by one manifest
The system SHALL organize every cohort under the active runtime root at `raw_data/cohorts/<cohort_id>/` with a deterministic, minimal layout. The cohort directories SHALL contain the localized cohort-owned runtime inputs without repeating project names or introducing extra cohort-wrapper trees. The canonical `manifest.csv` SHALL live once at `raw_data/cohorts/manifest.csv`.

#### Scenario: Cohort-owned localized inputs have one obvious home
- **WHEN** a cohort is materialized into the repository layout
- **THEN** its localized runtime inputs SHALL live under the active runtime root's `raw_data/cohorts/<cohort_id>/` with predictable subdirectories such as `images/`, `masks/`, `scores/`, and `metadata/` only as needed by that cohort

#### Scenario: Unified manifest sits at the cohort registry root
- **WHEN** the system writes the canonical manifest
- **THEN** it SHALL write `raw_data/cohorts/manifest.csv` under the active runtime root rather than creating per-cohort manifest files

#### Scenario: Source trees are not migrated in place
- **WHEN** the system builds the local cohort directory layout for scored quantification data
- **THEN** it SHALL NOT rename or reorganize the original PhD / cloud source trees in place as part of this change

#### Scenario: Localized cohort assets are copied, not moved
- **WHEN** the system consolidates source images, masks, score files, or metadata into `raw_data/cohorts/<cohort_id>/`
- **THEN** it SHALL copy those assets into the runtime cohort directory and SHALL NOT move them out of their original source locations

#### Scenario: Current accessible cohorts are materialized
- **WHEN** this change is implemented against the user's current accessible data holdings
- **THEN** the repository SHALL populate cohort directories and the unified manifest for the currently relevant accessible scored quantification cohorts rather than leaving the new layout empty

#### Scenario: Segmentation-install datasets remain separate
- **WHEN** the repository contains downloaded segmentation datasets such as Lucchi that are used only for segmentation installation or training layout
- **THEN** those datasets SHALL remain outside `raw_data/cohorts/<cohort_id>/` and SHALL NOT require `manifest.csv` under this change unless a future quantification change explicitly brings them into scope

#### Scenario: Repo-local data roots are not treated as active runtime storage
- **WHEN** the current machine's repo checkout has no active `data/raw_data` or `data/derived_data` storage while the active working state lives under `$EQ_RUNTIME_ROOT`
- **THEN** the cohort layout for this change SHALL target the active runtime root rather than creating a competing second data home inside the repo checkout

### Requirement: Repo-style cohort harmonization
The system SHALL harmonize each inventoried scored-only cohort into the repository's current naming, layout, and provenance styling before segmentation audit or downstream grading use. Harmonization SHALL produce deterministic staged identifiers, repo-style localized file organization, and a manifest that describes the copied runtime cohort without requiring original source paths.

#### Scenario: Heterogeneous external names are normalized into repo style
- **WHEN** a scored-only cohort uses experiment-specific image names, sample IDs, or folder layouts that do not follow the repository's canonical styling
- **THEN** the harmonization step SHALL assign deterministic repo-style identifiers and staged paths before any transport audit or grading dataset build

#### Scenario: Harmonization writes runtime-local asset references
- **WHEN** the system stages a scored-only cohort into repo-native organization
- **THEN** the harmonization manifest SHALL record the localized asset path, runtime-visible identifiers such as source image name or sample ID when needed, and the harmonized identifier for each staged item

#### Scenario: Unharmonizable rows are excluded explicitly
- **WHEN** a scored-only cohort row cannot be deterministically organized into the repo's current styling
- **THEN** the system SHALL mark that row as excluded in the harmonization manifest and SHALL NOT pass it forward as an admitted grading input

### Requirement: Localized cohort dataset is self-contained
The system SHALL build each in-scope runtime cohort directory as a self-contained localized working dataset rather than leaving the cohort as a manifest-only view over distributed source roots. Future pipeline stages and future users SHALL be able to work from the localized cohort directories plus the unified `manifest.csv` without re-discovering the original source layout.

#### Scenario: Consolidated cohort assets are present
- **WHEN** a cohort is brought under `raw_data/cohorts/<cohort_id>/`
- **THEN** the implementation SHALL materialize the localized image assets and any available mask, score, or metadata artifacts needed to support the cohort's intended workflow under that cohort directory

#### Scenario: Consolidated assets preserve source availability
- **WHEN** the localized cohort dataset has been built
- **THEN** the original source assets SHALL remain present at their source locations, but `manifest.csv` SHALL remain a runtime-local table rather than carrying those original paths

#### Scenario: Manifest maps runtime identifiers to localized assets
- **WHEN** a localized cohort row is written to `manifest.csv`
- **THEN** the row SHALL record the harmonized identifier, runtime-local asset paths, and any runtime-visible linkage fields needed to connect scores, images, and masks

#### Scenario: Future workflows do not depend on distributed source traversal
- **WHEN** a downstream stage such as segmentation inference, predicted-ROI generation, or grading dataset build runs against an admitted cohort
- **THEN** it SHALL be able to operate from the localized cohort directory and manifest without requiring ad hoc traversal of the original PhD / cloud directories

### Requirement: Model-derived artifacts live under result-family outputs
The system SHALL write model-derived artifacts under result-family directories, not under a generic cohort-named output tree. Segmentation artifacts such as mask-quality panels, transport audits, prediction reviews, and candidate-comparison evidence SHALL live under `output/segmentation_results/`. Quantification artifacts such as predicted ROI grading inputs, embeddings, grader outputs, and review reports SHALL live under `output/quantification_results/`.

#### Scenario: Segmentation artifact layout is deterministic
- **WHEN** the system materializes model-derived cohort artifacts
- **THEN** it SHALL write segmentation evidence to deterministic subdirectories under the runtime root's `output/segmentation_results/<result_or_cohort_id>/`

#### Scenario: Quantification artifact layout is deterministic
- **WHEN** the system materializes predicted-ROI grading, embedding, grader, or review artifacts
- **THEN** it SHALL write quantification evidence to deterministic subdirectories under the runtime root's `output/quantification_results/<cohort_id>/`

### Requirement: High-fidelity score-image mapping verification
The system SHALL require multiple high-fidelity verification checks before any harmonized scored-only row is admitted into downstream grading. Verification SHALL prove that each admitted staged image maps one-to-one to the intended score row and source identifier, record the evidence used for that decision, and fail closed on ambiguity or contradiction.

#### Scenario: Full admission-check bundle is required
- **WHEN** a row is marked `admission_status=admitted`
- **THEN** it SHALL have passed all of the following checks:
  - unique `harmonized_id` within `cohort_id`
  - exactly one runtime-local `image_path`
  - exactly one reconciled score assignment
  - no conflicting duplicate scores for the same runtime image
  - required cohort-specific locator fields present
  - runtime-local assets readable
  - runtime-local asset hashes recorded
  - sampled manual mapping review passed for the cohort

#### Scenario: Ambiguous score-to-image join is rejected
- **WHEN** a harmonized staged image matches more than one plausible source score row or source identifier
- **THEN** the system SHALL mark that item as excluded and SHALL record the ambiguity in a mapping-verification artifact rather than choosing a row heuristically

#### Scenario: Contradictory source fields block admission
- **WHEN** filename-derived identifiers, workbook sample IDs, treatment-assignment metadata, or other required source fields disagree for a staged item
- **THEN** the system SHALL mark that item as excluded and SHALL record which verification checks failed

#### Scenario: Verified row records row-level evidence
- **WHEN** a harmonized scored-only row passes mapping verification
- **THEN** the verification artifact SHALL record the harmonized identifier, runtime-local image path, runtime-local score linkage, any retained source-visible identifiers needed for adjudication, and any file-level integrity fields required by the cohort contract

#### Scenario: Duplicate score conflicts are not silently resolved
- **WHEN** the source cohort contains duplicate rows for the same staged image or biological unit with conflicting score values
- **THEN** the system SHALL block admission for that item until an explicit adjudication rule or exclusion decision is recorded

### Requirement: Duplicate and conflict adjudication is explicit
The system SHALL distinguish legitimate multiplicity from true duplicates and conflicts rather than treating every repeated identifier as the same failure mode.

#### Scenario: Same image with same score is deduplicated
- **WHEN** the same runtime image is discovered more than once with the same score value
- **THEN** the system SHALL retain one canonical row, preserve duplicate evidence in cohort artifacts, and SHALL NOT treat the repetition as an admission conflict

#### Scenario: Same biological sample with multiple images is allowed
- **WHEN** one biological sample maps to multiple runtime image files
- **THEN** the system SHALL allow multiple image-level manifest rows that share the cohort-visible sample identifier while preserving unique `harmonized_id` values per image

#### Scenario: One score row maps to multiple plausible images
- **WHEN** a reconciled score row still maps plausibly to more than one runtime image after iterative discovery
- **THEN** the system SHALL mark the affected row or rows unresolved rather than selecting one image heuristically

#### Scenario: Latest Dox export wins over older duplicates
- **WHEN** the same Dox task or image appears in multiple dated exports
- **THEN** the latest authoritative export SHALL supply the canonical row while older exports MAY be used only to recover missing fields

### Requirement: Segmentation transport audit gate
The system SHALL require a cohort-specific segmentation transport audit before any scored-only cohort is admitted into downstream grading. The audit SHALL identify the segmentation artifact used, stratify reviewed examples by cohort and treatment group, and record whether predictions remain non-degenerate and grading-usable on the audited slice.

#### Scenario: Cohort fails transport audit
- **WHEN** the segmenter produces degenerate, missing, or grading-invalid glomerulus predictions on the audited scored-only cohort slice
- **THEN** the cohort SHALL be marked excluded for grading use and the system SHALL write an explicit exclusion artifact with the failure reason

#### Scenario: Cohort passes transport audit
- **WHEN** the segmenter produces non-degenerate, review-approved glomerulus predictions on the audited harmonized scored-only cohort slice
- **THEN** the cohort SHALL be eligible for predicted-ROI grading artifact generation

#### Scenario: MR transport audit tests high-resolution preprocessing
- **WHEN** the cohort uses giant whole-field source images such as the MR TIFF batches
- **THEN** the transport audit SHALL explicitly record the preprocessing or tiling strategy used for inference and SHALL fail closed if the cohort cannot be processed reliably under that strategy

### Requirement: Masked external rows require explicit mask-quality admission
The system SHALL require an explicit mask-quality and contract check before recovered masked external rows are used for segmentation evaluation or training improvement. Recovered external masks SHALL NOT be treated as contract-equivalent to masked-core rows by default.

#### Scenario: Recovered masked Dox rows remain separate from masked-core
- **WHEN** the implementation ingests recovered masked Dox rows
- **THEN** it SHALL keep them in a distinct masked-external lane rather than silently merging them into the existing masked-core cohort

#### Scenario: Masked external rows fail mask-quality admission
- **WHEN** recovered external masks are unreadable, structurally implausible, or otherwise fail the mask-quality gate
- **THEN** those rows SHALL be excluded from segmentation improvement even if they remain usable for other documented purposes

#### Scenario: Masked external rows clear admission
- **WHEN** recovered external masks pass harmonization, mapping verification, and mask-quality review
- **THEN** those rows SHALL be eligible for external segmentation evaluation and, if explicitly allowed by the cohort policy, segmentation training augmentation

### Requirement: MR image-level scoring contract
The system SHALL treat MR kidney inputs as an image-level cohort with within-image repeated glomerular grades, not as a simple one-score-per-sample table. The canonical MR manifest rows SHALL map to the copied giant TIFF image files, while raw workbook replicates SHALL be preserved in a sidecar ingest artifact and reduced deterministically into one image-level median score per manifest row.

#### Scenario: MR row unit is image-level
- **WHEN** the implementation ingests the MR kidney cohort
- **THEN** each canonical MR manifest row SHALL correspond to one copied TIFF image rather than to one raw replicate value

#### Scenario: MR replicate reduction is explicit
- **WHEN** the implementation computes the canonical MR image-level score
- **THEN** it SHALL compute the median of the within-image replicate grades and SHALL record that reduction method in cohort artifacts

#### Scenario: MR raw replicate vector is preserved outside the manifest
- **WHEN** the implementation ingests the MR workbook replicates
- **THEN** it SHALL preserve the raw replicate values in a sidecar ingest artifact rather than flattening them away completely or forcing them into the canonical manifest columns

### Requirement: MR inferred ROI acceptance is repo-aligned
The system SHALL use repo-aligned inferred-ROI acceptance semantics for MR phase 1 evaluation rather than inventing a second ROI concept. Accepted MR inferred ROIs SHALL be derived from thresholded connected components that satisfy the quantification pipeline's minimum component-area gate, and the extracted ROI metadata SHALL follow the existing component/union ROI style used by the quantification pipeline.

#### Scenario: MR accepted ROI uses component-area gate
- **WHEN** the system converts MR segmentation outputs into candidate glomerulus ROIs
- **THEN** it SHALL reject thresholded components below the current minimum component-area gate used by the quantification pipeline

#### Scenario: MR accepted ROI metadata mirrors current quantification style
- **WHEN** the system extracts an accepted MR inferred ROI
- **THEN** it SHALL record component-count and component/union-style ROI metadata in the same style already used by the quantification pipeline for image-level ROI extraction

#### Scenario: MR image with zero accepted inferred ROIs is non-evaluable
- **WHEN** an MR image yields zero accepted inferred ROIs after tiling, segmentation, and component filtering
- **THEN** the system SHALL mark that image as non-evaluable for concordance rather than assigning it an inferred image-level grade

### Requirement: MR phase-1 use is external concordance only
The system SHALL treat MR as an external concordance and transport-evaluation cohort in phase 1 rather than as a training-expansion cohort. MR rows may be harmonized, verified, and used for segmentation inference and concordance reporting, but SHALL NOT be surfaced as training-admitted predicted-ROI grading rows in phase 1.

#### Scenario: MR is blocked from training admission in phase 1
- **WHEN** the implementation builds downstream training inputs from the unified manifest
- **THEN** it SHALL exclude `cohort_id=vegfri_mr` rows from training admission even if MR harmonization and transport audit succeed

#### Scenario: MR concordance compares human and inferred medians
- **WHEN** the implementation runs the phase 1 MR evaluation workflow
- **THEN** it SHALL compare the human image-level median derived from the workbook replicates against the inferred image-level median derived from segmented glomerulus predictions on the copied giant TIFF images

#### Scenario: MR inference path is explicit
- **WHEN** the implementation runs the phase 1 MR evaluation workflow
- **THEN** it SHALL tile the copied giant TIFF images, run segmentation on those tiles, extract accepted inferred glomerulus ROIs, grade the accepted ROIs, and aggregate inferred ROI grades to an image-level median before concordance reporting

#### Scenario: MR concordance metrics are fixed
- **WHEN** the implementation emits the phase 1 MR concordance report
- **THEN** it SHALL report at least MAE, Spearman correlation, exact agreement, and within-one-step agreement between human and inferred image-level medians, stratified by batch and treatment group when those strata are available

#### Scenario: MR outputs are evaluation artifacts
- **WHEN** the implementation completes for the current MR cohort
- **THEN** the output artifacts for `cohort_id=vegfri_mr` SHALL be transport-audit and concordance-evaluation artifacts rather than training-set expansion artifacts

### Requirement: Predicted-ROI grading artifact contract
The system SHALL represent admitted scored-only cohorts as predicted-ROI grading artifacts that remain distinct from manual-mask quantification artifacts. These artifacts SHALL record image provenance, predicted ROI provenance, score provenance, and the segmentation artifact identity used to generate the ROI inputs.

#### Scenario: Predicted-ROI artifacts preserve segmentation provenance
- **WHEN** the system generates grading inputs for an admitted scored-only cohort
- **THEN** each output row SHALL identify the source image, unified-manifest row, predicted ROI artifact, and segmentation artifact provenance rather than presenting the cohort as manual-mask ground truth

#### Scenario: Manual-mask and predicted-ROI artifacts stay separate
- **WHEN** the repository prepares downstream grading inputs
- **THEN** scored-only predicted-ROI artifacts SHALL remain distinguishable from canonical masked-core artifacts in both file outputs and provenance fields

### Requirement: Admission and exclusion states are explicit
The system SHALL persist explicit admission or exclusion decisions for each scored-only cohort. A cohort that is not admitted SHALL NOT be silently included in downstream grading datasets, reports, or evaluation summaries.

#### Scenario: Excluded cohort is blocked from grading dataset build
- **WHEN** a cohort has no admission artifact or has an explicit exclusion artifact
- **THEN** the grading dataset build SHALL omit that cohort and SHALL report the omission reason

#### Scenario: Admitted cohort is reported separately from masked-core data
- **WHEN** a grading dataset includes an admitted scored-only cohort
- **THEN** the resulting report SHALL state that the cohort entered through the scored-only predicted-ROI path and SHALL NOT describe it as masked segmentation supervision

### Requirement: Dox is the first external cohort with two admission lanes
The system SHALL treat `cohort_id=vegfri_dox` as the first external cohort with both:
- a masked-external lane for recovered verified masks that may improve segmentation evaluation or training
- a scored-only lane for predicted-ROI grading expansion when verified masks are absent

#### Scenario: Dox mask recovery prefers direct Label Studio export
- **WHEN** the authoritative Dox Label Studio project is accessible from the user’s setup
- **THEN** the system SHALL prefer direct Label Studio mask export over reconstructed brushlabel decoding for mask recovery

#### Scenario: Existing decoded Dox runtime masks are reused
- **WHEN** the current machine already has decoded Dox brushlabel masks materialized under `$EQ_RUNTIME_ROOT/raw_data/cohorts/vegfri_dox/`
- **THEN** the implementation SHALL treat that runtime-local `images/`, `masks/`, and `metadata/decoded_brushlabel_masks.csv` surface as the starting point for Dox mask-aware reconciliation rather than rebuilding an alternate active mask tree elsewhere

#### Scenario: Verified masked Dox rows can improve segmentation
- **WHEN** `cohort_id=vegfri_dox` rows have verified masks and clear the mask-quality and verification gates
- **THEN** those rows SHALL be eligible for masked-external segmentation evaluation and, if explicitly approved by the workflow, segmentation training augmentation

#### Scenario: All admitted masked rows can train one glomeruli candidate
- **WHEN** the glomeruli trainer is pointed at the active runtime `raw_data/cohorts` registry root
- **THEN** it SHALL enumerate admitted rows from the manual-mask and masked-external manifest lanes across current cohorts, including the active preeclampsia masked-core rows and approved Dox masked-external rows, while excluding scored-only, foreign, unresolved, and MR concordance-only rows from segmentation supervision

#### Scenario: Requested transfer training fails closed on base-load failure
- **WHEN** a glomeruli transfer run is given a base model path that cannot load in the active environment or cannot contribute any compatible weights
- **THEN** the run SHALL stop with an explicit error and SHALL NOT proceed as a no-mitochondria-base scratch candidate; the no-base ImageNet-initialized comparator SHALL require the explicit scratch-training path

#### Scenario: Scored-only Dox rows remain grading-only
- **WHEN** `cohort_id=vegfri_dox` rows do not have verified masks but clear transport and verification gates
- **THEN** those rows SHALL be eligible only for the scored-only predicted-ROI grading lane

### Requirement: Current data holdings are built out under the new contract
The system SHALL apply the cohort-first layout and unified manifest contract to the user's currently relevant accessible scored quantification data holdings, not just to future cohorts. At minimum, implementation SHALL cover Lauren's preeclampsia cohort, the currently identified VEGFRi Dox cohort, and the VEGFRi MR scored-only cohort, with explicit exclusion or unresolved states where a cohort cannot yet pass the required checks.

#### Scenario: Current cohorts receive unified-manifest rows
- **WHEN** the implementation processes the user's current accessible cohorts
- **THEN** each such cohort SHALL receive localized cohort-owned runtime directories under the active runtime root's `raw_data/cohorts/<cohort_id>/` and populated rows in `raw_data/cohorts/manifest.csv`

#### Scenario: Unsupported current cohorts remain explicit
- **WHEN** one of the current accessible cohorts cannot be fully harmonized, verified, or admitted
- **THEN** the implementation SHALL still create the cohort directory and manifest with explicit unresolved or excluded rows rather than skipping that cohort silently

### Requirement: Cohort-specific execution is explicit
The system SHALL implement a cohort-specific rollout plan for the current accessible data rather than relying on one generic ingestion path to describe all cases.

#### Scenario: Lauren preeclampsia is named by cohort identity
- **WHEN** the implementation processes Lauren's active preeclampsia runtime data
- **THEN** it SHALL produce `cohort_id=lauren_preeclampsia` rows in the unified manifest with `lane_assignment=manual_mask_core`
- **AND** it SHALL NOT use a generic cohort identity such as `masked_core` to encode mask availability

#### Scenario: Manual-mask lanes distinguish role, not drawing method
- **WHEN** Lauren preeclampsia and Dox rows both have manually drawn Label Studio-style masks
- **THEN** the manifest SHALL distinguish them as `manual_mask_core` versus `manual_mask_external` based on cohort role and admission gate rather than implying different mask provenance

#### Scenario: `vegfri_dox` recovers both scores and brush masks
- **WHEN** the implementation processes the current Dox Label Studio data
- **THEN** it SHALL start from the latest dated export, recover image-level score choices and brushlabel glomerulus masks where present, explicitly classify foreign mixed rows rather than silently treating the export as a clean single cohort, and partition reconciled Dox rows into `manual_mask_external` versus `scored_only` lanes

#### Scenario: `vegfri_mr` documents replicate reduction
- **WHEN** the implementation processes the current MR scoring workbook
- **THEN** it SHALL document the image-level median reduction used to produce manifest rows, preserve the raw replicate vector in sidecar ingest artifacts, and explicitly acknowledge the giant whole-field TIFF acquisition regime used by the copied MR images

### Requirement: Success is artifact- and count-based
The system SHALL define success for this change in terms of populated cohort artifacts, explicit status counts, and masked-core regression safety, not merely in terms of shipping reusable code.

#### Scenario: Unified manifest is required
- **WHEN** the change is considered complete
- **THEN** `raw_data/cohorts/manifest.csv` SHALL contain populated rows for each in-scope current cohort rather than leaving the runtime layout as directories alone

#### Scenario: Localized cohort datasets are required
- **WHEN** the change is considered complete
- **THEN** each in-scope current cohort SHALL have localized working assets under its runtime cohort directory rather than only a manifest that points into distributed external source roots

#### Scenario: Recovery counts are reported
- **WHEN** the implementation completes for the current cohorts
- **THEN** it SHALL emit cohort-level counts that show how many rows were recovered as masked-scored, scored-only, unresolved, excluded, or foreign where those categories apply

#### Scenario: MR preprocessing path is reported
- **WHEN** the implementation completes for the current MR cohort
- **THEN** it SHALL report the preprocessing or tiling path used for giant TIFF handling and the concordance between human image-level median and inferred image-level median in addition to the normal recovery counts

#### Scenario: Existing masked-core behavior remains intact
- **WHEN** the change is considered successful
- **THEN** regression checks SHALL confirm that the pre-existing masked-core quantification path still works without depending implicitly on the new unified manifest unless explicitly routed through it

### Requirement: Old runtime input surfaces are archived and retired
The system SHALL archive overlapping pre-existing runtime quantification-input directories once the new cohort tree is populated and verified, and SHALL treat those archived directories as retired reference surfaces rather than supported active inputs.

#### Scenario: Archived runtime inputs are not reused
- **WHEN** the new runtime cohort tree has been built and verified
- **THEN** downstream cohort-aware workflows SHALL use only the new cohort tree and SHALL NOT read retired overlapping runtime input directories as active sources

### Requirement: Shared repo path helpers use the new runtime contract
The system SHALL refactor the shared repo path helpers so the active runtime root, unified cohort manifest, cohort input tree, segmentation result tree, and quantification result tree are resolved consistently across the repository.

#### Scenario: Shared path helpers resolve active cohort surfaces
- **WHEN** a repo workflow requests the runtime root or cohort data/output locations
- **THEN** the shared path helpers SHALL resolve the active `$EQ_RUNTIME_ROOT` cohort surfaces rather than defaulting to placeholder repo-local `data/` roots for this workflow

### Requirement: Touched pipeline surfaces have regression coverage
The system SHALL add explicit tests for every quantification pipeline surface changed by the scored-only cohort workflow and SHALL preserve the existing masked-core contract behavior.

#### Scenario: New scored-only logic does not break masked-core scoring
- **WHEN** scored-only manifest, harmonization, verification, audit, or predicted-ROI features are added
- **THEN** regression tests SHALL confirm that the existing canonical masked-core quantification path still produces the expected scored-example joins and does not start reading scored-only artifacts implicitly

#### Scenario: Stage-level artifacts are validated
- **WHEN** the system writes scored-only inventory, manifest, harmonization, mapping-verification, transport-audit, or predicted-ROI artifacts
- **THEN** tests SHALL validate both the artifact schema and the expected failure behavior for invalid or excluded rows
