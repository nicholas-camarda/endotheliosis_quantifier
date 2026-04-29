## Context

The current quantification stack already contains the surfaces needed for a label-free atlas:

- ROI crops and scored row provenance: `roi_crops/roi_scored_examples.csv`
- frozen glomeruli encoder embeddings: `embeddings/roi_embeddings.csv`
- learned ROI feature tables: `burden_model/learned_roi/feature_sets/learned_roi_features.csv`
- ROI/QC features and identity columns used by supervised burden and grade-model candidates
- review evidence patterns in `src/eq/quantification/feature_review.py`, `src/eq/quantification/learned_roi_review.py`, and the generated `quantification_review/` subtree
- workflow dispatch and durable logging through `eq run-config`

The supervised grade-model path is useful but still tied to human image-level labels. Recent rubric-review experiments show that label adjudication can materially change model readiness. That means a new workflow should not be framed as "better prediction of current labels." It should be a label-blinded structure-discovery and review-prioritization workflow that helps decide whether the image data contain reproducible morphology groups that a revised grading rubric can anchor.

The atlas must stay separate from existing supervised outputs. It should not replace `primary_burden_index`, `learned_roi`, `source_aware_estimator`, `severe_aware_ordinal_estimator`, or `endotheliosis_grade_model`.

## Goals / Non-Goals

**Goals:**

- Build a reproducible label-free ROI embedding atlas from existing ROI/embedding artifacts.
- Discover morphology clusters and low-dimensional structure without using human grade, cohort, source, treatment, or reviewer fields during clustering.
- Reveal human grade and source metadata only after cluster assignments are finalized.
- Quantify cluster stability with subject-aware resampling and method sensitivity checks.
- Separate possible severity-like morphology groups from artifact, cohort/source, mask-geometry, or low-quality clusters.
- Generate reviewer-facing evidence and a prioritized adjudication queue for cases where cluster-neighbor structure disagrees with original labels or reviewed anchors.
- Keep every output traceable to exact ROI rows, feature columns, source tables, methods, package versions, and runtime root.
- Preserve current supervised workflows unchanged when atlas artifacts are absent.

**Non-Goals:**

- Do not produce calibrated probabilities for no-low, mid-mod, or mod-severe classes.
- Do not relabel rows automatically.
- Do not treat clusters as ground truth severity classes.
- Do not add new manual labels or require external pathology anchors before the workflow can run.
- Do not expand segmentation training, regenerate masks, or promote any segmentation model.
- Do not use t-SNE as the primary clustering space or probability surface.
- Do not weaken existing supervised grade-model gates.

## Decisions

### 1. Add a separate atlas workflow instead of folding this into P3

The implementation will add `label_free_roi_embedding_atlas` as a run-config workflow backed by `src/eq/quantification/embedding_atlas.py`.

Rationale: P3 is a supervised grade-model product-selection workflow. The atlas has a different estimand: descriptive morphology structure and review prioritization. Keeping it separate prevents label-free discovery artifacts from being accidentally promoted as supervised prediction evidence.

Alternatives considered:

- Extend P3 directly. Rejected because P3 selection, thresholds, severe gates, and deployment smoke tests are supervised and label-dependent.
- Extend `learned_roi.py` directly. Rejected because learned ROI is already a supervised candidate screen with readiness gates tied to human score coverage.
- Create a notebook. Rejected because the output needs durable runtime artifacts, testable contracts, and reproducible run logging.

### 2. Reuse the current quantification output root and add one role-specific subtree

The atlas will write under `burden_model/embedding_atlas/` inside an existing quantification output root. It will use grouped subfolders:

- `summary/`
- `feature_space/`
- `clusters/`
- `stability/`
- `diagnostics/`
- `evidence/`
- `review_queue/`

Rationale: This follows the existing burden-model artifact layout while making clear that the atlas is not a deployed supervised model.

Alternatives considered:

- Write under `quantification_review/`. Rejected because the atlas includes machine-readable feature-space and cluster artifacts, not only review HTML.
- Write under `learned_roi/`. Rejected because learned ROI has supervised candidate semantics.
- Write at top-level `embedding_atlas/`. Rejected because quantification-derived model evidence belongs under `burden_model/`.

### 3. Use explicit label blinding

The atlas will construct feature matrices from approved feature columns only. It will explicitly exclude:

- `score`
- banded labels and severe indicators
- `cohort_id`
- `lane_assignment`
- treatment/source fields
- reviewer/adjudication columns
- path strings and raw identifiers

Those excluded fields may be preserved as metadata and joined after clustering for diagnostics.

Rationale: This makes the clustering independent of the user's human grades and avoids accidentally discovering cohort or reviewer labels through direct feature leakage.

Alternatives considered:

- Allow grade-stratified dimensionality reduction. Rejected because it would contaminate the label-free claim.
- Allow cohort-aware clustering. Rejected for the clustering phase because cohort/source separation is a confounding diagnostic, not the target structure.

### 4. Use PCA-denoised and standardized spaces for clustering, not t-SNE

The workflow will produce at least these feature spaces:

- `encoder_standardized`
- `encoder_pca`
- `learned_roi_standardized` when learned ROI features exist
- `roi_qc_standardized`

Clustering will run in original standardized or PCA-denoised spaces. UMAP may be used for visualization if available. t-SNE may be generated only as optional visualization if explicitly enabled, but it is not an approved cluster-input method for this change.

Rationale: PCA and standardized feature spaces are reproducible and auditable enough for cluster stability checks. t-SNE can distort distances and neighborhood relationships, making it inappropriate as the primary analytic surface.

Alternatives considered:

- t-SNE clustering. Rejected because t-SNE geometry is not stable enough for primary cluster assignments.
- Raw unscaled embeddings only. Rejected because scale and near-zero-variance features need explicit diagnostics.

### 5. Use method availability audits instead of fallback substitution

The workflow will audit method availability before fitting:

- required: `sklearn` PCA, k-means, Gaussian mixture, nearest neighbors
- optional: `hdbscan`
- optional visualization: `umap-learn`

Unavailable optional methods will be recorded in `diagnostics/method_availability.json`. A missing optional method will not be silently replaced under the same method ID.

Rationale: The repo forbids fallback and patchwork logic. The correct behavior is to record method availability and run only methods with truthful IDs.

Alternatives considered:

- Always require HDBSCAN and UMAP. Rejected because the core label-free atlas can run without those optional packages.
- Silently replace HDBSCAN with DBSCAN. Rejected because that would make method identity untruthful.

### 6. Stability is subject-aware

Cluster stability will be evaluated with subject-aware resampling where possible. The primary stability unit is `subject_id`. Stability artifacts will record:

- resampling unit
- number of resamples
- row and subject support
- adjusted Rand index or another explicit assignment stability metric
- cluster-level recurrence
- non-estimable reasons

Rationale: repeated ROI/image rows from the same subject are not independent. Stability claims must not treat 707 ROI rows as 707 independent biological units.

Alternatives considered:

- row bootstrap only. Rejected as primary evidence because it inflates stability for repeated images from the same subject.
- no stability checks. Rejected because cluster maps without stability are not actionable.

### 7. Cluster interpretation is post hoc and gated

After cluster assignments are finalized, diagnostics will reveal:

- original score distribution
- severe/nonsevere distribution
- cohort/source/treatment distribution
- ROI/QC summaries
- mask/ROI adequacy indicators
- reviewed anchor distribution when available

The atlas verdict will classify clusters as one of:

- `candidate_morphology_group`
- `candidate_severity_like_group`
- `artifact_or_quality_group`
- `source_sensitive_group`
- `unstable_group`
- `insufficient_support`

Rationale: a cluster can be biologically interesting, severity-like, or merely an artifact/source separator. The workflow should make that distinction explicit before review artifacts are used to guide a rubric.

Alternatives considered:

- Map every cluster to a severity band automatically. Rejected because unsupervised clusters do not have intrinsic disease-severity meaning.
- Ignore source and ROI/QC distributions. Rejected because source-sensitive clusters are a known failure mode.

### 8. Generate evidence and review queues, not automatic labels

The workflow will generate:

- cluster representative tables
- ROI montage assets
- nearest-neighbor and boundary-example tables
- an atlas HTML review report
- a cluster disagreement queue
- a severe-boundary review queue
- an anchor-expansion queue if reviewed anchors exist

Rationale: the best immediate use of the atlas is to make human adjudication more reproducible and efficient, not to replace adjudication.

Alternatives considered:

- Write label overrides directly. Rejected because this would automate relabeling from an exploratory unsupervised method.
- Only output numeric cluster labels. Rejected because cluster assignments without representative evidence are not scientifically interpretable.

### 9. Use shared utilities where they already own the concept

Implementation should first inspect and reuse:

- `src/eq/run_config.py` for workflow dispatch
- `src/eq/utils/paths.py` for runtime path resolution
- `src/eq/utils/execution_logging.py` for durable run logs
- `src/eq/quantification/modeling_contracts.py` for finite numeric matrices, warning capture, and JSON writing where applicable
- `src/eq/quantification/embeddings.py` for embedding provenance conventions
- `src/eq/quantification/learned_roi.py` for learned feature table conventions
- `src/eq/quantification/feature_review.py` and `src/eq/quantification/learned_roi_review.py` for review/report rendering patterns

New code belongs in `src/eq/quantification/embedding_atlas.py` because no existing module owns label-free clustering, cluster stability, and atlas review queues.

## Risks / Trade-offs

- [Risk] Clusters reflect cohort, stain, scanner, mask geometry, or RBC contamination rather than severity. -> Mitigation: post hoc cohort/source/ROI-QC diagnostics are required, source-sensitive clusters cannot be severity-like without review, and the verdict records artifact/source blocks.
- [Risk] The atlas is mistaken for calibrated severity prediction. -> Mitigation: first-read verdict and review report must state that clusters are descriptive morphology groups, not class probabilities or validated disease severity.
- [Risk] Human labels leak into the clustering feature matrix through derived columns. -> Mitigation: explicit deny-list and allow-list checks must fail closed when label/source/reviewer columns enter feature matrices.
- [Risk] High-dimensional embeddings create unstable or numerically degenerate structure. -> Mitigation: feature diagnostics, PCA-denoised spaces, finite-matrix checks, and subject-aware stability summaries are required.
- [Risk] Optional packages differ between macOS and WSL environments. -> Mitigation: method availability is audited and recorded; optional methods run only under truthful method IDs.
- [Risk] Review montages expose broken ROI paths or stale runtime artifacts. -> Mitigation: required provenance checks fail closed for missing ROI image paths and record the exact missing path counts.
- [Risk] Cluster review creates pressure to auto-relabel the dataset. -> Mitigation: review queues are advisory artifacts; label override files remain explicit human-reviewed inputs owned by the supervised quantification workflow.

## Migration Plan

1. Add the new workflow and config without changing existing supervised configs.
2. Run the atlas against the current full-cohort quantification output root.
3. Verify all atlas artifacts are written under `burden_model/embedding_atlas/`.
4. Verify existing `configs/endotheliosis_quantification.yaml` behavior remains unchanged when the atlas is absent.
5. If the atlas fails, preserve failure diagnostics under `embedding_atlas/summary/atlas_verdict.json` and keep supervised outputs unaffected.

No data migration is required. Existing runtime outputs remain valid. The atlas is an additive generated artifact family.

## Explicit Decisions

- Workflow ID: `label_free_roi_embedding_atlas`.
- Config: `configs/label_free_roi_embedding_atlas.yaml`.
- Primary module: `src/eq/quantification/embedding_atlas.py`.
- Output root: `burden_model/embedding_atlas/`.
- Required first-read verdict: `burden_model/embedding_atlas/summary/atlas_verdict.json`.
- Required reviewer report: `burden_model/embedding_atlas/evidence/embedding_atlas_review.html`.
- Required cluster assignment table: `burden_model/embedding_atlas/clusters/cluster_assignments.csv`.
- Required review queue: `burden_model/embedding_atlas/review_queue/atlas_adjudication_queue.csv`.
- Required method audit: `burden_model/embedding_atlas/diagnostics/method_availability.json`.
- Clustering feature matrices must be label-blinded.
- Human scores are diagnostic metadata only in this workflow.
- Cluster membership is descriptive evidence, not severity probability.

## Open Questions

- [audit_first_then_decide] Which existing report renderer should own atlas HTML output? Decide by inspecting `src/eq/quantification/feature_review.py`, `src/eq/quantification/learned_roi_review.py`, and generated review HTML conventions before implementation.
- [audit_first_then_decide] Whether `hdbscan` and `umap-learn` are available in the active environments. Decide from `diagnostics/method_availability.json` generated by import audits.
- [defer_ok] Whether an external pathology-reviewed anchor table should become a future input. This change emits an anchor-ready review queue but does not require the anchor table.
