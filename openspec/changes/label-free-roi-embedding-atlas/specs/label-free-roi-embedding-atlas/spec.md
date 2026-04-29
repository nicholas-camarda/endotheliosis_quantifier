## ADDED Requirements

### Requirement: Label-free atlas runs through the supported config entrypoint
The system SHALL expose a `label_free_roi_embedding_atlas` workflow through `eq run-config` and SHALL write atlas artifacts under the configured quantification output root without changing existing supervised quantification outputs.

#### Scenario: Atlas workflow runs from YAML config
- **WHEN** `eq run-config --config configs/label_free_roi_embedding_atlas.yaml` is executed
- **THEN** the workflow SHALL dispatch to the label-free ROI embedding atlas implementation
- **AND** it SHALL write artifacts under `burden_model/embedding_atlas/`
- **AND** it SHALL use the configured runtime root and quantification output root rather than writing generated artifacts under the Git checkout

#### Scenario: No direct CLI alias is added
- **WHEN** this change is implemented
- **THEN** the atlas SHALL use `eq run-config --config configs/label_free_roi_embedding_atlas.yaml` as the sole supported entrypoint
- **AND** it SHALL NOT add a direct `eq` CLI alias for the atlas workflow

#### Scenario: Existing quantification remains independent
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` is executed without atlas configuration
- **THEN** the supervised quantification workflow SHALL NOT require `burden_model/embedding_atlas/` artifacts
- **AND** missing atlas artifacts SHALL NOT block primary burden, learned ROI, source-aware, severe-aware, or P3 grade-model outputs

### Requirement: Atlas inputs preserve ROI identity and provenance
The atlas workflow SHALL read existing ROI, embedding, and learned ROI artifacts while preserving row identity, ROI path provenance, feature provenance, and biological grouping fields needed for diagnostics.

#### Scenario: Required embedding input is loaded
- **WHEN** the atlas workflow starts from a completed quantification run
- **THEN** it SHALL load an embedding table such as `embeddings/roi_embeddings.csv`
- **AND** the table SHALL include finite embedding feature columns and row identity sufficient to preserve `subject_id`, `subject_image_id`, `cohort_id` when present, ROI image path provenance, and original `score` when present

#### Scenario: Stale ROI or embedding provenance blocks clustering
- **WHEN** ROI or embedding input artifacts lack hardened ROI geometry, preprocessing, threshold, ROI status, or artifact-provenance fields from the completed fail-closed quantification contract
- **THEN** the atlas SHALL fail before clustering
- **AND** `burden_model/embedding_atlas/summary/atlas_verdict.json` SHALL record the stale or incomplete provenance fields

#### Scenario: Missing identity fails closed
- **WHEN** the atlas input table lacks required identity columns needed for subject-aware stability or ROI review evidence
- **THEN** the workflow SHALL fail before clustering
- **AND** `burden_model/embedding_atlas/summary/atlas_verdict.json` SHALL record the missing columns and failure status

#### Scenario: Optional learned ROI features are joined truthfully
- **WHEN** `burden_model/learned_roi/feature_sets/learned_roi_features.csv` is present and join keys are unique
- **THEN** the atlas MAY include learned ROI feature spaces
- **AND** the feature-space metadata SHALL record source table path, join keys, row counts, feature counts, and any excluded rows
- **AND** duplicate or ambiguous joins SHALL block the affected learned ROI feature space rather than silently falling back to image-level joins

### Requirement: Clustering feature matrices are label-blinded
The atlas workflow SHALL exclude human labels and source-identifying metadata from clustering feature matrices and SHALL use those fields only after cluster assignments are finalized.

#### Scenario: Human scores are withheld during clustering
- **WHEN** feature matrices are constructed for clustering
- **THEN** columns such as `score`, grade bands, severe indicators, reviewer scores, label overrides, adjudication fields, and prediction targets SHALL be excluded
- **AND** the workflow SHALL write `burden_model/embedding_atlas/diagnostics/label_blinding_audit.json`
- **AND** the audit SHALL list excluded label-like columns, approved feature columns, metadata-only columns, and whether any denied column entered a cluster feature matrix

#### Scenario: Source metadata is withheld during clustering
- **WHEN** feature matrices are constructed for clustering
- **THEN** columns such as `cohort_id`, `lane_assignment`, source workbook fields, treatment fields, path strings, and reviewer metadata SHALL be excluded from clustering features
- **AND** those fields MAY be preserved as metadata for post hoc source-sensitivity diagnostics

#### Scenario: Leakage blocks cluster fitting
- **WHEN** a denied label, source, treatment, reviewer, path, or target column is detected in a cluster feature matrix
- **THEN** the workflow SHALL stop before cluster fitting
- **AND** the atlas verdict SHALL mark the run failed with a label-leakage or source-leakage reason

#### Scenario: Unapproved feature lineage blocks clustering
- **WHEN** a numeric clustering feature lacks approved lineage proving it derives from frozen encoder embeddings, ROI/QC measurements, or learned ROI features
- **THEN** the workflow SHALL stop before cluster fitting
- **AND** the label-blinding audit SHALL record the unapproved feature and its source artifact

### Requirement: Atlas feature spaces are explicit and finite
The atlas SHALL generate named feature-space artifacts with finite numeric matrices and diagnostics before clustering.

#### Scenario: Required feature spaces are written
- **WHEN** atlas feature preparation succeeds
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/feature_space/feature_space_manifest.json`
- **AND** it SHALL include at least `encoder_standardized`, `encoder_pca`, and `roi_qc_standardized` when their source columns are present
- **AND** it SHALL include `learned_roi_standardized` only when learned ROI feature columns are available and valid

#### Scenario: Feature diagnostics are written
- **WHEN** a feature space is prepared
- **THEN** the workflow SHALL write feature diagnostics that include row count, subject count, feature count, missingness, nonfinite counts, zero-variance counts, near-zero-variance counts, scaling policy, PCA component count where applicable, and package versions

#### Scenario: Nonfinite selected features fail closed
- **WHEN** a selected feature space contains nonfinite values after approved preprocessing
- **THEN** that feature space SHALL be excluded or the workflow SHALL fail with an explicit reason
- **AND** the workflow SHALL NOT impute or silently coerce nonfinite values without recording the preprocessing method and affected counts

### Requirement: Atlas method availability is audited
The atlas workflow SHALL audit clustering and visualization method availability and SHALL not silently substitute one method for another under the same method ID.

#### Scenario: Method availability artifact is written
- **WHEN** the atlas workflow initializes methods
- **THEN** it SHALL write `burden_model/embedding_atlas/diagnostics/method_availability.json`
- **AND** the artifact SHALL record availability, package/module version where available, method role, fit eligibility, and failure reason for each configured method

#### Scenario: Required methods are available or the workflow fails
- **WHEN** required sklearn methods for PCA, k-means, Gaussian mixture, and nearest-neighbor review are unavailable
- **THEN** the workflow SHALL fail before clustering
- **AND** the atlas verdict SHALL identify the missing required method

#### Scenario: Optional methods are truthfully skipped
- **WHEN** optional methods such as `hdbscan` or `umap-learn` are unavailable
- **THEN** the workflow SHALL record them as unavailable
- **AND** it SHALL NOT write outputs using those method IDs
- **AND** it SHALL NOT silently replace them with another method under the same method ID

### Requirement: Clustering is performed in approved feature spaces
The atlas SHALL compute cluster assignments in standardized or PCA-denoised feature spaces and SHALL not use t-SNE as a primary clustering or probability surface.

#### Scenario: Cluster assignments are written
- **WHEN** clustering completes for an approved method and feature space
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/clusters/cluster_assignments.csv`
- **AND** each row SHALL include row identity, `subject_id`, feature-space ID, cluster method ID, cluster ID, assignment confidence or distance where available, outlier/noise status where applicable, and no inferred severity label

#### Scenario: t-SNE is not a cluster input
- **WHEN** optional t-SNE visualization is enabled in a future config
- **THEN** t-SNE coordinates SHALL be labeled visualization-only
- **AND** the workflow SHALL NOT use t-SNE coordinates as the primary input for cluster assignment, cluster stability, or severity probabilities

#### Scenario: Multiple cluster resolutions are recorded
- **WHEN** a method evaluates multiple cluster counts or density parameters
- **THEN** the workflow SHALL record the parameter grid, selected parameter set, selection rule, and non-selected alternatives in machine-readable artifacts
- **AND** parameter selection SHALL use label-free structure and stability criteria, not human grade performance

### Requirement: Cluster stability is subject-aware
The atlas SHALL evaluate cluster stability with biological grouping awareness and SHALL report row and subject support separately.

#### Scenario: Stability artifacts are written
- **WHEN** cluster stability is evaluated
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/stability/cluster_stability.json`
- **AND** it SHALL record resampling unit, number of resamples, row count, subject count, feature-space ID, cluster method ID, stability metrics, and non-estimable reasons

#### Scenario: Subject grouping is primary
- **WHEN** `subject_id` is present
- **THEN** stability resampling SHALL treat `subject_id` as the primary biological grouping unit
- **AND** the report SHALL state row count and independent subject count separately
- **AND** row-level bootstrap stability SHALL NOT be the sole readiness evidence

#### Scenario: Unstable clusters are blocked from severity-like interpretation
- **WHEN** a cluster or method family fails configured stability support
- **THEN** the cluster interpretation SHALL be marked `unstable_group` or `insufficient_support`
- **AND** the atlas SHALL NOT describe that cluster as a candidate severity-like phenotype

### Requirement: Cluster interpretation is post hoc and source-aware
The atlas SHALL reveal labels, source metadata, and ROI/QC summaries only after cluster assignments are fixed and SHALL use those diagnostics to block artifact or source-sensitive severity claims.

#### Scenario: Post hoc cluster diagnostics are written
- **WHEN** cluster assignments are finalized
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/diagnostics/cluster_posthoc_diagnostics.json`
- **AND** the artifact SHALL include original score distribution, severe/nonsevere distribution where score exists, cohort/source distribution, ROI/QC summaries, mask/ROI adequacy indicators, and reviewed-anchor distribution where available

#### Scenario: Source-sensitive clusters are labeled as such
- **WHEN** a cluster is dominated by cohort, lane, source batch, treatment proxy, mask geometry, ROI size, RBC-heavy flags, low-quality flags, or other artifact indicators
- **THEN** the cluster SHALL be labeled `source_sensitive_group` or `artifact_or_quality_group`
- **AND** the atlas verdict SHALL block severity-like interpretation for that cluster unless explicit reviewer evidence later reclassifies it

#### Scenario: Severity-like interpretation remains bounded
- **WHEN** a cluster is stable and post hoc label distributions suggest a severity gradient
- **THEN** the cluster MAY be labeled `candidate_severity_like_group`
- **AND** the report SHALL state that this is descriptive and associational evidence, not calibrated severity probability, causal mechanism, or externally validated disease severity

#### Scenario: Severity-like interpretation requires explicit thresholds
- **WHEN** a cluster fails predeclared thresholds for minimum row/subject support, stability, source imbalance, ROI/QC artifact dominance, grade-association strength, or representative-asset completeness
- **THEN** the atlas SHALL NOT label the cluster `candidate_severity_like_group`
- **AND** the post hoc diagnostics SHALL record which threshold blocked the label

### Requirement: Atlas review evidence is generated
The atlas SHALL generate reviewer-facing evidence that makes cluster structure inspectable without requiring manual path chasing.

#### Scenario: Representative evidence is written
- **WHEN** cluster interpretation completes
- **THEN** the workflow SHALL write representative tables under `burden_model/embedding_atlas/evidence/`
- **AND** each representative row SHALL include ROI identity, cluster ID, method ID, feature-space ID, distance or medoid status, ROI image path, mask path when available, and metadata needed to inspect the example

#### Scenario: Atlas HTML review is written
- **WHEN** representative ROI paths are available
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/evidence/embedding_atlas_review.html`
- **AND** the report SHALL show cluster representatives, boundary examples, nearest neighbors, cluster size, subject support, post hoc original-grade distribution, and artifact/source warnings

#### Scenario: Missing ROI assets are explicit
- **WHEN** ROI image or mask paths referenced by selected representative rows are missing
- **THEN** the report SHALL record missing asset counts and affected row IDs
- **AND** it SHALL NOT silently omit all broken examples while presenting the cluster as fully reviewable

### Requirement: Atlas review queues prioritize adjudication without relabeling
The atlas SHALL generate review queues for human adjudication and SHALL NOT write label overrides or replace original scores.

#### Scenario: Adjudication queue is written
- **WHEN** cluster assignments and post hoc diagnostics are available
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/review_queue/atlas_adjudication_queue.csv`
- **AND** each row SHALL include review priority, reason code, row identity, cluster ID, nearest-neighbor evidence, original score when present, reviewed-anchor evidence when present, and ROI path provenance

#### Scenario: Queue includes severe-boundary disagreements
- **WHEN** original scores or reviewed anchors indicate a severe/nonsevere boundary and cluster-neighbor structure disagrees with that boundary
- **THEN** the row SHALL be eligible for a severe-boundary review reason
- **AND** the queue SHALL preserve the original score rather than replacing it

#### Scenario: No automatic label overrides are written
- **WHEN** the atlas workflow completes
- **THEN** it SHALL NOT write score label override files for supervised quantification
- **AND** any future label override file SHALL require explicit reviewed labels from a separate human-adjudicated input

### Requirement: Atlas verdict states readiness and claim boundary
The atlas SHALL write first-read summary artifacts that distinguish descriptive atlas readiness from supervised prediction readiness.

#### Scenario: First-read verdict is written
- **WHEN** the atlas workflow completes or fails after creating the output root
- **THEN** it SHALL write `burden_model/embedding_atlas/summary/atlas_verdict.json`
- **AND** the verdict SHALL include workflow status, candidate feature spaces, candidate clustering methods, selected atlas view if any, blockers, source/artifact warnings, review queue count, and next action

#### Scenario: Index and manifest are written
- **WHEN** the atlas workflow completes successfully
- **THEN** it SHALL write `burden_model/embedding_atlas/INDEX.md`
- **AND** it SHALL write `burden_model/embedding_atlas/summary/artifact_manifest.json`
- **AND** the manifest SHALL map all generated atlas artifacts relative to `burden_model/embedding_atlas/`

#### Scenario: Claim boundary is explicit
- **WHEN** the atlas summary or review report describes the output
- **THEN** it SHALL state that the atlas supports descriptive morphology clustering and review prioritization only
- **AND** it SHALL NOT claim calibrated no-low, mid-mod, or mod-severe probabilities
- **AND** it SHALL NOT claim external validity, clinical deployment, causal mechanism, or automatic replacement of human review

### Requirement: Atlas tests cover leakage, stability, artifacts, and failure paths
The implementation SHALL include tests that prevent silent leakage, unstable output schemas, and misleading cluster interpretation.

#### Scenario: Label-blinding tests fail on leaked labels
- **WHEN** a test fixture includes denied label or source columns in the feature allow-list
- **THEN** atlas feature construction SHALL fail before clustering
- **AND** the failure SHALL identify the denied column

#### Scenario: Stability tests use subject grouping
- **WHEN** a fixture contains repeated rows per `subject_id`
- **THEN** the stability evaluation SHALL record `subject_id` as the resampling unit
- **AND** the test SHALL verify that row count and subject count are reported separately

#### Scenario: Artifact schema tests validate first-read outputs
- **WHEN** a small atlas fixture completes
- **THEN** tests SHALL verify `INDEX.md`, `summary/atlas_verdict.json`, `summary/artifact_manifest.json`, `clusters/cluster_assignments.csv`, `stability/cluster_stability.json`, and `review_queue/atlas_adjudication_queue.csv`

#### Scenario: Existing quantification tests remain independent
- **WHEN** existing supervised quantification tests run without atlas artifacts
- **THEN** they SHALL not fail solely because `burden_model/embedding_atlas/` is absent
