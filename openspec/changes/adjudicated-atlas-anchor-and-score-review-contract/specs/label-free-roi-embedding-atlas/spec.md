## ADDED Requirements

### Requirement: Atlas adjudication inputs are validated before use
The atlas workflow SHALL treat reviewer adjudication exports as optional explicit evidence inputs and SHALL validate them against the current atlas artifacts before using them.

#### Scenario: Missing adjudication inputs do not block baseline atlas generation
- **WHEN** `eq run-config --config configs/label_free_roi_embedding_atlas.yaml` runs without adjudication exports configured or present
- **THEN** the workflow SHALL still write the baseline atlas artifacts under `burden_model/embedding_atlas/`
- **AND** it SHALL record that adjudication evidence was not provided
- **AND** it SHALL NOT infer score corrections, recovered anchors, or blocked clusters from missing reviewer evidence

#### Scenario: Provided adjudication rows match atlas identity
- **WHEN** an atlas adjudication export is provided
- **THEN** every adjudication row SHALL match exactly one row in the current atlas by `atlas_row_id`
- **AND** any provided `subject_image_id`, `cluster_id`, `original_score`, `roi_image_path`, and `roi_mask_path` SHALL match the current atlas artifacts
- **AND** unmatched or contradictory rows SHALL block adjudication ingestion

#### Scenario: Required adjudication columns are enforced
- **WHEN** an atlas score-correction or anchor-decision export is provided
- **THEN** the workflow SHALL require columns for row identity, original score, decision type, reviewed decision, reviewer timestamp, and ROI path provenance
- **AND** missing required columns SHALL be reported in an adjudication diagnostics artifact

### Requirement: Atlas score corrections are evidence, not canonical label replacement
The atlas workflow SHALL write adjudicated score suggestions as separate review evidence and SHALL NOT overwrite original human scores in atlas metadata, cluster assignments, or upstream quantification inputs.

#### Scenario: Score corrections are summarized separately
- **WHEN** reviewer decisions change one or more scores
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/evidence/atlas_score_corrections.csv`
- **AND** each row SHALL include `atlas_row_id`, row identity, `original_score`, adjudicated or corrected score, decision reason, reviewer note when present, reviewed timestamp, and ROI path provenance

#### Scenario: Original scores remain immutable in atlas outputs
- **WHEN** score-correction evidence is ingested
- **THEN** existing original score fields in cluster assignments, representatives, review queues, and post hoc diagnostics SHALL remain unchanged
- **AND** any output that compares original and adjudicated scores SHALL use distinct column names for adjudicated evidence

#### Scenario: Score correction verdict updates next action
- **WHEN** validated score-correction evidence is present
- **THEN** `burden_model/embedding_atlas/summary/atlas_verdict.json` SHALL report the count of score changes and kept reviewed scores
- **AND** `burden_model/embedding_atlas/INDEX.md` or summary Markdown SHALL point reviewers to the score-correction artifact

### Requirement: Atlas anchor manifests distinguish cluster anchors from row-level recovered anchors
The atlas workflow SHALL write explicit anchor evidence that distinguishes reviewed cluster-level anchor candidates, blocked clusters, and row-level recovered anchor examples.

#### Scenario: Reviewed cluster anchor candidates are written
- **WHEN** cluster-level adjudication identifies a real morphology cluster with sufficient accepted representative cases
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/evidence/atlas_adjudicated_anchor_manifest.csv`
- **AND** the manifest SHALL include cluster ID, feature space, method, dominant reviewed morphology, accepted case count, representative row IDs, source/cohort warnings, review confidence, and claim boundary

#### Scenario: Blocked clusters are written
- **WHEN** cluster-level adjudication identifies a mixed, source/batch, ROI/mask, or otherwise non-anchor cluster
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/evidence/atlas_blocked_cluster_manifest.csv`
- **AND** blocked clusters SHALL NOT be promoted as cluster-level anchors even if individual rows from the cluster are recovered as useful examples

#### Scenario: Recovered row-level anchors are written separately
- **WHEN** a reviewer allows a case as an anchor despite wrong-cluster, atypical, RBC-confounded, or outlier context
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/evidence/atlas_recovered_anchor_examples.csv`
- **AND** recovered rows SHALL include the row-level anchor decision, score, reason, reviewer note, cluster context, and ROI path provenance
- **AND** recovered rows SHALL NOT promote the source cluster as an anchor

### Requirement: Atlas final adjudication outcome is first-read evidence
The atlas workflow SHALL write a first-read final adjudication outcome whenever validated atlas adjudication evidence is present.

#### Scenario: Final outcome JSON and Markdown are written
- **WHEN** score-correction or anchor-decision evidence is ingested successfully
- **THEN** the workflow SHALL write `burden_model/embedding_atlas/evidence/atlas_final_adjudication_outcome.json`
- **AND** it SHALL write `burden_model/embedding_atlas/evidence/atlas_final_adjudication_outcome.md`
- **AND** the outcome SHALL summarize score changes, scores kept after review, recovered anchors, blocked clusters, candidate anchor clusters, source evidence paths, and next implementation action

#### Scenario: Artifact manifest includes adjudication outputs
- **WHEN** adjudication outcome artifacts are written
- **THEN** `burden_model/embedding_atlas/summary/artifact_manifest.json` SHALL include the final outcome, score-correction, recovered-anchor, anchor-manifest, and blocked-cluster artifacts
- **AND** missing optional adjudication outputs SHALL be represented as not provided rather than silently omitted when no review evidence exists

#### Scenario: Claim boundary remains descriptive
- **WHEN** atlas summaries describe adjudication outcomes
- **THEN** they SHALL state that adjudication evidence supports reviewed morphology anchors and score-review evidence
- **AND** they SHALL NOT describe the outputs as externally validated labels, causal mechanism evidence, calibrated severity probabilities, or automatic replacement of human review

### Requirement: Atlas focused review pages use the adjudication review workflow standard
Focused atlas review pages SHALL follow the reusable adjudication review workflow standard so reviewers can complete task-specific decisions from the images themselves.

#### Scenario: Focused flagged-case review is reviewable
- **WHEN** the atlas generates a focused flagged-case review HTML
- **THEN** it SHALL contain static visible case cards for each flagged case
- **AND** each case SHALL show ROI image and ROI mask beside the task-specific score or anchor decision controls
- **AND** it SHALL include an export button that writes structured decisions with row identity and ROI provenance

#### Scenario: Focused review postflight catches blank pages
- **WHEN** tests exercise focused atlas review generation
- **THEN** they SHALL verify case-card count, image count, select-control count, and export-control text
- **AND** they SHALL fail if the HTML contains only JavaScript data without static visible case markup
