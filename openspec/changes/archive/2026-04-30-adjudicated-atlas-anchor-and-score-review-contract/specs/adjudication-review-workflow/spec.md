## ADDED Requirements

### Requirement: Review HTML presents visible case evidence without JavaScript dependency
Image-based adjudication workflows SHALL write reviewer-facing HTML in which review cases are present as static HTML elements and each case shows the relevant image assets next to the decision controls.

#### Scenario: Review cases are visible in static HTML
- **WHEN** a review workflow writes an adjudication HTML artifact
- **THEN** the HTML SHALL contain one static case element per selected review case
- **AND** each case element SHALL include the review identity, source row identity, prior machine or reviewer decision context, and the specific decision prompt
- **AND** the page SHALL NOT depend entirely on JavaScript rendering to show the cases

#### Scenario: Image assets are adjacent to controls
- **WHEN** a review case references ROI image, ROI mask, raw image, overlay, or other visual assets
- **THEN** the HTML SHALL show those assets in the same case card as the decision controls
- **AND** the asset paths SHALL be explicit enough for postflight validation to confirm that the referenced files exist

#### Scenario: Blank review pages fail postflight
- **WHEN** a review HTML artifact is generated
- **THEN** tests or postflight checks SHALL verify the expected case count, image count, decision-control count, and export-control presence
- **AND** a page with only a header and no visible cases SHALL be treated as a failed review artifact

### Requirement: Review controls are task-specific and exported as structured decisions
Adjudication review workflows SHALL use decision controls that match the question being asked and SHALL export reviewer decisions in a structured file with provenance.

#### Scenario: Score review fields are explicit
- **WHEN** a case is selected for score review
- **THEN** the review artifact SHALL ask whether to keep the original score, change the score, or request a second reviewer
- **AND** if the score changes, the export SHALL include the original score, adjudicated score, reason, reviewer note, reviewed timestamp, row identity, and image provenance

#### Scenario: Anchor review fields are explicit
- **WHEN** a case is selected for anchor eligibility review
- **THEN** the review artifact SHALL ask whether the row is allowed as an anchor, excluded from anchor use, or requires a second reviewer
- **AND** the export SHALL include anchor decision, exclusion or recovery reason, reviewer note, reviewed timestamp, row identity, and image provenance

#### Scenario: Cluster-level and case-level decisions are separate
- **WHEN** a workflow asks reviewers to interpret a cluster and also review individual cases
- **THEN** cluster-level fields SHALL be stored separately from case-level fields
- **AND** case-level exports SHALL include cluster context without implying that one accepted case promotes the whole cluster

### Requirement: Review exports preserve original evidence and reviewer provenance
Adjudication exports SHALL preserve original machine or human evidence and SHALL add reviewer decisions as separate columns rather than replacing source values.

#### Scenario: Original values are preserved
- **WHEN** a reviewer exports adjudication decisions
- **THEN** original score, original prediction, original cluster, original action, or original failure fields supplied to the review SHALL remain present in the export
- **AND** adjudicated values SHALL use distinct fields such as `adjudicated_score`, `corrected_score`, `score_decision`, `anchor_decision`, or workflow-specific reviewed-decision names

#### Scenario: Reviewer provenance is captured
- **WHEN** the review export is created
- **THEN** it SHALL include a reviewed timestamp and enough case identity to join the decisions back to the generated workflow artifacts
- **AND** reviewer notes SHALL be preserved when provided

### Requirement: Missing or invalid review inputs are explicit
Adjudication-consuming workflows SHALL distinguish missing optional review input from invalid provided review input.

#### Scenario: Missing optional review input writes empty summaries
- **WHEN** no optional adjudication export is present
- **THEN** the workflow SHALL continue to generate its baseline artifacts
- **AND** it SHALL write or report an explicit no-review status rather than pretending review occurred

#### Scenario: Invalid provided review input fails closed
- **WHEN** an adjudication export is provided but lacks required columns, contains conflicting duplicate decisions, or cannot be joined to source workflow artifacts
- **THEN** the consuming workflow SHALL fail closed or mark adjudication ingestion failed
- **AND** it SHALL write diagnostics that identify the missing columns, conflicting rows, or unmatched row identities
