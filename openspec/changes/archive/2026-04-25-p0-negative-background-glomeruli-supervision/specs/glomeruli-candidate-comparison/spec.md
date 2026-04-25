## ADDED Requirements

### Requirement: Candidate reports disclose negative/background supervision
Glomeruli candidate comparison reports SHALL disclose negative/background supervision state for each candidate.

#### Scenario: Candidate was trained with mask-derived background crops
- **WHEN** a candidate report is generated for a model whose metadata records mask-derived background supervision
- **THEN** `candidate_summary.csv`, `promotion_report.json`, and `promotion_report.md` include mask-derived background crop count, manifest path, manifest hash, and sampler weight

#### Scenario: Candidate was trained with curated MR/TIFF negatives
- **WHEN** a candidate report is generated for a model whose metadata records curated reviewed negative crops
- **THEN** the report includes curated negative crop count, source image count, review protocol version, manifest path, and manifest hash
- **AND** it states that the negatives are crop-level evidence, not whole-image negative evidence

#### Scenario: Candidate lacks negative crop supervision
- **WHEN** a candidate has no validated negative/background supervision metadata
- **THEN** the report records `negative_crop_supervision_status=absent`

### Requirement: Background category gates remain promotion blockers
Promotion gates SHALL continue to block candidates with background false-positive excess even when aggregate Dice is high.

#### Scenario: Aggregate Dice is high but background crops fail
- **WHEN** candidate aggregate Dice clears a nominal threshold
- **AND** deterministic background crops show false-positive foreground excess
- **THEN** the candidate remains not promotion eligible
- **AND** the report lists the background failure reason

### Requirement: Candidate reports disclose augmentation policy
Candidate comparison reports SHALL record the actual augmentation policy used for each candidate.

#### Scenario: Candidate metadata includes augmentation policy
- **WHEN** candidate metadata includes augmentation policy fields
- **THEN** candidate reports include the selected augmentation variant, FastAI transform settings, and whether config-declared augmentation fields were active
