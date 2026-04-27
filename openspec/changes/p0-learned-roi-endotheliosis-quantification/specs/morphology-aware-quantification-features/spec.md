## ADDED Requirements

### Requirement: Blocked deterministic morphology remains QC-only
Deterministic morphology features SHALL remain reviewer-visible QC/evidence when their visual feature-readiness gate is blocked, but SHALL NOT be used as biological proof of closed-lumen burden.

#### Scenario: Blocked morphology is excluded from learned biological claims
- **WHEN** `burden_model/candidates/morphology_candidate_summary.json` reports `selection_status` as `blocked_by_visual_feature_readiness`
- **THEN** learned ROI reports SHALL treat deterministic morphology features as QC/evidence only
- **AND** learned ROI reports SHALL NOT use deterministic slit features to justify a closed-lumen mechanistic claim

#### Scenario: Morphology covariates carry readiness status
- **WHEN** deterministic morphology features are included in any learned ROI candidate feature set
- **THEN** the learned ROI candidate summary SHALL include the deterministic morphology readiness status and blockers
- **AND** a blocked morphology readiness status SHALL prevent any claim that the combined model is mechanistically using validated slit features

#### Scenario: Future supervised morphology labels are separate
- **WHEN** a future workflow adds supervised labels for open lumen, RBC-filled patent lumen, collapsed/slit-like lumen, nuclear/mesangial false positives, or border false positives
- **THEN** those labels SHALL be represented as a separate supervised morphology-labeling capability
- **AND** they SHALL NOT be conflated with the current deterministic morphology feature-readiness gate
