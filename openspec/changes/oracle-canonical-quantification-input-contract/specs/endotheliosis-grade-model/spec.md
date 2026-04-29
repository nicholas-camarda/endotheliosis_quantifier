## ADDED Requirements

### Requirement: Grade model consumes canonical label contract
The P3 endotheliosis grade model SHALL consume only the canonical resolved quantification label contract and SHALL write that contract into final selector, verdict, and model metadata artifacts.

#### Scenario: Final model metadata records label contract
- **WHEN** P3 writes `model/final_model_metadata.json`
- **THEN** the metadata includes the resolved score source, reviewed label override path or `none`, override content hash when present, effective target-definition version, and path to the score override audit

#### Scenario: Final verdict records label contract
- **WHEN** P3 writes `summary/final_product_verdict.json`
- **THEN** the verdict includes the resolved label contract reference so a reader can distinguish reviewed-rubric runs from unreviewed scored-example runs

#### Scenario: Candidate features cannot reintroduce stale scores
- **WHEN** optional feature tables contain score-like columns from older artifacts
- **THEN** P3 ignores those stale score columns and uses only labels from the canonical resolved label contract

#### Scenario: Label contract changes invalidate comparability
- **WHEN** two P3 runs use different reviewed override hashes or target-definition versions
- **THEN** their model-selection and final-verdict artifacts record the difference so metrics are not presented as directly comparable without that context
