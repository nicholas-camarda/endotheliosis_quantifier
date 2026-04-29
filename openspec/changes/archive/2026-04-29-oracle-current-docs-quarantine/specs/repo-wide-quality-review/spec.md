## ADDED Requirements

### Requirement: Active docs quarantine unsupported historical operations
Active documentation SHALL NOT present unsupported historical modules, legacy namespace shims, fallback inference paths, or compatibility rescue logic as executable current workflow guidance.

#### Scenario: Active doc contains historical module import
- **WHEN** an active doc outside `docs/archive/` includes `eq.inference.historical_glomeruli_inference`
- **THEN** the docs quarantine check fails and reports the file path

#### Scenario: Active doc contains historical setup instruction
- **WHEN** an active doc outside `docs/archive/` instructs users to call `setup_historical_environment`
- **THEN** the docs quarantine check fails and reports the file path

#### Scenario: Active doc recommends fallback model loading
- **WHEN** an active doc outside `docs/archive/` recommends automatic fallback loading for historical glomeruli inference
- **THEN** the docs quarantine check fails and reports the unsupported guidance

#### Scenario: Historical archive contains reference material
- **WHEN** a file under `docs/archive/` contains historical fallback content
- **THEN** the docs quarantine check allows it only when the file has reference-only framing and is indexed from `docs/HISTORICAL_NOTES.md`

### Requirement: Active docs remain current-state only
Active documentation SHALL describe the current supported `eq` workflow directly and MUST NOT use migration framing, workaround language, or historical fallback examples as the main operational path.

#### Scenario: README describes supported path
- **WHEN** `README.md` describes quantification or segmentation execution
- **THEN** it points to current supported commands and configs rather than historical fallback modules

#### Scenario: Onboarding describes supported path
- **WHEN** `docs/ONBOARDING_GUIDE.md` describes setup or execution
- **THEN** it uses the current environment, `eq run-config`, and supported artifact contracts

#### Scenario: Integration guide remains active
- **WHEN** `docs/INTEGRATION_GUIDE.md` remains outside `docs/archive/`
- **THEN** it contains only current operational guidance and links historical details through `docs/HISTORICAL_NOTES.md`
