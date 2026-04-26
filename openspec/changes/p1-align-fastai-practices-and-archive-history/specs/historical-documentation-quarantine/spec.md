## ADDED Requirements

### Requirement: Historical FastAI material is retained outside current workflow docs
Historical FastAI integration, migration, legacy artifact, and older pipeline-planning material SHALL remain accessible and SHALL be retained only in explicitly historical documentation surfaces.

#### Scenario: Historical docs are moved into archive surfaces
- **WHEN** historical FastAI docs such as `docs/INTEGRATION_GUIDE.md`, `docs/PIPELINE_INTEGRATION_PLAN.md`, or `docs/HISTORICAL_IMPLEMENTATION_ANALYSIS.md` are retained
- **THEN** their historical content lives under `docs/HISTORICAL_NOTES.md` or `docs/archive/`
- **AND** their first screen clearly states that the material is historical/reference-only and not current operational guidance

#### Scenario: Historical archive is accessible
- **WHEN** a user starts from `docs/HISTORICAL_NOTES.md` or `docs/README.md`
- **THEN** they can find the archived FastAI historical integration and implementation-analysis documents through explicit links
- **AND** those links are labeled as historical/reference material rather than current workflow entrypoints

#### Scenario: Current docs link to archive without adopting it
- **WHEN** current docs need to mention historical FastAI context
- **THEN** they link to `docs/HISTORICAL_NOTES.md` or a `docs/archive/` page
- **AND** they do not reproduce historical setup commands, fallback loaders, legacy module shims, or old preprocessing recommendations as active instructions

### Requirement: Current workflow docs describe present behavior only
Current user-facing docs SHALL describe the supported present workflow directly without migration framing, historical rescue instructions, legacy compatibility shims, or workaround paths.

#### Scenario: README is inspected
- **WHEN** `README.md` is inspected after implementation
- **THEN** it describes current `eq run-config`, `eq-mac`, runtime-root, supported artifact, and segmentation-validation behavior
- **AND** it does not instruct users to use historical FastAI helper modules, legacy namespace shims, or fallback model-loading paths

#### Scenario: Implementation guide is inspected
- **WHEN** `docs/INTEGRATION_GUIDE.md` exists after implementation
- **THEN** it describes current implementation status and supported commands only
- **AND** historical integration content has been extracted into `docs/archive/fastai_legacy_integration.md`
- **AND** it does not present historical FastAI helper modules, workaround branches, or compatibility shims as active implementation guidance

#### Scenario: Segmentation engineering guide is inspected
- **WHEN** `docs/SEGMENTATION_ENGINEERING_GUIDE.md` is inspected after implementation
- **THEN** it describes current full-image dynamic patching, trusted current-namespace artifacts, promotion evidence, and validation gates
- **AND** it treats legacy FastAI pickle artifacts as historical unless a separate compatibility change exists

#### Scenario: Onboarding docs are inspected
- **WHEN** `docs/ONBOARDING_GUIDE.md` and `docs/README.md` are inspected after implementation
- **THEN** they point new users to current setup, current workflow configs, and current validation commands
- **AND** historical material is discoverable only through a clearly labeled archive/reference link

### Requirement: Historical-current drift is testable
The repository SHALL include a test or validation check that prevents archived historical FastAI instructions from reappearing as current workflow guidance.

#### Scenario: Historical quarantine check runs
- **WHEN** the documentation quarantine validation is run
- **THEN** it scans `README.md`, `docs/README.md`, `docs/INTEGRATION_GUIDE.md`, `docs/ONBOARDING_GUIDE.md`, `docs/SEGMENTATION_ENGINEERING_GUIDE.md`, and `docs/TECHNICAL_LAB_NOTEBOOK.md`
- **AND** it fails if current docs contain active instructions to use `historical_glomeruli_inference`, `setup_historical_environment`, legacy namespace shims, or historical fallback loading

#### Scenario: Archive docs are allowed historical language
- **WHEN** the same validation check scans `docs/HISTORICAL_NOTES.md` or files under `docs/archive/`
- **THEN** historical FastAI terms and legacy examples are allowed only if the document is explicitly marked historical/reference-only
