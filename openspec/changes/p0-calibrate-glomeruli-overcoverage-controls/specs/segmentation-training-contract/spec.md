# segmentation-training-contract Delta Specification

## ADDED Requirements

### Requirement: Glomeruli training signal controls are auditable before full retraining
Glomeruli training workflows SHALL expose and record the training-signal controls needed to investigate foreground overcoverage before another full production training run is treated as justified.

#### Scenario: Training provenance records overcoverage controls
- **WHEN** a glomeruli transfer or scratch candidate is trained
- **THEN** its provenance records negative crop manifest path, negative crop sampler weight, `positive_focus_p`, loss name, resolved loss class, false-positive penalty parameters when applicable, crop size, output size, augmentation variant, and whether the run is a short screening run or a production training run

#### Scenario: Short screening run changes a training-signal axis
- **WHEN** a short overcoverage-control screening run changes sampler weight, `positive_focus_p`, loss, resize policy, or augmentation
- **THEN** the run provenance names the changed axis
- **AND** `training_signal_ablation_summary.csv` records the changed axis, candidate family, run id, epochs, model path, comparison artifact path, and category-level overcoverage outcome

#### Scenario: Combined ablation is run
- **WHEN** a screening run changes more than one training-signal axis at once
- **THEN** the provenance labels the run as `combined_non_diagnostic`
- **AND** the result is not used as sole evidence for selecting a production training policy

### Requirement: Glomeruli scratch and transfer training honor requested loss settings
Both canonical glomeruli candidate families SHALL honor requested loss settings or fail closed before training.

#### Scenario: Transfer training receives a loss setting
- **WHEN** transfer training is invoked with a supported loss name and parameters
- **THEN** the learner uses that loss
- **AND** model metadata records the requested loss and resolved loss implementation

#### Scenario: Scratch training receives a loss setting
- **WHEN** scratch/no-mitochondria-base glomeruli training is invoked with a supported loss name and parameters
- **THEN** the learner uses that loss
- **AND** model metadata records the requested loss and resolved loss implementation
- **AND** the training path does not silently ignore the requested loss

#### Scenario: Unsupported loss setting is requested
- **WHEN** transfer or scratch training receives an unsupported loss name or invalid false-positive penalty parameters
- **THEN** training fails before model fitting
- **AND** the failure identifies the unsupported loss setting

### Requirement: False-positive-penalizing loss settings are explicit
False-positive-penalizing loss behavior SHALL be configured with explicit parameters rather than inferred from a generic loss family name.

#### Scenario: Tversky-style loss is requested for overcoverage control
- **WHEN** a Tversky-style loss is selected to penalize foreground false positives
- **THEN** provenance records the exact false-positive and false-negative penalty parameters
- **AND** the run report states whether the parameters penalize false positives more than false negatives

#### Scenario: Default Tversky parameters are used
- **WHEN** Tversky-style loss uses equal false-positive and false-negative weights
- **THEN** the report does not describe it as a false-positive-penalizing configuration

### Requirement: Resize-policy ablations are required before production resize changes
Glomeruli resize-policy changes SHALL be screened and recorded before a full production retrain changes `crop_size` or `output_size`.

#### Scenario: Current resize policy is audited
- **WHEN** the current `crop_size=512` and `output_size=256` policy is evaluated for overcoverage
- **THEN** the audit records crop size, output size, crop-to-output resize ratio, resize method, image interpolation, mask interpolation, prediction resize-back method, and threshold/resize order

#### Scenario: Less-downsampled policy is screened
- **WHEN** local hardware permits a less-downsampled screening run
- **THEN** the run records the tested crop size, output size, batch size, device, memory-related failures when present, and category-level overcoverage metrics
- **AND** `resize_policy_comparison.csv` compares the policy against the current resize policy

#### Scenario: Less-downsampled policy cannot run
- **WHEN** MPS memory or supported operations prevent a less-downsampled screening run
- **THEN** the failure is recorded in `resize_policy_comparison.csv` and `audit-results.md`
- **AND** the implementation does not silently skip the resize-policy question

### Requirement: Augmentation variants are explicit and non-silent
Glomeruli augmentation changes SHALL be named, configured, and recorded before they are used to interpret overcoverage behavior.

#### Scenario: Default augmentation is used
- **WHEN** training uses the current FastAI default augmentation path
- **THEN** provenance records `augmentation_variant=fastai_default`, the FastAI transform settings, and whether config controls were active

#### Scenario: Spatial-only or no-augmentation variant is used
- **WHEN** `spatial_only` or `no_aug` is selected for an overcoverage-control screening run
- **THEN** provenance records the selected variant
- **AND** the report identifies augmentation as the changed axis

#### Scenario: Gaussian noise is claimed as active
- **WHEN** a run records Gaussian noise or noise augmentation as active
- **THEN** the training transform path must actually include that transform
- **AND** tests verify that the provenance and transform pipeline agree
