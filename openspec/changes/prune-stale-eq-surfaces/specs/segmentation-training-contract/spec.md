## MODIFIED Requirements

### Requirement: Static patch roots are retired and not active code paths
Supported segmentation training SHALL use full-image `images/` and `masks/` roots with dynamic patching, and SHALL NOT expose active source-code paths that create, load, audit, or train from pre-generated `image_patches/` and `mask_patches/` trees.

#### Scenario: Static patch root is supplied to training
- **WHEN** mitochondria or glomeruli training is started with a data root whose active inputs are `image_patches/` and `mask_patches/`
- **THEN** training fails before model construction and instructs the user to use the full-image `images/` and `masks/` root

#### Scenario: Static patch utility is requested from the active CLI or package
- **WHEN** CLI help or active package exports are inspected
- **THEN** no command or public helper is exposed for patchifying, loading, or auditing static patch datasets

#### Scenario: Retired static patch code or data is preserved
- **WHEN** stale static patch code or runtime data is retired
- **THEN** it is moved to a dated `_retired/` location outside the active source/runtime input tree
- **AND** the implementation records the original path, retired path, size, and move timestamp

### Requirement: Dynamic patching is the only supported segmentation loading path
The active segmentation data loader SHALL provide only full-image dynamic-patching builders for supported model training.

#### Scenario: Active dataloader exports are inspected
- **WHEN** `eq.data_management.datablock_loader` or `eq.data_management` exports are inspected
- **THEN** they expose dynamic full-image training helpers and validators, not static patch dataloaders
