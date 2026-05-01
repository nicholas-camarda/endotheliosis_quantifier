## ADDED Requirements

### Requirement: Collaborator distribution SHALL separate Git-tracked software from heavyweight artifacts

The repository SHALL treat Git-tracked surfaces as **source code, configs, contracts, tests, and illustrative stubs only**. Collaborators MUST obtain segmentation weights, MedSAM checkpoints, generated-mask releases, cohort imagery, and quantification runtime outputs via **documented non-Git channels** unless an explicit pinned Git LFS allowlist applies.

#### Scenario: Collaborator reads distribution capability spec

- **WHEN** an operator seeks supported collaborator onboarding paths
- **THEN** documentation MUST cite **Git clone + Release bundle download + runtime layout** as the default narrative
- **AND** MUST NOT imply GitHub hosts full collaborator datasets by default

### Requirement: Canonical artifact manifests SHALL accompany pinned collaborator bundles

Pinned collaborator bundles distributed outside raw Git commits MUST ship or reference an **`artifacts_manifest`** JSON listing artifact names, cryptographic hashes (minimum **`sha256`** where computing hashes is feasible), logical roles (`medsam_checkpoint`, `mask_release_archive`, `quant_model_export`, etc.), and compatible **`eq` git tags or semver**.

#### Scenario: Manifest-driven verification

- **WHEN** a collaborator unpacks a Release bundle adjacent to the cloned repository
- **THEN** they MUST be able to verify integrity using the manifest hashes before copying artifacts into **`EQ_RUNTIME_ROOT`** paths referenced by YAML configs

### Requirement: Documentation SHALL contrast Label Studio labeling execution versus eq quantification execution

Collaborator-facing documentation MUST state that **Label Studio** covers **grading UI / annotation lifecycle** while **`eq run-config` / quantification workflows** cover **burden modeling outputs derived from authoritative exported grades**, unless a future capability explicitly merges predictors into LS tasks under revised grading contracts.

#### Scenario: Prevent mistaken expectation that LS hosts quant training loops

- **WHEN** a collaborator configures hybrid grading plus burden quantification review
- **THEN** documentation MUST clarify MedSAM companion endpoints serve **interactive segmentation assistance**, not replacement for **`eq` quantification pipelines**

### Requirement: Git LFS usage SHALL remain optional and explicitly constrained

Any Git LFS guidance MUST describe quota/bandwidth risks and MUST recommend **GitHub Releases** (or institutional mirrors) for large checkpoints when LFS economics are unfavorable.

#### Scenario: Choosing channels

- **WHEN** repository maintainers publish a new collaborator-facing bundle exceeding comfortable LFS thresholds
- **THEN** Releases (or institutional object storage with manifest URLs) MUST be documented as preferred over expanding LFS footprint

### Requirement: Packaging guidance SHOULD document Docker Compose topology for Label Studio plus MedSAM companion

The capability spec MUST reference optional **Compose** snippets (git-tracked when implementation tasks land) orchestrating **`heartexlabs/label-studio:*` pinned tag** plus the **MedSAM companion** service described by **`label-studio-medsam-hybrid-grading`**, using **volume mounts** for downloaded weights—not committing weights to Git.

#### Scenario: Compose snippet discovery

- **WHEN** hybrid grading enables companion-backed labeling sessions
- **THEN** collaborator docs MUST link Compose packaging guidance adjacent to **`configs/label_studio_medsam_hybrid.yaml`** instructions once those snippets exist

### Requirement: Quantification predictor injection into Stage 1 grading UX SHALL remain out of scope until separately governed

Until an OpenSpec change explicitly modifies **`label-studio-medsam-hybrid-grading`** grading contracts, collaborator packaging docs MUST preserve the stance that **Stage 2 quantification predictors MUST NOT surface inside Stage 1 primary grading UX**.

#### Scenario: Align expectations with hybrid grading non-goals

- **WHEN** collaborators ask whether burden models run automatically inside Label Studio scoring forms
- **THEN** documentation MUST cite hybrid grading non-goals and direct burden workflows to **`eq` exports**
