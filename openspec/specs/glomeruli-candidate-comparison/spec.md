# glomeruli-candidate-comparison Specification

## Purpose
Define the supported mitochondria-transfer-versus-no-base comparison workflow, deterministic promotion evidence, and explicit promotion decision contract for glomeruli segmentation artifacts.

## Requirements
### Requirement: Glomeruli promotion compares transfer and no-base candidates on the same evidence set
The repository SHALL compare at least one mitochondria-transfer glomeruli candidate and one no-mitochondria-base glomeruli candidate using the same deterministic promotion-evaluation evidence.

#### Scenario: Candidate-comparison run is started
- **WHEN** the glomeruli candidate-comparison workflow is executed
- **THEN** it trains or loads a canonical mitochondria-transfer candidate and a canonical no-mitochondria-base candidate under the supported `raw_data/cohorts` all-admitted-masked contract or an explicitly selected active paired project root
- **AND** the canonical initial workflow uses one explicit seed per candidate family
- **AND** both candidates are evaluated on the same deterministic validation manifest
- **AND** each manifest crop is assigned to exactly one review category rather than being reused across `background`, `boundary`, and `positive`
- **AND** the manifest selection prefers spanning multiple source images when enough qualifying evidence exists
- **AND** the workflow does not silently fall back from transfer to no-base training or from no-base training to transfer during the comparison run
- **AND** the no-base candidate records its encoder initialization as the ImageNet-pretrained ResNet34 baseline rather than implying literal all-random initialization

#### Scenario: One candidate family cannot be completed
- **WHEN** either the transfer or no-base candidate cannot be trained or evaluated successfully
- **THEN** the comparison report records that failure explicitly
- **AND** the workflow does not silently substitute a different candidate family

### Requirement: Candidate comparison uses a canonical CLI-first control surface
The repository SHALL treat the dedicated glomeruli training module CLI as the authoritative control surface for candidate comparison and promotion provenance.

#### Scenario: Candidate comparison is configured
- **WHEN** transfer and scratch candidate runs are defined for promotion comparison
- **THEN** the authoritative recipe is expressed as explicit training-module invocation plus recorded provenance
- **AND** any YAML config file acts only as an optional override source rather than the canonical promotion-workflow contract

#### Scenario: Candidate comparison output location is not supplied
- **WHEN** the glomeruli candidate-comparison workflow is executed without an explicit `--output-dir`
- **THEN** it SHALL write promotion reports, review assets, deterministic manifests, and comparison metrics under the active runtime output root's `segmentation_evaluation/glomeruli_candidate_comparison/` subtree on this machine
- **AND** the workflow MAY still accept an explicit caller-supplied override path when the user intentionally wants a different destination

#### Scenario: Candidate comparison trains model artifacts
- **WHEN** the glomeruli candidate-comparison workflow trains transfer or no-base candidates
- **THEN** trained candidate model artifacts SHALL be written under the configured model root's `segmentation/glomeruli/{transfer,scratch}/` subtrees
- **AND** the comparison output tree SHALL reference those model artifacts rather than duplicating them under the evaluation report directory

### Requirement: Candidate comparison produces a deterministic promotion report
The glomeruli candidate-comparison workflow SHALL write a promotion report artifact that makes the comparison and decision auditable.

#### Scenario: Promotion report is generated
- **WHEN** candidate comparison completes
- **THEN** the report records each candidate's provenance, deterministic-manifest metrics, trivial-baseline comparisons, and prediction-review results
- **AND** the report records the seed used for each candidate
- **AND** the report states the final decision outcome as `promoted`, `blocked`, or `insufficient_evidence`
- **AND** the report includes a manifest-coverage summary that records total crops plus unique-image and unique-subject coverage
- **AND** the HTML review surface labels each panel with category, crop provenance, per-panel metrics, and explicit panel-order semantics
- **AND** the report structure preserves a clean path for future repeated-seed candidate rows without breaking the initial artifact contract

#### Scenario: Shared-manifest prediction semantics are applied
- **WHEN** candidate probabilities are converted into binary segmentation masks for promotion comparison
- **THEN** the comparison workflow SHALL use learner-consistent preprocessing rather than a bespoke evaluation-only image transform
- **AND** it SHALL use the supported underconfident-model inference threshold semantics for the current segmentation path rather than assuming a generic `0.5` threshold

#### Scenario: Compatibility artifact is available
- **WHEN** a current compatibility-era glomeruli artifact is available for comparison
- **THEN** the promotion report includes it as a non-promoted comparison artifact alongside the transfer and scratch candidates

#### Scenario: Candidate family failures are reported
- **WHEN** a candidate family is unavailable and the workflow still emits a structured promotion report
- **THEN** that report SHALL record the family failure explicitly
- **AND** report generation alone SHALL NOT be treated as evidence that the supported transfer-versus-scratch comparison executed successfully end to end

### Requirement: Promotion decision is explicit and non-automatic
The workflow SHALL NOT treat the most recent runtime-compatible glomeruli artifact as promoted unless the promotion report explicitly clears it.

#### Scenario: Candidate clears promotion gates
- **WHEN** exactly one candidate beats the required baselines, avoids degeneracy, passes deterministic prediction review, and satisfies the promotion report criteria
- **THEN** the report marks that candidate as the promoted glomeruli artifact

#### Scenario: Neither candidate clears promotion gates
- **WHEN** both candidates fail the required promotion checks or remain too close to trivial or compatibility baselines
- **THEN** the report blocks promotion
- **AND** no new artifact is labeled as scientifically promoted

#### Scenario: Candidates are scientifically indistinguishable
- **WHEN** both transfer and scratch clear the hard promotion gates but remain within an absolute practical tie margin of `0.02` or less on both shared-manifest Dice and shared-manifest Jaccard
- **THEN** the report records `insufficient_evidence` with an explicit tie reason
- **AND** neither candidate becomes the sole promoted default
- **AND** both artifacts remain available as explicit runtime-compatible research candidates for downstream segmentation and endotheliosis quantification

### Requirement: Scratch candidate training honors the requested crop-size contract
The scratch glomeruli candidate workflow SHALL propagate the caller-supplied crop size through batch-size selection, dynamic cropping, and provenance rather than silently collapsing it to `image_size`.

#### Scenario: Scratch candidate is configured with larger crop context
- **WHEN** scratch training is started with `image_size=256` and `crop_size=512`
- **THEN** the stage-aware batch-size recommendation uses `512` as the crop context for the glomeruli candidate family
- **AND** dynamic patching crops `512`-pixel regions before resizing to the `256`-pixel model input
- **AND** the resulting candidate provenance records both the requested crop size and the output image size
