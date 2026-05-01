## ADDED Requirements

### Requirement: Scored cohort construction SHALL support iterative authoritative grading refresh cycles

The quantification cohort pipeline SHALL support repeated cycles where updated authoritative human grades supersede prior grades for the same logical scoring units without ambiguous silent merges.

#### Scenario: Operator reruns cohort build after new Label Studio export

- **WHEN** a newer authoritative grading export or reviewed-label override batch replaces grades for overlapping scored rows compared to an earlier ingest
- **THEN** cohort-building workflows MUST reconcile scores using explicit precedence documented for Label Studio loaders and reviewed overrides such that the latest authoritative grade wins per reconciliation rules already governing score recovery
- **AND** MUST emit audit metadata identifying that grading inputs changed relative to any referenced prior cohort snapshot when operators configure snapshot comparison or lineage audits

#### Scenario: Quantification rerun after grading expansion

- **WHEN** additional images or glomerulus instances receive first-time authoritative grades without altering previously finalized rows
- **THEN** cohort-building workflows MUST admit new scored rows under existing manifest pairing rules
- **AND** downstream quantification reruns MUST remain reproducible given the same grading-export inputs

### Requirement: Quantification SHALL record grading-input lineage sufficient for iterative learning cycles

Runs that consume Label Studio-derived scores or reviewed label overrides MUST persist durable lineage fields suitable for auditing whether burden-model outputs correspond to grading snapshot **A** versus **B**, including stable identifiers for annotation or export provenance supplied by loaders or explicitly declared override files.

#### Scenario: Auditor compares two burden_model trees

- **WHEN** two quantification output roots exist from different grading-export generations
- **THEN** each root MUST expose lineage artifacts or admitted manifest fields that allow an auditor to determine which grading inputs were consumed without relying solely on informal README notes

### Requirement: Operator documentation SHALL describe the authoritative grading → cohort → quantification loop

Repository-maintained quantification onboarding documentation MUST describe the offline iterative cadence: finalize authoritative grades under **`label-studio-glomerulus-grading`**, rebuild or extend scored cohort artifacts under **`scored-only-quantification-cohort`**, rerun contract-first quantification including P3 under **`endotheliosis-grade-model`**, and archive outputs with pointers to grading-input lineage sufficient to detect stale reuse.

#### Scenario: Collaborator reads quant onboarding section

- **WHEN** a collaborator reads documented quantification onboarding guidance maintained for this repository
- **THEN** that guidance MUST explicitly state that burden modeling refresh follows refreshed authoritative grades and explicit reruns—not implicit reuse of stale cohort snapshots

### Requirement: Cohort pipelines SHALL distinguish scoring-unit eras without inferring per-glom grades from legacy aggregates

Quantification and cohort admission MUST support **both** (a) historical **image-level** scored examples collected under legacy rules and (b) **per-glomerulus** authoritative rows from `label-studio-glomerulus-grading` or hybrid exports, using explicit manifests, loaders, or schema markers so operators do not silently merge incompatible scoring units.

#### Scenario: Legacy image-level cohort remains valid

- **WHEN** a scored cohort was built from image-level Label Studio or spreadsheet scores predating per-glom instance exports
- **THEN** reproducibility artifacts MUST remain valid for that cohort definition without requiring retroactive per-glom rows

#### Scenario: Operator migrates images to per-glom scoring

- **WHEN** operators re-label legacy images in a per-glom workflow and produce a new authoritative export
- **THEN** cohort-building workflows MUST admit the new per-glom rows under per-glom reconciliation rules and MUST NOT treat a legacy single image-level score as automatically decomposed into per-glom training targets without explicit new annotations

#### Scenario: No authoritative decomposition from aggregate alone

- **WHEN** only a historical image-level aggregate score exists for a multi-glomerulus image and no new per-glom annotations were collected
- **THEN** cohort and quantification workflows MUST NOT present synthetic per-glom scores derived solely from that aggregate as scientifically equivalent to brush-finalized hybrid exports unless a **separate, explicitly scoped** modeling change defines and validates such inference
