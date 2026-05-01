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
