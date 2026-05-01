## ADDED Requirements

### Requirement: P3 burden-model outputs SHALL reference grading-input lineage when scores originate from authoritative exports

When P3 consumes scored examples derived from Label Studio exports or reviewed label overrides, the workflow MUST persist references to grading-input lineage consistent with **`scored-only-quantification-cohort`** iterative-loop lineage requirements, using summaries or indexes under `burden_model/endotheliosis_grade_model/summary/` (or successor paths) or equivalent audit artifacts emitted during the quantification workflow.

#### Scenario: P3 completes after grading-export refresh

- **WHEN** P3 completes following a grading-export refresh that changed authoritative scores upstream of scored examples
- **THEN** written summaries or lineage indexes MUST record identifiers sufficient to correlate the fit with the grading snapshot consumed upstream

#### Scenario: Per-glom versus image-level scoring units

- **WHEN** scored examples mix or transition between **image-level** legacy grades and **per-glomerulus** hybrid or instance exports
- **THEN** lineage artifacts MUST preserve enough scoring-unit metadata that downstream readers can tell which rows belong to which era, consistent with **`scored-only-quantification-cohort`** dual-era rules, and MUST NOT imply that per-glom targets were obtained by automatic decomposition of legacy image-level aggregates alone
