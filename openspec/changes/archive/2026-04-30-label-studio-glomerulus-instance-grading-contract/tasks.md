## 1. Reuse Audit And Fixture Design

- 1.1 Audit `src/eq/quantification/labelstudio_scores.py`, `src/eq/quantification/cohorts.py`, `src/eq/quantification/input_contract.py`, and related tests for reusable score coercion, provenance, and fail-closed validation behavior.
- 1.2 Create minimal Label Studio export fixtures covering multiple complete glomeruli per image, cutoff exclusions, missing grade-to-region links, duplicate active grades, and named grader provenance.
- 1.3 Document the selected initial ingestion source from audited Label Studio evidence: JSON export, SDK/API sync, webhooks, or a staged combination.

## 2. Label Studio Grading Contract

- 2.1 Add `configs/label_studio_glomerulus_grading.xml` for primary blind grading of complete glomerulus instances and cutoff exclusions.
- 2.2 Ensure the Label Studio config can represent grade-to-region linkage for each complete glomerulus and explicit `cutoff_partial_glomerulus` exclusion for partial glomeruli.
- 2.3 Add documentation or inline config notes that primary grading must not show model grade, confidence, disagreement, or decision-state fields.

## 3. Ingestion And Validation

- 3.1 Add the minimal `src/eq/labelstudio/` module surface for parsing the selected Label Studio export/API shape without duplicating existing quantification score helpers.
- 3.2 Implement canonical glomerulus-instance records containing `image_id`, `glomerulus_instance_id`, region/ROI reference, completeness status, exclusion reason, human grade, grader provenance, annotation provenance, and export provenance.
- 3.3 Implement fail-closed validation for grades without linked complete regions, complete regions without grades, grades on excluded cutoff regions, duplicate active grades from one grader, missing provenance, and unstable region mappings.
- 3.4 Ensure image-level averaged historical scores are rejected for this per-glomerulus contract with a clear legacy-baseline diagnostic.

## 4. Export And Rollup Preparation

- 4.1 Emit glomerulus-level records that preserve Label Studio task, annotation, user, region, and grading provenance.
- 4.2 Emit rollup-ready tables that preserve source glomerulus record IDs and exclude cutoff candidates from image, kidney, animal, training, and model-performance aggregates.
- 4.3 Keep generated exports under configured runtime/output roots rather than Git-tracked data directories.

## 5. CLI, Docs, And Validation

- 5.1 Add focused CLI or workflow entrypoints only after confirming they reuse existing `eq` command patterns and do not bypass `eq run-config` governance unnecessarily.
- 5.2 Add user/admin documentation describing the Label Studio collaborator workflow, the `eq` ingestion boundary, and the distinction between legacy image-average data and new per-glomerulus records.
- 5.3 Add unit tests for all fixture and validation scenarios listed in the spec.
- 5.4 Run focused tests for the new Label Studio contract.
- 5.5 Run `openspec validate label-studio-glomerulus-instance-grading-contract --strict`.
- 5.6 Run `python3 scripts/check_openspec_explicitness.py label-studio-glomerulus-instance-grading-contract`.