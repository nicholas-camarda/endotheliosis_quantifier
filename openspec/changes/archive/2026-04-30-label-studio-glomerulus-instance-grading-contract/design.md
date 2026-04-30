## Context

The repo currently supports Label Studio-derived image-level grades joined to image/mask pairs, followed by quantification, embedding, atlas, and binary review-triage workflows. That is useful for the current image-level baseline, but the intended lab workflow is per-glomerulus: collaborators should grade each complete glomerulus, cut-off glomeruli should be excluded, and downstream animal-level scores should be derived from traceable glomerulus-instance records.

Multiple glomeruli per image are common. Therefore, the key correctness problem is not simply importing a Label Studio score. The system must preserve which grade belongs to which complete glomerulus region within the image, who made that grade, and whether the region was excluded because it was cut off.

This change defines the Stage 1 contract. It does not deploy a model, add MedSAM/SAM assistance, or create a live Label Studio ML backend. Those later workflows must consume this contract rather than inventing a parallel schema.

## Goals / Non-Goals

**Goals:**

- Define a Label Studio primary grading contract for images with multiple glomeruli.
- Define the canonical `image_id + glomerulus_instance_id` grading unit.
- Require grade-to-ROI linkage for every complete glomerulus grade.
- Require explicit cutoff exclusion semantics.
- Preserve named grader provenance from Label Studio annotations.
- Define validation and export behavior that can later support model second review, adjudication, variability analysis, and animal-level rollups.
- Keep Label Studio as a collaborator-facing app and `eq` as the repo-owned ingestion, validation, and export engine.

**Non-Goals:**

- Do not expose model grades during primary human grading.
- Do not implement batch model second review in this change.
- Do not implement a live Label Studio ML backend in this change.
- Do not implement MedSAM/SAM-assisted interaction in this change.
- Do not treat image-level averaged historical scores as per-glomerulus ground truth.
- Do not require collaborators to run `eq`, conda, Python scripts, or model commands.
- Do not install Label Studio into the `eq-mac` environment.

## Explicit Decisions

- The first implementation owner should be `src/eq/labelstudio/` because the contract is Label Studio-specific and should not overload the existing image-level `src/eq/quantification/labelstudio_scores.py` path.
- Existing `src/eq/quantification/labelstudio_scores.py`, `src/eq/quantification/cohorts.py`, and `src/eq/quantification/input_contract.py` must be audited and reused where they already own score coercion, provenance reference, or quantification input validation concepts.
- The first Label Studio config filename should be `configs/label_studio_glomerulus_grading.xml`.
- A later second-review config should use `configs/label_studio_glomerulus_second_review.xml`, but that file is not part of Stage 1 unless needed for schema validation examples.
- A future CLI surface should use `eq labelstudio ...` subcommands for import, validation, and export. This exact CLI surface is allowed only after the OpenSpec tasks define its command names and reuse audit.
- Stage 1 output should be glomerulus-level records first. Image, kidney, and animal rollups are derived outputs and must preserve the source glomerulus record IDs.
- Cut-off or partial glomeruli use `exclusion_reason = cutoff_partial_glomerulus`.
- Model-related fields may be reserved for future compatibility, but Stage 1 must not populate model grades or model decision states.

## Decisions

### Label Studio owns interaction; `eq` owns contracts

Collaborators should interact only with Label Studio. This keeps the daily workflow accessible and avoids asking lab members to manage runtime environments. `eq` ingests Label Studio exports or API records, validates the contract, and emits clean outputs.

Alternative considered: make a custom grading UI in this repo. That would provide tighter control, but it would duplicate Label Studio assignment, login, annotation, export, and review features. It is not justified for Stage 1.

### The atomic unit is a glomerulus instance

The canonical unit is `image_id + glomerulus_instance_id`. Whole-image average scores are preserved only as legacy baseline data. This is necessary because multi-glomerulus images are common and grade-to-region linkage is the scientific invariant.

Alternative considered: keep one image-level grade and let future models infer glomerulus contributions. That cannot verify which grade belongs to which glomerulus and cannot support reliable per-glomerulus training, adjudication, or variability analysis.

### Primary grading remains model-blind

The first workflow must collect independent human grades before any model comparison. This protects against anchoring bias and leaves room to measure whether the model truly improves workload and quality.

Alternative considered: import model preannotations before human grading. That may save time, but it weakens the ability to measure inter-observer variability and model-human disagreement. It belongs in a later, explicitly evaluated workflow.

### Cutoff exclusion is explicit

Every visible candidate must become either a complete gradeable glomerulus or an excluded cutoff/partial glomerulus. Exclusions are data, not silent omissions.

Alternative considered: ignore partial glomeruli without recording them. That makes it hard to audit whether graders skipped true complete glomeruli or appropriately excluded partial anatomy.

### Validation fails closed

`eq` must reject ambiguous or incomplete Label Studio exports rather than silently coerce them. Hard failures are appropriate for missing grade-to-region links, duplicate active grades, grades attached to excluded glomeruli, missing grader provenance, or unstable region identifiers.

Alternative considered: warn and continue. That would allow silent corruption of the primary scientific data contract.

## Risks / Trade-offs

- Region/choice linkage may be awkward in Label Studio → Mitigation: audit a minimal project with multiple regions and per-region choices before implementation, then encode any Label Studio-specific constraints in tests.
- Adding `src/eq/labelstudio/` could duplicate existing score parsing → Mitigation: audit and reuse `labelstudio_scores.py`, `cohorts.py`, and `input_contract.py` before adding new helpers.
- Collaborators may find per-glomerulus region assignment slower than image-level grading → Mitigation: Stage 1 prioritizes correctness; MedSAM/SAM or imported candidate regions can be added later to reduce friction.
- Historical image-average data will not satisfy the new contract → Mitigation: preserve it as a baseline and do not relabel it as per-glomerulus evidence.
- Named grader fields may expose personally identifying information → Mitigation: preserve Label Studio identity internally, and add pseudonymized export options in a later governance or export-focused change if needed.

## Migration Plan

1. Create the Stage 1 Label Studio config and fixtures in the implementation change.
2. Implement ingestion and validation for exported or API-synced Label Studio annotations.
3. Validate sample Label Studio exports covering complete glomeruli, cutoff exclusions, missing links, duplicate grades, and multiple graders.
4. Emit glomerulus-level records and rollup-ready exports under configured runtime output roots, not in Git.
5. Keep existing image-level quantification workflows unchanged.

Rollback is straightforward because Stage 1 adds a new contract and new surfaces. If the workflow is not ready, stop using the new Label Studio config and continue using the existing image-level workflows.

## Open Questions

- [audit_first_then_decide] Which Label Studio source is the initial ingestion authority: JSON export, SDK/API sync, webhooks, or a combination? Audit sample Label Studio exports and project API access before implementation.
- [audit_first_then_decide] Can Label Studio region/result grouping enforce grade-to-region linkage for this task, or must `eq` enforce it post-export? Audit a minimal Label Studio project with multiple image regions and per-region choices.
- [audit_first_then_decide] Which existing code in `src/eq/quantification/labelstudio_scores.py`, `src/eq/quantification/cohorts.py`, and `src/eq/quantification/input_contract.py` should be reused before introducing helpers in `src/eq/labelstudio/`?
- [defer_ok] Should the first reproducible Label Studio deployment recipe use Docker Compose, a documented standalone install, or both?
- [defer_ok] Should later glomerulus-assistance use MedSAM/SAM through Label Studio interactive prediction, offline candidate generation, or a separate service?