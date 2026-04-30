## Why

The intended lab workflow is quantification-forward: collaborators should grade complete glomeruli in Label Studio, while `eq` independently validates the grade-to-glomerulus linkage and later acts as a second reviewer. The current usable pipeline is built around image-level averaged scores, which cannot reliably support per-glomerulus grading, named grader variability analysis, or assignment of the correct grade to the correct glomerulus within multi-glomerulus images.

This change establishes the first contract needed to align the repo with that goal: a Label Studio glomerulus-instance grading data model and validation boundary. Model deployment and second-review inference are intentionally deferred until the human annotation contract is stable.

## Explicit Decisions

- The OpenSpec change name is `label-studio-glomerulus-instance-grading-contract`.
- The new capability is `label-studio-glomerulus-grading`.
- The primary workflow unit is `image_id + glomerulus_instance_id`, not whole-image score.
- Primary grading remains human-first and blind to model output.
- Only complete glomeruli are gradeable. Cut-off or partial glomeruli must be excluded with reason `cutoff_partial_glomerulus`.
- Label Studio user identity and annotation provenance are the source of truth for named grader records.
- `eq-mac` remains the project/model runtime; Label Studio runs as a separate app or service configured by repo-owned files and commands.
- Stage 1 introduces the data contract only. Batch model second review, live Label Studio ML backend integration, and MedSAM/SAM assistance are later changes.

## What Changes

- Add a Label Studio primary grading contract for images that may contain multiple glomeruli.
- Add a canonical glomerulus-instance record schema that links each human grade to exactly one complete glomerulus ROI or region.
- Add explicit exclusion semantics for cut-off or partial glomeruli so they cannot enter animal averages, training labels, or model-performance metrics.
- Add named grader provenance requirements based on Label Studio annotation fields such as `completed_by`, annotation ID, task ID, timestamps, and lead time.
- Add validation requirements that fail closed when grades and glomerulus regions are not linked correctly.
- Add export requirements for glomerulus-level records and downstream rollups that preserve provenance.
- Defer model second-review inference until a later OpenSpec change can consume the validated glomerulus-instance contract.

## Capabilities

### New Capabilities

- `label-studio-glomerulus-grading`: Defines the Label Studio-facing glomerulus-instance grading contract, grade-to-region validation rules, cutoff exclusion semantics, grader provenance, and export schemas for per-glomerulus records.

### Modified Capabilities

- None. Existing `endotheliosis-grade-model`, `adjudication-review-workflow`, `label-free-roi-embedding-atlas`, `scored-only-quantification-cohort`, and `workflow-config-entrypoints` requirements are not changed by this first contract proposal.

## Impact

- Affected future code surfaces:
  - `src/eq/labelstudio/` for Label Studio import, parsing, validation, and export helpers.
  - `src/eq/__main__.py` for future `eq labelstudio ...` CLI subcommands after the contract is accepted.
  - `src/eq/run_config.py` only if this workflow later becomes YAML-first through `eq run-config`.
  - `src/eq/quantification/` only through explicit adapters that consume validated glomerulus-instance records.
- Affected future config and docs surfaces:
  - `configs/label_studio_glomerulus_grading.xml`
  - `configs/label_studio_glomerulus_second_review.xml` in a later second-review change.
  - Documentation for collaborator and admin Label Studio workflows.
- Affected test surfaces:
  - Label Studio export fixtures with multiple complete glomeruli, cutoff exclusions, duplicate grades, missing grade-to-region links, and named grader provenance.
  - Validation tests that ensure image-level averaged historical scores are not treated as per-glomerulus ground truth.
- Data contract impact:
  - Introduces explicit `image_id`, `glomerulus_instance_id`, ROI/region reference, completeness status, exclusion reason, grader provenance, human grade, and export provenance fields.
  - Preserves raw data, derived data, Label Studio exports, model files, and generated outputs outside Git.
- Compatibility impact:
  - Existing image-level quantification and binary triage workflows remain valid as legacy baseline and QA workflows.
  - Historical image-average labels cannot satisfy the new per-glomerulus contract without new Label Studio annotations.

## logging-contract

This change proposes a new Label Studio ingestion and export surface but does not add a new durable logging root, subprocess teeing behavior, or independent file-handler system. Any future CLI entrypoints must use the existing `eq` logging conventions and, if promoted to YAML-first execution, must participate in the existing `eq run-config` durable command-capture surface.

## docs-impact

Documentation must explain the collaborator-facing Label Studio workflow, the admin/developer ingestion boundary, the difference between legacy image-average scores and new per-glomerulus records, and the fact that collaborators do not run `eq` or manage model environments. README-level documentation should remain current-state only and must not claim model-assisted second review until a later OpenSpec change implements it.

## Open Questions

- [audit_first_then_decide] Which Label Studio source is the initial ingestion authority: JSON export, SDK/API sync, webhooks, or a combination? Audit sample exports and local collaborator access before implementation.
- [audit_first_then_decide] Can Label Studio's native region/result grouping enforce grade-to-region linkage for this task, or must `eq` enforce it entirely post-export? Audit a minimal Label Studio project with multiple image regions and per-region choices.
- [audit_first_then_decide] Which existing helpers in `src/eq/quantification/labelstudio_scores.py`, `src/eq/quantification/cohorts.py`, and `src/eq/quantification/input_contract.py` can be reused for Label Studio provenance and score validation before adding `src/eq/labelstudio/`.
- [defer_ok] Should the first deployment recipe use Docker Compose, documented standalone Label Studio setup, or both?
- [defer_ok] Should MedSAM/SAM assistance be wrapped through Label Studio interactive prediction, an offline pre-candidate generator, or a separate later service? This is not part of Stage 1.