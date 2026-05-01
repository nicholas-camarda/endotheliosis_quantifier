## ADDED Requirements

### Requirement: Hybrid mode SHALL preload MedSAM proposal regions without blocking cold-start labeling

When hybrid grading is enabled, the system SHALL import Label Studio tasks with MedSAM-derived proposal regions from the configured generated-mask release whenever matching mask artifacts exist, and MUST NOT refuse task import solely because some images lack preload predictions.

#### Scenario: Warm start with preload coverage

- **WHEN** `configs/label_studio_medsam_hybrid.yaml` binds a valid `mask_release_id` whose release manifest resolves masks or polygons for a subset of images under `<image-dir>`
- **THEN** each matching task MUST include preload proposal geometries consumable by the labeling UI prior to submission
- **AND** lineage metadata tying proposals to `{mask_release_id, mask_source}` MUST be persisted for deterministic export ingestion

#### Scenario: Cold start with missing preload masks

- **WHEN** an image exists under `<image-dir>` but the mask release lacks a preload artifact or metadata entry for it
- **THEN** bootstrap MUST still create a Label Studio task for that image unless an explicit YAML policy rejects missing rows
- **AND** lineage MUST record absent preload artifacts so downstream parsers can distinguish inferred vs hydrated proposals

### Requirement: Bootstrap SHALL audit available releases and default to the latest valid release

Hybrid bootstrap MUST inspect the generated-mask registry before task construction, determine the latest valid release using explicit ordering rules, and use that release when no pinned `mask_release_id` is provided.

#### Scenario: No release pinned in YAML

- **WHEN** `configs/label_studio_medsam_hybrid.yaml` omits a pinned `mask_release_id`
- **THEN** bootstrap MUST audit `derived_data/generated_masks/glomeruli/manifest.csv`
- **AND** select the latest valid release by documented precedence
- **AND** emit the selected `mask_release_id` in startup logs and provenance metadata

#### Scenario: Pinned release missing or invalid

- **WHEN** YAML pins `mask_release_id` that is absent or invalid in the registry
- **THEN** bootstrap MUST fail closed with an actionable error listing audit results and candidate valid releases

### Requirement: Box-assisted MedSAM MUST be reachable when hybrid mode requires interactive proposals

When hybrid grading is configured with companion enforcement (default-on), startup MUST probe the companion health endpoint referenced in YAML and MUST refuse to finalize bootstrap if the probe fails unless `offline_manual_only_allowed: true` is explicitly enabled for admin-only contingency.

#### Scenario: Companion healthy

- **WHEN** YAML sets `require_box_assisted_medsam: true` or equivalent default-on hybrid flag AND the companion returns success on the documented health probe
- **THEN** bootstrap completes and emits operator messaging that hybrid assist is operational

#### Scenario: Companion unavailable with strict hybrid policy

- **WHEN** strict hybrid companion enforcement is enabled AND the health probe fails
- **THEN** bootstrap terminates with actionable diagnostics before collaborators open the labeling queue

### Requirement: Collaborative CLI surface MUST remain minimally flagged

Collaborator documentation MUST cite `eq labelstudio start <image-dir>` without requiring ancillary flags; advanced knobs MUST reside in YAML, optional `--config`, or explicit environment overrides surfaced only for administrators.

#### Scenario: Collaborator runbook path

- **WHEN** a collaborator executes `eq labelstudio start /path/to/images` with default repository config resolution
- **THEN** YAML supplies MedSAM preload binding, companion URL, LS/Docker defaults, and authentication references without prompting for additional positional parameters

### Requirement: Exported hybrid lineage MUST capture proposal derivation and edits

Hybrid exports MUST annotate each authoritative record with enumerated lineage fields distinguishing auto preload vs box-assisted origin, classify human edit fidelity, preserve `mask_release_id`, tie to Label Studio `region_id`/internal instance mapping, and keep references suitable for rollup or training ingestion audits.

#### Scenario: Auto preload refined by annotator brush edits

- **WHEN** a grader submits a finalized complete-glomerulus region originally imported from preload predictions and edits segmentation boundaries prior to assigning a discrete grade
- **THEN** the emitted record MUST include proposal kind auto, edit state reflecting human refinement, authoritative grade, preserved mask release lineage, stable `glomerulus_instance_id`

#### Scenario: Box-assisted region graded after interactive assist

- **WHEN** a grader submits a finalized complete-glomerulus region synthesized through box-assisted MedSAM after preload absence
- **THEN** lineage MUST denote box-assisted derivation with companion provenance timestamps and authoritative grade linkage identical to preload-derived rows

### Requirement: Hybrid workflow SHALL treat brush refinement as the canonical graded mask primitive

MedSAM proposals imported from releases and masks returned from box-assisted inference MUST be presented and edited as **brush-compatible** region geometry in the labeling configuration (for example `brushlabels` with RLE or equivalent), so graders refine boundaries with the brush tool rather than committing final science masks as axis-aligned rectangles alone.

#### Scenario: Grader refines preload proposal

- **WHEN** a task includes preload MedSAM geometry and the grader adjusts segmentation before assigning a grade
- **THEN** the labeling configuration MUST allow brush-based edits on that geometry without forcing a rectangle-only workflow

#### Scenario: Cold image relies on box-assist

- **WHEN** an image has no usable preload for one or more glomeruli and the grader uses box-assisted MedSAM to propose initial geometry
- **THEN** the resulting region MUST remain brush-editable to the same standard as preload-derived regions

#### Scenario: Repeat review of preloaded tasks

- **WHEN** operators re-open tasks that already had preload or prior annotations (QA, grade change, boundary correction)
- **THEN** hybrid bootstrap and Label Studio project behavior MUST continue to support labeling on those tasks without requiring net-new images only

### Requirement: Hybrid mode MUST support both cold coverage and repeat review

The integration SHALL NOT assume workflows are limited to first-pass consumption of preload masks: box-assisted and manual-brush paths remain available for **unseen** coverage, while tasks with existing preload or prior work remain valid for **repeat** grading sessions.

#### Scenario: New image without release coverage

- **WHEN** an image under `<image-dir>` has no matching preload artifact in the bound mask release
- **THEN** the task MUST still be importable and graders MUST be able to obtain complete-glomerulus regions via box-assisted MedSAM (when companion policy allows), manual brush, or both

#### Scenario: Previously reviewed image returns to the queue

- **WHEN** an image already associated with a mask release or prior export re-enters the labeling queue
- **THEN** operators MUST be able to run another authoritative grading pass without a special “new-only” restriction imposed by hybrid bootstrap logic