## Why

After `medsam-glomeruli-fine-tuning` produces a usable generated-mask release, the grading workflow needs to turn those masks into editable Label Studio regions so graders can accurately assign endotheliosis scores to final accepted glomerulus regions. Without this integration, fine-tuned MedSAM can improve segmentation artifacts but still fail the actual lab workflow goal: faster, region-linked, human-owned glomerulus scoring.

## What Changes

- Add a Label Studio MedSAM region-review workflow that imports fine-tuned MedSAM generated-mask releases as preannotated glomerulus regions.
- Preserve region-level provenance from auto MedSAM proposals through human edits, exclusions, additions, deletion/rejection, and final region-level scores.
- Support the hybrid user flow selected during brainstorming: auto-generated regions first, editable/fixable masks in Label Studio, and box-assisted MedSAM region creation for missed glomeruli.
- Require final scores to attach to final accepted complete-glomerulus regions, not to image-level tasks or untracked mask files.
- Add validation and export contracts for accepted, edited, excluded, deleted/rejected, and box-assisted regions.
- Gate this workflow on a usable `medsam-glomeruli-fine-tuning` generated-mask release rather than assuming a fine-tuned checkpoint is available before the current change completes.
- Keep model grade suggestions absent during primary grading; MedSAM is used for region generation/edit support only, not visible grade assistance.

## Capabilities

### New Capabilities

- `label-studio-medsam-region-review`: Import fine-tuned MedSAM generated masks as editable Label Studio glomerulus regions, preserve region provenance, support box-assisted added regions, and export final accepted region-score records.

### Modified Capabilities

- `label-studio-glomerulus-grading`: Extend glomerulus-instance grading from manual-only region creation to MedSAM-proposed, human-edited, box-assisted, and rejected/excluded region states while preserving final region-linked scoring.
- `label-studio-local-bootstrap`: Extend the one-command bootstrap to optionally prepare MedSAM region-review projects from a generated-mask release manifest.
- `medsam-glomeruli-fine-tuning`: Require generated-mask releases intended for Label Studio review to expose enough manifest fields for import as editable regions.

## Impact

- Affected modules:
  - `src/eq/labelstudio/bootstrap.py`
  - `src/eq/labelstudio/glomerulus_grading.py`
  - likely new `src/eq/labelstudio/medsam_region_review.py` for region task construction and export validation if existing owners cannot absorb it cleanly
  - `src/eq/utils/paths.py` only if additional generated-mask or Label Studio runtime helpers are needed beyond existing runtime-root helpers
- Affected CLI:
  - `eq labelstudio start` gains an opt-in generated-mask release argument, for example `--generated-mask-release <manifest-or-release-root>`, or an explicit sibling subcommand if design proves clearer.
- Affected configs:
  - A new Label Studio XML config for MedSAM region review if the current `configs/label_studio_glomerulus_grading.xml` cannot express preannotated/editable region provenance and box-assisted region states cleanly.
- Affected tests:
  - Region task construction from generated-mask manifests.
  - Label Studio export ingestion with auto, edited, box-assisted, excluded, and rejected region states.
  - Fail-closed behavior for missing generated-mask provenance or stale/non-reviewable releases.
  - CLI/bootstrap wiring without requiring a live Label Studio server.
- Affected artifact roots:
  - Reads from `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`.
  - Writes Label Studio runtime artifacts under the existing active runtime `labelstudio/` root.
  - Writes review exports outside Git-tracked roots.
- Scientific interpretation:
  - This change supports practical annotation workflow acceleration and region-score provenance.
  - It does not by itself prove downstream grading validity, reduce observer variability, or scientifically promote the MedSAM model.

## Explicit Decisions

- Change name: `label-studio-medsam-region-review`.
- New capability: `label-studio-medsam-region-review`.
- Primary dependency: a completed or explicitly selected generated-mask release from `medsam-glomeruli-fine-tuning`.
- Default user experience: hybrid auto-first plus user correction, deletion/rejection, exclusion, and box-assisted MedSAM addition.
- Atomic scoring unit: final accepted complete-glomerulus Label Studio region.
- Required source states: `auto_medsam`, `human_edited_medsam`, `box_assisted_medsam`, `excluded_partial_glomerulus`, and `rejected_medsam_candidate`.
- Stage remains human-first for grading: model grade suggestions remain absent during primary scoring.

## Open Questions

- [audit_first_then_decide] Which exact Label Studio annotation JSON shape best represents imported MedSAM mask regions plus later human edits? Audit Label Studio prediction/preannotation export behavior before implementation.
- [audit_first_then_decide] Can box-assisted MedSAM be implemented inside the local Label Studio workflow without a custom frontend/plugin, or should the first implementation generate box-assisted masks through an `eq` sidecar/API action? Audit Label Studio extension points and local API constraints before applying.
- [resolve_before_apply] Should this be an extension of `eq labelstudio start` via `--generated-mask-release`, or a distinct command such as `eq labelstudio start-medsam-review`? Decide before implementation to avoid parallel CLI surfaces.
