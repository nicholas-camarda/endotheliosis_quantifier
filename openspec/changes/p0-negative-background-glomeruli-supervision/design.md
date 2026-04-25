## Context

P1 established the scientific rule: unmasked MR/TIFF crops are not true negatives unless reviewed at the crop level. P3 then produced a 5-epoch MPS quick run showing both transfer and scratch candidates remain blocked by background false positives. This change moves from contract to implementation.

The current DataBlock uses `CropTransform` with `positive_focus_p=0.6`; the remaining 40% of crops are random, but there is no explicit target for mask-proven zero-foreground background crops. The current DataBlock also always applies FastAI `aug_transforms` with repo constants, while `configs/glomeruli_finetuning_config.yaml` contains an `augmentation:` section that is not a real control surface for candidate comparison. Therefore augmentation has not been rigorously audited or ablated.

## Implementation Shape

## Explicit Decisions

- Implement mask-derived background crop generation before attempting production retraining.
- Keep curated MR/TIFF negative crops review-gated; proposals are not trainable.
- Preserve dynamic full-image patching as the canonical training mode.
- Add negative/background supervision as manifest-backed sampler input, not as active static patch directories.
- Treat augmentation as an audited secondary axis; do not change augmentation and negative supervision simultaneously in the first production run.
- Reject unsupported augmentation names rather than ignoring them.

### 1. Negative Crop Types

Mask-derived background crops:
- Source image has a paired segmentation mask.
- Candidate crop box is accepted only when the mask crop has zero foreground pixels after the same coordinate-frame interpretation used for training.
- These crops can be trainable without manual review because the paired mask supplies direct negative evidence.

Curated MR/TIFF negative crops:
- Source image has no full segmentation mask.
- The generator may create proposal rows and review panels.
- A crop becomes trainable only after a reviewed manifest row records `label=negative_glomerulus`, `annotation_status=reviewed_negative`, and `negative_scope=crop_only`.

### 2. Storage Contract

Generated artifacts live under runtime `derived_data/glomeruli_negative_crops/`:
- `manifests/<curation_id>.csv`
- `audits/<curation_id>.json`
- `review_assets/<curation_id>/`

The implementation must not copy these crops into active training-root `images/` and `masks/` directories.

### 3. Sampler Contract

The canonical training dataset remains full-image dynamic patching. The sampler gains an additional source of supervised crop specs:

- normal positive-aware dynamic crops from full image/mask pairs
- optional mask-derived background crops from admitted paired rows
- optional curated reviewed negative crops from unmasked sources

For negative crop samples, the returned mask must be an all-zero mask at the same network output size as the current training crop. The provenance must record counts and manifest hash.

### 4. Configuration Contract

`configs/glomeruli_candidate_comparison.yaml` gains a `negative_background_supervision:` section with explicit defaults:

```yaml
negative_background_supervision:
  enabled: true
  mask_derived_background:
    enabled: true
    curation_id: latest_run_mask_background
    crops_per_image_limit: 2
    sampler_weight: 0.5
    min_foreground_pixels: 0
  curated_negative_manifest:
    enabled: false
    manifest_path: null
    sampler_weight: 0.5
```

If `curated_negative_manifest.enabled` is true, missing or invalid manifests are hard failures.

### 5. Augmentation Audit Contract

The implementation records the actual augmentation policy in training provenance and comparison reports:

- current FastAI `aug_transforms` settings from constants
- whether config augmentation controls were honored
- selected augmentation variant
- whether gaussian noise is active

Initial implementation should not silently add gaussian noise or arbitrary stain jitter. Augmentation ablations should be explicit configs or command flags because otherwise they will confound the negative-supervision intervention.

### 6. Promotion Interpretation

Reports may say candidates are `available_research_use` if they train and load. They may not say promoted unless background category gates clear after negative/background supervision and deterministic review artifacts show non-degenerate behavior.

## Risks / Trade-offs

- [Risk] Overweighting background crops can suppress positive recall. Mitigation: require category-level recall and positive/boundary metrics in quick ablations before production training.
- [Risk] Curated MR/TIFF proposal rows could be mistaken for reviewed negatives. Mitigation: proposal status is non-trainable, and validator rejects anything but reviewed negative rows.
- [Risk] Augmentation changes could hide whether background supervision fixed the failure. Mitigation: first implementation keeps augmentation fixed and records later ablations separately.
- [Risk] The sampler implementation could become a hidden second training root. Mitigation: manifests contain crop specs; source pixels are read from canonical source images at runtime.

## Required Validation

- Unit tests for manifest validation.
- Unit tests for mask-derived background crop generation with zero-mask-overlap guarantees.
- Unit tests for rejecting unreviewed MR/TIFF proposal rows as training input.
- Integration test that a dry-run config exposes negative/background supervision controls and provenance fields.
- Focused training smoke test on a tiny synthetic dataset proving the sampler can emit all-zero negative masks without static patch roots.
- Candidate-comparison smoke run with short epochs after implementation, followed by review of background false-positive fractions.
