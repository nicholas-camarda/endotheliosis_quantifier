## Context

The repository's supported glomeruli training contract uses dynamic patching from full-image roots. The current all-data training surface is the manifest-backed `raw_data/cohorts` registry root, which enumerates admitted `manual_mask_core` and `manual_mask_external` manifest rows across localized cohorts. Lauren-only training can use the localized cohort root `raw_data/cohorts/lauren_preeclampsia`. Current training inputs do not provide explicit negative supervision at the crop level. The current admitted masked full images contain glomeruli somewhere, and the larger MR/TIFF source images that could provide additional background-only regions do not have full masks.

That means the repo is currently missing a clean way to say "this crop is a true negative for glomerulus presence" unless the crop comes from a masked image and has verified zero overlap. The next change should define a supported negative-crop annotation contract for larger source images without pretending that unlabeled crops are automatically negative.

## Source Inventory Result

Runtime evidence is recorded in `source-provenance-inventory.md`.

- The active runtime root is `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier`.
- `raw_data/cohorts/manifest.csv` contains 707 admitted paired image/mask rows: 88 from `lauren_preeclampsia` and 619 from `vegfri_dox`.
- `vegfri_mr` contains 127 runtime TIFF images and source-audit paths back to `/Volumes/USB EXT2020/.../kidney/images/...`, but it has no masks and no admitted trainable image/mask rows.
- Lauren's `source_audit.csv` maps localized JPG/mask pairs to themselves; it does not provide larger MR/TIFF provenance or crop boxes.
- `derived_data/` contains no current negative-crop manifest, audit, or review-asset tree.

Current conclusion: the repo has source material that can be curated into negative crops later, but it does not currently have supported true-negative glomeruli crops.

## Goals / Non-Goals

**Goals**
- Define a supported contract for curating negative glomeruli crop annotations from larger MR/TIFF source images without full segmentation masks.
- Keep `raw_data` source images distinct from generated manifests, review artifacts, and audits under `derived_data`.
- Make explicit annotation or source mapping mandatory before a crop is treated as a true negative.
- Preserve the full-image dynamic training contract as the canonical training mode; negative crops should enter through manifests and samplers, not through a revived static patch dataset tree.
- Record enough provenance that later training and promotion artifacts can state whether curated negative crop supervision was used.

**Non-Goals**
- Implement the full negative-crop curation UI in this change.
- Rebuild the glomeruli promotion decision logic from scratch.
- Treat pseudo-negative unlabeled MR/TIFF crops as scientifically equivalent to annotated negatives.
- Require full segmentation masks for the larger MR/TIFF source images.

## Explicit Decisions

1. **Negative glomeruli crops from unmasked source images require explicit annotation.**
   - Rationale: without a mask, box annotation, or reliable source-to-mask mapping, the repo cannot defensibly claim that a crop contains no glomerulus.
   - Consequence:
     - unlabeled crops from larger MR/TIFF images remain source material only
     - a crop becomes a supported negative example only after explicit annotation or equivalent provenance-backed source mapping

2. **Negative crop curation uses manifests, not active static patch directories.**
   - Rationale: the project already retired static patch datasets as active training inputs; solving negative supervision should not recreate that architecture.
   - Consequence:
     - canonical outputs are annotation manifests, audits, and review assets
     - future training code may sample from these manifests while keeping full-image roots as the source of truth

3. **Raw source images stay in `raw_data`; generated manifests and audits belong in `derived_data`.**
   - Rationale: the repository already distinguishes source material from generated data products, and negative crop curation should follow the same rule.
   - Canonical source roots:
     - `raw_data/cohorts/vegfri_mr/images/` for localized MR TIFF source material
     - `raw_data/cohorts/vegfri_mr/metadata/source_audit.csv` for source-to-runtime image provenance
   - Canonical generated roots:
     - `derived_data/glomeruli_negative_crops/manifests/<curation_id>.csv`
     - `derived_data/glomeruli_negative_crops/audits/<curation_id>.json`
     - `derived_data/glomeruli_negative_crops/review_assets/<curation_id>/`
   - Generated crop images or thumbnails, if any, are review assets, not active training roots.

4. **Future training integration should treat negative crop supervision as crop-level evidence, not whole-image negativity.**
   - Rationale: these larger MR/TIFF files come from kidneys that may contain glomeruli elsewhere; the scientifically valid claim is that a specific annotated crop is negative, not that the entire source image is glomerulus-free.

5. **Negative-crop provenance must remain visible in downstream promotion artifacts.**
   - Rationale: if negative supervision is added later, the repo needs to say so directly rather than leaving readers to infer it from changed metrics.

6. **Minimum supported negative-crop manifest fields are fixed.**
   - Required identity and source fields:
     - `negative_crop_id`
     - `source_image_path`
     - `source_image_sha256`
     - `source_cohort_id`
   - Required crop geometry fields:
     - `crop_x_min`
     - `crop_y_min`
     - `crop_x_max`
     - `crop_y_max`
     - `coordinate_frame`
   - Required annotation fields:
     - `label`
     - `annotation_status`
     - `reviewer_id`
     - `reviewed_at_utc`
     - `review_batch_id`
     - `review_protocol_version`
   - Required interpretation/provenance fields:
     - `negative_scope`
     - `source_mapping_method`
     - `source_mapping_status`
     - `notes`
   - `label` must be `negative_glomerulus` for supported negative crops.
   - `negative_scope` must be `crop_only`; whole-image negativity is not a supported inference.

7. **Training provenance must disclose negative supervision state.**
   - Required downstream provenance fields:
     - `negative_crop_supervision_status`: `absent`, `present`, or `unsupported`
     - `negative_crop_manifest_path`
     - `negative_crop_manifest_sha256`
     - `negative_crop_count`
     - `negative_crop_source_image_count`
     - `negative_crop_review_protocol_version`
     - `negative_crop_sampler_weight`
   - If no curated manifest is supplied, reports must say `negative_crop_supervision_status=absent`; they must not infer negatives from unmasked MR TIFFs.

## Risks / Trade-offs

- [Risk] Annotating negative crops adds manual curation work. → Mitigation: allow lightweight crop-level negative annotation without requiring full-slide masks.
- [Risk] Poorly curated negative crops could include missed glomeruli and poison the training signal. → Mitigation: require explicit review provenance and auditable manifests rather than unlabeled crop mining.
- [Risk] The repo could drift back toward static patch training through convenience exports. → Mitigation: keep manifests as the contract and preserve full-image dynamic patching as the canonical training mode.
- [Risk] Users may over-interpret negative crop supervision as proof that whole-image negative detection is solved. → Mitigation: keep the contract explicitly crop-level in wording and provenance.

## Required Follow-On Implementation

This change deliberately stops at the contract boundary. The current runtime inventory found MR TIFF source material but no supported true-negative crop annotations. A follow-on implementation change is required before retraining should be expected to address the observed false-positive / over-segmentation failure mode.

That follow-on change should implement:

1. a review-batch generator that proposes candidate crop boxes from `raw_data/cohorts/vegfri_mr/images/`;
2. a manifest validator for the required `derived_data/glomeruli_negative_crops/manifests/<curation_id>.csv` fields;
3. review assets under `derived_data/glomeruli_negative_crops/review_assets/<curation_id>/`;
4. training sampler support that consumes validated negative-crop manifests without static patch roots;
5. model and comparison-report provenance fields for `negative_crop_supervision_status=present`;
6. a fresh glomeruli candidate-comparison run after curated negative crops exist.
