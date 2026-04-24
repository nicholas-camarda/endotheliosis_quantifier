## Context

The repository's supported glomeruli training contract uses dynamic patching from full-image roots. The current all-data training surface is the manifest-backed `raw_data/cohorts` registry root, which enumerates admitted `manual_mask_core` and `manual_mask_external` manifest rows across localized cohorts. Lauren-only training can use the localized cohort root `raw_data/cohorts/lauren_preeclampsia`. Current training inputs do not provide explicit negative supervision at the crop level. The current admitted masked full images contain glomeruli somewhere, and the larger MR/TIFF source images that could provide additional background-only regions do not have full masks.

That means the repo is currently missing a clean way to say "this crop is a true negative for glomerulus presence" unless the crop comes from a masked image and has verified zero overlap. The next change should define a supported negative-crop annotation contract for larger source images without pretending that unlabeled crops are automatically negative.

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

## Decisions

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

4. **Future training integration should treat negative crop supervision as crop-level evidence, not whole-image negativity.**
   - Rationale: these larger MR/TIFF files come from kidneys that may contain glomeruli elsewhere; the scientifically valid claim is that a specific annotated crop is negative, not that the entire source image is glomerulus-free.

5. **Negative-crop provenance must remain visible in downstream promotion artifacts.**
   - Rationale: if negative supervision is added later, the repo needs to say so directly rather than leaving readers to infer it from changed metrics.

## Risks / Trade-offs

- [Risk] Annotating negative crops adds manual curation work. → Mitigation: allow lightweight crop-level negative annotation without requiring full-slide masks.
- [Risk] Poorly curated negative crops could include missed glomeruli and poison the training signal. → Mitigation: require explicit review provenance and auditable manifests rather than unlabeled crop mining.
- [Risk] The repo could drift back toward static patch training through convenience exports. → Mitigation: keep manifests as the contract and preserve full-image dynamic patching as the canonical training mode.
- [Risk] Users may over-interpret negative crop supervision as proof that whole-image negative detection is solved. → Mitigation: keep the contract explicitly crop-level in wording and provenance.
