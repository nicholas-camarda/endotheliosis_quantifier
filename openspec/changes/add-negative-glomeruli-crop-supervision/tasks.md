## 1. Negative Crop Contract

- [ ] 1.1 Inventory what source provenance exists, if any, from the admitted `raw_data/cohorts` manifest rows and the direct `raw_data/preeclampsia_project/data` paired root back to larger MR/TIFF source images.
- [ ] 1.2 Define the minimum annotation fields required for a supported negative crop manifest, including source image path, crop box, label, and review provenance.
- [ ] 1.3 Define canonical storage locations for source images versus generated negative-crop manifests, audits, and review assets.

## 2. Training And Promotion Integration

- [ ] 2.1 Specify how future glomeruli training may consume curated negative crop manifests without reviving static patch dataset directories as active training inputs.
- [ ] 2.2 Specify what downstream provenance fields must record whether curated negative crop supervision was used.
- [ ] 2.3 Specify what glomeruli promotion or comparison reports must disclose about negative-crop coverage when such supervision is present.

## 3. Validation And Documentation

- [ ] 3.1 Add or update specs so unlabeled large-image crops are never treated as supported true negatives.
- [ ] 3.2 Document the crop-level, not whole-image, interpretation of curated negative supervision.
- [ ] 3.3 Run `env OPENSPEC_TELEMETRY=0 openspec validate add-negative-glomeruli-crop-supervision --strict`.
