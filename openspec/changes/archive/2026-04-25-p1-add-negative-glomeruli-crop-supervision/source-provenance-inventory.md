## Source Provenance Inventory

Inventory date: 2026-04-24

Runtime root inspected:

- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier`

Evidence inspected:

- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/manifest.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/lauren_preeclampsia/metadata/source_audit.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/vegfri_mr/metadata/source_audit.csv`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data`

## Findings

The admitted glomeruli training surface is paired image/mask data, not explicit negative-crop data.

- `raw_data/cohorts/manifest.csv` has 1,079 rows.
- 707 rows are `admitted`.
- 88 admitted rows are from `lauren_preeclampsia`; all 88 have image and mask paths.
- 619 admitted rows are from `vegfri_dox`; all admitted training rows have image and mask paths.
- `vegfri_mr` contributes 127 rows, but those rows are `mr_concordance_only` / `evaluation_only`, not admitted glomeruli training rows, and they have no mask paths.

Lauren's localized root does not currently provide provenance back to larger MR/TIFF source images.

- `raw_data/cohorts/lauren_preeclampsia/metadata/source_audit.csv` has 88 rows.
- Its `source_image_path` and `source_mask_path` point to the localized runtime JPG/mask files under `raw_data/cohorts/lauren_preeclampsia`.
- It does not identify larger MR/TIFF source fields, crop boxes, or crop-level negative annotations.

The MR TIFF surface is source/evaluation material, not trainable negative supervision.

- `raw_data/cohorts/vegfri_mr/images/` contains 127 TIFF files.
- `raw_data/cohorts/vegfri_mr/metadata/source_audit.csv` maps those runtime TIFFs back to `/Volumes/USB EXT2020/.../kidney/images/...`.
- `source_mask_path` and `runtime_mask_path` are empty for the MR TIFF rows.
- No `masks/` directory or mask-like files were found under `raw_data/cohorts/vegfri_mr`.

No generated negative-crop contract artifacts exist yet.

- `derived_data/` currently contains `cohort_manifest/` only.
- There is no current `derived_data/glomeruli_negative_crops/` manifest, audit, or review-asset tree.

## Conclusion

The current runtime has plausible larger-source TIFF material for future background/negative curation, but it does not have supported true-negative glomeruli crops. Any future negative supervision must be created as crop-level annotations with source path, crop box, negative label, and review provenance. Current training and promotion workflows must continue to report curated negative-crop supervision as absent until those artifacts exist.
