## Why

The current glomeruli promotion workflow exposed a real limitation in the available supervision: the supported all-data training surface now comes from the manifest-backed `raw_data/cohorts` registry root, which enumerates admitted `manual_mask_core` and `manual_mask_external` rows, and Lauren-only training can use `raw_data/cohorts/lauren_preeclampsia`. These full images contain glomeruli somewhere, and the larger MR/TIFF source images that could supply background-only crop context do not have masks. That means the repository cannot currently treat arbitrary crops from those larger source images as defensible true negatives.

This should be tracked as a separate change rather than left as an implicit complaint inside glomeruli promotion. The problem is not "transfer versus scratch" anymore. The problem is that the repo lacks a supported contract for curating, auditing, and using explicitly annotated negative glomeruli crops from larger source images without turning the project back into a static-patch training workflow.

## What Changes

- Define a supported negative-glomeruli-crop annotation contract for larger MR/TIFF source images that do not have full segmentation masks.
- Specify where negative-crop manifests, review artifacts, and provenance belong relative to `raw_data` and `derived_data`.
- Require explicit negative-crop annotation or source mapping before a crop may be treated as a true negative; unannotated large-image crops must not silently become negative supervision.
- Define how future glomeruli training and promotion workflows may consume these negative crops without recreating static exported patch datasets as the active training input.

## Explicit Decisions

- Change ID: `p2-add-negative-glomeruli-crop-supervision`.
- Negative-crop source material: larger MR/TIFF glomeruli source images that do not have full segmentation masks.
- Supported negative example contract: a crop may be treated as a true negative only when an auditable manifest row records source image path, crop box, negative label, and review provenance or equivalent source-to-mask mapping.
- Source/artifact boundary: source images remain under `raw_data`; generated negative-crop manifests, audits, and review assets belong under `derived_data`.
- Training contract: curated negative crops must enter future glomeruli training through manifests and samplers, not through active static patch dataset directories.
- Interpretation contract: curated negative supervision is crop-level evidence only and must not be represented as whole-image negativity.

## Impact

- Affected specs: `segmentation-training-contract`, `glomeruli-candidate-comparison`, `negative-glomeruli-crop-supervision`
- Affected code: future glomeruli data curation, crop-manifest generation, training samplers, provenance, and promotion audits
- Affected artifacts: negative-crop annotation manifests, review panels, curation audit summaries, and future training-side provenance that records whether curated negative crops were used
