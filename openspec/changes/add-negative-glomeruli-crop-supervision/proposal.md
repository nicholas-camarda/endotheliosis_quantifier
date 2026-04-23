## Why

The current glomeruli promotion workflow exposed a real limitation in the available supervision: the supported `training_pairs` cohort contains glomerulus-positive full images only, and the larger MR/TIFF source images that could supply background-only crop context do not have masks. That means the repository cannot currently treat arbitrary crops from those larger source images as defensible true negatives.

This should be tracked as a separate change rather than left as an implicit complaint inside glomeruli promotion. The problem is not "transfer versus scratch" anymore. The problem is that the repo lacks a supported contract for curating, auditing, and using explicitly annotated negative glomeruli crops from larger source images without turning the project back into a static-patch training workflow.

## What Changes

- Define a supported negative-glomeruli-crop annotation contract for larger MR/TIFF source images that do not have full segmentation masks.
- Specify where negative-crop manifests, review artifacts, and provenance belong relative to `raw_data` and `derived_data`.
- Require explicit negative-crop annotation or source mapping before a crop may be treated as a true negative; unannotated large-image crops must not silently become negative supervision.
- Define how future glomeruli training and promotion workflows may consume these negative crops without recreating static exported patch datasets as the active training input.

## Impact

- Affected specs: `segmentation-training-contract`, `glomeruli-candidate-comparison`, `negative-glomeruli-crop-supervision`
- Affected code: future glomeruli data curation, crop-manifest generation, training samplers, provenance, and promotion audits
- Affected artifacts: negative-crop annotation manifests, review panels, curation audit summaries, and future training-side provenance that records whether curated negative crops were used
