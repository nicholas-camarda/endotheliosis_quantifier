## Why

The t52 no-mask-smoke review showed that MedSAM-assisted masks may capture glomerular regions the current glomeruli segmenter under-covers or misses, but t52 has no independent manual reference and therefore cannot answer whether MedSAM improves segmentation quality. We need a small, isolated pilot on existing admitted manual-mask rows to compare MedSAM masks, current tiled segmenter masks, and manual masks without polluting raw cohort directories.

## What Changes

- Add a pilot MedSAM/manual glomeruli comparison workflow that samples a deterministic subset of admitted manual-mask manifest rows from `raw_data/cohorts/manifest.csv`.
- Generate MedSAM oracle-prompt masks from manual-mask-derived bounding boxes using the existing MedSAM environment at `/Users/ncamarda/mambaforge/envs/medsam`, repository `/Users/ncamarda/Projects/MedSAM`, and checkpoint `/Users/ncamarda/Projects/MedSAM/work_dir/MedSAM/medsam_vit_b.pth`.
- Generate current glomeruli segmenter masks on the same pilot images using the canonical tiled high-resolution inference path rather than whole-field single-pass resize.
- Write masks, boxes, overlays, metrics, manifest selections, and provenance under `output/segmentation_evaluation/medsam_manual_glomeruli_comparison/<run_id>/`.
- Add tests that prevent high-resolution glomeruli evaluation from accidentally using the low-level whole-image resize helper as a full-field inference path.
- Do not write generated MedSAM masks, current-model masks, overlays, or metrics into `raw_data/cohorts/**/masks`.

## Capabilities

### New Capabilities
- `medsam-manual-glomeruli-comparison`: Defines the pilot workflow, artifact contract, prompt provenance, mask-output isolation, and comparison metrics for MedSAM/manual/current-segmenter glomeruli audits.

### Modified Capabilities
- `glomeruli-candidate-comparison`: No requirement changes. Existing candidate-promotion reports may be reused as model inputs, but this pilot does not promote or demote current segmentation artifacts.
- `glomeruli-overcoverage-audit`: No requirement changes. Existing threshold and category-gate evidence may inform the current-segmenter baseline, but this pilot has a separate artifact root and decision status.
- `segmentation-validation-audit`: No requirement changes. Existing validation concepts are reused where possible for metrics and provenance rather than duplicated as promotion gates.

## Impact

- Affected modules to inspect for reuse: `src/eq/quantification/endotheliosis_grade_model.py` for `_predict_tiled_segmentation_probability`, `src/eq/inference/prediction_core.py` for low-level preprocessing behavior, `src/eq/evaluation/segmentation_metrics.py` for metrics, `src/eq/training/promotion_gates.py` for binary mask and component helpers, and `src/eq/utils/paths.py` for runtime output roots.
- New or extended CLI/config surface should be explicit, likely `configs/medsam_manual_glomeruli_comparison.yaml` through `eq run-config` plus a worker module under `src/eq/evaluation/`.
- Runtime dependency boundary: project Python remains `eq-mac` for current-model loading and project metrics; MedSAM inference uses `/Users/ncamarda/mambaforge/envs/medsam/bin/python` and `/Users/ncamarda/Projects/MedSAM/MedSAM_Inference.py` or a thin equivalent adapter that records the same checkpoint and repo provenance.
- Generated output schemas: `inputs.csv`, `oracle_boxes.csv`, `medsam_masks/`, `current_segmenter_masks/`, `overlays/`, `metrics.csv`, `metric_by_source.csv`, `prompt_failures.csv`, and `summary.json`.
- Compatibility risks: MedSAM environment and checkpoint are local external dependencies, so the pilot output is audit evidence only until dependency setup and weight provenance are documented. The pilot does not alter existing raw data, model artifacts, quantification outputs, or README-facing model claims.

## Explicit Decisions

- Change name: `pilot-medsam-manual-glomeruli-comparison`.
- New capability name: `medsam-manual-glomeruli-comparison`.
- Pilot output root: `output/segmentation_evaluation/medsam_manual_glomeruli_comparison/<run_id>/` under the active runtime root.
- Default pilot run ID: `pilot_medsam_manual_glomeruli_comparison`.
- Pilot input source: admitted rows in `raw_data/cohorts/manifest.csv` with `lane_assignment` in `manual_mask_core` or `manual_mask_external`, non-empty `image_path`, non-empty `mask_path`, and existing files.
- Pilot size: 20 image/mask rows when available, sampled deterministically across cohorts and subjects with an intended split of 10 `vegfri_dox` and 10 `lauren_preeclampsia` rows when both cohorts have enough eligible rows.
- MedSAM environment: `/Users/ncamarda/mambaforge/envs/medsam/bin/python`.
- MedSAM repository: `/Users/ncamarda/Projects/MedSAM`.
- MedSAM checkpoint: `/Users/ncamarda/Projects/MedSAM/work_dir/MedSAM/medsam_vit_b.pth`.
- MedSAM prompt mode for the first pilot: oracle bounding boxes derived from manual-mask connected components.
- Current segmenter baseline: tiled full-field inference using both current glomeruli candidate artifacts, not single-pass full-field resize through `PredictionCore`.
- Transfer candidate baseline: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/glomeruli_candidate_transfer-transfer_loss-custom_s1lr1e-3_s2lr_lrfind_e20_b12_lr1e-3_sz256/glomeruli_candidate_transfer-transfer_loss-custom_s1lr1e-3_s2lr_lrfind_e20_b12_lr1e-3_sz256.pkl`.
- Scratch candidate baseline: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/scratch/glomeruli_candidate_no_mito_base-scratch_e25_b12_lr1e-3_sz256/glomeruli_candidate_no_mito_base-scratch_e25_b12_lr1e-3_sz256.pkl`.

## Open Questions

- [audit_first_then_decide] Should automatic MedSAM prompts from current segmenter proposals be included in the initial pilot or deferred until oracle-prompt performance is reviewed?
- [defer_ok] Should the pilot include downstream ROI-feature deltas in the first implementation, or should it stop at segmentation metrics and overlays before deciding whether feature stability is worth running?
