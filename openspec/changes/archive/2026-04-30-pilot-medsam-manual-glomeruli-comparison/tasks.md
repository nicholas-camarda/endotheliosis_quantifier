## 1. Reuse Audit And Control Surface

- [x] 1.1 Inspect `src/eq/quantification/endotheliosis_grade_model.py`, `src/eq/inference/prediction_core.py`, `src/eq/evaluation/segmentation_metrics.py`, `src/eq/training/promotion_gates.py`, and `src/eq/utils/paths.py` for reusable tiled inference, mask loading, metrics, component, and runtime-path owners before adding new helpers.
- [x] 1.2 Add `configs/medsam_manual_glomeruli_comparison.yaml` with `workflow: medsam_manual_glomeruli_comparison`, explicit MedSAM environment paths, current segmenter artifact paths, pilot sampling settings, tiling settings, and output roots.
- [x] 1.3 Register the workflow in the existing `eq run-config` dispatch path without creating a competing shell-script orchestration surface.

## 2. Input Selection And Box Contracts

- [x] 2.1 Implement deterministic admitted manual-mask manifest selection from `raw_data/cohorts/manifest.csv` with cohort/subject diversity and a default 20-row pilot target.
- [x] 2.2 Write `inputs.csv` with manifest, cohort, lane, subject, image, mask, selection rank, and selection reason fields.
- [x] 2.3 Implement manual-mask connected-component box derivation with minimum area, padding, image-bound clipping, and multi-component support.
- [x] 2.4 Write `oracle_boxes.csv` and prompt provenance fields for every selected component and skipped component.

## 3. MedSAM And Current Segmenter Generation

- [x] 3.1 Implement MedSAM dependency preflight for `/Users/ncamarda/mambaforge/envs/medsam/bin/python`, `/Users/ncamarda/Projects/MedSAM`, `/Users/ncamarda/Projects/MedSAM/MedSAM_Inference.py`, and `/Users/ncamarda/Projects/MedSAM/work_dir/MedSAM/medsam_vit_b.pth`.
- [x] 3.2 Implement MedSAM oracle-box inference as an external-runtime adapter that saves one union mask per selected image under `medsam_masks/` and records prompt command, checkpoint, device, and return-code provenance.
- [x] 3.3 Record MedSAM prompt and image failures in `prompt_failures.csv` without substituting successful-looking fallback masks.
- [x] 3.4 Reuse or centralize tiled current-segmenter inference so pilot masks are generated under `current_segmenter_masks/` with tile size, stride, threshold, expected size, model path, and preprocessing provenance.
- [x] 3.5 Add a guard or test-covered contract that rejects whole-field single-pass `PredictionCore` resizing for high-resolution pilot images.

## 4. Metrics, Overlays, And Reports

- [x] 4.1 Compute per-image segmentation metrics against the same manual masks for MedSAM oracle masks and every configured current-segmenter candidate.
- [x] 4.2 Write `metrics.csv` with method, candidate artifact, manifest row, cohort, lane, Dice, Jaccard, precision, recall, pixel accuracy, foreground fractions, area ratio, component counts, and bbox summaries.
- [x] 4.3 Write `metric_by_source.csv` grouped by method, candidate artifact, cohort, and lane assignment.
- [x] 4.4 Write visual overlays or review panels showing raw image, manual mask, MedSAM mask, current-segmenter mask, oracle boxes, and method labels.
- [x] 4.5 Write `summary.json` with workflow ID, run ID, config path, runtime root, package availability, MedSAM provenance, current artifact provenance, selected inputs, generated outputs, aggregate metrics, prompt failures, and audit-scoped interpretation.

## 5. Tests And Validation

- [x] 5.1 Add unit tests for deterministic pilot selection and cohort/subject balancing from a fixture manifest.
- [x] 5.2 Add unit tests for connected-component oracle box generation, padding, skipped components, and clipped image bounds.
- [x] 5.3 Add unit tests proving generated output paths remain under the runtime segmentation-evaluation output root and never under `raw_data/cohorts/**/masks`.
- [x] 5.4 Add unit tests for the high-resolution current-segmenter guard so whole-field single-pass resize is not accepted as pilot evidence.
- [x] 5.5 Add unit tests for metrics/report schema generation using small synthetic masks without invoking MedSAM.
- [x] 5.6 Run `python -m pytest -q` or the focused test subset plus import/CLI smoke checks for the new workflow.
- [x] 5.7 Run `openspec validate pilot-medsam-manual-glomeruli-comparison --strict`.

## 6. Pilot Execution

- [x] 6.1 Run a dry-run config execution to verify selected inputs, dependency preflight, output paths, and provenance without writing generated masks.
- [x] 6.2 Run the real 20-row MedSAM/manual pilot using the local `medsam` environment and configured current segmenter artifacts.
- [x] 6.3 Review `summary.json`, `metrics.csv`, `metric_by_source.csv`, `prompt_failures.csv`, and overlays for whether oracle MedSAM performance justifies automatic-prompt MedSAM or downstream ROI-feature delta work.
- [x] 6.4 Record the pilot command, output path, key metrics, failures, and next decision in the change notes before treating implementation as complete.