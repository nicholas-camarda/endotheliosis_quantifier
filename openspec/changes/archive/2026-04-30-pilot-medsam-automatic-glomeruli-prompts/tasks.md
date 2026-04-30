## 1. Reuse And Control Surface

- [x] 1.1 Inspect `src/eq/evaluation/run_medsam_manual_glomeruli_comparison_workflow.py`, `src/eq/quantification/endotheliosis_grade_model.py`, `src/eq/inference/prediction_core.py`, `src/eq/training/promotion_gates.py`, and `src/eq/utils/paths.py` for reusable input selection, MedSAM execution, tiled inference, metric, component, and output-isolation owners.
- [x] 1.2 Factor shared MedSAM/manual workflow helpers into `src/eq/evaluation/medsam_glomeruli_workflow.py` when reuse would otherwise duplicate selection, mask loading, MedSAM batch execution, metric row generation, or output-isolation behavior.
- [x] 1.3 Add `configs/medsam_automatic_glomeruli_prompts.yaml` with `workflow: medsam_automatic_glomeruli_prompts`, MedSAM environment paths, current segmenter artifact paths, proposal thresholds, proposal post-processing settings, pilot settings, and output roots.
- [x] 1.4 Register `medsam_automatic_glomeruli_prompts` in the existing `eq run-config` dispatch path.

## 2. Proposal Box Generation

- [x] 2.1 Implement tiled probability generation for configured current segmenter candidates using the existing high-resolution tiled inference owner, with model loading once per candidate.
- [x] 2.2 Implement connected-component proposal box derivation from probability masks with threshold, minimum area, maximum area, padding, clipping, merge IoU, and max-box guardrails.
- [x] 2.3 Write `proposal_boxes.csv` with generated, skipped, merged, and overflow proposal decisions and enough provenance to trace each box to candidate artifact and threshold.
- [x] 2.4 Compute `proposal_recall.csv` by matching proposal boxes to manual connected components on the 20-row pilot.

## 3. MedSAM Automatic Inference

- [x] 3.1 Reuse the MedSAM dependency preflight and external-runtime adapter from the manual pilot without adding MedSAM to the project package dependencies.
- [x] 3.2 Run MedSAM using automatic proposal boxes and save one union mask per selected image, candidate artifact, and selected threshold under `medsam_auto_masks/`.
- [x] 3.3 Record prompt failures in `prompt_failures.csv` without substituting manual, oracle, current-segmenter, or empty masks as successful automatic outputs.
- [x] 3.4 Reuse or reference oracle MedSAM outputs from the completed manual pilot when configured, so automatic-prompt performance can be compared to oracle-prompt performance without rerunning oracle masks unnecessarily.

## 4. Metrics, Gates, Overlays, And Derived Outputs

- [x] 4.1 Compute per-image segmentation metrics for automatic MedSAM masks, current-segmenter masks, and oracle MedSAM masks or references against the same manual masks.
- [x] 4.2 Write `metrics.csv` and `metric_by_source.csv` with method, prompt mode, candidate artifact, proposal threshold, cohort, lane, Dice, Jaccard, precision, recall, pixel accuracy, foreground fractions, area ratio, component counts, and bbox summaries.
- [x] 4.3 Write visual overlays or review panels showing raw image, manual mask, oracle MedSAM mask when available, automatic MedSAM mask, current-segmenter evidence, proposal boxes, missed manual components, and method labels.
- [x] 4.4 Write `summary.json` with workflow ID, run ID, config path, runtime root, MedSAM provenance, current artifact provenance, proposal recall, oracle gap, prompt failures, aggregate metrics, gate decisions, and audit-scoped interpretation.
- [x] 4.5 Implement broad derived-mask generation only as an opt-in gated mode that writes to `output/derived_masks/medsam_automatic_glomeruli/<run_id>/` and never to raw cohort mask directories.
- [x] 4.6 Add summary gate fields for `recommended_generated_mask_source`, `mask_source`, `primary_segmenter_transition_status`, and `fine_tuning_recommendation`.
- [x] 4.7 Classify failure mode as `proposal_localization`, `medsam_boundary_quality`, `downstream_integration`, or `none_detected` so the next step is box-proposer improvement, MedSAM/SAM fine-tuning, or primary generated-mask transition.

## 5. Primary Segmenter Transition And Documentation

- [x] 5.1 Add derived-mask manifest fields that make downstream opt-in explicit: source image, derived mask path, `mask_source=medsam_automatic_glomeruli`, proposal source, candidate artifact, threshold, MedSAM checkpoint, run ID, and generation status.
- [x] 5.2 Add config or summary documentation for how a later workflow should opt into `medsam_automatic_glomeruli` while retaining current segmenter artifacts as fallback/comparator evidence.
- [x] 5.3 If pilot gates pass, update `docs/TECHNICAL_LAB_NOTEBOOK.md` to describe MedSAM automatic masks as the current preferred generated glomeruli segmentation candidate and state that raw/manual masks are not overwritten.
- [x] 5.4 If pilot gates fail because proposal recall is inadequate, record that the next change should improve the automatic box proposer before any MedSAM fine-tuning work.
- [x] 5.5 If pilot gates fail despite adequate proposal recall, record that the next change should propose MedSAM/SAM fine-tuning using admitted manual masks and current MedSAM checkpoint provenance.

## 6. Tests And Validation

- [x] 6.1 Add unit tests for proposal box derivation, padding, clipping, merge IoU, max-box overflow, and skipped component reasons from synthetic probability masks.
- [x] 6.2 Add unit tests for proposal recall matching against manual connected components, including missed components and multi-component masks.
- [x] 6.3 Add unit tests proving all generated automatic MedSAM masks and broad derived masks remain under configured runtime output roots and never under `raw_data/cohorts/**/masks`.
- [x] 6.4 Add unit tests for automatic metrics, oracle-gap summary fields, primary-segmenter transition fields, fine-tuning recommendation fields, and gate decision schema using synthetic masks without invoking MedSAM.
- [x] 6.5 Run focused tests for automatic prompt helpers plus existing MedSAM/manual workflow tests and the high-resolution `PredictionCore` guard tests.
- [x] 6.6 Run `python -m eq run-config --config configs/medsam_automatic_glomeruli_prompts.yaml --dry-run`.
- [x] 6.7 Run `openspec validate pilot-medsam-automatic-glomeruli-prompts --strict`.

## 7. Pilot Execution And Decision

- [x] 7.1 Run the real 20-row automatic-prompt pilot using the local `medsam` environment and configured current segmenter candidates.
- [x] 7.2 Review `summary.json`, `proposal_recall.csv`, `metrics.csv`, `metric_by_source.csv`, `prompt_failures.csv`, and overlays for whether automatic prompts close enough of the oracle gap to justify broad derived-mask generation.
- [x] 7.3 Record the pilot command, output path, selected proposal source, key metrics, proposal recall, failures, gate decisions, primary-segmenter transition recommendation, fine-tuning recommendation, and next recommendation in the change notes before treating implementation as complete.
