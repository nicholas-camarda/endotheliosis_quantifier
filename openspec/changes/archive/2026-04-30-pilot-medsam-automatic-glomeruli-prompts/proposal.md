## Why

The oracle-box MedSAM/manual pilot showed substantially better agreement with admitted manual glomeruli masks than the current tiled segmenters, but oracle boxes are derived from the answer mask and cannot be used for deployment or broad mask regeneration. We need an automatic-prompt pilot that tests whether current-segmenter proposal boxes can localize glomeruli well enough for MedSAM to replace or augment generated glomeruli masks as isolated derived artifacts.

## What Changes

- Add an automatic-prompt MedSAM pilot workflow that uses tiled current-segmenter probability maps to generate high-recall glomerulus proposal boxes.
- Reuse the completed manual-mask pilot selection and MedSAM external-runtime adapter to compare automatic-box MedSAM, oracle-box MedSAM, current segmenter masks, and manual masks on the same admitted 20-row pilot first.
- Write proposal boxes, automatic MedSAM masks, metrics, overlays, prompt failures, and provenance under `output/segmentation_evaluation/medsam_automatic_glomeruli_prompts/<run_id>/`.
- Add explicit gates before any broader mask replacement: automatic-box performance, proposal recall against manual components, prompt failure rate, and visual overlay review.
- Add a documented transition plan for making MedSAM-derived segmentation the primary generated-mask path if automatic prompts pass gates, including config names, artifact labels, docs to update, and rollback behavior.
- Add a fine-tuning decision track that records when MedSAM prompt engineering is sufficient versus when a MedSAM/SAM fine-tuning proposal should be opened.
- Do not overwrite manual masks or write generated MedSAM masks into `raw_data/cohorts/**/masks`.

## Capabilities

### New Capabilities

- `medsam-automatic-glomeruli-prompts`: Defines automatic proposal-box generation, MedSAM automatic-prompt inference, comparison against manual and oracle evidence, derived-mask artifact isolation, and gates before broader MedSAM mask generation.

### Modified Capabilities

- `medsam-manual-glomeruli-comparison`: No requirement changes. The automatic-prompt workflow reuses its selection, MedSAM runtime, metrics, and artifact-isolation contracts but remains a separate capability because automatic localization is a distinct behavioral claim.

## Impact

- Affected modules to inspect for reuse: `src/eq/evaluation/run_medsam_manual_glomeruli_comparison_workflow.py` for input selection, MedSAM batch execution, metrics, overlays, output isolation, and config parsing; `src/eq/quantification/endotheliosis_grade_model.py` for `_predict_tiled_segmentation_probability`; `src/eq/inference/prediction_core.py` for the high-resolution direct-resize guard; `src/eq/training/promotion_gates.py` for binary metrics and component helpers; `src/eq/utils/paths.py` for runtime-root conventions.
- New config path: `configs/medsam_automatic_glomeruli_prompts.yaml`.
- New workflow ID: `medsam_automatic_glomeruli_prompts`, dispatched through `eq run-config`.
- New worker module: `src/eq/evaluation/run_medsam_automatic_glomeruli_prompts_workflow.py`, with shared helpers factored from the manual workflow if reuse would otherwise duplicate selection, box, metric, MedSAM, or output-isolation behavior.
- Runtime dependency boundary remains unchanged: project logic runs in `eq-mac`; MedSAM inference runs with `/Users/ncamarda/mambaforge/envs/medsam/bin/python`, `/Users/ncamarda/Projects/MedSAM`, and `/Users/ncamarda/Projects/MedSAM/work_dir/MedSAM/medsam_vit_b.pth`.
- Generated output schemas include `inputs.csv`, `proposal_boxes.csv`, `medsam_auto_masks/`, `current_segmenter_masks/`, `overlays/`, `metrics.csv`, `metric_by_source.csv`, `prompt_failures.csv`, `proposal_recall.csv`, and `summary.json`.
- Documentation impact, conditional on pilot gates passing: update `docs/TECHNICAL_LAB_NOTEBOOK.md` and workflow-facing README/config notes to describe MedSAM-derived masks as the primary generated glomeruli segmentation candidate, while keeping manual masks and raw cohort masks distinct.
- Future model-development impact: if automatic prompts fail because localization is poor, the next change should improve/replace the box proposer; if MedSAM boundaries fail despite good boxes, the next change should propose MedSAM/SAM fine-tuning using admitted manual masks.
- Compatibility risk is limited to new derived artifacts under the runtime output root. Raw cohort masks, manual masks, current model artifacts, and existing quantification outputs are not modified.

## Explicit Decisions

- Change name: `pilot-medsam-automatic-glomeruli-prompts`.
- New capability name: `medsam-automatic-glomeruli-prompts`.
- Workflow ID: `medsam_automatic_glomeruli_prompts`.
- Config path: `configs/medsam_automatic_glomeruli_prompts.yaml`.
- Worker module: `src/eq/evaluation/run_medsam_automatic_glomeruli_prompts_workflow.py`.
- Default output root: `output/segmentation_evaluation/medsam_automatic_glomeruli_prompts/<run_id>/` under the active runtime root.
- Default run ID: `pilot_medsam_automatic_glomeruli_prompts`.
- Validation input source: the same admitted manual-mask manifest selection used by `medsam_manual_glomeruli_comparison`, defaulting to 20 rows.
- Initial proposal source: the better-recall current candidate selected from the completed oracle pilot comparison, with both transfer and scratch proposal metrics recorded when configured.
- Default proposal threshold sweep: `0.20`, `0.35`, and `0.50`; the selected default threshold is the highest-recall threshold that does not exceed configured proposal-count and area-ratio guardrails.
- Default proposal post-processing: connected components on tiled probability masks, minimum component area `2000`, maximum component area `750000`, padding `16`, overlapping-box merge IoU `0.25`, and maximum boxes per image `20`.
- Broader replacement output, if enabled after pilot review, is derived-only and must use `output/derived_masks/medsam_automatic_glomeruli/<run_id>/`, not raw cohort mask directories.
- Primary segmenter transition target, if gates pass: `medsam_automatic_glomeruli` becomes the preferred generated glomeruli mask source in configs and docs, while manual masks remain the reference source.
- Fine-tuning trigger: open a separate change only if automatic prompt localization passes but MedSAM automatic-mask boundary metrics or overlay review remain insufficient.
- Documentation update gate: update documentation only after the automatic-prompt pilot reports passing gates in `summary.json`; before that, docs may mention the workflow as experimental audit evidence only.

## Open Questions

- [audit_first_then_decide] Which current candidate and threshold should become the default automatic box source after reviewing the proposal-recall sweep on the 20-row manual pilot?
- [audit_first_then_decide] Should broader derived-mask generation first target no-mask-smoke/t52-style folders only, or all admitted cohort images with existing images, after the automatic-prompt pilot passes?
- [audit_first_then_decide] If automatic prompts pass, which downstream configs should switch first to `medsam_automatic_glomeruli`: evaluation-only configs, no-mask-smoke generation, or quantification candidate configs?
- [audit_first_then_decide] If fine-tuning is needed, should the training target be MedSAM box-prompt fine-tuning, SAM adapter fine-tuning, or a separate glomeruli-specific detector plus MedSAM boundary refiner?
- [defer_ok] Should a later change add human review/adjudication UI for accepting MedSAM automatic masks before downstream quantification uses them?