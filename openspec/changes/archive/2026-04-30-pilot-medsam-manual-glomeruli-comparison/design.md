## Context

The current glomeruli segmentation stack has two distinct concerns that should not be conflated. The maintained FastAI glomeruli segmenters are trained from full-image `images/` and `masks/` roots with dynamic crop sampling, and current high-resolution scored-no-mask inference already contains a tiled probability path in `src/eq/quantification/endotheliosis_grade_model.py`. However, the low-level `PredictionCore` API still accepts arbitrary PIL images and resizes them to the model input size, so ad-hoc high-resolution evaluation can accidentally collapse a full `2448x2048` image to `256x256`.

The t52 MedSAM-assisted no-mask-smoke review exposed that distinction: full-field single-pass resize is an invalid evaluation path for the current crop-trained segmenter, while crop/tile inference is the correct way to evaluate high-resolution images. t52 itself has no independent manual reference, so it cannot be used to decide whether MedSAM is better than the current segmenter. The pilot must therefore use existing admitted manual-mask rows from `raw_data/cohorts/manifest.csv`, generate MedSAM artifacts outside raw data, and compare all methods against the same manual masks.

## Goals / Non-Goals

**Goals:**

- Implement a small, deterministic pilot workflow named `medsam_manual_glomeruli_comparison`.
- Select 20 admitted manual-mask rows when available, split across `vegfri_dox` and `lauren_preeclampsia` with subject diversity.
- Generate MedSAM oracle-prompt masks from manual-mask connected-component bounding boxes.
- Generate current segmenter baseline masks for both current transfer and scratch candidates using tiled high-resolution inference, not whole-field single-pass resize.
- Write all generated masks, overlays, metrics, prompt failures, input selections, and provenance under the active runtime output root.
- Add tests that make raw-data pollution and invalid full-field single-pass current-segmenter evaluation hard to reintroduce.

**Non-Goals:**

- Do not train or fine-tune MedSAM.
- Do not install MedSAM into `eq-mac` or make it a normal package dependency.
- Do not label MedSAM, the current transfer candidate, or the current scratch candidate as scientifically promoted from this pilot alone.
- Do not use t52 no-mask-smoke masks as manual truth.
- Do not rewrite the existing glomeruli candidate-promotion workflow.

## Decisions

### Decision: Add a dedicated evaluation workflow

The new top-level workflow SHALL be `medsam_manual_glomeruli_comparison`, configured by `configs/medsam_manual_glomeruli_comparison.yaml` and run through the existing config execution surface. The worker implementation SHOULD live in `src/eq/evaluation/run_medsam_manual_glomeruli_comparison_workflow.py` so it stays near other evaluation workflows instead of quantification model code.

Alternatives considered:

- Extend `glomeruli_candidate_comparison`: rejected because this pilot evaluates a foundation-model audit path and prompt provenance, not candidate promotion.
- Extend `glomeruli_overcoverage_audit`: rejected because that workflow audits thresholds/probabilities for existing current-namespace artifacts, not external MedSAM prompts.
- Keep this as a notebook or one-off script: rejected because the pilot needs reproducible artifact contracts and raw-data isolation.

### Decision: Treat MedSAM as an external local runtime

MedSAM inference SHALL use `/Users/ncamarda/mambaforge/envs/medsam/bin/python`, `/Users/ncamarda/Projects/MedSAM`, and `/Users/ncamarda/Projects/MedSAM/work_dir/MedSAM/medsam_vit_b.pth`. The project SHALL record these paths, command lines, checkpoint hash when readable, and package/module availability in `summary.json`.

The implementation MAY either call `/Users/ncamarda/Projects/MedSAM/MedSAM_Inference.py` as a subprocess or import the MedSAM repository from the MedSAM environment through a small adapter. In both cases, the project code remains responsible for selecting inputs, deriving boxes, writing output contracts, and computing metrics.

Alternatives considered:

- Add MedSAM as a project dependency: rejected for the pilot because this is audit evidence and cross-env dependency promotion is a separate decision.
- Vendor MedSAM code into `src/eq`: rejected because it would blur ownership and make external checkpoint provenance harder to audit.

### Decision: Start with oracle boxes

The first pilot SHALL use bounding boxes derived from manual-mask connected components. Each component above a configured `min_component_area` gets a padded box, MedSAM produces a mask for that prompt, and component masks are unioned into a full-image MedSAM mask for comparison against the manual union mask.

This is upper-bound prompt evidence. It tests whether MedSAM can trace manual-localized glomeruli better, not whether MedSAM can find glomeruli without help. Automatic prompts from the current segmenter are deferred until oracle-prompt results are reviewed.

Alternatives considered:

- Use one bounding box around the whole manual union mask: rejected as the default because multi-component masks can produce too-large prompts and merge distinct glomeruli.
- Use current segmenter boxes first: rejected for the pilot because it mixes localization failures with MedSAM boundary quality before the oracle upper bound is known.

### Decision: Reuse current tiled inference, not low-level whole-image resize

The current segmenter baseline SHALL use a canonical tiled full-field function for both current glomeruli candidate artifacts. The existing `_predict_tiled_segmentation_probability` behavior in `src/eq/quantification/endotheliosis_grade_model.py` is the current implementation owner to inspect first; if generalized, it should be moved or wrapped centrally rather than copied. The workflow SHALL record tile size, stride, expected model input size, threshold, current model artifact path, and preprocessing contract.

Any full-field call that sends an entire high-resolution image directly to `PredictionCore.predict_segmentation_probability` SHALL be treated as invalid for this pilot.

### Decision: Isolate generated artifacts from raw data

The pilot SHALL write only under `output/segmentation_evaluation/medsam_manual_glomeruli_comparison/<run_id>/` by default. Raw cohort `images/` and `masks/` directories remain read-only inputs. Generated MedSAM masks and current-model masks SHALL be stored in dedicated subdirectories of the output root.

### Decision: Report descriptive evidence only

The pilot SHALL report segmentation agreement metrics and visual review artifacts. It SHALL not make causal, prognostic, or external-validity claims. It SHALL not update README-facing current-performance claims.

## Risks / Trade-offs

- MedSAM local environment or checkpoint is missing on another machine -> The workflow records dependency failure in `summary.json` and fails without writing partial masks as if they were valid.
- Oracle boxes overstate deployable performance -> Reports label oracle-prompt results as upper-bound evidence and keep automatic-prompt evidence separate.
- Manual masks contain omissions or mixed quality -> The pilot reports agreement with existing manual masks and includes overlays for review, but it does not treat manual masks as immutable scientific truth.
- Multi-component masks may cause prompt explosion -> The pilot records component counts, box counts, skipped components, and prompt failures per image.
- Current-model baseline can accidentally use whole-field resize -> Tests cover the high-resolution guard and workflow provenance records tiled inference.
- Running MedSAM per component may be slow on CPU/MPS -> The pilot is intentionally 20 rows; runtime is recorded before deciding whether to scale up.

## Migration Plan

1. Add config validation, input selection, MedSAM oracle-mask generation, current tiled baseline generation, metrics, overlays, and provenance for the pilot workflow.
2. Run unit tests for deterministic pilot selection, box generation, output isolation, metric schema, and high-resolution inference guard.
3. Run a dry-run workflow to verify paths and provenance without MedSAM mask generation.
4. Run the real pilot using the local `medsam` environment.
5. Review `summary.json`, `metrics.csv`, and overlays before deciding whether to add automatic-prompt MedSAM or downstream ROI-feature deltas.

Rollback is simple: remove the pilot output directory. No raw data, model artifacts, or existing quantification outputs are modified.

## Explicit Decisions

- Workflow ID: `medsam_manual_glomeruli_comparison`.
- Config path: `configs/medsam_manual_glomeruli_comparison.yaml`.
- Worker module: `src/eq/evaluation/run_medsam_manual_glomeruli_comparison_workflow.py`.
- Default output root: `output/segmentation_evaluation/medsam_manual_glomeruli_comparison/<run_id>/`.
- Default run ID: `pilot_medsam_manual_glomeruli_comparison`.
- Pilot sample size: 20 admitted manual-mask rows, cohort-balanced when possible.
- MedSAM prompt mode in first implementation: `oracle_component_boxes_from_manual_mask`.
- MedSAM environment path: `/Users/ncamarda/mambaforge/envs/medsam/bin/python`.
- MedSAM repo path: `/Users/ncamarda/Projects/MedSAM`.
- MedSAM checkpoint path: `/Users/ncamarda/Projects/MedSAM/work_dir/MedSAM/medsam_vit_b.pth`.
- Transfer candidate baseline path: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/glomeruli_candidate_transfer-transfer_loss-custom_s1lr1e-3_s2lr_lrfind_e20_b12_lr1e-3_sz256/glomeruli_candidate_transfer-transfer_loss-custom_s1lr1e-3_s2lr_lrfind_e20_b12_lr1e-3_sz256.pkl`.
- Scratch candidate baseline path: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/scratch/glomeruli_candidate_no_mito_base-scratch_e25_b12_lr1e-3_sz256/glomeruli_candidate_no_mito_base-scratch_e25_b12_lr1e-3_sz256.pkl`.
- Generated artifact directories: `medsam_masks/`, `current_segmenter_masks/`, `overlays/`, and `review_panels/`.

## Open Questions

- [audit_first_then_decide] Whether automatic MedSAM prompts are worth adding will be decided from the oracle pilot's `metrics.csv`, `prompt_failures.csv`, overlays, and runtime summary.
- [defer_ok] Downstream ROI-feature deltas are deferred until segmentation-level MedSAM/manual/current agreement is reviewed.