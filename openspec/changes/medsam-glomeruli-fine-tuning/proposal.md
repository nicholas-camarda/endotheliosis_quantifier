## Why

The automatic MedSAM pilot showed that proposal localization is adequate for admitted manual-mask rows, but boundary quality remains below the oracle-prompt ceiling and below the threshold for adopting MedSAM as the preferred generated glomeruli mask source. A practical glomeruli domain-adaptation change is needed now to target oracle-level MedSAM/SAM boundary quality inside a reproducible, reusable generated-mask workflow, without assuming full foundation-model retraining is necessary or appropriate for this project.

## What Changes

- Add a `medsam_glomeruli_fine_tuning` workflow config and `eq run-config` dispatch path for fixed-split evaluation, constrained MedSAM/SAM domain adaptation, and post-adaptation automatic-prompt evaluation.
- Build deterministic admitted manual-mask train/validation/test manifests from `raw_data/cohorts/manifest.csv`, preserving cohort, subject, lane, and source-path provenance.
- Evaluate current automatic MedSAM, fine-tuned MedSAM checkpoints, current segmenter candidates, and trivial baselines on fixed validation examples with metrics and overlay review artifacts.
- Gate fine-tuned checkpoints against oracle-prompt MedSAM reference metrics or a configured maximum oracle gap, not only against current segmenter improvement.
- Report a three-tier decision: `oracle_level_preferred`, `improved_candidate_not_oracle`, or `blocked`, so a checkpoint that improves over current segmenters but misses oracle-level gates is explicit rather than ambiguously failed.
- Store training checkpoints under `models/medsam_glomeruli/<checkpoint_id>/` with complete data, code, package, checkpoint, and command provenance.
- Store disposable run diagnostics under `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/<run_id>/`.
- Store reusable generated-mask releases under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` with masks, release-local `manifest.csv`, `INDEX.md`, and `provenance.json`.
- Maintain the centralized generated glomeruli mask registry at `derived_data/generated_masks/glomeruli/manifest.csv`, analogous to `raw_data/cohorts/manifest.csv`, so usable generated masks are discoverable without searching run folders.
- Keep `raw_data/cohorts/**/images` and `raw_data/cohorts/**/masks` read-only for this workflow; generated MedSAM masks SHALL NOT be written there.
- Add explicit gates for practical generated-mask adoption separately from stricter scientific model promotion.

## Capabilities

### New Capabilities

- `medsam-glomeruli-fine-tuning`: Adapt and evaluate MedSAM/SAM as a practical generated glomeruli mask source, including fixed admitted manual-mask splits, constrained fine-tuning modes, checkpoint provenance, generated-mask release layout, oracle-level target gates, and adoption gates.

### Modified Capabilities

- `medsam-automatic-glomeruli-prompts`: Replace the earlier reusable broad-mask output root with the canonical generated-mask release root under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` when a fine-tuned checkpoint passes adoption gates.
- `segmentation-training-contract`: Extend the training contract to cover MedSAM/SAM fine-tuning artifacts, fixed manual-mask splits, and generated-mask release provenance without treating generated masks as raw training data.

## Impact

- Affected modules: `src/eq/run_config.py`, `src/eq/utils/paths.py`, `src/eq/evaluation/medsam_glomeruli_workflow.py`, a new `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`, and any shared MedSAM command/provenance helpers needed for reuse.
- Affected configs: new `configs/medsam_glomeruli_fine_tuning.yaml`.
- Affected tests: focused tests for split generation, path isolation, checkpoint provenance, generated-mask release manifests, gate decisions, and baseline/fine-tuned metric schemas.
- Affected artifact roots: `models/medsam_glomeruli/`, `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/`, and `derived_data/generated_masks/glomeruli/medsam_finetuned/` under the active runtime root.
- External dependency risk: MedSAM/SAM domain adaptation may require the existing `/Users/ncamarda/mambaforge/envs/medsam/bin/python`, `/Users/ncamarda/Projects/MedSAM`, and checkpoint assets. If the local clone lacks a usable constrained fine-tuning entrypoint, implementation must adapt to the official MedSAM repository training path documented at `https://github.com/bowang-lab/MedSAM`, including `pre_CT_MR.py`, `train_one_gpu.py`, `train_multi_gpus.sh`, and `utils/ckpt_convert.py` where applicable, while preserving this change's intent of partial/lightweight adaptation rather than full original-scale retraining.
- Scientific interpretation: passing generated-mask adoption gates supports practical downstream opt-in only; it does not promote the model scientifically without downstream grading stability and broader non-degenerate prediction review.

## Explicit Decisions

- Workflow ID: `medsam_glomeruli_fine_tuning`.
- Config filename: `configs/medsam_glomeruli_fine_tuning.yaml`.
- Proposed runner module: `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`.
- Model root: `models/medsam_glomeruli/<checkpoint_id>/` under the active runtime root.
- Evaluation root: `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/<run_id>/` under the active runtime root.
- Reusable generated-mask release root: `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` under the active runtime root.
- Central generated-mask registry: `derived_data/generated_masks/glomeruli/manifest.csv` under the active runtime root.
- Fine-tuning mode preference: freeze the MedSAM image encoder where feasible and prioritize mask-decoder, prompt-related, adapter, or other lightweight domain-adaptation modes before considering broader model updates.
- Generated masks remain separate from `raw_data`; raw/manual masks remain source/reference data only.

## Open Questions

- [audit_first_then_decide] Which exact MedSAM/SAM fine-tuning entrypoint should the workflow wrap? Audit `/Users/ncamarda/Projects/MedSAM` and installed `medsam` environment before implementation.
- [audit_first_then_decide] Which admitted manual-mask rows are sufficient for the initial train/validation/test split sizes? Audit `raw_data/cohorts/manifest.csv` and lane coverage before applying the change.
- [defer_ok] Whether a passing fine-tuned generated-mask release should update README-facing model status immediately can be decided after downstream grading stability is reviewed.