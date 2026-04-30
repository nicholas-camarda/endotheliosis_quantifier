## 1. Reuse Audit And Adaptation Entry Point

- [x] 1.1 Inspect `src/eq/utils/paths.py`, `src/eq/run_config.py`, `src/eq/evaluation/medsam_glomeruli_workflow.py`, `src/eq/evaluation/run_medsam_automatic_glomeruli_prompts_workflow.py`, and `src/eq/quantification/endotheliosis_grade_model.py` for reusable path, config, MedSAM execution, proposal-box, metrics, and tiled-inference owners before adding new helpers.
- [x] 1.2 Audit `/Users/ncamarda/Projects/MedSAM` and `/Users/ncamarda/mambaforge/envs/medsam/bin/python` for constrained MedSAM/SAM adaptation options, including frozen-image-encoder mask-decoder tuning, prompt-related tuning, adapter-style updates, required arguments, checkpoint save format, and environment constraints; if no ready entrypoint exists, map the implementation path to the official MedSAM training scripts `pre_CT_MR.py`, `train_one_gpu.py`, `train_multi_gpus.sh`, and `utils/ckpt_convert.py` without expanding scope to full original-scale retraining.
- [x] 1.3 Audit `raw_data/cohorts/manifest.csv` under the active runtime root for admitted `manual_mask_core` and `manual_mask_external` row counts, cohort/subject coverage, and feasible train/validation/test split sizes.
- [x] 1.4 Record the selected adaptation mode, fine-tuning entrypoint, frozen/trainable component policy, upstream MedSAM training-script decision, split-size decision, split grouping key, and central generated-mask registry decision in the change notes before implementing training execution.

## 2. Config, Paths, And Dispatch

- [x] 2.1 Add `configs/medsam_glomeruli_fine_tuning.yaml` with `workflow: medsam_glomeruli_fine_tuning`, MedSAM environment paths, base checkpoint path, adaptation mode, frozen/trainable component policy, split settings, proposal-box settings, training hyperparameters, oracle-level evaluation gates with initial defaults `min_dice=0.90`, `min_jaccard=0.82`, `max_oracle_dice_gap=0.05`, model root, evaluation root, generated-mask release root, and central generated-mask registry path.
- [x] 2.2 Add or extend `src/eq/utils/paths.py` helpers for `models/medsam_glomeruli/<checkpoint_id>/`, `derived_data/generated_masks/glomeruli/manifest.csv`, and `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` without hardcoding machine-specific paths.
- [x] 2.3 Register `medsam_glomeruli_fine_tuning` in `src/eq/run_config.py` and dispatch it to `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`.
- [x] 2.4 Ensure all configured generated-mask release paths fail closed if they resolve under `raw_data/cohorts/**/images` or `raw_data/cohorts/**/masks`.

## 3. Fixed Splits And Baseline Evaluation

- [x] 3.1 Implement deterministic admitted manual-mask train/validation/test split generation with cohort/subject-aware selection, grouping by `source_sample_id` or the strongest available source identifier to prevent leakage, and explicit split manifests.
- [x] 3.2 Validate explicit split manifest inputs and record manifest paths and hashes in run provenance.
- [x] 3.3 Reuse automatic proposal-box generation to evaluate current automatic MedSAM on fixed validation/test examples.
- [x] 3.4 Include or compute oracle-prompt MedSAM reference metrics for the fixed validation/test examples when configured.
- [x] 3.5 Reuse tiled current-segmenter inference to evaluate configured current segmenter candidates on the same fixed examples.
- [x] 3.6 Add all-background and all-foreground trivial baselines to the fixed-split metric outputs.

## 4. Domain Adaptation And Checkpoint Provenance

- [x] 4.1 Implement dependency preflight for the audited MedSAM/SAM adaptation entrypoint, Python environment, repository path, base checkpoint, adaptation mode, frozen/trainable component policy, and output checkpoint directory.
- [x] 4.2 Implement a local feasibility smoke run (1-2 examples) for constrained adaptation and record backend/device, image size, batch size, elapsed time, and memory-related failures.
- [x] 4.3 Set and record `local_feasibility_status` as `local_feasible` or `requires_external_accelerator`; if `requires_external_accelerator`, keep split/config/provenance contracts identical for external execution.
- [x] 4.4 Wrap the external constrained adaptation command without vendoring MedSAM code into `src/eq`; if the local clone lacks a ready command, implement the repository adapter against the official MedSAM training path identified in task 1.2 while preserving frozen/partial adaptation as the first intended mode.
- [x] 4.5 Write fine-tuned checkpoint artifacts under `models/medsam_glomeruli/<checkpoint_id>/`.
- [x] 4.6 Write checkpoint provenance with training command, environment, MedSAM repo path, base checkpoint path/hash, code version, package versions, split manifest paths/hashes, adaptation mode, frozen and trainable component names, hyperparameters, checkpoint files, training status, and local feasibility status.
- [x] 4.7 Fail closed when fine-tuning dependencies or checkpoint outputs are missing, without writing success-like checkpoint provenance.

## 5. Fine-Tuned Evaluation, Gates, And Release Packaging

- [ ] 5.1 Run automatic proposal-box MedSAM inference with fine-tuned checkpoints on fixed validation/test examples.
- [ ] 5.2 Write fine-tuned masks, prompt provenance, metrics, overlays, and prompt failures under `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/<run_id>/`.
- [ ] 5.3 Compare fine-tuned checkpoints against oracle-prompt MedSAM reference metrics, current automatic MedSAM, current segmenter candidates, and trivial baselines using the same fixed examples.
- [x] 5.4 Implement generated-mask adoption gates for oracle-level Dice/Jaccard and maximum oracle gap, metric improvement, prompt failures, foreground fraction, area ratio, trivial-baseline comparisons, and overlay review status, producing `oracle_level_preferred`, `improved_candidate_not_oracle`, or `blocked`.
- [x] 5.5 Package reusable generated-mask releases under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` when the adoption tier is `oracle_level_preferred` or explicit downstream comparison is requested for `improved_candidate_not_oracle`.
- [x] 5.6 Write generated-mask release `manifest.csv`, `INDEX.md`, and `provenance.json` with `generated_mask_id`, `mask_release_id`, `mask_source=medsam_finetuned_glomeruli`, `adoption_tier`, checkpoint ID, proposal source, threshold, run ID, source image, reference mask, generated mask path, and generation status.
- [x] 5.7 Update or create the central generated-mask registry `derived_data/generated_masks/glomeruli/manifest.csv` with one row per generated mask or release-manifest entry.

## 6. Tests, Documentation, And Validation

- [x] 6.1 Add unit tests for fixed split generation, explicit split validation, cohort/subject coverage fields, and split manifest hashing.
- [x] 6.2 Add unit tests for generated-mask release path isolation, central generated-mask registry updates, adoption-tier fields, and raw-data write rejection.
- [x] 6.3 Add unit tests for checkpoint provenance schema and fail-closed fine-tuning dependency handling without invoking real MedSAM training.
- [x] 6.4 Add unit tests for baseline/fine-tuned metric schemas, oracle-gap summary fields, trivial baseline rows, gate decisions, and release manifest fields using synthetic masks.
- [x] 6.5 Run focused tests for MedSAM fine-tuning helpers plus existing MedSAM automatic/manual workflow tests and `PredictionCore` high-resolution guard tests.
- [x] 6.6 Run `python -m eq run-config --config configs/medsam_glomeruli_fine_tuning.yaml --dry-run`.
- [x] 6.7 Run `ruff check .` or the focused changed-file equivalent.
- [x] 6.8 Run `openspec validate medsam-glomeruli-fine-tuning --strict`.

## 7. Pilot Execution And Decision

- [x] 7.1 Run the baseline-only dry run to verify splits, output roots, generated-mask release roots, and dependency preflight before training.
- [ ] 7.2 Run and review the local feasibility smoke run first; if status is `local_feasible`, proceed locally, and if status is `requires_external_accelerator`, execute the full pilot on the selected external accelerator path.
- [x] 7.3 Run the first fine-tuning pilot using the feasible execution path and admitted manual-mask split manifests.
- [ ] 7.4 Review `summary.json`, fixed-split metrics, oracle-gap fields, prompt failures, checkpoint provenance, local feasibility status, generated-mask release manifest when present, and overlays.
- [ ] 7.5 Record the pilot command, output path, checkpoint ID, split manifest paths, local feasibility status, key metrics, gate decisions, generated-mask release status, and next recommendation in the change notes before treating implementation as complete.

