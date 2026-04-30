## Context

The completed MedSAM pilots established a practical starting point for a constrained domain-adaptation loop: oracle-box MedSAM reached mean Dice 0.922948, automatic-box MedSAM reached mean Dice 0.784805, proposal recall was 1.0, and prompt failures were zero on the 20-row admitted manual-mask pilot. That points away from localization as the first blocker and toward MedSAM/SAM boundary quality under automatic prompts. The target is oracle-level automatic-prompt performance, measured by a configured maximum gap from oracle-prompt MedSAM on fixed admitted manual-mask examples. This change is not a full MedSAM retraining effort; it should first test whether lightweight or partial adaptation can make automatic prompts behave closer to the already-strong oracle prompt result.

The repository already has several relevant owners that should be reused rather than bypassed:

- `src/eq/utils/paths.py` owns active runtime root, raw-data, model, derived-data, and output path resolution.
- `src/eq/run_config.py` owns workflow YAML dispatch.
- `src/eq/evaluation/medsam_glomeruli_workflow.py` owns shared MedSAM pilot helpers for input selection, MedSAM preflight/execution, metrics, mask loading, output-path isolation, and CSV writing.
- `src/eq/evaluation/run_medsam_automatic_glomeruli_prompts_workflow.py` owns automatic proposal-box generation, proposal recall, MedSAM automatic mask metrics, gate fields, and failure-mode classification.
- `openspec/specs/segmentation-training-contract/spec.md` owns supported training data, artifact provenance, and promotion constraints.

This change should extend those surfaces where possible. The new implementation surface is justified for the fine-tuning orchestration itself because it combines training, fixed-split evaluation, checkpoint provenance, and generated-mask release packaging in a workflow that is distinct from the completed oracle/automatic pilot evaluators.

## Goals / Non-Goals

**Goals:**

- Adapt MedSAM/SAM for glomeruli masks using admitted manual-mask rows, prioritizing lightweight or partial fine-tuning over full model retraining.
- Target oracle-level automatic-prompt performance rather than settling for any improvement over the current segmenter.
- Build deterministic train/validation/test manifests from `raw_data/cohorts/manifest.csv`.
- Evaluate current automatic MedSAM, fine-tuned checkpoints, current segmenter candidates, and trivial baselines on fixed examples.
- Produce complete checkpoint provenance under `models/medsam_glomeruli/<checkpoint_id>/`.
- Keep disposable diagnostics in `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/<run_id>/`.
- Package reusable generated masks under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`.
- Maintain a central generated-mask registry at `derived_data/generated_masks/glomeruli/manifest.csv` so generated glomeruli masks are as discoverable as cohort rows in `raw_data/cohorts/manifest.csv`.
- Separate practical generated-mask adoption from scientific model promotion.

**Non-Goals:**

- Do not overwrite `raw_data/cohorts/**/masks` or move generated masks into raw cohort mask directories.
- Do not run full original-scale MedSAM retraining as part of this change.
- Do not claim causal, prognostic, or downstream grading validity from segmentation metrics alone.
- Do not vendor MedSAM into `src/eq` or add it as a package dependency unless a later dependency-management change explicitly decides that.
- Do not support legacy FastAI pickle artifacts or legacy namespace shims as part of this change.
- Do not update README-facing model-status claims unless the generated-mask release and downstream opt-in gates pass.

## Decisions

### Decision: Treat fine-tuned MedSAM masks as reusable derived data, not raw data

Reusable masks intended for downstream workflows SHALL live under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`, with `masks/`, release-local `manifest.csv`, `INDEX.md`, and `provenance.json`. The canonical generated-mask registry SHALL live at `derived_data/generated_masks/glomeruli/manifest.csv` and include one row per generated mask or release-manifest entry. This avoids scattering usable masks across run-specific `output/` directories while preserving the meaning of `raw_data` as source/reference material.

Alternative considered: write generated masks under `raw_data/cohorts/**/masks`. Rejected because it would make generated masks visually indistinguishable from manual/reference masks and would undermine downstream provenance.

Alternative considered: leave generated masks under `output/segmentation_evaluation/**`. Rejected for reusable releases because those directories are run diagnostics and are already complex to navigate.

### Decision: Use adoption tiers rather than a single pass/fail label

Fine-tuned checkpoints SHALL be classified as `oracle_level_preferred`, `improved_candidate_not_oracle`, or `blocked`. `oracle_level_preferred` requires configured oracle-level Dice/Jaccard and maximum oracle-gap gates, plus reliability and overlay review gates. `improved_candidate_not_oracle` is allowed when a checkpoint beats current automatic MedSAM and the current segmenter candidates but remains outside the configured oracle gap; it may be packaged for explicit downstream comparison, but it SHALL NOT become the preferred generated-mask source. `blocked` means the checkpoint fails improvement, reliability, or safety gates and SHALL remain in run diagnostics only.

Initial configurable gate defaults SHALL be `min_dice: 0.90`, `min_jaccard: 0.82`, and `max_oracle_dice_gap: 0.05`, based on the prior oracle-prompt pilot mean Dice 0.922948 and mean Jaccard 0.857703. These defaults are implementation starting points, not scientific promotion thresholds.

Alternative considered: require oracle-level gates before any reusable release. Rejected because an improved-but-not-oracle checkpoint may be useful as an explicit candidate for downstream stability experiments, provided its status is recorded unambiguously.

### Decision: Prefer constrained domain adaptation over full retraining

The implementation SHALL prefer feasible local adaptation modes before considering broader model updates: frozen image encoder with mask-decoder tuning, prompt/mask-decoder fine-tuning, adapter-style training, or other lightweight MedSAM/SAM adaptation supported by the audited upstream code. Full MedSAM retraining comparable to the original MedSAM training regime is out of scope for this change because the oracle-prompt result already shows the base model has useful boundary capacity, and the practical question is whether automatic-prompt behavior can be adapted to glomeruli masks.

Alternative considered: full MedSAM retraining from the upstream training scripts. Rejected for this change because it is likely unnecessary for the observed failure mode and mismatched to local hardware; if partial adaptation cannot close the oracle gap, a later change can evaluate external CUDA/cloud training explicitly.

### Decision: Gate full pilot execution behind a local feasibility smoke run

Before any full local pilot, the workflow SHALL run a tiny local adaptation smoke run and record backend, device, image size, batch size, elapsed time, and memory stability evidence. If the smoke run indicates local execution is unreliable or impractically slow, the workflow SHALL record `requires_external_accelerator` and treat external CUDA/cloud execution as the supported path for the full pilot while preserving identical split, config, and provenance contracts.

Alternative considered: attempt the full pilot first and infer feasibility from failure. Rejected because it wastes runtime, obscures root cause, and produces inconsistent failure artifacts.

### Decision: Add one fine-tuning workflow while reusing existing MedSAM helpers

The workflow ID SHALL be `medsam_glomeruli_fine_tuning`, configured by `configs/medsam_glomeruli_fine_tuning.yaml` and dispatched through `eq run-config`. The runner module SHALL be `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`. It SHOULD reuse `medsam_glomeruli_workflow.py` for shared I/O, MedSAM preflight, CSV writing, metric rows, and path isolation, and reuse automatic-prompt proposal helpers rather than duplicating proposal-box logic.

Alternative considered: extend `run_medsam_automatic_glomeruli_prompts_workflow.py` to train models. Rejected because training/checkpoint ownership would make the automatic-prompt evaluator too broad.

### Decision: Fixed splits are promotion evidence, dynamic runs are diagnostics

The workflow SHALL create deterministic split manifests from admitted manual-mask rows before training or evaluation. Split assignment SHALL group by `source_sample_id` when present, and otherwise by the strongest available subject/source identifier, so near-duplicate rows do not leak across train, validation, and test. Validation and test metrics SHALL use fixed examples; stochastic or augmented training may be used inside the fine-tuning process but cannot replace fixed validation evidence.

Alternative considered: use only a random validation split emitted by the fine-tuning script. Rejected because it would make before/after comparison and promotion review unstable.

### Decision: Fine-tuning is evaluated as an improvement arm, not a one-off run

The workflow SHALL evaluate current automatic MedSAM, oracle-prompt MedSAM reference metrics when available, the current segmenter candidates, trivial masks, and each fine-tuned checkpoint using the same fixed examples and metric schema. A fine-tuned checkpoint can become a preferred generated-mask candidate only if it closes the configured oracle gap and improves over current automatic MedSAM and current segmenters without failing foreground-fraction, area-ratio, prompt-failure, or overlay review gates.

Alternative considered: run fine-tuning first and inspect qualitative results manually. Rejected because the pilot already showed the need for comparable, repeatable evidence.

### Decision: External MedSAM training is wrapped, not absorbed

The implementation SHALL audit `/Users/ncamarda/Projects/MedSAM` and the configured `medsam` conda environment to identify the exact fine-tuning entrypoint. The repository workflow SHALL wrap that entrypoint with explicit command and environment provenance rather than copying external training code into `src/eq`. If the local clone does not expose a ready constrained fine-tuning command, implementation SHALL adapt the official MedSAM training path documented by `https://github.com/bowang-lab/MedSAM`, including `pre_CT_MR.py`, `train_one_gpu.py`, `train_multi_gpus.sh`, and `utils/ckpt_convert.py` where applicable, but only in service of the constrained domain-adaptation goal.

Alternative considered: implement a repository-native MedSAM trainer immediately. Rejected until the external fine-tuning API and checkpoint format are audited.

## Explicit Decisions

- Workflow ID: `medsam_glomeruli_fine_tuning`.
- Config filename: `configs/medsam_glomeruli_fine_tuning.yaml`.
- Runner module: `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`.
- Model root: `models/medsam_glomeruli/<checkpoint_id>/`.
- Evaluation root: `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/<run_id>/`.
- Reusable generated-mask release root: `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`.
- Central generated-mask registry: `derived_data/generated_masks/glomeruli/manifest.csv`.
- Split manifests: `splits/train.csv`, `splits/validation.csv`, and `splits/test.csv` under the fine-tuning run output root, copied or referenced in checkpoint provenance.
- Downstream opt-in field: `mask_source=medsam_finetuned_glomeruli`.
- Preferred adaptation mode order: frozen image encoder plus mask-decoder/prompt-related tuning first; adapter or similarly lightweight updates second; broader encoder updates only after lightweight modes fail feasibility or oracle-gap gates.
- Local feasibility status field: `local_feasibility_status` with values `local_feasible` or `requires_external_accelerator`.
- Adoption gate target: oracle-level automatic-prompt MedSAM performance, expressed as configured minimum Dice/Jaccard and maximum oracle-prompt gap on fixed validation/test examples.
- Initial oracle-level gate defaults: `min_dice: 0.90`, `min_jaccard: 0.82`, `max_oracle_dice_gap: 0.05`.

## Risks / Trade-offs

- Fine-tuning overfits the small admitted manual-mask set -> Use fixed validation/test splits, cohort/subject-aware split manifests, trivial baselines, and overlay review before releasing masks.
- External MedSAM fine-tuning API differs from inference API -> Audit the MedSAM repository and record the exact training entrypoint before implementation.
- Lightweight adaptation cannot close the oracle gap -> Record `improved_candidate_not_oracle` or `blocked`, preserve diagnostics, and open a later external CUDA/cloud or broader retraining change only if evidence justifies it.
- Generated masks are hard to find later -> Publish reusable releases under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` with an `INDEX.md`.
- Generated masks are mistaken for manual truth -> Never write generated masks to `raw_data`; require `mask_source=medsam_finetuned_glomeruli` and source manual-mask lineage in release manifests.
- A checkpoint improves Dice but remains below oracle-prompt MedSAM -> Classify it as `improved_candidate_not_oracle`; allow explicit downstream comparison only if reliability gates pass and the central manifest records the candidate status.
- A checkpoint reaches oracle-level segmentation but harms downstream grading stability -> Treat generated-mask adoption and scientific promotion as separate gates, and require downstream opt-in review before README/model-status updates.

## Migration Plan

1. Add path helpers in `src/eq/utils/paths.py` for the generated-mask release root and MedSAM model root if no existing helper covers them.
2. Add the workflow config and `eq run-config` dispatch.
3. Implement fixed split generation and baseline evaluation without training first.
4. Wrap the audited MedSAM/SAM fine-tuning entrypoint with fail-closed dependency checks.
5. Evaluate fine-tuned checkpoints on fixed examples and write diagnostics under `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/<run_id>/`.
6. If gates pass, package a reusable generated-mask release under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`.
7. Downstream quantification workflows may opt into a release by manifest path and `mask_source=medsam_finetuned_glomeruli`; rollback is selecting the previous current-segmenter or MedSAM automatic manifest.

## Open Questions

- [audit_first_then_decide] Which exact MedSAM/SAM fine-tuning command should be wrapped? Audit `/Users/ncamarda/Projects/MedSAM`, the configured `medsam` environment, and available checkpoint save/load behavior.
- [audit_first_then_decide] What split sizes are feasible for the first fine-tuning run? Audit admitted `manual_mask_core` and `manual_mask_external` rows in `raw_data/cohorts/manifest.csv`.
- [defer_ok] Whether the first passing generated-mask release updates README-facing status should wait until downstream grading stability is reviewed.
