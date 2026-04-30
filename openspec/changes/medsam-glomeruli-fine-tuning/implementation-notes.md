## Task 1 Audit Notes (2026-04-30)

### 1.1 Reuse/Centralization Audit

- `src/eq/utils/paths.py` already owns runtime-root path contracts; extend this file for:
  - `models/medsam_glomeruli/<checkpoint_id>/`
  - `derived_data/generated_masks/glomeruli/manifest.csv`
  - `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`
- `src/eq/run_config.py` already owns workflow dispatch; add `medsam_glomeruli_fine_tuning` there.
- `src/eq/evaluation/medsam_glomeruli_workflow.py` already centralizes MedSAM preflight, batch execution, CSV writing, and input selection (re-exported from the manual workflow); reuse for shared helper behavior.
- `src/eq/evaluation/run_medsam_automatic_glomeruli_prompts_workflow.py` already implements proposal boxes, proposal recall, gate-style summary fields, and path isolation; reuse proposal and metrics patterns rather than duplicating.
- `src/eq/quantification/endotheliosis_grade_model.py` already owns `_predict_tiled_segmentation_probability` and `PredictionCore` tile-role usage; reuse this for current segmenter baseline probabilities.

### 1.2 MedSAM Entry Point Audit

- Confirmed upstream/local MedSAM training surfaces exist:
  - `train_one_gpu.py`
  - `train_multi_gpus.py`
  - `train_multi_gpus.sh`
  - `pre_CT_MR.py`
  - `utils/ckpt_convert.py`
- Existing scripts show constrained adaptation behavior (prompt encoder freezing and prompt/image encoder freeze policies) and are compatible with the change intent of partial adaptation over full retraining.
- Decision: use constrained adaptation mode first (frozen image encoder + mask-decoder/prompt-related adaptation where feasible), wrapping local entrypoint if available; otherwise adapt official script path while preserving constrained scope.

### 1.3 Manifest Coverage Audit

- Audited runtime manifest: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/cohorts/manifest.csv`
- Eligible admitted manual-mask rows:
  - total: `707`
  - `manual_mask_external`: `619` (`vegfri_dox`)
  - `manual_mask_core`: `88` (`lauren_preeclampsia`)
- Distinct coverage:
  - `cohort_id`: `2`
  - `source_sample_id`: `140`
  - `subject_id`: `60`
- Feasible deterministic split policy:
  - group by `source_sample_id` (fallback to strongest subject/source identifier)
  - target initial split sizing around 70/15/15 with lane-aware minimum coverage constraints.

### 1.4 Selected Decisions Recorded

- Adaptation mode: constrained domain adaptation, not full original-scale retraining.
- Entry point policy:
  1. use local constrained entrypoint if present/usable
  2. fallback to official MedSAM training scripts (`pre_CT_MR.py`, `train_one_gpu.py`, `train_multi_gpus.sh`, `utils/ckpt_convert.py`) while keeping constrained scope.
- Frozen/trainable policy:
  - prefer frozen image encoder with mask-decoder/prompt-related updates first.
- Split grouping key:
  - primary `source_sample_id`, fallback strongest source/subject key.
- Central manifest policy:
  - maintain canonical generated-mask registry at `derived_data/generated_masks/glomeruli/manifest.csv`.

## Implementation Status (2026-04-30)

- Added `medsam_glomeruli_fine_tuning` config and `eq run-config` dispatch.
- Added runtime path helpers for MedSAM glomeruli checkpoints and generated-mask registry/release roots.
- Added fail-closed generated-mask raw-data path isolation.
- Added deterministic fixed split generation, explicit split validation, split hashing, and dry-run provenance.
- Added trivial all-background/all-foreground baseline metrics for fixed validation/test rows.
- Added dependency preflight for MedSAM Python, repo, base checkpoint, upstream entrypoint, adaptation mode, and frozen/trainable policy.
- Added wrapper command construction for upstream `train_one_gpu.py` without vendoring MedSAM code into `src/eq`.
- Added checkpoint provenance writing that does not claim a supported checkpoint unless completed checkpoint files exist.
- Added local feasibility smoke-run helper that records backend/device, image size, batch size, elapsed time, stderr/stdout, memory failures, and `local_feasibility_status`.
- Added adoption-tier helper for `oracle_level_preferred`, `improved_candidate_not_oracle`, and `blocked`.
- Added reusable generated-mask release packaging and central generated-mask registry update helpers.

Verification run:

- `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/test_medsam_glomeruli_fine_tuning_workflow.py tests/test_medsam_automatic_glomeruli_prompts.py tests/test_prediction_core_highres_guard.py`
  - Result: `18 passed`
- `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/medsam_glomeruli_fine_tuning.yaml --dry-run`
  - Result: completed; summary at `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/medsam_glomeruli_fine_tuning/pilot_medsam_glomeruli_fine_tuning/summary.json`
- `ruff check src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py src/eq/utils/paths.py src/eq/run_config.py tests/test_medsam_glomeruli_fine_tuning_workflow.py`
  - Result: `All checks passed!`
- `openspec validate medsam-glomeruli-fine-tuning --strict`
  - Result: valid

Remaining blocker before closing the change:

- The remaining tasks require real fixed-example MedSAM/current-segmenter evaluation and a real constrained fine-tuning pilot. The upstream MedSAM training command expects a MedSAM `npy_data` training root with `imgs/` and `gts/`; this implementation records the intended `derived_data/medsam_glomeruli/npy_data` path but does not yet generate that upstream-format training root or produce an adapted checkpoint. Do not mark checkpoint artifact, fine-tuned inference, or pilot decision tasks complete until that data-preparation/training execution path is run and reviewed.

## Pilot Run Update (2026-04-30)

- Runtime root: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier`
- Workflow run: `eq run-config --config configs/medsam_glomeruli_fine_tuning.yaml` (non-dry-run)
- Baseline completion:
  - current segmenter metrics written on fixed 42-row validation/test slice
  - automatic MedSAM masks/metrics written (42 rows)
  - oracle MedSAM masks/metrics written (42 rows)
  - trivial baselines written
- Training completion:
  - adapter backend on `mps` completed 10 epochs on 665 examples
  - artifacts at `models/medsam_glomeruli/pilot_medsam_glomeruli_fine_tuning/medsam_glomeruli_{latest,best}.pth`
  - provenance updated with `training_status=completed` and `supported_checkpoint=true`
- Not yet complete:
  - fine-tuned checkpoint inference/metrics/overlay comparison tasks (5.1/5.2/5.3)
  - decision-note closure tasks (7.4/7.5)

See `continuation-action-plan.md` in this change for the execution checklist to complete the remaining tasks.
