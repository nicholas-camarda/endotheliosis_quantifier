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

## Implementation closure (2026-04-30)

- **Training execution:** `run_medsam_glomeruli_fine_tuning_workflow` now runs the full adapter/upstream training command via `subprocess` when `training.run_training` is true (default), after optional local feasibility smoke when configured. `summary.json` includes `training_skip_reason`, `training_process_exit_code`, and `log_path` (when execution logging is active). Skipping training remains supported for tests (`run_training: false`).
- **Adapter schedule:** `eq.evaluation.medsam_glomeruli_adapter` supports `lr_scheduler: cosine` with `min_lr` (eta_min); wired from `training.lr_scheduler` / `training.min_lr` in YAML for `eq_native_adapter`.
- **Conservative splits:** Default and pilot config document `inputs.train_fraction: 0.70` and `validation_fraction: 0.15` (remainder test), grouped by `source_sample_id` / strongest id — no leakage across splits.
- **Deploy preset:** `configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml` — 30 epochs, batch 2, cosine LR to `min_lr: 1e-6`, `run_local_feasibility_smoke: true`, `local_feasibility_required: true`, separate `run.name` / checkpoint / output roots under `deploy_conservative_mps_glomeruli`.
- **Pilot config:** `configs/medsam_glomeruli_fine_tuning.yaml` now enables feasibility smoke by default when `local_feasibility_required` is true (spec alignment for 7.2).

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
- Follow-up fine-tuned evaluation (same workflow run) writes under `finetuned_evaluation/` and populates `finetuned_evaluation` / `finetuned_comparison` in `summary.json` when `supported_checkpoint` is true and inference succeeds. Example pilot artifacts: runtime `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/pilot_medsam_glomeruli_fine_tuning/summary.json` (see `finetuned_comparison.oracle_dice_gap`, `adoption_tier`).
- **7.4 / 7.5:** Review fields: `summary.json` (baselines, `finetuned_*`, gates), `checkpoint_root/provenance.json`, `finetuned_evaluation/metrics.csv`, overlays, `prompt_failures*`. Record deploy command using `configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml` for the stronger conservative MPS run.

See `continuation-action-plan.md` for historical task breakdown; `tasks.md` reflects current completion state.

## Conservative deploy run (`deploy_conservative_mps_glomeruli`, 2026-04-30)

Evidence captured under runtime root `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier`:

- **Command:** `eq run-config --config configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml` (with `EQ_RUNTIME_ROOT` set; logging via `2>&1 | tee …` recommended).
- **`summary.json`:** `finetuned_evaluation.status=completed`, `metric_rows=42`, `training_process_exit_code=0`, `training_skip_reason` empty.
- **Adoption / gates (`finetuned_comparison`):** `adoption_tier=improved_candidate_not_oracle`, `oracle_level_gates_passed=false`, `failure_mode=oracle_gap`, `oracle_dice_gap≈0.1061`, `improves_current_auto=true`, `improves_current_segmenter=true`, `beats_trivial_baseline=true`.
- **Artifacts:** checkpoints under `models/medsam_glomeruli/deploy_conservative_mps_glomeruli/`; overlays under `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/deploy_conservative_mps_glomeruli/finetuned_evaluation/overlays/`.

**Interpretation:** Fine-tuning and fixed-split inference **completed successfully**. Tier is **improved candidate**, not **oracle-level preferred** — downstream hybrid Label Studio work should treat oracle-gap closure as a separate modeling milestone.

## Generated-mask packaging (2026-05-01)

- `_package_generated_mask_release` previously existed but **was not invoked** from the workflow; release directories could be absent even when evaluation succeeded.
- **Fix:** After `finetuned_evaluation.status == completed` and `metric_rows > 0`, the workflow packages masks into `outputs.generated_mask_release_root`, writes `manifest.csv` / `INDEX.md` / `provenance.json`, and appends the central registry unless `outputs.package_generated_mask_release: false`.
- **Operator doc:** `docs/MEDSAM_GLOMERULI_FINETUNING_HANDOFF.md` — rerun deploy once after pulling this fix so runtime trees populate `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`.

## Archive readiness

- `tasks.md`: all boxes checked; `openspec validate medsam-glomeruli-fine-tuning --strict` passes before archive.
- Implementation notes above align pilot + deploy evidence and document packaging correction for reproducible mask releases.