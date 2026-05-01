# MedSAM glomerulus fine-tuning — operator handoff

This document is the **minimal usable handoff** for running the repo workflow, inspecting results, and consuming **generated-mask releases** under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`.

## What “usable” means here

- **Code + configs** live in Git (`eq`, YAML under `configs/`).
- **Data, checkpoints, masks, and evaluation outputs** live under **`EQ_RUNTIME_ROOT`** (not committed).
- After a successful run you should have:
  - **Checkpoints** under `models/medsam_glomeruli/<checkpoint_id>/`
  - **`summary.json`** under `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/<run_id>/`
  - **Fine-tuned metrics + overlays** under `…/finetuned_evaluation/`
  - **Packaged masks + release manifest** under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` when evaluation completes (see `outputs.package_generated_mask_release` in YAML; default **true** in code).

Heavy artifacts (MedSAM base `.pth`, MedSAM repo path, cohort imagery) remain **machine-local** — configure them in YAML.

## Prerequisites (macOS / `eq-mac`)

- Conda env: `environment-macos.yml` → `eq-mac`
- Docker Desktop (only if you use Label Studio separately)
- Local **MedSAM** clone + **`medsam_vit_b.pth`** (or equivalent) at paths referenced in your YAML

Canonical interpreter (see `AGENTS.md`):

```text
/Users/ncamarda/mambaforge/envs/eq-mac/bin/python
```

## Conservative deploy preset (recommended)

Config: `configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml`

Example full command with logging (adjust `EQ_RUNTIME_ROOT`):

```bash
export EQ_RUNTIME_ROOT=/path/to/your/runtime
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR=/tmp/mpl_eq

/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config \
  --config configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml \
  2>&1 | tee "${EQ_RUNTIME_ROOT}/logs/medsam_deploy_conservative_mps_run.log"
```

Run **outside** sandboxes that block Metal/MPS if training must use Apple GPU.

## Where to look after a run

Under `${EQ_RUNTIME_ROOT}` (paths mirror YAML defaults for `deploy_conservative_mps_glomeruli`):

| Artifact | Typical path |
| --- | --- |
| Run summary | `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/deploy_conservative_mps_glomeruli/summary.json` |
| Fine-tuned metrics | `…/finetuned_evaluation/metrics.csv` |
| Review overlays | `…/finetuned_evaluation/overlays/*_overlay.png` (green = manual mask, red = fine-tuned prediction, blue = proposal boxes) |
| Training provenance | `models/medsam_glomeruli/deploy_conservative_mps_glomeruli/provenance.json` |
| Mask release bundle | `derived_data/generated_masks/glomeruli/medsam_finetuned/deploy_conservative_mps_glomeruli/` (`manifest.csv`, `INDEX.md`, `masks/`) |
| Central registry append | `derived_data/generated_masks/glomeruli/manifest.csv` |

## Reading gates honestly

Open `summary.json` → `finetuned_comparison`:

- **`adoption_tier`**: `oracle_level_preferred` vs `improved_candidate_not_oracle` vs `blocked`
- **`oracle_dice_gap`**, **`oracle_level_gates_passed`**, **`failure_mode`**

`improved_candidate_not_oracle` means **engineering success** (beats automatic baselines, etc.) but **oracle-tier promotion may still fail** — that is expected until oracle-gap gates close.

## Pilot vs deploy

- **Pilot** config: `configs/medsam_glomeruli_fine_tuning.yaml` → default run id `pilot_medsam_glomeruli_fine_tuning`
- **Deploy** config: `configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml` → run id `deploy_conservative_mps_glomeruli`

## Collaborator artifact manifest (optional)

Example bundle manifest schema (illustrative only):

- `docs/examples/artifacts_manifest.example.json`

## Next OpenSpec threads (not implemented here)

- **`label-studio-medsam-hybrid-grading`** — consume mask releases inside hybrid Label Studio workflows.
- **`collaborator-distribution-packaging`** — Releases/LFS/Compose packaging narrative.
