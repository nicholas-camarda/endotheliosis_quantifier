# Endotheliosis Quantifier

Endotheliosis Quantifier (`eq`) is a research pipeline for glomeruli segmentation and image-level endotheliosis review triage in kidney histology.

The current usable endpoint is a binary review-triage workflow:

- `no_low`: score `0` or `0.5`
- `moderate_severe`: score `1.5`, `2`, or `3`
- `borderline_review`: score `1.0`, routed for review

The model is useful for prioritizing human review of no/low versus moderate/severe cases. It is not a clinical diagnostic device, an autonomous grader, or external validation evidence.


| Current performance summary | Review workload summary |
| --------------------------- | ----------------------- |
| Selected triage model: `roi_qc_binary_logistic` | Primary target: `no_low` vs `moderate_severe` |
| Balanced accuracy `0.657`, recall `0.705`, AUROC `0.695` | `371` no/low, `220` moderate/severe, `116` borderline-review rows |


## Quick Start

Install the package in the environment that matches the machine.

Linux/CUDA:

```bash
mamba env create -f environment.yml
conda activate eq
pip install -e .[dev]
```

macOS Apple Silicon/MPS:

```bash
mamba env create -f environment-macos.yml
conda activate eq-mac
pip install -e .[dev]
```

Check that the CLI imports:

```bash
python -m eq --help
eq capabilities
eq mode --show
```

## Start A Label Studio Grading Project

Use this when collaborators need to grade complete glomeruli from a directory of images. Docker Desktop is required because Label Studio runs as a separate local web app, not inside the `eq-mac` environment.

macOS setup:

```bash
brew install --cask docker
open -a Docker
```

Wait for Docker Desktop to finish starting. For active hybrid development, use the two-terminal demo loop below as the default path.

Terminal 1 starts the MedSAM companion on the Mac host so MPS/Metal is available:

```bash
cd /Users/ncamarda/Projects/endotheliosis_quantifier
conda activate eq-mac
PYTORCH_ENABLE_MPS_FALLBACK=1 /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq.labelstudio.medsam_companion \
  --checkpoint /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/medsam_glomeruli_fine_tuning/deploy_conservative_mps_glomeruli/finetuned_evaluation/medsam_glomeruli_best_sam_state_dict.pth \
  --device mps \
  --port 8098
```

Terminal 2 creates or refreshes the Label Studio project:

```bash
cd /Users/ncamarda/Projects/endotheliosis_quantifier
conda activate eq-mac
eq labelstudio start /tmp/eq_hybrid_demo/images --project-name "Hybrid Demo Clean Components"
```

The command recursively imports `.jpg`, `.jpeg`, `.png`, `.tif`, and `.tiff` files, creates a local Label Studio project with `configs/label_studio_glomerulus_grading.xml`, and applies hybrid bootstrap settings from `configs/label_studio_medsam_hybrid.yaml` (or `--config` override). It prints the Label Studio URL, project URL, and hybrid companion status.

Open the `Project URL` printed by the command. Do not hardcode the project id; each fresh demo project can receive a new `/projects/<id>/data` URL.

Default local login:

```text
Email: eq-admin@example.local
Password: eq-labelstudio
```

Preview without starting Docker or importing tasks:

```bash
eq labelstudio start /path/to/images --dry-run
```

Legacy automation remains valid:

```bash
eq labelstudio start --images /path/to/images
```

For details, see [docs/LABEL_STUDIO_GLOMERULUS_GRADING.md](docs/LABEL_STUDIO_GLOMERULUS_GRADING.md).
For MedSAM box-assist companion launch/contract, see [docs/LABEL_STUDIO_MEDSAM_COMPANION.md](docs/LABEL_STUDIO_MEDSAM_COMPANION.md).

### Current Hybrid Status

This loop is usable for development iteration, but still half-finished from an operator UX standpoint. The current gap is preload quality/instance semantics, not core import/export plumbing.

Where this is stuck as of 2026-05-01:

- Runtime logs confirmed that the confusing green blobs in Label Studio are coming from the selected MedSAM mask release, not from Label Studio inventing geometry.
- The active preload release is `deploy_conservative_mps_glomeruli`.
- The demo images resolve to generated mask PNGs under `derived_data/generated_masks/glomeruli/medsam_finetuned/deploy_conservative_mps_glomeruli/masks/`.
- Connected-component splitting and RLE encoding preserve the source mask geometry; the bad shapes already exist in the release masks before import.
- Label Studio hover/selection makes bad regions look like they appear/disappear because it changes opacity and highlights the full region extent.
- The current auto-preload path is therefore not a good operator demo for these images.

Next work should decide between these paths before further UI polish:

1. Prefer box-assist/manual-first for new images: do not preload the full-image release masks by default; let the operator draw a box, call MedSAM, then brush/erase and grade the resulting region.
2. Keep auto-preload only when a release has passed a per-image quality gate; add explicit config for minimum area, border-touch rejection, maximum area, and whether to materialize or strip prediction overlays.
3. Add explicit policy for images with existing manual masks: if trusted manual masks already exist, import those as editable annotations and do not run or preload model inference for that image unless requested.

Debug status:

- Instrumentation remains in `src/eq/labelstudio/bootstrap.py` and `src/eq/labelstudio/medsam_companion.py` for the next session.
- The next evidence gap is box-assist quality: restart the companion from current source and run `/v1/box_infer` on a known glomerulus box to log whether the box-prompt output is tight enough or whether the companion/checkpoint preprocessing is also poor.

Local login:

```text
Email: eq-admin@example.local
Password: eq-labelstudio
```

Suggested annotation mode while iterating:

1. Keep `Compare All` off.
2. Work only in `eq-admin_local`.
3. Correct masks with brush/eraser, delete obvious false positives, and relabel cutoffs as `cutoff_partial_glomerulus`.
4. Grade complete glomeruli per-region.

Export and run quant contract-first check:

```bash
conda activate eq-mac
eq run-config --config configs/endotheliosis_quantification.yaml
```

If you need to quickly redeploy a fresh demo project after code changes, rerun:

```bash
eq labelstudio start /tmp/eq_hybrid_demo/images --project-name "Hybrid Demo Clean Components"
```

## MedSAM glomerulus fine-tuning (domain adaptation)

YAML-first workflow: trains/fine-tunes MedSAM glomeruli masks on admitted manual-mask cohort rows, evaluates on fixed splits, optionally **packages** prediction masks into `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` when fine-tuned evaluation completes.

Operator handoff (paths, logs, gates, mask releases):

- [docs/MEDSAM_GLOMERULI_FINETUNING_HANDOFF.md](docs/MEDSAM_GLOMERULI_FINETUNING_HANDOFF.md)

Quick dry-run:

```bash
eq run-config --config configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml --dry-run
```

## Run The Current Quantification Workflow

The main entrypoint is YAML-first:

```bash
eq run-config --config configs/endotheliosis_quantification.yaml
eq run-config --config configs/label_free_roi_embedding_atlas.yaml
```

The first command builds the scored ROI, embedding, burden, comparator, learned-ROI, and review artifact tree. The second command builds the label-free ROI embedding atlas and the binary no/low versus moderate/severe triage handoff.

The committed configs expect the runtime root to contain the required data, masks, labels, and model artifacts. Use `EQ_RUNTIME_ROOT` or edit the YAML when running against a different runtime tree.

Authoritative grading loop for reruns:

1. Finalize Label Studio export (image-level legacy or per-glomerulus hybrid export).
2. Rebuild scored examples through contract-first quantification (`eq run-config --config ...`).
3. Review lineage in `scored_examples/lineage_summary.json` to confirm which grading snapshot and scoring unit were consumed.
4. Compare new burden/quant outputs against prior run roots before promoting decisions.

## Review The Result

Open the atlas output in this order:

1. `burden_model/embedding_atlas/INDEX.md`
2. `burden_model/embedding_atlas/summary/atlas_verdict.json`
3. `burden_model/embedding_atlas/evidence/embedding_atlas_review.html`
4. `burden_model/embedding_atlas/binary_review_triage/INDEX.md`
5. `burden_model/embedding_atlas/binary_review_triage/evidence/binary_triage_review.html`

The binary review HTML is a bounded QA sample, not a request to review every prediction row. Review the cards shown, export the CSV beside the HTML file, and inspect the full prediction table only if the sample reveals a systematic failure pattern.

## Current Result

Current selected triage model:

```text
roi_qc_binary_logistic
```

Current grouped-development metrics:


| Metric                 | Value |
| ---------------------- | ----- |
| Balanced accuracy      | 0.657 |
| Moderate/severe recall | 0.705 |
| Precision              | 0.517 |
| Specificity            | 0.609 |
| AUROC                  | 0.695 |
| Average precision      | 0.538 |


Target support:


| Group             | Count |
| ----------------- | ----- |
| no/low            | 371   |
| moderate/severe   | 220   |
| borderline review | 116   |


## Head-To-Head Results Snapshot

These tables summarize the main internal comparisons this repo has produced. They are useful for handoff and orientation, but they are not external validation or clinical-performance claims.

### Segmentation Comparisons

What actually drives hybrid Label Studio preload and box-assist (current defaults). Rows are ordered **canonical weights first**, then **packaged preload PNGs** derived from the same deploy run:

| Role | Model / artifact | Where it lives |
| --- | --- | --- |
| Live box-assist inference | Segment Anything **ViT-B** (`vit_b`) weights loaded from the deploy-run evaluation checkpoint | `${EQ_RUNTIME_ROOT}/output/segmentation_evaluation/medsam_glomeruli_fine_tuning/deploy_conservative_mps_glomeruli/finetuned_evaluation/medsam_glomeruli_best_sam_state_dict.pth` |
| Full-image preload masks | Fine-tuned MedSAM mask release `deploy_conservative_mps_glomeruli` (`mask_source=medsam_finetuned_glomeruli`) | `${EQ_RUNTIME_ROOT}/derived_data/generated_masks/glomeruli/medsam_finetuned/deploy_conservative_mps_glomeruli/` (see `manifest.csv`) |

The transfer/scratch rows below are **not** what Label Studio hybrid preload imports by default. Those `.pkl` paths are explicit comparison baselines wired into `configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml` under `current_segmenter` for MedSAM evaluation reports — legacy FastAI pickles are historical unless separately promoted as supported artifacts.

Glomeruli transfer-vs-scratch comparison used the deterministic adjudication-aware candidate-comparison panel (`30` crops across `27` images and `5` subjects; threshold `0.75`):

| Segmentation candidate | Dice | Jaccard | Precision | Recall | Result status |
| --- | ---: | ---: | ---: | ---: | --- |
| Transfer glomeruli candidate | 0.8719 | 0.7729 | 0.7957 | 0.9643 | Promotion-eligible, but tied |
| Scratch/no-mitochondria-base glomeruli candidate | 0.8674 | 0.7658 | 0.7975 | 0.9507 | Promotion-eligible, but tied |

Rows sorted by Dice descending.

MedSAM oracle-box pilot compared manual-box MedSAM against the current glomeruli segmenters on `20` admitted manual-mask rows:

| Segmentation path | Mean Dice | Mean Jaccard | Interpretation |
| --- | ---: | ---: | --- |
| MedSAM oracle box | 0.9229 | 0.8577 | Strong upper-bound boundary-quality evidence |
| Current scratch segmenter | 0.7102 | 0.5714 | Similar to transfer on this subset |
| Current transfer segmenter | 0.7088 | 0.5675 | Weaker than oracle-box MedSAM on this subset |

Rows sorted by mean Dice descending (oracle reference first, then CNN baselines).

Current caveat: the latest Label Studio hybrid debug run showed that the active full-image auto-preload release (`deploy_conservative_mps_glomeruli`) can produce poor operator-facing masks on demo images. The next segmentation UX direction should be box-assist/manual-first unless a release passes a quality gate.

### Quantification Comparisons

Binary no/low vs moderate/severe review triage is the current usable endpoint. Score `1.0` is routed to borderline review and excluded from primary binary metrics.

| Binary triage candidate | Feature family | Threshold | Balanced accuracy | Recall | Precision | Specificity | AUROC | Average precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `roi_qc_binary_logistic` | ROI/QC | 0.45 | 0.657 | 0.705 | 0.517 | 0.609 | 0.695 | 0.538 |
| `atlas_hybrid_binary_logistic` | embedding + ROI/QC + atlas anchor | 0.25 | 0.650 | 0.659 | 0.522 | 0.642 | 0.692 | 0.552 |
| `embedding_binary_logistic` | frozen embedding | 0.25 | 0.641 | 0.655 | 0.511 | 0.628 | 0.673 | 0.533 |
| `atlas_cluster_anchor_mapping` | atlas anchor cluster | 0.50 | 0.663 | 0.933 | 0.620 | 0.393 | 0.663 | 0.613 |

Selected model first; remaining rows sorted by AUROC descending.

Exploratory endotheliosis burden/grade screens are kept as review evidence, not promoted claims:

| Quantification screen | Best candidate | Level | Rows/subjects | Stage-index MAE | Grade-scale MAE | Status |
| --- | --- | --- | --- | ---: | ---: | --- |
| Learned ROI candidate screen | `subject_simple_roi_qc` | subject | 60 / 60 | 11.134 | 0.334 | Blocked: cohort predictability and numerical-warning diagnostics |
| Learned ROI candidate screen | `image_simple_roi_qc` | image | 707 / 60 | 23.590 | 0.608 | Blocked: broad prediction sets, score coverage gaps, numerical warnings, cohort predictability |
| Morphology-aware screen | `subject_morphology_only_ridge` | subject | 60 / 60 | 12.671 | 0.380 | Blocked by visual feature readiness |
| Morphology-aware screen | `image_morphology_only_ridge` | image | 707 / 60 | 21.999 | 0.660 | Blocked by visual feature readiness |

Within each screen, **subject**-level row precedes **image**-level (cohort rollup before per-image rows).

For the full checkpoint and release policy, see [docs/REPRODUCIBILITY_HANDOFF_2026-04-30.md](docs/REPRODUCIBILITY_HANDOFF_2026-04-30.md).

## Other Maintained Workflows

All maintained workflows use the same `eq run-config` entrypoint.


| Task                                                   | Config                                                              |
| ------------------------------------------------------ | ------------------------------------------------------------------- |
| Endotheliosis quantification                           | `configs/endotheliosis_quantification.yaml`                         |
| Glomeruli candidate comparison                         | `configs/glomeruli_candidate_comparison.yaml`                       |
| Glomeruli fine-tuning                                  | `configs/glomeruli_finetuning_config.yaml`                          |
| Glomeruli transport audit                              | `configs/glomeruli_transport_audit.yaml`                            |
| High-resolution concordance                            | `configs/highres_glomeruli_concordance.yaml`                        |
| Label-free atlas and binary triage                     | `configs/label_free_roi_embedding_atlas.yaml`                       |
| MedSAM glomeruli fine-tuning (conservative MPS deploy) | `configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml` |
| MedSAM glomeruli fine-tuning (pilot)                   | `configs/medsam_glomeruli_fine_tuning.yaml`                         |
| Mitochondria pretraining                               | `configs/mito_pretraining_config.yaml`                              |


Use `--dry-run` before long-running training or audit jobs:

```bash
eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run
```

## Repository Layout

```text
configs/      Runnable workflow YAMLs
docs/         Supporting guides, methods notes, and handoff docs
openspec/     Spec-driven change history and current contracts
src/eq/       Python package and CLI implementation
tests/        Unit and integration tests
assets/       Public-safe images for README/docs
```

Large data, trained models, logs, generated reports, notebooks, and review exports stay out of Git. Runtime paths are configured through `EQ_RUNTIME_ROOT`, `analysis_registry.yaml`, and `src/eq/utils/paths.py`.

## Documentation Map

- [docs/BINARY_REVIEW_TRIAGE_GUIDE.md](docs/BINARY_REVIEW_TRIAGE_GUIDE.md): how to review the binary triage HTML and interpret its dropdowns.
- [docs/REPRODUCIBILITY_HANDOFF_2026-04-30.md](docs/REPRODUCIBILITY_HANDOFF_2026-04-30.md): source checkpoint, metrics, release policy, and resume plan.
- [docs/ONBOARDING_GUIDE.md](docs/ONBOARDING_GUIDE.md): longer walkthrough for collaborators and future-you.
- [docs/OUTPUT_STRUCTURE.md](docs/OUTPUT_STRUCTURE.md): runtime directory layout and artifact locations.
- [docs/SEGMENTATION_ENGINEERING_GUIDE.md](docs/SEGMENTATION_ENGINEERING_GUIDE.md): segmentation engineering details.
- [docs/TECHNICAL_LAB_NOTEBOOK.md](docs/TECHNICAL_LAB_NOTEBOOK.md): detailed lab notes and current internal evidence.
- [docs/HISTORICAL_NOTES.md](docs/HISTORICAL_NOTES.md): archived implementation history.

## Development Checks

Run before committing:

```bash
ruff check .
python -m pytest -q
openspec validate --specs --strict
```

