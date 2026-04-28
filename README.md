# Endotheliosis Quantifier

Endotheliosis Quantifier (`eq`) is a FastAI/PyTorch toolkit for glomeruli segmentation and endotheliosis quantification in kidney histology, with manifest-backed multi-cohort intake and candidate-evaluated segmentation training. Mitochondria pretraining is one supported transfer-learning path for segmentation. The repository supports active development in both WSL/Linux with CUDA and macOS Apple Silicon with MPS, with local data directories and Git-tracked code/config only.

If you want the friendlier long-form introduction and workflow explanation, see [docs/ONBOARDING_GUIDE.md](docs/ONBOARDING_GUIDE.md).
For the full curated documentation set, see [docs/README.md](docs/README.md).

## Quick Start

The main workflow entrypoint is `eq run-config`. If you want the easiest supported way to run the segmentation workflows in this repo, use one of the committed YAML configs:

```bash
eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run
eq run-config --config configs/glomeruli_candidate_comparison.yaml
```

Other supported workflow configs use the same entrypoint:

```bash
eq run-config --config configs/mito_pretraining_config.yaml
eq run-config --config configs/glomeruli_finetuning_config.yaml
```

The YAML is the control surface. In the common case, you should not need to stitch the workflow together manually from separate shell commands.

## Operating Contract

| Area | Current contract |
| --- | --- |
| Supported development environments | WSL/Linux with CUDA via `eq` and macOS Apple Silicon/MPS via `eq-mac` |
| Package source | `src/eq/` |
| Workflow control surface | YAML workflow files in `configs/`, run with `eq run-config --config <file>` |
| Runtime root | `EQ_RUNTIME_ROOT`, with this checkout's local default recorded in `analysis_registry.yaml` |
| Runtime inputs | Raw datasets under `$EQ_RUNTIME_ROOT/raw_data/` |
| Runtime outputs | Derived data, trained models, logs, and generated reports under `$EQ_RUNTIME_ROOT/derived_data/`, `$EQ_RUNTIME_ROOT/models/`, `$EQ_RUNTIME_ROOT/logs/`, and `$EQ_RUNTIME_ROOT/output/` |
| Scored cohort registry | `$EQ_RUNTIME_ROOT/raw_data/cohorts/manifest.csv` |
| Current quantification supervision | Image-level grades joined to image/mask pairs in the active scored cohort workflow |
| Quantification ROI semantics | Full multi-component union ROI |
| Quantification outputs | Frozen segmentation-encoder embeddings, exploratory burden-index predictions, learned ROI candidate screens, comparator outputs, subject/cohort summaries, and combined review artifacts |

## Environment Contract

This repository has two supported Python environments. Use the one that matches the machine you are on.

| Machine | Environment | Setup file | Use for |
| --- | --- | --- | --- |
| WSL/Linux with CUDA | `eq` | `environment.yml` | CUDA development, CUDA training, general Linux tests |
| macOS Apple Silicon with MPS | `eq-mac` | `environment-macos.yml` | Mac execution, MPS segmentation training, segmentation validation, model export, model loading |

On macOS, use the Mac environment explicitly:

```bash
conda activate eq-mac
python -m eq --help
```

Real MPS training or validation should run in a normal macOS terminal with the `eq-mac` interpreter. Sandboxed terminal results are not authoritative for local Metal execution. For segmentation training and validation on Mac, use:

```bash
env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq \
  python ...
```

Do not use the generic `eq` environment for Mac MPS segmentation work. Transfer training with `--base-model` must load that artifact and copy compatible weights. The `--from-scratch` glomeruli path is the no-mitochondria-base comparator and uses FastAI's explicit ImageNet-pretrained ResNet34 encoder initialization.

## Environment Setup

For WSL/Linux with CUDA:

```bash
git clone https://github.com/nicholas-camarda/endotheliosis_quantifier.git
cd endotheliosis_quantifier

mamba env create -f environment.yml
conda activate eq

pip install -e .[dev]
```

For macOS Apple Silicon with MPS:

```bash
git clone https://github.com/nicholas-camarda/endotheliosis_quantifier.git
cd endotheliosis_quantifier

mamba env create -f environment-macos.yml
conda activate eq-mac

pip install -e .[dev]
```

If `conda activate eq` fails inside a fresh shell, initialize Conda for the shell first:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eq
```

## Sanity Checks

```bash
python -m eq --help
python -m pytest -q
eq capabilities
eq mode --show
```

`python -m eq --help` is the fastest check that the package imports cleanly. `eq capabilities` and `eq mode --show` help confirm whether the current machine is being detected as CUDA, MPS, or CPU.

## Recommended Repository Layout

```text
endotheliosis_quantifier/
├── configs/
├── docs/
├── openspec/
├── src/eq/
└── tests/
```

The repo checkout is for code, configs, tests, and docs. Active raw data, derived data, trained models, logs, and generated outputs live under the active runtime root and stay out of Git.

Committed configs and docs use runtime-root-relative paths such as `raw_data/...`, `derived_data/...`, `models/...`, and `output/...`. Code resolves those through `src/eq/utils/paths.py`, using `EQ_RUNTIME_ROOT` or the local default recorded in `analysis_registry.yaml`.

## Scored Cohort Manifest

Segmentation and quantification workflows use one runtime-local scored-cohort manifest:

```text
$EQ_RUNTIME_ROOT/
├── raw_data/
│   └── cohorts/
│       ├── manifest.csv
│       └── <cohort_id>/
├── models/segmentation/
└── output/
    ├── segmentation_evaluation/
    ├── predictions/
    └── quantification_results/
```

Build or refresh the runtime manifest with:

```bash
eq cohort-manifest
```

The manifest is the project-local data contract for runtime-local image assets, optional masks, score linkage, workflow lane, verification or admission state, and file hashes. It is not a generic public dataset format. Original source-folder provenance belongs in sidecar ingest artifacts, while normal training and quantification use the localized runtime cohort directories.

Lucchi and other segmentation-install datasets stay outside `raw_data/cohorts/manifest.csv`.

For the generic runtime layout and more detailed manifest semantics, see [docs/OUTPUT_STRUCTURE.md](docs/OUTPUT_STRUCTURE.md#runtime-scored-cohort-layout). Current local cohort counts and source-specific notes live in the lab notebook rather than this front-door README.

## YAML-First Workflow

The normal segmentation workflow is controlled by YAML files in `configs/`. Edit the YAML for run names, roots, paths, model names, training settings, and comparison outputs; then run it through one CLI entrypoint.

The candidate-comparison run is:

```bash
eq run-config --config configs/glomeruli_candidate_comparison.yaml
```

That command reads `configs/glomeruli_candidate_comparison.yaml`, refreshes the cohort manifest, trains or loads the mitochondria base needed for glomeruli transfer, trains the no-mitochondria-base comparator, writes trained models under `$EQ_RUNTIME_ROOT/models/segmentation/`, writes comparison evidence under `$EQ_RUNTIME_ROOT/output/segmentation_evaluation/`, and tees the workflow output to `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/`.

Dry-run the resolved commands before launching training:

```bash
eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run
```

Supported workflow configs use the same entrypoint:

```bash
eq run-config --config configs/mito_pretraining_config.yaml
eq run-config --config configs/glomeruli_finetuning_config.yaml
eq run-config --config configs/glomeruli_candidate_comparison.yaml
eq run-config --config configs/glomeruli_transport_audit.yaml
eq run-config --config configs/highres_glomeruli_concordance.yaml
eq run-config --config configs/endotheliosis_quantification.yaml
```

Use `configs/glomeruli_candidate_comparison.yaml` when you want candidate evidence for the current glomeruli segmentation baseline. Use the transport, high-resolution concordance, or quantification YAMLs only when you already have the explicit upstream artifacts required by those stages.

The YAML owns the routine settings:

- `run.name`, `run.seed`, `run.python`, and `run.required_env`
- `run.runtime_root_default`, unless `EQ_RUNTIME_ROOT` is set for this shell
- runtime-relative input/output paths under `paths:` or `data:`
- model names, batch sizes, learning rates, image sizes, crop sizes, and comparison settings

Environment variables are only local overrides. In the common case, you do not need to define model names, model directories, training roots, annotation paths, or MPS fallback flags by hand. Set `EQ_RUNTIME_ROOT` only when you want to run the same YAML against a different runtime tree than the checkout default:

```bash
EQ_RUNTIME_ROOT=/path/to/runtime eq run-config --config configs/glomeruli_candidate_comparison.yaml
```

Site-specific source-location overrides are defined in `src/eq/utils/paths.py` for local cohort ingestion. Treat those as local data plumbing, not as part of the ordinary run recipe.

## Current Segmentation Snapshot

The current checked-in segmentation snapshot comes from the April 25, 2026 P0 workflow artifacts under `$EQ_RUNTIME_ROOT/models/segmentation/` and `$EQ_RUNTIME_ROOT/output/segmentation_evaluation/glomeruli_candidate_comparison/production_glomeruli_candidate_p0_contract_20260425_adjudicated/`.

- Current candidate artifacts are recorded by the comparison report and model sidecars under the runtime model root.
- Current deterministic glomeruli review panel: `30` crops across `27` images and `5` subjects
- Review-panel category balance: `10` background, `10` boundary, `10` positive

The current glomeruli candidates are available as research-use artifacts, but the repository does not currently select a single default candidate because transfer and scratch remain within the configured practical tie margin. For the checked-in internal evidence and interpretation, see [docs/TECHNICAL_LAB_NOTEBOOK.md](docs/TECHNICAL_LAB_NOTEBOOK.md#current-segmentation-training-snapshot).

## Typical Run

1. Activate the supported environment for the machine.
2. Run the sanity checks.
3. Review or edit the workflow YAML.
4. Dry-run the workflow.
5. Run the workflow from YAML.

```bash
eq capabilities
eq mode --show
eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run
eq run-config --config configs/glomeruli_candidate_comparison.yaml
```

The comparison output reports explicit decision and evidence states. Transfer and no-mitochondria-base candidates can remain explicit research-use comparators even when the evidence is `insufficient_evidence`, `audit_missing`, or `not_promotion_eligible` for front-page performance claims.

## Utility Commands

These commands are useful for inspection, data preparation, or targeted checks. They are not the normal way to stitch together a full training run.

```bash
eq cohort-manifest
eq validate-naming --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>"
eq organize-lucchi \
  --input-dir "$EQ_RUNTIME_ROOT/raw_data/lucchi" \
  --output-dir "$EQ_RUNTIME_ROOT/raw_data/mitochondria_data"
eq visualize --mask path/to/mask.png --output "$EQ_RUNTIME_ROOT/output/mask_preview.png"
```

## Quantification

Quantification runs against an explicit segmentation model artifact produced by the YAML workflow. Use the model path from the completed run's model directory or comparison report.

```bash
eq prepare-quant-contract \
  --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>" \
  --segmentation-model /absolute/path/to/glomeruli_model.pkl \
  --score-source labelstudio \
  --annotation-source "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>/scores/labelstudio_annotations.json"

eq quant-endo \
  --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>" \
  --segmentation-model /absolute/path/to/glomeruli_model.pkl \
  --score-source labelstudio \
  --annotation-source "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>/scores/labelstudio_annotations.json" \
  --output-dir "$EQ_RUNTIME_ROOT/output/quantification_results/<cohort_id>"
```

To quantify with a different candidate, use that candidate's `.pkl` path.

The current maintained quantification path uses image-level grades joined to each image or mask pair. ROI extraction uses the full multi-component mask bounding box with context padding, then builds frozen segmentation-backbone embeddings. The primary model output is an exploratory endotheliosis burden index on a `0-100` ordinal stage scale, with learned ROI candidate screens, direct stage-index regression, and ordinal/multiclass outputs retained as comparators.

The current burden-index model is not README/docs-ready as an operational model claim. It is useful for review and method development, but the latest full-cohort run still has broad per-image prediction sets, slight undercoverage against the nominal prediction-set target, and finite-output backend matrix warnings. Subject-level ROI aggregation is the strongest current follow-up direction for cohort summaries, while per-image prediction remains a separate calibration and feature-modeling problem.

Endotheliosis is graded by assessing the relative amount of open versus collapsed capillary or arteriole lumina within the glomerulus. The maintained quantification path writes deterministic morphology features for open/pale lumina, collapsed or slit-like structures, ridge/line signals, erythrocyte-like patent-lumen confounding, and ROI quality. These morphology features are candidate evidence, not a deployed mechanistic model; the generated feature-review HTML and operator adjudication template must be inspected before using them for a shareable claim.

The learned ROI branch is a capped phase-1 screen under `burden_model/learned_roi/`. It fits only the current glomeruli encoder embeddings, simple ROI QC features, and their hybrid. Optional backbone or foundation providers are audited but not fitted. A learned ROI result is shareable only if its generated candidate summary passes the explicit uncertainty, numerical-stability, ordinal/grade-scale, and cohort-confounding gates.

Current quantification implementation surfaces:

- Primary burden-index estimator surface: `src/eq/quantification/burden.py`
- Learned ROI candidate surface: `src/eq/quantification/learned_roi.py`
- Ordinal comparator surface: `src/eq/quantification/ordinal.py`
- Orchestration caller: `src/eq/quantification/pipeline.py` via `evaluate_embedding_table()` and the contract-first quantification entrypoints
- CLI entrypoint: `eq quant-endo`
- Regression surfaces: `tests/unit/test_quantification_pipeline.py` and `tests/integration/test_local_runtime_quantification_pipeline.py`

`quant-endo` writes:

- `labelstudio_scores/` with recovered per-image grades and duplicate-resolution audit tables
- `roi_crops/` with union-ROI crops over the full multi-component mask
- `embeddings/` with frozen segmentation-encoder embeddings
- `burden_model/` with grouped `primary_model/`, `validation/`, `calibration/`, `summaries/`, `evidence/`, `candidates/`, `diagnostics/`, `feature_sets/`, and `learned_roi/` subfolders for exploratory burden predictions, support gates, uncertainty calibration, cohort summaries, nearest examples, candidate screens, morphology features, learned ROI screens, and review diagnostics
- `ordinal_model/` with comparator predictions, probabilities, metrics, confusion matrix, and `review_report/ordinal_review.html`
- `quantification_review/` with combined HTML review, reviewer examples, concrete result summaries, and a README/docs snippet from the current run; reuse the snippet only when the reported readiness flag and uncertainty checks pass

## Configuration Files

The main workflow configs live here:

- `configs/mito_pretraining_config.yaml`
- `configs/glomeruli_finetuning_config.yaml`
- `configs/glomeruli_candidate_comparison.yaml`
- `configs/glomeruli_transport_audit.yaml`
- `configs/highres_glomeruli_concordance.yaml`
- `configs/endotheliosis_quantification.yaml`

Path helpers centralize repo-local defaults, runtime roots, and external cohort sources. Prefer YAML-relative paths and the path helpers in `src/eq/utils/paths.py` over hardcoded machine-specific paths.

## Development Notes

- Use `ruff check .` and `ruff format .` before committing formatting-heavy changes.
- Use `python -m pytest -q` for the local test pass.
- Avoid hardcoded machine-specific paths in code, configs, or docs.
- Keep datasets, trained models, notebooks, logs, and temporary artifacts out of Git.
