# Endotheliosis Quantifier

Endotheliosis Quantifier (`eq`) is a FastAI/PyTorch project for binary segmentation workflows around glomeruli histology and mitochondria pretraining, plus a maintained Label Studio-first image-level endotheliosis scoring baseline. The repository is a WSL-first development environment with local GPU training, local data directories, and Git-tracked code/config only.

If you want the friendlier long-form introduction and workflow explanation, see [docs/ONBOARDING_GUIDE.md](docs/ONBOARDING_GUIDE.md).
For the full curated documentation set, see [docs/README.md](docs/README.md).

## Operating Contract

| Area | Current contract |
| --- | --- |
| Development target | WSL on Windows with CUDA-capable PyTorch |
| macOS local execution | Apple Silicon/MPS through the `eq-mac` conda environment |
| Package source | `src/eq/` |
| Workflow control surface | YAML workflow files in `configs/`, run with `eq run-config --config <file>` |
| Runtime root | `EQ_RUNTIME_ROOT`, with this checkout's local default recorded in `analysis_registry.yaml` |
| Runtime inputs | Raw datasets under `$EQ_RUNTIME_ROOT/raw_data/` |
| Runtime outputs | Derived data, trained models, logs, and generated reports under `$EQ_RUNTIME_ROOT/derived_data/`, `$EQ_RUNTIME_ROOT/models/`, `$EQ_RUNTIME_ROOT/logs/`, and `$EQ_RUNTIME_ROOT/output/` |
| Scored cohort registry | `$EQ_RUNTIME_ROOT/raw_data/cohorts/manifest.csv` |
| Preeclampsia quantification labels | Label Studio-derived image-level grades joined to image/mask pairs |
| Quantification ROI semantics | Full multi-component union ROI |
| Quantification outputs | Frozen segmentation-encoder embeddings, ordinal predictions, and an HTML review artifact with example cases |

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

The manifest is the project-local data contract for cohort ID, runtime-local image paths, optional mask paths, score linkage, lane assignment, mapping verification, hashes, and admission state. It is not a generic public dataset format. Original source-folder provenance belongs in sidecar ingest artifacts, while normal training and quantification use the localized runtime cohort directories.

Manifest rows are image-level. Admitted rows require `cohort_id`, `image_path`, `score`, and a score locator such as `source_sample_id` or `source_score_row`; the pipeline appends `manifest_row_id`, `harmonized_id`, `join_status`, `verification_status`, `lane_assignment`, `admission_status`, `exclusion_reason`, `image_sha256`, and `mask_sha256`. Placeholder rows may be present before enrichment, but rows without a resolved runtime-local `image_path` cannot be admitted for training or quantification.

Manifest naming separates cohort identity from workflow role:

- `cohort_id` names the biological or project cohort.
- `lane_assignment` names the workflow lane, such as `manual_mask_core`, `manual_mask_external`, `scored_only`, or `mr_concordance_only`.
- Manual-mask lanes are first-class glomeruli training inputs when admitted; lane names preserve provenance and workflow role.

Lucchi and other segmentation-install datasets stay outside `raw_data/cohorts/manifest.csv`.

For the generic runtime layout, see [docs/OUTPUT_STRUCTURE.md](docs/OUTPUT_STRUCTURE.md#runtime-scored-cohort-layout). For this checkout's current local cohort counts and unresolved-source notes, see [docs/TECHNICAL_LAB_NOTEBOOK.md](docs/TECHNICAL_LAB_NOTEBOOK.md#local-cohort-manifest-snapshot).

## YAML-First Workflow

The normal segmentation workflow is controlled by YAML files in `configs/`. Edit the YAML for run names, roots, paths, model names, training settings, and comparison outputs; then run it through one CLI entrypoint.

The all-in-one segmentation run is:

```bash
eq run-config --config configs/segmentation_fixedloader_full_retrain.yaml
```

That command reads `configs/segmentation_fixedloader_full_retrain.yaml`, refreshes the cohort manifest, trains a fresh mitochondria base, uses that exported base artifact for glomeruli transfer, trains the no-mitochondria-base comparator, writes trained models under `$EQ_RUNTIME_ROOT/models/segmentation/`, writes comparison evidence under `$EQ_RUNTIME_ROOT/output/segmentation_evaluation/`, and tees the workflow output to `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/`.

Dry-run the resolved commands before launching training:

```bash
eq run-config --config configs/segmentation_fixedloader_full_retrain.yaml --dry-run
```

Supported workflow configs use the same entrypoint:

```bash
eq run-config --config configs/mito_pretraining_config.yaml
eq run-config --config configs/glomeruli_finetuning_config.yaml
eq run-config --config configs/segmentation_fixedloader_full_retrain.yaml
```

Use the full retraining YAML when you want candidate evidence for the current glomeruli segmentation baseline. Use the mitochondria or glomeruli YAMLs only when you intentionally want to run one stage.

The YAML owns the routine settings:

- `run.name`, `run.seed`, `run.python`, and `run.required_env`
- `run.runtime_root_default`, unless `EQ_RUNTIME_ROOT` is set for this shell
- runtime-relative input/output paths under `paths:` or `data:`
- model names, batch sizes, learning rates, image sizes, crop sizes, and comparison settings

Environment variables are only local overrides. In the common case, you do not need to define model names, model directories, training roots, annotation paths, or MPS fallback flags by hand. Set `EQ_RUNTIME_ROOT` only when you want to run the same YAML against a different runtime tree than the checkout default:

```bash
EQ_RUNTIME_ROOT=/path/to/runtime eq run-config --config configs/segmentation_fixedloader_full_retrain.yaml
```

Site-specific source-location overrides are defined in `src/eq/utils/paths.py` for local cohort ingestion. Treat those as local data plumbing, not as part of the ordinary run recipe.

## Typical Run

1. Activate the supported environment for the machine.
2. Run the sanity checks.
3. Review or edit the workflow YAML.
4. Dry-run the workflow.
5. Run the workflow from YAML.

```bash
eq capabilities
eq mode --show
eq run-config --config configs/segmentation_fixedloader_full_retrain.yaml --dry-run
eq run-config --config configs/segmentation_fixedloader_full_retrain.yaml
```

The comparison output reports explicit decision states: `promoted`, `blocked`, or `insufficient_evidence`. If transfer and no-mitochondria-base candidates are within the configured practical tie margin, neither becomes the sole promoted default and both remain explicit research-use comparators.

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

This path treats the Label Studio image-level grade as the supervised target for each image/mask pair. ROI extraction uses the full multi-component mask bounding box with context padding, then builds frozen segmentation-backbone embeddings and a canonical penalized multiclass ordinal baseline from `src/eq/quantification/ordinal.py`.

The pipeline reports cohort-shape and target-support metadata with each run. Treat these outputs as a predictive audit baseline unless the scored cohort provides the target support and validation evidence needed for the intended scientific claim.

Current ordinal implementation surfaces:

- Canonical estimator surface: `src/eq/quantification/ordinal.py`
- Orchestration caller: `src/eq/quantification/pipeline.py` via `evaluate_embedding_table()` and the contract-first quantification entrypoints
- CLI entrypoint: `eq quant-endo`
- Regression surfaces: `tests/unit/test_quantification_pipeline.py` and `tests/integration/test_local_runtime_quantification_pipeline.py`

`quant-endo` writes:

- `labelstudio_scores/` with recovered per-image grades and duplicate-resolution audit tables
- `roi_crops/` with union-ROI crops over the full multi-component mask
- `embeddings/` with frozen segmentation-encoder embeddings
- `ordinal_model/` with predictions, probabilities, metrics, confusion matrix, and `review_report/ordinal_review.html`

## Configuration Files

The main workflow configs live here:

- `configs/mito_pretraining_config.yaml`
- `configs/glomeruli_finetuning_config.yaml`
- `configs/segmentation_fixedloader_full_retrain.yaml`

Path helpers centralize repo-local defaults, runtime roots, and external cohort sources. Prefer YAML-relative paths and the path helpers in `src/eq/utils/paths.py` over hardcoded machine-specific paths.

## Development Notes

- Use `ruff check .` and `ruff format .` before committing formatting-heavy changes.
- Use `python -m pytest -q` for the local test pass.
- Avoid hardcoded machine-specific paths in code, configs, or docs.
- Keep datasets, trained models, notebooks, logs, and temporary artifacts out of Git.
