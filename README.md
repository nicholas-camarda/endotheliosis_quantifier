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

Scored cohorts use one runtime-local manifest:

```text
$EQ_RUNTIME_ROOT/
├── raw_data/
│   └── cohorts/
│       ├── manifest.csv
│       ├── lauren_preeclampsia/
│       ├── vegfri_dox/
│       │   ├── images/
│       │   ├── masks/
│       │   └── metadata/
│       └── vegfri_mr/
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

The manifest is the canonical linking surface for cohort ID, runtime-local image paths, optional mask paths, score linkage, lane assignment, mapping verification, hashes, and admission state. It does not carry original PhD or cloud source paths; source-location audit belongs in sidecar ingest artifacts.
Source-location overrides are centralized in the shared path helpers: `EQ_DOX_LABEL_STUDIO_EXPORT`, `EQ_MR_SCORE_WORKBOOK`, and `EQ_MR_IMAGE_ROOT`.

Manifest rows are image-level. Admitted rows require `cohort_id`, `image_path`, `score`, and a score locator such as `source_sample_id` or `source_score_row`; the pipeline appends `manifest_row_id`, `harmonized_id`, `join_status`, `verification_status`, `lane_assignment`, `admission_status`, `exclusion_reason`, `image_sha256`, and `mask_sha256`. Placeholder rows may be present before enrichment, but rows without a resolved runtime-local `image_path` cannot be admitted for training or quantification.

Manifest naming separates cohort identity from admission lane:

- `cohort_id` names the biological/project cohort, such as `lauren_preeclampsia`, `vegfri_dox`, or `vegfri_mr`.
- `lane_assignment` names the workflow lane, such as `manual_mask_core`, `manual_mask_external`, `scored_only`, or `mr_concordance_only`.
- Lauren's preeclampsia masks and Dox masks are equivalent-stature manual-mask glomeruli training labels. The lane names preserve provenance; they do not make Dox a lower-stature or separately gated training source.

Current cohort states:

- `lauren_preeclampsia`: 88 current preeclampsia image/mask rows are localized as `manual_mask_core` and `admitted`.
- `vegfri_dox`: 864 Label Studio export rows are represented. The 626 decoded brushlabel image/mask rows include 619 accepted `manual_mask_external` rows used as first-class glomeruli training labels and 7 `unresolved` missing-score rows. The remaining scored-only rows include 228 `foreign` mixed-export rows and 10 unresolved rows without decoded runtime images.
- `vegfri_mr`: 127 workbook image-level rows are represented from the external-drive whole-field TIFF batches. The 126 rows with localized TIFFs are `evaluation_only`; workbook row `8570-5` is unresolved because no matching TIFF was found in the discovered image root. Phase 1 use is concordance/evaluation only.

Lucchi and other segmentation-install datasets stay outside `raw_data/cohorts/manifest.csv`.

## Set These Once

The workflow below is much easier if you choose your project name and model names once, up front, and then reuse shell variables instead of retyping placeholder fragments.

```bash
export EQ_RUNTIME_ROOT="/path/to/endotheliosis_quantifier_runtime"
export GLOMERULI_ROOT="$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia"
export GLOMERULI_TRAIN_ROOT="$EQ_RUNTIME_ROOT/raw_data/cohorts"
export GLOMERULI_ANNOTATIONS="$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia/scores/labelstudio_annotations.json"

export MITO_MODEL_DIR="$EQ_RUNTIME_ROOT/models/segmentation/mitochondria"
export GLOMERULI_MODEL_DIR="$EQ_RUNTIME_ROOT/models/segmentation/glomeruli"

# Base names passed to --model-name during training.
export MITO_MODEL_NAME="mitochondria_baseline"
export GLOMERULI_TRANSFER_MODEL_NAME="glomeruli_transfer_baseline"
export GLOMERULI_SCRATCH_MODEL_NAME="glomeruli_scratch_baseline"
```

What these mean:

- `GLOMERULI_ROOT` is Lauren's active preeclampsia cohort root with direct `images/` and `masks/`
- `GLOMERULI_TRAIN_ROOT` is the manifest-backed cohort registry root used to train on all admitted masked rows across current cohorts
- `MITO_MODEL_NAME`, `GLOMERULI_TRANSFER_MODEL_NAME`, and `GLOMERULI_SCRATCH_MODEL_NAME` are the only values you pass to `--model-name`

The training code appends the descriptive run suffix automatically when it writes the artifact directory and `.pkl`. Those full artifact paths are outputs of training, not inputs you should hardcode up front.

## Workflow

### Configured full segmentation run

The full fixed-loader retraining workflow is run from its YAML config:

```bash
eq run-config --config configs/segmentation_fixedloader_full_retrain.yaml
```

That command refreshes the cohort manifest, trains the mitochondria base, selects the exported base artifact, trains the glomeruli transfer and no-mitochondria-base candidates, and writes comparison evidence under `$EQ_RUNTIME_ROOT/output/segmentation_evaluation/glomeruli_candidate_comparison/<run_id>/`.

All YAML workflow configs in `configs/` use the same entrypoint:

```bash
eq run-config --config configs/mito_pretraining_config.yaml
eq run-config --config configs/glomeruli_finetuning_config.yaml
eq run-config --config configs/segmentation_fixedloader_full_retrain.yaml
```

### 1. Inspect hardware and mode

```bash
eq capabilities
eq mode --show
```

### 2. Validate raw glomeruli naming

```bash
eq validate-naming --data-dir "$GLOMERULI_ROOT"
```

### 3. Prepare Lucchi mitochondria images

```bash
eq organize-lucchi \
  --input-dir "$EQ_RUNTIME_ROOT/raw_data/lucchi" \
  --output-dir "$EQ_RUNTIME_ROOT/raw_data/mitochondria_data"
```

### 4. Train mitochondria model

```bash
python -m eq.training.train_mitochondria \
  --data-dir "$EQ_RUNTIME_ROOT/raw_data/mitochondria_data/training" \
  --model-dir "$MITO_MODEL_DIR" \
  --model-name "$MITO_MODEL_NAME" \
  --epochs 50 \
  --batch-size 24 \
  --learning-rate 1e-3 \
  --image-size 256
```

On the powerful Apple Silicon MPS machine class, `24` is the current starting batch-size recommendation for `256x256` mitochondria training. Override it when throughput or stability requires a different value.

After training finishes, inspect the produced artifact path and export the one you want to reuse:

```bash
find "$MITO_MODEL_DIR" -path "*${MITO_MODEL_NAME}-*" -name '*.pkl' | sort
export MITO_BASE_MODEL="/absolute/path/to/the/mitochondria_model.pkl"
```

### 5. Train glomeruli model

Transfer candidate:

```bash
python -m eq.training.train_glomeruli \
  --data-dir "$GLOMERULI_TRAIN_ROOT" \
  --model-dir "$GLOMERULI_MODEL_DIR" \
  --model-name "$GLOMERULI_TRANSFER_MODEL_NAME" \
  --base-model "$MITO_BASE_MODEL" \
  --epochs 30 \
  --batch-size 12 \
  --learning-rate 1e-3 \
  --image-size 256 \
  --crop-size 512 \
  --seed 42
```

Scratch candidate:

```bash
python -m eq.training.train_glomeruli \
  --data-dir "$GLOMERULI_TRAIN_ROOT" \
  --model-dir "$GLOMERULI_MODEL_DIR" \
  --model-name "$GLOMERULI_SCRATCH_MODEL_NAME" \
  --from-scratch \
  --epochs 50 \
  --batch-size 12 \
  --learning-rate 1e-3 \
  --image-size 256 \
  --crop-size 512 \
  --seed 42
```

On the powerful Apple Silicon MPS machine class, `12` is the current starting batch-size recommendation for `512x512` glomeruli crops. Override it when throughput or stability requires a different value.

For all-data glomeruli training, pass the manifest-backed cohort registry root: `$EQ_RUNTIME_ROOT/raw_data/cohorts`. The loader enumerates admitted manifest rows in the `manual_mask_core` and `manual_mask_external` lanes, so the current training set is Lauren's 88 preeclampsia manual-mask rows plus the 619 Dox manual-mask rows. For a Lauren-only run, use `$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia`. Raw backup trees are source material, not direct training roots. Generated manifest summaries, caches, evaluation artifacts, and prediction outputs belong under `derived_data` or `output`.

For heavy training runs, the dedicated training modules above are the safest entrypoints. The `eq` CLI is still useful for validation, inspection, capabilities, and utility commands.

The dedicated training module CLI is the canonical control surface for glomeruli candidate work. Choose the family explicitly with `--base-model` or `--from-scratch`; optional YAML files are overlays, not the authoritative promotion contract. A transfer run with `--base-model` must load that artifact and copy compatible weights or it stops with an error. The scratch candidate is the no-mitochondria-base baseline with an ImageNet-pretrained ResNet34 encoder, not a literal all-random initialization baseline.
`--model-name` is only the base name. The trainer adds the run suffix automatically when it creates the output directory and `.pkl`, so inspect the produced artifacts after training instead of guessing the final directory name in advance.

After transfer and scratch training finish, inspect the produced artifacts and export the exact paths you want to compare or quantify:

```bash
find "$GLOMERULI_MODEL_DIR/transfer" -path "*${GLOMERULI_TRANSFER_MODEL_NAME}-*" -name '*.pkl' | sort
export GLOMERULI_TRANSFER_MODEL="/absolute/path/to/the/transfer_model.pkl"

find "$GLOMERULI_MODEL_DIR/scratch" -path "*${GLOMERULI_SCRATCH_MODEL_NAME}-*" -name '*.pkl' | sort
export GLOMERULI_SCRATCH_MODEL="/absolute/path/to/the/scratch_model.pkl"
```

### 6. Compare glomeruli candidates

Compare existing trained artifacts:

```bash
python -m eq.training.compare_glomeruli_candidates \
  --data-dir "$GLOMERULI_TRAIN_ROOT" \
  --transfer-model-path "$GLOMERULI_TRANSFER_MODEL" \
  --scratch-model-path "$GLOMERULI_SCRATCH_MODEL" \
  --seed 42 \
  --crop-size 512
```

Train fresh transfer and scratch candidates inside the comparison workflow:

```bash
python -m eq.training.compare_glomeruli_candidates \
  --data-dir "$GLOMERULI_TRAIN_ROOT" \
  --transfer-base-model "$MITO_BASE_MODEL" \
  --transfer-model-name "$GLOMERULI_TRANSFER_MODEL_NAME" \
  --scratch-model-name "$GLOMERULI_SCRATCH_MODEL_NAME" \
  --seed 42 \
  --crop-size 512
```

If `--output-dir` is omitted, this workflow writes its deterministic validation manifest, per-candidate metrics, review panels, and promotion report under the active runtime output root's `glomeruli_candidate_comparison/` subtree on this machine. The decision states remain explicit: `promoted`, `blocked`, or `insufficient_evidence`. If transfer and scratch are within the practical tie margin, neither becomes the sole promoted default and both remain explicit research-use comparators.

### 7. Run the Label Studio-first quantification baseline

```bash
eq prepare-quant-contract \
  --data-dir "$GLOMERULI_ROOT" \
  --segmentation-model "$GLOMERULI_TRANSFER_MODEL" \
  --score-source labelstudio \
  --annotation-source "$GLOMERULI_ANNOTATIONS"

eq quant-endo \
  --data-dir "$GLOMERULI_ROOT" \
  --segmentation-model "$GLOMERULI_TRANSFER_MODEL" \
  --score-source labelstudio \
  --annotation-source "$GLOMERULI_ANNOTATIONS" \
  --output-dir "output/quantification/$GLOMERULI_PROJECT"
```

If you want to quantify with the scratch candidate instead, replace `"$GLOMERULI_TRANSFER_MODEL"` with `"$GLOMERULI_SCRATCH_MODEL"`.

This path currently treats the Label Studio image-level grade as the supervised target for each image/mask pair. ROI extraction uses the full multi-component mask bounding box with context padding, then builds frozen segmentation-backbone embeddings and a canonical penalized multiclass ordinal baseline from `src/eq/quantification/ordinal.py`.

The current local runtime audit cohort is numerically stable under that estimator, but it only populates scores `0.0`, `0.5`, `1.0`, and `1.5`. The pipeline therefore reports incomplete seven-bin target support until a richer scored cohort resolves the missing upper-score support. Treat the current outputs as a predictive audit baseline with explicit cohort-shape metadata, not as full target-support validation.

Current ordinal implementation surfaces:

- Canonical estimator surface: `src/eq/quantification/ordinal.py`
- Orchestration caller: `src/eq/quantification/pipeline.py` via `evaluate_embedding_table()` and the contract-first quantification entrypoints
- CLI entrypoint: `eq quant-endo`
- Regression surfaces: `tests/unit/test_quantification_pipeline.py` and `tests/integration/test_local_runtime_quantification_pipeline.py`

## Useful Commands

```bash
eq cohort-manifest
eq prepare-quant-contract --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia" --segmentation-model "$GLOMERULI_TRANSFER_MODEL" --score-source labelstudio --annotation-source "$GLOMERULI_ANNOTATIONS"
eq quant-endo --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia" --segmentation-model "$GLOMERULI_TRANSFER_MODEL" --score-source labelstudio --annotation-source "$GLOMERULI_ANNOTATIONS"
eq visualize --mask path/to/mask.png --output "$EQ_RUNTIME_ROOT/output/mask_preview.png"
```

`quant-endo` now writes:

- `labelstudio_scores/` with recovered per-image grades and duplicate-resolution audit tables
- `roi_crops/` with union-ROI crops over the full multi-component mask
- `embeddings/` with frozen segmentation-encoder embeddings
- `ordinal_model/` with predictions, probabilities, metrics, confusion matrix, and `review_report/ordinal_review.html`

## Configuration

The main project configs live here:

- `configs/mito_pretraining_config.yaml`
- `configs/glomeruli_finetuning_config.yaml`

These files are optional overlays and engineering references. They are not the canonical control surface for glomeruli promotion or candidate comparison; use the dedicated training-module CLI commands above for authoritative execution and provenance.

Path helpers centralize repo-local defaults, runtime roots, and external cohort sources. Use these environment variables instead of hardcoded local paths:

Repo-local compatibility overrides:

- `EQ_DATA_PATH`
- `EQ_OUTPUT_PATH`
- `EQ_CACHE_PATH`
- `EQ_MODEL_PATH`
- `EQ_LOG_PATH` or `EQ_LOGS_PATH`

Runtime-root overrides:

- `EQ_RUNTIME_ROOT`
- `EQ_RUNTIME_OUTPUT_PATH`
- `EQ_RUNTIME_MODEL_PATH`

External cohort source overrides:

- `EQ_DOX_LABEL_STUDIO_EXPORT`
- `EQ_MR_SCORE_WORKBOOK`
- `EQ_MR_IMAGE_ROOT`

## Development Notes

- Use `ruff check .` and `ruff format .` before committing formatting-heavy changes.
- Use `python -m pytest -q` for the local test pass.
- Avoid hardcoded machine-specific paths in code, configs, or docs.
- Keep datasets, trained models, notebooks, logs, and temporary artifacts out of Git.
