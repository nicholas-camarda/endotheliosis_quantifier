# Endotheliosis Quantifier

Endotheliosis Quantifier (`eq`) is a FastAI/PyTorch project for binary segmentation workflows around glomeruli histology and mitochondria pretraining, plus a maintained Label Studio-first image-level endotheliosis scoring baseline. This repository is currently maintained as a WSL-first development environment with local GPU training, local data directories, and Git-tracked code/config only.

If you want the friendlier long-form introduction and workflow explanation, see [docs/ONBOARDING_GUIDE.md](docs/ONBOARDING_GUIDE.md).
For the full curated documentation set, see [docs/README.md](docs/README.md).

## Current Baseline

- Primary development target: WSL on Windows with CUDA-capable PyTorch
- Package source: `src/eq/`
- Raw datasets: `data/raw_data/` (gitignored)
- Generated artifacts: `data/derived_data/`, `models/`, `logs/`, `output/` (gitignored)
- Main operational branch today: `master`
- Quantification default for preeclampsia: Label Studio-derived image-level grades joined to image/mask pairs
- Current quantification input semantics: full multi-component union ROI, not largest-component-only crops
- Current quantification outputs: frozen segmentation-encoder embeddings, ordinal predictions, and an HTML review artifact with example cases

The repo contains older experimental branches and historical notes. Treat the files in this branch as the source of truth unless you are deliberately comparing against another branch.

## Environment Setup

```bash
git clone https://github.com/nicholas-camarda/endotheliosis_quantifier.git
cd endotheliosis_quantifier

mamba env create -f environment.yml
conda activate eq

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
├── data/
│   ├── raw_data/
│   │   ├── lucchi/
│   │   └── my_glomeruli_project/
│   │       ├── images/
│   │       ├── masks/
│   │       ├── annotations/
│   │       │   └── annotations.json
│   │       └── subject_metadata.xlsx
│   └── derived_data/
├── models/
│   └── segmentation/
│       ├── mitochondria/
│       └── glomeruli/
├── src/eq/
└── tests/
```

Everything under `data/`, `models/`, `logs/`, and `output/` is intentionally local-only and should stay out of Git.

## Set These Once

The workflow below is much easier if you choose your project name and model names once, up front, and then reuse shell variables instead of retyping placeholder fragments.

```bash
export GLOMERULI_PROJECT="my_glomeruli_project"
export GLOMERULI_ROOT="data/raw_data/$GLOMERULI_PROJECT"
export GLOMERULI_TRAIN_ROOT="$GLOMERULI_ROOT/training_pairs"
export GLOMERULI_ANNOTATIONS="$GLOMERULI_ROOT/annotations/annotations.json"

export MITO_MODEL_DIR="models/segmentation/mitochondria"
export GLOMERULI_MODEL_DIR="models/segmentation/glomeruli"

export MITO_MODEL_NAME="mitochondria_baseline"
export MITO_RUN="${MITO_MODEL_NAME}-pretrain_e50_b24_lr1e-3_sz256"
export MITO_BASE_MODEL="$MITO_MODEL_DIR/$MITO_RUN/$MITO_RUN.pkl"

export GLOMERULI_TRANSFER_MODEL_NAME="glomeruli_transfer_baseline"
export GLOMERULI_TRANSFER_RUN="${GLOMERULI_TRANSFER_MODEL_NAME}-transfer_s1lr1e-3_s2lr_lrfind_e30_b12_lr1e-3_sz256"
export GLOMERULI_TRANSFER_MODEL="$GLOMERULI_MODEL_DIR/transfer/$GLOMERULI_TRANSFER_RUN/$GLOMERULI_TRANSFER_RUN.pkl"

export GLOMERULI_SCRATCH_MODEL_NAME="glomeruli_scratch_baseline"
export GLOMERULI_SCRATCH_RUN="${GLOMERULI_SCRATCH_MODEL_NAME}-scratch_e50_b12_lr1e-3_sz256"
export GLOMERULI_SCRATCH_MODEL="$GLOMERULI_MODEL_DIR/scratch/$GLOMERULI_SCRATCH_RUN/$GLOMERULI_SCRATCH_RUN.pkl"
```

What these mean:

- `GLOMERULI_PROJECT` is your raw-data project folder under `data/raw_data/`
- `MITO_BASE_MODEL` is the supported mitochondria `.pkl` produced in step 4
- `GLOMERULI_TRANSFER_MODEL` is the transfer candidate `.pkl` produced in step 5
- `GLOMERULI_SCRATCH_MODEL` is the scratch candidate `.pkl` produced in step 5

Define them at shell startup even if the later files do not exist yet. The point is that the paths become predictable and the later commands stay copy-pasteable.
If you change `--model-name`, `--epochs`, `--batch-size`, `--learning-rate`, or `--image-size`, update the corresponding `*_RUN` variable so it still matches the artifact name the training code will write.

## Workflow

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
  --input-dir data/raw_data/lucchi \
  --output-dir data/derived_data/mitochondria_data
```

### 4. Train mitochondria model

```bash
python -m eq.training.train_mitochondria \
  --data-dir data/derived_data/mitochondria_data/training \
  --model-dir "$MITO_MODEL_DIR" \
  --model-name "$MITO_MODEL_NAME" \
  --epochs 50 \
  --batch-size 24 \
  --learning-rate 1e-3 \
  --image-size 256
```

On the powerful Apple Silicon MPS machine class, `24` is the current starting batch-size recommendation for `256x256` mitochondria training. Override it when throughput or stability requires a different value.

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

The glomeruli training root must contain paired full-image `images/` and `masks/` directories under `raw_data`. Raw project backups such as `clean_backup` are source material; curate paired files into `training_pairs` before running model training. Generated manifests, audits, caches, and metrics belong under `derived_data`.

For heavy training runs, the dedicated training modules above are the safest entrypoints. The `eq` CLI is still useful for validation, inspection, capabilities, and utility commands.

The dedicated training module CLI is the canonical control surface for glomeruli candidate work. Choose the family explicitly with `--base-model` or `--from-scratch`; optional YAML files are overlays, not the authoritative promotion contract.

### 6. Compare glomeruli candidates

```bash
python -m eq.training.compare_glomeruli_candidates \
  --data-dir "$GLOMERULI_TRAIN_ROOT" \
  --transfer-base-model "$MITO_BASE_MODEL" \
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
eq process-data --input-dir "$GLOMERULI_ROOT" --output-dir "data/derived_data/$GLOMERULI_PROJECT"
eq metadata-process --input-file "$GLOMERULI_ROOT/subject_metadata.xlsx" --output-dir "data/derived_data/$GLOMERULI_PROJECT/metadata"
eq prepare-quant-contract --data-dir "$GLOMERULI_ROOT" --segmentation-model "$GLOMERULI_TRANSFER_MODEL" --score-source labelstudio --annotation-source "$GLOMERULI_ANNOTATIONS"
eq quant-endo --data-dir "$GLOMERULI_ROOT" --segmentation-model "$GLOMERULI_TRANSFER_MODEL" --score-source labelstudio --annotation-source "$GLOMERULI_ANNOTATIONS"
eq audit-derived --data-dir "data/derived_data/$GLOMERULI_PROJECT"
eq visualize --mask path/to/mask.png --output output/mask_preview.png
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

Path defaults should stay relative to the repo root unless there is a strong reason to use environment overrides. Supported overrides include:

- `EQ_DATA_PATH`
- `EQ_OUTPUT_PATH`
- `EQ_CACHE_PATH`
- `EQ_MODEL_PATH`

## Development Notes

- Use `ruff check .` and `ruff format .` before committing formatting-heavy changes.
- Use `python -m pytest -q` for the local test pass.
- Avoid hardcoded machine-specific paths in code, configs, or docs.
- Keep datasets, trained models, notebooks, logs, and temporary artifacts out of Git.

## Git Hygiene

- Create focused branches for real work instead of piling changes into long-lived dirty working trees.
- Treat branch comparisons explicitly when reviving old work. This repo has multiple historical lines of development.
- Snapshot risky states before cleanup with a safety branch or stash.
