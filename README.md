# Endotheliosis Quantifier

Endotheliosis Quantifier (`eq`) is a FastAI/PyTorch project for binary segmentation workflows around glomeruli histology and mitochondria pretraining. This repository is currently maintained as a WSL-first development environment with local GPU training, local data directories, and Git-tracked code/config only.

If you want the friendlier long-form introduction and workflow explanation, see [docs/ONBOARDING_GUIDE.md](/home/ncamarda/endotheliosis_quantifier/docs/ONBOARDING_GUIDE.md).
For the full curated documentation set, see [docs/README.md](/home/ncamarda/endotheliosis_quantifier/docs/README.md).

## Current Baseline

- Primary development target: WSL on Windows with CUDA-capable PyTorch
- Package source: `src/eq/`
- Raw datasets: `data/raw_data/` (gitignored)
- Generated artifacts: `data/derived_data/`, `models/`, `logs/`, `output/` (gitignored)
- Main operational branch today: `master`

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
│   │   └── <your_glomeruli_project>/
│   │       ├── images/
│   │       ├── masks/
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

## Workflow

### 1. Inspect hardware and mode

```bash
eq capabilities
eq mode --show
```

### 2. Validate raw glomeruli naming

```bash
eq validate-naming --data-dir data/raw_data/<your_glomeruli_project>
```

### 3. Prepare Lucchi mitochondria images

```bash
eq extract-images \
  --input-dir data/raw_data/lucchi \
  --output-dir data/derived_data/mito
```

### 4. Train mitochondria model

```bash
python -m eq.training.train_mitochondria \
  --data-dir data/derived_data/mito \
  --model-dir models/segmentation/mitochondria \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --image-size 256
```

### 5. Train glomeruli model

```bash
python -m eq.training.train_glomeruli \
  --data-dir data/raw_data/<your_glomeruli_project> \
  --model-dir models/segmentation/glomeruli \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --image-size 256 \
  --crop-size 512
```

For heavy training runs, the dedicated training modules above are the safest entrypoints. The `eq` CLI is still useful for validation, inspection, capabilities, and utility commands.

## Useful Commands

```bash
eq process-data --input-dir data/raw_data/<project> --output-dir data/derived_data/<project>
eq metadata-process --input-file data/raw_data/<project>/subject_metadata.xlsx --output-dir data/derived_data/<project>/metadata
eq audit-derived --data-dir data/derived_data/<project>
eq visualize --mask path/to/mask.png --output output/mask_preview.png
```

## Configuration

The main project configs live here:

- `configs/mito_pretraining_config.yaml`
- `configs/glomeruli_finetuning_config.yaml`

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
