# Endotheliosis Quantifier

Endotheliosis Quantifier (`eq`) is a FastAI/PyTorch toolkit for glomeruli segmentation and endotheliosis quantification in kidney histology, with manifest-backed multi-cohort intake and candidate-evaluated segmentation training. Mitochondria pretraining is one supported transfer-learning path for segmentation. The repository supports active development in both WSL/Linux with CUDA and macOS Apple Silicon with MPS, with local data directories and Git-tracked code/config only.

If you want the friendlier long-form introduction and workflow explanation, see [docs/ONBOARDING_GUIDE.md](docs/ONBOARDING_GUIDE.md).
For the full curated documentation set, see [docs/README.md](docs/README.md).

Current quantification handoff: binary no/low versus moderate/severe review triage. The current grouped-development result is useful for review prioritization and QA, with honest limits around external validation and autonomous grading. Use [docs/BINARY_REVIEW_TRIAGE_GUIDE.md](docs/BINARY_REVIEW_TRIAGE_GUIDE.md) for reviewer workflow and [docs/REPRODUCIBILITY_HANDOFF_2026-04-30.md](docs/REPRODUCIBILITY_HANDOFF_2026-04-30.md) for the reproducibility checkpoint.

![Binary triage performance](assets/quantification/binary_triage_performance.svg)

![Binary triage review queue](assets/quantification/binary_triage_review_queue.svg)

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

`eq run-config` writes durable workflow logs under `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/`. Supported direct module entrypoints such as training, candidate comparison, transport audit, high-resolution concordance, and quantification workflows write durable logs under `$EQ_RUNTIME_ROOT/logs/direct/<surface>/<run_id>/`. Generic `eq` utility subcommands use explicit `eq --log-file <path>` capture when you want a file for those commands.

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
| Quantification outputs | Frozen segmentation-encoder embeddings, exploratory burden/comparator evidence, learned ROI screens, label-free embedding atlas, adjudicated anchor evidence, binary no/low versus moderate/severe review-triage artifacts, and combined review reports |

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

The same workflow functions emit logger events when called from direct module entrypoints. Direct supported module runs write to `$EQ_RUNTIME_ROOT/logs/direct/<surface>/<run_id>/`; imported functions emit events to the caller's configured logger and do not create log files by themselves.

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
eq run-config --config configs/label_free_roi_embedding_atlas.yaml
```

Use `configs/glomeruli_candidate_comparison.yaml` when you want candidate evidence for the current glomeruli segmentation baseline. Use the transport, high-resolution concordance, quantification, or label-free atlas YAMLs only when you already have the explicit upstream artifacts required by those stages.

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

Quantification is a YAML-first, two-stage workflow. The first stage builds the score-linked ROI, embedding, burden, comparator, learned-ROI, and review artifact tree from an explicit segmentation model. The second stage builds the label-free ROI embedding atlas and binary review-triage handoff from that quantification output root.

```bash
eq run-config --config configs/endotheliosis_quantification.yaml
eq run-config --config configs/label_free_roi_embedding_atlas.yaml
```

`configs/endotheliosis_quantification.yaml` names the active scored-cohort registry, reviewed label-overrides file, explicit glomeruli segmentation artifact, and output root. Missing upstream paths fail closed. To quantify with a different segmentation candidate, edit the YAML to point at that candidate's supported `.pkl` artifact and its evidence bundle.

The maintained quantification input contract uses image-level grades joined to image/mask rows in the scored cohort manifest. ROI extraction uses the full multi-component mask union with context padding, then builds frozen segmentation-backbone embeddings. The primary supervised outputs are evidence surfaces, not a deployed autonomous grader: exploratory burden-index predictions on a `0-100` ordinal stage scale, direct stage-index and ordinal/multiclass comparators, source-aware and severe-aware estimator diagnostics, learned ROI candidate screens, morphology feature reviews, and combined review reports.

The burden-index and ordinal outputs are useful for method development and error review. They are not calibrated clinical probabilities, independent validation evidence, or a replacement for human-reviewed labels. Per-image grade prediction remains limited by label quality, source/cohort structure, and calibration. Subject-level aggregation and review-triage are the current practical directions for shareable summaries.

Endotheliosis is graded by assessing the relative amount of open versus collapsed capillary or arteriole lumina within the glomerulus. The maintained quantification path writes deterministic morphology features for open/pale lumina, collapsed or slit-like structures, ridge/line signals, erythrocyte-like patent-lumen confounding, and ROI quality. These morphology features are candidate evidence, not a deployed mechanistic model; the generated feature-review HTML and operator adjudication template must be inspected before using them for a shareable claim.

The learned ROI branch lives under `burden_model/learned_roi/`. It fits only the current glomeruli encoder embeddings, simple ROI QC features, and their hybrid. Optional backbone or foundation providers are audited but not fitted. Learned ROI outputs are candidate evidence gated by uncertainty, numerical-stability, ordinal/grade-scale, and cohort-confounding checks.

The label-free ROI embedding atlas runs from the quantification output root named in `configs/label_free_roi_embedding_atlas.yaml`. It clusters approved feature spaces without using human grade, cohort, source, treatment, reviewer, adjudication, prediction, or path fields during clustering.

The atlas writes under `burden_model/embedding_atlas/`. Open these first:

- `INDEX.md`
- `summary/atlas_verdict.json`
- `evidence/embedding_atlas_review.html`
- `evidence/atlas_final_adjudication_outcome.md`
- `evidence/atlas_score_corrections.csv`
- `evidence/atlas_recovered_anchor_examples.csv`
- `evidence/atlas_adjudicated_anchor_manifest.csv`
- `evidence/atlas_blocked_cluster_manifest.csv`
- `binary_review_triage/INDEX.md`
- `binary_review_triage/evidence/binary_triage_review.html`

The current review target is binary triage: score `0` or `0.5` is `no_low`, score `1.5`, `2`, or `3` is `moderate_severe`, and score `1.0` is `borderline_review` outside the primary binary training target. Binary triage outputs include grouped-development metrics, subject-bootstrap confidence intervals where estimable, uncertainty labels, source/cohort warnings, nearest reviewed-anchor evidence, blocked-cluster indicators, and coefficient-based feature summaries for review. These artifacts support descriptive morphology clustering and review prioritization only. They are not calibrated multi-ordinal probabilities, independent validation evidence, mechanistic evidence, or replacements for human-reviewed labels. For the reviewer workflow, reproducibility checklist, binary target math, explanation interpretation, and model artifact policy, see [docs/BINARY_REVIEW_TRIAGE_GUIDE.md](docs/BINARY_REVIEW_TRIAGE_GUIDE.md).

Current quantification implementation surfaces:

- Primary burden-index estimator surface: `src/eq/quantification/burden.py`
- Learned ROI candidate surface: `src/eq/quantification/learned_roi.py`
- Label-free atlas and binary review-triage surface: `src/eq/quantification/embedding_atlas.py`
- Ordinal comparator surface: `src/eq/quantification/ordinal.py`
- Orchestration caller: `src/eq/quantification/pipeline.py` via `evaluate_embedding_table()` and the contract-first quantification entrypoints
- YAML entrypoints: `configs/endotheliosis_quantification.yaml` and `configs/label_free_roi_embedding_atlas.yaml`
- Direct CLI utilities: `eq prepare-quant-contract` and `eq quant-endo`
- Regression surfaces: `tests/unit/test_quantification_pipeline.py` and `tests/integration/test_local_runtime_quantification_pipeline.py`

The quantification stage writes:

- `labelstudio_scores/` with recovered per-image grades and duplicate-resolution audit tables
- `roi_crops/` with union-ROI crops over the full multi-component mask
- `embeddings/` with frozen segmentation-encoder embeddings
- `burden_model/` with `INDEX.md`, a contained `primary_burden_index/` subtree for exploratory burden predictions, support gates, uncertainty calibration, cohort summaries, nearest examples, candidate screens, morphology features, and review diagnostics, plus contained estimator subtrees such as `learned_roi/`, `source_aware_estimator/`, `severe_aware_ordinal_estimator/`, and grade-model diagnostic subtrees
- `ordinal_model/` with comparator predictions, probabilities, metrics, confusion matrix, and `review_report/ordinal_review.html`
- `quantification_review/` with combined HTML review, reviewer examples, concrete result summaries, and a README/docs snippet from the current run; reuse the snippet only when the reported readiness flag and uncertainty checks pass

The atlas stage adds:

- `burden_model/embedding_atlas/` with label-blinding diagnostics, method availability, feature-space manifest, cluster assignments, stability summaries, posthoc source/artifact diagnostics, representative cases, nearest neighbors, adjudication queue, score-correction evidence, recovered anchors, adjudicated anchor manifests, blocked cluster manifests, and first-read HTML/Markdown review artifacts
- `burden_model/embedding_atlas/binary_review_triage/` with binary no/low versus moderate/severe predictions, grouped-development metrics, bootstrap intervals where estimable, model manifest, explanations, support diagnostics, verdict files, and the reviewer-facing HTML handoff

## Configuration Files

The main workflow configs live here:

- `configs/mito_pretraining_config.yaml`
- `configs/glomeruli_finetuning_config.yaml`
- `configs/glomeruli_candidate_comparison.yaml`
- `configs/glomeruli_transport_audit.yaml`
- `configs/highres_glomeruli_concordance.yaml`
- `configs/endotheliosis_quantification.yaml`
- `configs/label_free_roi_embedding_atlas.yaml`

Path helpers centralize repo-local defaults, runtime roots, and external cohort sources. Prefer YAML-relative paths and the path helpers in `src/eq/utils/paths.py` over hardcoded machine-specific paths.

## Development Notes

- Use `ruff check .` and `ruff format .` before committing formatting-heavy changes.
- Use `python -m pytest -q` for the local test pass.
- Avoid hardcoded machine-specific paths in code, configs, or docs.
- Keep datasets, trained models, notebooks, logs, and temporary artifacts out of Git.
