# Preflight Path And Artifact Map

## Scope

- Repo path: `/Users/ncamarda/Projects/endotheliosis_quantifier`
- Runtime root: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier`
- Cloud publish home: `/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/SideProjects/endotheliosis_quantifier`
- Cloud publish root name: `Analysis`
- Canonical Mac interpreter: `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python`

## Active OpenSpec Changes

Direct evidence: `env OPENSPEC_TELEMETRY=0 openspec list --json`.

- `p0-harden-glomeruli-segmentation-validation`: complete, 41/41.
- `p1-add-negative-glomeruli-crop-supervision`: complete, 9/9.
- `p2-split-segmentation-and-quantification-workflows`: complete, 23/23.
- `p3-repo-wide-quality-review-and-streamlining`: in progress at preflight.

## Committed Configs

Direct evidence: `find configs -maxdepth 1 -type f`.

- `configs/mito_pretraining_config.yaml`
- `configs/glomeruli_finetuning_config.yaml`
- `configs/glomeruli_candidate_comparison.yaml`
- `configs/glomeruli_transport_audit.yaml`
- `configs/highres_glomeruli_concordance.yaml`
- `configs/endotheliosis_quantification.yaml`

## Key Docs

Direct evidence: `find docs -maxdepth 1 -type f`.

- `README.md`
- `docs/README.md`
- `docs/ONBOARDING_GUIDE.md`
- `docs/OUTPUT_STRUCTURE.md`
- `docs/SEGMENTATION_ENGINEERING_GUIDE.md`
- `docs/TECHNICAL_LAB_NOTEBOOK.md`
- `docs/HISTORICAL_IMPLEMENTATION_ANALYSIS.md`
- `docs/HISTORICAL_NOTES.md`
- `docs/INTEGRATION_GUIDE.md`
- `docs/PIPELINE_INTEGRATION_PLAN.md`

## CLI Entrypoints

Direct evidence: `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`.

Top-level commands:

- `orchestrator`
- `extract-images`
- `organize-lucchi`
- `validate-naming`
- `quant-endo`
- `prepare-quant-contract`
- `cohort-manifest`
- `run-config`
- `dox-mask-quality-audit`
- `backup-project-data`
- `metadata-process`
- `capabilities`
- `mode`
- `visualize`

`eq run-config --help` exposes only `--config` and `--dry-run`.

## Direct Module Entrypoints

Direct evidence: source files and documentation-wizard interface extraction.

- `python -m eq.training.train_mitochondria`
- `python -m eq.training.train_glomeruli`
- `python -m eq.training.compare_glomeruli_candidates`
- `python -m eq.training.run_glomeruli_candidate_comparison_workflow`
- `python -m eq.evaluation.run_glomeruli_transport_audit_workflow`
- `python -m eq.evaluation.run_highres_glomeruli_concordance_workflow`
- `python -m eq.quantification.run_endotheliosis_quantification_workflow`

Inference: direct module entrypoints remain implementation/targeted-run surfaces; README and onboarding should keep `eq run-config` as the normal workflow front door.

## Runtime Artifact Roots

Direct evidence: path helpers, `analysis_registry.yaml`, and runtime-root directory listing.

- `raw_data/cohorts/`
- `raw_data/mitochondria_data/`
- `derived_data/cohort_manifest/`
- `models/segmentation/`
- `output/segmentation_evaluation/`
- `output/predictions/`
- `output/quantification_results/`
- `logs/run_config/`
- `_retired/`

Runtime still contains `output/segmentation_results/`; this is not a current documented output root and should remain a legacy/runtime cleanup candidate rather than a repo source path.

## Repo Clutter And Generated Files

Direct evidence: `find . -maxdepth 2` and `git ls-files`.

- Repo-local generated/cache directories found: `.pytest_cache`, `tests/__pycache__`, `.history/output`.
- No tracked `__pycache__`, `.pyc`, `.DS_Store`, `raw_data`, `derived_data`, `models`, `logs`, or `output` files were found.
- Current p3 review generated one large raw workspace-governor JSON dump; it was summarized into `workspace-governor-report.md` and removed from the review dossier.

## Baseline Command Evidence

- `eq --help`: passed.
- `eq run-config --help`: passed.
- `openspec list --json`: passed.
- Split workflow dry-runs for p2 configs passed before p3 began.

## Quick Test Plan

The p3 migration-success smoke command is a shortened real candidate-comparison run:

```bash
env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src \
  /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config \
  --config openspec/changes/p3-repo-wide-quality-review-and-streamlining/review/quicktest_glomeruli_candidate_comparison_5epoch.yaml
```

The quick config preserves the full workflow shape while capping mitochondria, transfer, and no-base candidate training at 5 epochs.
