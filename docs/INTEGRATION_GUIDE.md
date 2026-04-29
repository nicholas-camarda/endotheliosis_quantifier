# Integration Guide

This page is current operational guidance. Historical FastAI integration notes were moved to [archive/fastai_legacy_integration.md](archive/fastai_legacy_integration.md) and are reference-only.

Use the maintained YAML workflow entrypoints from the repository root:

```bash
conda activate eq-mac
eq run-config --config configs/glomeruli_finetuning_config.yaml --dry-run
eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run
eq run-config --config configs/endotheliosis_quantification.yaml --dry-run
```

Supported model artifacts must load from the current `src/eq` namespace in the certified environment and must carry provenance for the training command, code version, package versions, data root, and training mode. Unsupported legacy artifacts are historical references, not current integration targets.
