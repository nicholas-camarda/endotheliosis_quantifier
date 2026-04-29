# Pipeline Integration Plan

This page is current operational guidance. Historical planning for an older FastAI glomeruli integration path was moved to [archive/fastai_pipeline_integration_plan.md](archive/fastai_pipeline_integration_plan.md) and is reference-only.

The maintained pipeline entrypoint is `eq run-config` with committed YAML configs. Current segmentation and quantification work should start from:

```bash
conda activate eq-mac
eq run-config --config configs/glomeruli_finetuning_config.yaml --dry-run
eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run
eq run-config --config configs/endotheliosis_quantification.yaml --dry-run
```

Current workflows fail closed when required model artifacts, provenance, masks, scores, or runtime paths are missing. Do not add compatibility branches or alternate loader paths to rescue unsupported historical artifacts.
