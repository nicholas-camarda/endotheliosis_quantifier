# Reproducibility Handoff 2026-04-30

This handoff records the current reproducible source state and the runtime evidence needed to resume the endotheliosis quantifier work without guessing which output is current.

## Current Direction

The current endotheliosis-grading direction is binary review triage:

- `no_low`: score `0` or `0.5`
- `moderate_severe`: score `1.5`, `2`, or `3`
- `borderline_review`: score `1.0`, routed for review and excluded from the primary binary target

This is a review-prioritization workflow. It is not an externally validated diagnostic model, autonomous grading system, causal morphology explanation, or calibrated multi-ordinal classifier.

## Source Checkpoint

Canonical repository:

```text
https://github.com/nicholas-camarda/endotheliosis_quantifier.git
```

Handoff tag:

```text
handoff-2026-04-30-binary-triage
```

The tag identifies the exact source code, configs, tests, docs, and OpenSpec archive state used for the handoff. Runtime outputs are generated artifacts and stay outside normal Git history.

## Canonical Commands

Install the supported environment for the machine, then run from the repository root.

macOS:

```bash
/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml
/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/label_free_roi_embedding_atlas.yaml
```

Generic installed environment:

```bash
eq run-config --config configs/endotheliosis_quantification.yaml
eq run-config --config configs/label_free_roi_embedding_atlas.yaml
```

The atlas config expects the configured quantification output root to contain ROI crops, masks, embeddings, and adjudication exports when those exports are part of the handoff result.

## Runtime Evidence Root

Current atlas and binary-triage evidence root:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/embedding_atlas
```

Current run logs:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/logs/run_config/label_free_roi_embedding_atlas_full_cohort_transfer_p0_adjudicated/
```

These runtime paths are not portable public paths. They identify the local evidence bundle used for this handoff.

## Primary Runtime Artifacts

Open these in order:

1. `INDEX.md`
2. `summary/atlas_verdict.json`
3. `evidence/embedding_atlas_review.html`
4. `evidence/atlas_flagged_case_review.html`
5. `binary_review_triage/INDEX.md`
6. `binary_review_triage/summary/binary_triage_verdict.md`
7. `binary_review_triage/evidence/binary_triage_review.html`

The binary review HTML is a bounded, route-stratified QA sample. It is not a request to review every row in `predictions/binary_triage_predictions.csv`.

## Current Binary Triage Result

Current selected candidate:

```text
roi_qc_binary_logistic
```

Current selected feature family:

```text
roi_qc
```

Current threshold:

```text
0.45
```

Current threshold objective:

```text
maximize_grouped_oof_balanced_accuracy_then_moderate_severe_recall
```

Primary target support:

```text
no_low: 371
moderate_severe: 220
borderline_review: 116
```

Current selected primary metrics:

```text
balanced_accuracy: 0.656855
recall: 0.704545
precision: 0.516667
specificity: 0.609164
AUROC: 0.694658
average_precision: 0.537940
false_negative_count: 65
false_positive_count: 145
```

These are grouped-development, current-data metrics. They do not establish external validity.

## What Belongs In Git

Commit and keep synchronized:

- source code under `src/eq/`
- tests under `tests/`
- runnable YAML configs under `configs/`
- environment files and packaging metadata
- OpenSpec current specs and archived changes
- reviewer and reproducibility docs under `docs/`

## What Stays Out Of Git

Do not commit these directly:

- raw source data
- derived cohort data
- runtime output trees
- generated review HTML and CSV exports
- run logs
- generated model files
- local machine paths

Generated model artifacts stay under the runtime root until a separate promotion decision names the model, schema, provenance manifest, validation bundle, and claim boundary. Git LFS is appropriate only after that promotion gate.

## Release Asset Policy

A GitHub Release may attach a sanitized aggregate evidence bundle for the handoff tag. It should include only public-safe summaries such as:

- binary triage verdict
- binary triage metrics
- metric intervals
- storage-policy model manifest with local paths redacted
- checksum manifest
- this handoff note

Do not attach full prediction tables, review HTML, ROI paths, image identifiers, subject identifiers, raw masks, raw images, or local absolute paths to a public release.

## Validation Checklist

Run before treating the handoff as reproducible:

```bash
ruff check .
python -m pytest -q
openspec validate --specs --strict
```

Then rerun:

```bash
eq run-config --config configs/label_free_roi_embedding_atlas.yaml
```

Confirm:

- `summary/atlas_verdict.json` reports the expected adjudication status.
- `binary_review_triage/summary/binary_triage_verdict.json` reports `binary_no_low_vs_moderate_severe_review_triage`.
- `binary_review_triage/evidence/binary_triage_review.html` shows the bounded QA sample and exports a review CSV.
- Runtime logs are present under the run-config log directory.

## Next Work After This Handoff

The next methodological work is source/batch-effect handling. The current atlas shows source-sensitive clusters, so the binary triage result should remain claim-bounded until a dedicated batch-effect correction and post-correction review spec is implemented.
