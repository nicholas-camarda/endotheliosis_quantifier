# Binary Review Triage Guide

This repository's current endotheliosis-grading direction is binary review triage:

- `no_low`: original score `0` or `0.5`
- `moderate_severe`: original score `1.5`, `2`, or `3`
- `borderline_review`: original score `1.0`, excluded from the primary binary target and routed for review

The output is a review-prioritization system. It is not an autonomous pathologist, independent validation evidence, causal mechanism evidence, calibrated multi-ordinal probability model, or replacement for human grading.

## Run The Workflow

Run the maintained atlas and triage workflow from YAML:

```bash
eq run-config --config configs/label_free_roi_embedding_atlas.yaml
```

The config expects an existing quantification output root with ROI crops, embeddings, and optional atlas adjudication exports. It writes under:

```text
$EQ_RUNTIME_ROOT/output/quantification_results/<quantification_run>/burden_model/embedding_atlas/
```

`eq run-config` also writes a durable log under:

```text
$EQ_RUNTIME_ROOT/logs/run_config/label_free_roi_embedding_atlas_full_cohort_transfer_p0_adjudicated/
```

## What To Open

Open these files in this order:

1. `burden_model/embedding_atlas/INDEX.md`
2. `burden_model/embedding_atlas/evidence/embedding_atlas_review.html`
3. `burden_model/embedding_atlas/evidence/atlas_flagged_case_review.html`
4. `burden_model/embedding_atlas/binary_review_triage/evidence/binary_triage_review.html`
5. `burden_model/embedding_atlas/binary_review_triage/summary/binary_triage_verdict.md`

Every review HTML page contains static case cards, ROI images, ROI masks, dropdown controls, reviewer notes, and a CSV export button. The export belongs in the same folder as the HTML file. A static HTML page cannot silently write into its own directory because browsers block that for security; the page uses a save-file prompt when the browser supports it and otherwise downloads the CSV. If it downloads automatically, move the exported CSV next to the HTML file before rerunning the workflow.

If a review HTML page opens with no visible cases, that artifact is invalid and the workflow should be rerun after fixing the generator.

## How To Review

`embedding_atlas_review.html` asks two separate questions:

- Cluster-level interpretation: decide whether the cluster is a real morphology cluster, source/batch artifact, ROI/mask artifact, or mixed/uninterpretable.
- Case-level interpretation: decide whether a row's score and anchor eligibility make sense inside that cluster context.

`atlas_flagged_case_review.html` is the focused cleanup page. Use it for selected rows that need a score correction, anchor recovery, second review, or exclusion.

`binary_triage_review.html` is the final usability surface. Each case starts with `Model recommendation`, which is the plain-language model answer or warning. The ROI image and ROI mask are the source of truth for the human answer. Numeric model details are kept under `Model diagnostics`.

Use `Your decision` as follows:

- `accept model recommendation`: the model recommendation matches the ROI image and mask.
- `human says no/low`: the human reviewer decides the case belongs in the no/low group.
- `human says moderate/severe`: the human reviewer decides the case belongs in the moderate/severe group.
- `needs second human review`: the current reviewer cannot confidently decide from the images and wants an independent human review.
- `exclude: bad ROI or mask`: the ROI crop or mask is not gradeable.

Use `Follow-up status` only as workflow-routing metadata:

- `routine`: ordinary review item.
- `urgent: possible model error`: prioritize because the model looks wrong or the case is scientifically important.
- `defer: not useful now`: keep the record but do not spend more review time on it now.

## Primary Artifacts

Adjudication evidence:

- `evidence/atlas_score_corrections.csv`
- `evidence/atlas_recovered_anchor_examples.csv`
- `evidence/atlas_adjudicated_anchor_manifest.csv`
- `evidence/atlas_blocked_cluster_manifest.csv`
- `evidence/atlas_final_adjudication_outcome.json`
- `evidence/atlas_final_adjudication_outcome.md`

Binary triage evidence:

- `binary_review_triage/summary/binary_triage_verdict.json`
- `binary_review_triage/summary/binary_triage_verdict.md`
- `binary_review_triage/summary/binary_triage_metrics.csv`
- `binary_review_triage/summary/binary_triage_metric_intervals.json`
- `binary_review_triage/predictions/binary_triage_predictions.csv`
- `binary_review_triage/evidence/binary_triage_explanations.csv`
- `binary_review_triage/evidence/binary_triage_review.html`
- `binary_review_triage/model/model_manifest.json`

The original score columns in atlas assignments, representatives, review queues, and diagnostics remain unchanged. Adjudicated scores are separate evidence fields.

## Method

The primary binary target is:

```text
y = 0 if score <= 0.5
y = 1 if score >= 1.5
score == 1.0 -> borderline_review, excluded from primary training and primary metrics
```

The sensitivity target is:

```text
y = 0 if score <= 1.0
y = 1 if score >= 1.5
```

Candidate evidence can include ROI/QC features, learned ROI features when present, embedding PCA coordinates, selected cluster ID, Gaussian-mixture or cluster evidence, blocked-cluster indicators, nearest reviewed anchor class and distance, and recovered-anchor evidence. The pure atlas cluster mapping is reported as a baseline only; blocked clusters route to review instead of being forced into a binary class.

The fitted logistic candidates use subject-grouped development folds. For a row with feature vector `x`, the model score is:

```text
p(moderate_severe | x) = 1 / (1 + exp(-(b0 + b1*x1 + ... + bk*xk)))
```

Operating thresholds are selected from subject-grouped out-of-fold development predictions using the recorded objective in `binary_triage_metrics.csv` and `binary_triage_verdict.json`. The current objective is balanced review-triage performance, using balanced accuracy with moderate/severe recall and specificity as tie breakers.

Metric intervals in `binary_triage_metric_intervals.json` are subject-grouped bootstrap uncertainty for the current data. They are not independent validation evidence.

Feature explanations in `binary_triage_explanations.csv` report model-decision evidence: top feature contributions and feature-family contribution summaries. They explain why the fitted triage surface routed a row; they do not prove biological mechanism.

## Model Artifact Policy

Generated model files stay under the runtime root and out of Git by default. The binary triage model artifact is written to:

```text
burden_model/embedding_atlas/binary_review_triage/model/binary_triage_selected_model.joblib
```

`model/model_manifest.json` records the selected candidate, target, threshold, feature family, storage policy, and claim boundary. Git LFS is appropriate only after a separate promotion decision names a stable model artifact, schema, environment, provenance manifest, and reproducible validation bundle. Until then, runtime model artifacts remain local generated outputs.

## Reproducibility Checklist

1. Install the supported environment for the machine.
2. Set `EQ_RUNTIME_ROOT` only if the default runtime root is not the intended artifact tree.
3. Run `eq run-config --config configs/endotheliosis_quantification.yaml` when the quantification root needs to be regenerated.
4. Run `eq run-config --config configs/label_free_roi_embedding_atlas.yaml`.
5. Confirm `summary/atlas_verdict.json` reports the expected adjudication status.
6. Confirm `binary_review_triage/summary/binary_triage_verdict.json` reports `binary_no_low_vs_moderate_severe_review_triage`.
7. Review `binary_review_triage/evidence/binary_triage_review.html` and export decisions when review is complete.
8. Keep the generated CSV, JSON, HTML, Markdown, log, and model-manifest artifacts together when sharing a run.
