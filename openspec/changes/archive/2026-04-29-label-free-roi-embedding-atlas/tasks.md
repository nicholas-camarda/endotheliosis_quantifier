## 1. Reuse and Interface Audit

- [x] 1.1 Inspect `src/eq/run_config.py`, `src/eq/__main__.py`, `src/eq/utils/paths.py`, and `src/eq/utils/execution_logging.py` and record the exact dispatcher, path, and durable-log hooks to reuse.
- [x] 1.2 Inspect `src/eq/quantification/modeling_contracts.py`, `src/eq/quantification/embeddings.py`, `src/eq/quantification/learned_roi.py`, `src/eq/quantification/feature_review.py`, and `src/eq/quantification/learned_roi_review.py` for reusable finite-matrix, JSON, feature provenance, nearest-neighbor, and report-rendering helpers.
- [x] 1.3 Inspect current runtime artifacts under the active quantification output root and confirm the input file paths, identity columns, embedding column prefixes, ROI/QC columns, and learned ROI feature columns that the atlas will support.
- [x] 1.4 Decide whether existing review-rendering helpers can own atlas HTML output or whether `embedding_atlas.py` needs a small atlas-specific renderer; record the decision in implementation notes or code comments.

## 2. Workflow and Configuration Surface

- [x] 2.1 Add `configs/label_free_roi_embedding_atlas.yaml` with workflow ID `label_free_roi_embedding_atlas`, active quantification output root inputs, method settings, feature-space settings, review limits, and output root settings.
- [x] 2.2 Extend `src/eq/run_config.py` to dispatch `label_free_roi_embedding_atlas` to the atlas runner without changing existing workflow IDs.
- [x] 2.3 Keep `eq run-config --config configs/label_free_roi_embedding_atlas.yaml` as the sole supported atlas entrypoint for this change; do not add a direct CLI alias.
- [x] 2.4 Ensure run logging captures the atlas command, config path, runtime root, output root, and terminal output through the existing durable logging contract.

## 3. Atlas Module and Artifact Layout

- [x] 3.1 Create `src/eq/quantification/embedding_atlas.py` with a single public runner, canonical output-path helper, and first-read artifact writer.
- [x] 3.2 Implement `burden_model/embedding_atlas/` grouped output directories: `summary/`, `feature_space/`, `clusters/`, `stability/`, `diagnostics/`, `evidence/`, and `review_queue/`.
- [x] 3.3 Write `INDEX.md`, `summary/atlas_verdict.json`, `summary/atlas_summary.md`, and `summary/artifact_manifest.json` for both successful and fail-closed runs where the output root is available.
- [x] 3.4 Keep all atlas artifacts under the configured runtime quantification output root and add tests that no repo-root generated artifact directories are created.

## 4. Input Loading and Label-Blinding Contract

- [x] 4.1 Implement input loading for `embeddings/roi_embeddings.csv`, `roi_crops/roi_scored_examples.csv`, and optional `burden_model/learned_roi/feature_sets/learned_roi_features.csv`.
- [x] 4.2 Validate required identity/provenance columns, including `subject_id`, row identity, ROI image path, and feature provenance; fail closed with explicit `atlas_verdict.json` blockers when required columns are missing.
- [x] 4.3 Implement feature allow-lists and denied metadata/label column checks so `score`, grade bands, severe indicators, cohort/source/treatment fields, reviewer fields, label overrides, path strings, and targets cannot enter clustering matrices.
- [x] 4.4 Write `diagnostics/label_blinding_audit.json` listing approved feature columns, denied columns, metadata-only columns, excluded label-like columns, and leakage status.
- [x] 4.5 Require input artifacts to carry hardened ROI geometry, embedding preprocessing, threshold, ROI status, and artifact-provenance fields from the completed Oracle hardening change; stale or provenance-incomplete inputs must fail before clustering with blockers in `atlas_verdict.json`.
- [x] 4.6 Require approved feature lineage for every numeric clustering feature. Reject feature columns without provenance proving they are derived from frozen encoder embeddings, ROI/QC measurements, or learned ROI features rather than labels, source metadata, review queues, post hoc diagnostics, supervised predictions, or target columns.
- [x] 4.7 Add regression tests where leaked label/source columns in the feature allow-list fail before clustering.
- [x] 4.8 Add regression tests where stale ROI/embedding artifacts or feature columns without approved lineage fail before clustering.

## 5. Feature Spaces and Method Availability

- [x] 5.1 Implement finite numeric feature-matrix construction using shared finite-matrix utilities where possible.
- [x] 5.2 Build named feature spaces: `encoder_standardized`, `encoder_pca`, `roi_qc_standardized`, and `learned_roi_standardized` when valid learned ROI columns exist.
- [x] 5.3 Write `feature_space/feature_space_manifest.json` and per-feature-space diagnostics covering row count, subject count, feature count, missingness, nonfinite counts, zero-variance counts, near-zero-variance counts, scaling policy, PCA policy, and package versions.
- [x] 5.4 Implement `diagnostics/method_availability.json` for sklearn PCA, k-means, Gaussian mixture, nearest neighbors, optional `hdbscan`, optional `umap-learn`, and optional visualization-only t-SNE.
- [x] 5.5 Ensure unavailable optional methods are skipped under truthful method IDs and are not silently replaced.

## 6. Clustering and Low-Dimensional Geometry

- [x] 6.1 Implement k-means clustering over configured cluster-count grids using label-free structure and stability criteria only.
- [x] 6.2 Implement Gaussian mixture clustering where support permits and record assignment probabilities.
- [x] 6.3 Implement optional HDBSCAN clustering only when the audited package is available.
- [x] 6.4 Generate PCA coordinates for feature-space inspection and optional UMAP coordinates for visualization when available.
- [x] 6.5 Write `clusters/cluster_assignments.csv` with row identity, `subject_id`, feature-space ID, method ID, cluster ID, confidence or distance where available, outlier/noise status, and no inferred severity label.
- [x] 6.6 Write cluster parameter-grid and selected-parameter artifacts so cluster count or density choices are auditable and not label-selected.

## 7. Stability and Post Hoc Diagnostics

- [x] 7.1 Implement subject-aware cluster stability resampling with `subject_id` as the primary biological grouping unit when present.
- [x] 7.2 Write `stability/cluster_stability.json` with resampling unit, number of resamples, row count, subject count, feature-space ID, method ID, stability metrics, and non-estimable reasons.
- [x] 7.3 Implement post hoc cluster diagnostics that reveal score, severe/nonsevere, cohort/source, treatment/source proxy, ROI/QC, mask adequacy, and reviewed-anchor distributions only after assignments are finalized.
- [x] 7.4 Define explicit interpretation thresholds for cluster support, subject count, stability, source imbalance, ROI/QC artifact dominance, grade-association strength, and missing-asset tolerance before assigning `candidate_severity_like_group`.
- [x] 7.5 Write `diagnostics/cluster_posthoc_diagnostics.json` and cluster interpretation labels: `candidate_morphology_group`, `candidate_severity_like_group`, `artifact_or_quality_group`, `source_sensitive_group`, `unstable_group`, and `insufficient_support`.
- [x] 7.6 Add tests that unstable, source-sensitive, artifact-dominated, under-supported, or threshold-incomplete clusters cannot be marked severity-like.

## 8. Review Evidence and Adjudication Queues

- [x] 8.1 Implement representative, medoid, nearest-neighbor, farthest, and boundary-example selection while excluding same-`subject_id` nearest-neighbor evidence where required.
- [x] 8.2 Write representative and nearest-neighbor evidence tables under `evidence/` with ROI image path, mask path when available, row identity, cluster ID, method ID, feature-space ID, distance, and medoid/boundary status.
- [x] 8.3 Generate `evidence/embedding_atlas_review.html` with cluster representative montages, cluster summaries, original-grade distributions, source/artifact warnings, and missing-asset counts.
- [x] 8.4 Generate `review_queue/atlas_adjudication_queue.csv` with priority, reason code, row identity, cluster evidence, nearest-neighbor evidence, original score when present, reviewed-anchor evidence when present, and ROI path provenance.
- [x] 8.5 Ensure the atlas never writes score label override files and document that future overrides require explicit human-reviewed labels.

## 9. Tests

- [x] 9.1 Add `tests/unit/test_quantification_embedding_atlas.py` covering label blinding, input validation, feature-space diagnostics, method availability, cluster assignment schema, stability schema, post hoc interpretation gates, and review queue generation.
- [x] 9.2 Add a small integration test for `eq run-config --config configs/label_free_roi_embedding_atlas.yaml` using a temporary fixture output root.
- [x] 9.3 Add regression coverage proving existing supervised quantification tests do not require `burden_model/embedding_atlas/` artifacts.
- [x] 9.4 Add fixture tests for missing ROI assets and missing identity columns that verify fail-closed verdict artifacts.

## 10. Documentation

- [x] 10.1 Add current-state documentation for running `eq run-config --config configs/label_free_roi_embedding_atlas.yaml`.
- [x] 10.2 Document the claim boundary: descriptive morphology clustering and review prioritization only, not calibrated severity probabilities, causal mechanism, external validity, or automatic human-label replacement.
- [x] 10.3 Document the first-read atlas artifacts and how a reviewer should inspect `INDEX.md`, `summary/atlas_verdict.json`, `evidence/embedding_atlas_review.html`, and `review_queue/atlas_adjudication_queue.csv`.
- [x] 10.4 Keep public docs current-state only and avoid migration framing or historical comparisons.

## 11. Validation

- [x] 11.1 Run `openspec validate label-free-roi-embedding-atlas --strict`.
- [x] 11.2 Run `openspec validate --specs --strict`.
- [x] 11.3 Run the repository explicitness checker if available for OpenSpec artifacts.
- [x] 11.4 Run `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_quantification_embedding_atlas.py -q`.
- [x] 11.5 Run focused integration tests for `eq run-config --config configs/label_free_roi_embedding_atlas.yaml`.
- [x] 11.6 Run relevant existing quantification tests to verify supervised workflows remain independent of atlas artifacts.
- [x] 11.7 Run `ruff check .` and `ruff format .` after implementation edits.

## 12. Postflight And Archive Lifecycle

- [x] 12.1 Complete the per-change postflight required by `openspec/changes/ACTIVE_EXECUTION_ORDER.md`, including spec-to-diff review, completed-task evidence review, `git diff --check`, `git diff --stat`, and unrelated-edit inspection.
- [x] 12.2 Commit the implementation as `implement label-free-roi-embedding-atlas`.
- [x] 12.3 Archive/sync with `openspec archive label-free-roi-embedding-atlas --yes`.
- [x] 12.4 Run `openspec validate --specs --strict` after archive/sync.
- [x] 12.5 Revalidate every remaining active change with `openspec validate <remaining-change> --strict` and `python3 scripts/check_openspec_explicitness.py <remaining-change>`.
- [x] 12.6 Commit the archive/sync as `archive label-free-roi-embedding-atlas`.
