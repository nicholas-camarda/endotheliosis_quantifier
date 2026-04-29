## Why

Current supervised endotheliosis quantification is limited by human image-level grading that appears practitioner-dependent, boundary-sensitive, and unstable across rubric review. A label-free ROI embedding atlas can test whether the ROI image data contain reproducible morphology structure independent of those labels, then use human review only to interpret clusters, anchor severity-like phenotypes, and prioritize adjudication.

This solves a different problem than the current grade model. It does not try to predict the original labels better. It asks whether frozen encoder embeddings and ROI/QC features support stable, auditable morphology groups, artifact groups, or severity-like axes before any claim is made about no-low, mid-mod, or mod-severe endotheliosis.

## What Changes

- Add a new label-free atlas workflow ID: `label_free_roi_embedding_atlas`.
- Add config file `configs/label_free_roi_embedding_atlas.yaml`.
- Add a new quantification module: `src/eq/quantification/embedding_atlas.py`.
- Extend the `eq run-config` workflow dispatcher to run `label_free_roi_embedding_atlas`.
- Reuse existing ROI, embedding, learned ROI, path, run logging, finite-matrix, and artifact-writing contracts rather than creating a second path system.
- Read the current quantification run's ROI/embedding artifacts, starting with:
  - `embeddings/roi_embeddings.csv`
  - `roi_crops/roi_scored_examples.csv`
  - `burden_model/learned_roi/feature_sets/learned_roi_features.csv` when present
- Write label-free atlas artifacts under:
  - `burden_model/embedding_atlas/summary/`
  - `burden_model/embedding_atlas/feature_space/`
  - `burden_model/embedding_atlas/clusters/`
  - `burden_model/embedding_atlas/stability/`
  - `burden_model/embedding_atlas/diagnostics/`
  - `burden_model/embedding_atlas/evidence/`
  - `burden_model/embedding_atlas/review_queue/`
- Generate clustering in a label-blinded phase using embedding/ROI features only; human grade, cohort, source, treatment, and reviewer metadata are withheld from clustering and exposed only in post hoc interpretation diagnostics.
- Evaluate at least these label-free structure families:
  - standardized frozen encoder embeddings
  - PCA-denoised frozen encoder embeddings
  - standardized learned ROI features when available
  - ROI/QC-only comparator features
- Use cluster methods with explicit stability checks, including a deterministic baseline such as `kmeans`, a density option such as `hdbscan` when locally available, and a model-based option such as Gaussian mixture when support permits.
- Use PCA and UMAP as visualization or dimensionality-control aids, but do not use t-SNE coordinates as the primary clustering or probability surface.
- Generate reviewer-facing cluster evidence:
  - representative ROI montages
  - medoids, nearest neighbors, and boundary examples
  - cluster-level ROI/QC summaries
  - cluster-level cohort/source/treatment distribution
  - cluster-level original score distribution revealed only after clustering
  - artifact/quality flags such as RBC-heavy, low-quality, multi-component, mask-adequacy, or source-sensitive behavior when available
- Generate a prioritized adjudication queue for cases where cluster-neighbor structure disagrees with original human score, severe/nonsevere status, or existing rubric anchors.
- Add gates that prevent artifact, cohort, source, or mask geometry clusters from being promoted as severity-like phenotypes.
- Add tests covering label blinding, subject-aware stability resampling, cluster artifact schemas, review queue generation, and failure behavior when required embeddings or identity columns are missing.
- Add current-state documentation for running and interpreting the atlas workflow without claiming external validity or automated disease severity.

## Capabilities

### New Capabilities

- `label-free-roi-embedding-atlas`: Defines the label-blinded ROI embedding atlas workflow, feature-space construction, clustering, stability checks, artifact/source diagnostics, reviewer evidence, adjudication queue, and claim boundaries for interpreting morphology clusters independent of original human labels.

### Modified Capabilities

- None.

## Impact

- Affected modules:
  - `src/eq/run_config.py`
  - `src/eq/quantification/embedding_atlas.py`
  - `src/eq/quantification/embeddings.py`
  - `src/eq/quantification/learned_roi.py`
  - `src/eq/quantification/modeling_contracts.py`
  - `src/eq/utils/paths.py`
  - `src/eq/utils/execution_logging.py`
- Affected configs:
  - `configs/label_free_roi_embedding_atlas.yaml`
  - `configs/endotheliosis_quantification.yaml` only if it records the latest atlas input/output roots for convenience
- Affected tests:
  - new `tests/unit/test_quantification_embedding_atlas.py`
  - integration coverage for `eq run-config --config configs/label_free_roi_embedding_atlas.yaml`
  - existing quantification tests should remain green without requiring atlas outputs
- Affected runtime artifact root:
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/<run_id>/burden_model/embedding_atlas/`
- Storage boundaries:
  - No raw data, derived data, models, logs, or generated outputs are written under the Git checkout.
  - Atlas outputs remain generated runtime artifacts under the configured quantification output root.
- Data contract changes:
  - No existing supervised grade-model schema is replaced.
  - Atlas inputs require row identity columns sufficient to preserve `subject_id`, `subject_image_id`, `cohort_id`, `score` when present, ROI path provenance, hardened ROI geometry/preprocessing/threshold provenance, and feature-column provenance.
  - Atlas feature columns require approved lineage proving they derive from frozen encoder embeddings, ROI/QC measurements, or learned ROI features rather than labels, source metadata, review queues, post hoc diagnostics, supervised predictions, or targets.
  - Human label columns may be present in input tables but must be excluded from clustering feature matrices and used only in post hoc diagnostics.
- Scientific claim boundary:
  - Descriptive: the atlas may describe reproducible ROI morphology clusters, cluster stability, and cluster-level feature distributions.
  - Associational: the atlas may report post hoc associations between clusters and human grades, cohorts, treatments, or review anchors.
  - Predictive/prognostic: the atlas must not claim calibrated severity probabilities, external validity, or outcome prediction.
  - Causal: the atlas must not claim that clusters cause, prove, or mechanistically establish endotheliosis.
- Compatibility risks:
  - Missing embedding tables, missing ROI path provenance, duplicate row identity, or nonfinite feature columns must fail closed with explicit diagnostics.
  - Stale ROI or embedding artifacts that do not carry the completed fail-closed ROI geometry, preprocessing, threshold, and artifact provenance contract must fail before clustering.
  - Existing supervised burden, learned ROI, source-aware, severe-aware, and P3 grade-model artifacts must continue to run without requiring atlas artifacts.
  - Optional packages such as `umap-learn` or `hdbscan` may be absent; their absence should be recorded as method unavailability rather than silently replacing the approved method under the same method ID.

## logging-contract

This change uses the existing `eq run-config` durable logging surface. It does not add a second logging root. The atlas run must record the config path, workflow ID, runtime root, quantification output root, method availability, and generated artifact manifest through the existing workflow-owned command capture.

## docs-impact

Docs must add current-state guidance for `eq run-config --config configs/label_free_roi_embedding_atlas.yaml`, first-read atlas artifacts, and the claim boundary. Public docs must describe the atlas as descriptive morphology clustering and review prioritization, not as calibrated severity probabilities or automatic replacement of human grading.

## Explicit Decisions

- Change name: `label-free-roi-embedding-atlas`.
- Workflow ID: `label_free_roi_embedding_atlas`.
- Config path: `configs/label_free_roi_embedding_atlas.yaml`.
- Primary implementation owner: `src/eq/quantification/embedding_atlas.py`.
- Run-config dispatcher owner: `src/eq/run_config.py`.
- Entry point: `eq run-config --config configs/label_free_roi_embedding_atlas.yaml` only. This change will not add a direct `eq` CLI alias.
- Runtime output subtree: `burden_model/embedding_atlas/`.
- First-read artifacts:
  - `burden_model/embedding_atlas/INDEX.md`
  - `burden_model/embedding_atlas/summary/atlas_verdict.json`
  - `burden_model/embedding_atlas/summary/atlas_summary.md`
  - `burden_model/embedding_atlas/summary/artifact_manifest.json`
- Cluster-feature matrices must exclude human grade, cohort/source/treatment, reviewer, and outcome columns during clustering.
- Cluster-feature matrices must reject unapproved derived-feature lineage, not only suspicious column names.
- Human grade and source metadata may be joined only after cluster assignments are finalized for interpretation and artifact/confounding diagnostics.
- `candidate_severity_like_group` requires explicit predeclared thresholds for stability, subject support, source imbalance, ROI/QC artifact dominance, grade-association strength, and missing-asset tolerance.
- t-SNE is not an approved primary clustering method or probability surface for this change.
- The atlas does not replace `endotheliosis_grade_model`, `learned_roi`, `source_aware_estimator`, or `primary_burden_index` artifacts.

## Open Questions

- [audit_first_then_decide] Whether `umap-learn` and `hdbscan` are available in `eq-mac` and WSL/CUDA environments. Decide from `importlib.util.find_spec()` provider audit in the new workflow and record unavailable methods in `diagnostics/method_availability.json`.
- [audit_first_then_decide] Whether montage generation should reuse an existing image-review helper or require a small atlas-specific renderer. Decide by inspecting `src/eq/quantification/feature_review.py`, `src/eq/quantification/learned_roi_review.py`, and existing quantification review HTML helpers before implementation.
- [defer_ok] Whether future externally reviewed pathology anchors should be imported as a separate anchor table. This change should emit an anchor-ready schema but should not require new external labels before implementation.
