## 1. Reuse Audit And Input Contract

- [x] 1.1 Inspect `src/eq/quantification/embedding_atlas.py`, `src/eq/quantification/feature_review.py`, and `src/eq/quantification/severe_aware_ordinal_estimator.py` for reusable adjudication loading, validation, HTML, and summary patterns; record whether the implementation stays atlas-local or extracts a shared helper.
- [x] 1.2 Add optional adjudication input settings to `configs/label_free_roi_embedding_atlas.yaml` without changing the workflow ID or adding a new CLI command.
- [x] 1.3 Define atlas adjudication input schemas for cluster review exports and flagged-case decisions, including required columns, allowed decision values, row identity fields, original-score fields, reviewed timestamp, reviewer notes, and ROI path provenance.
- [x] 1.4 Implement fail-closed validation for present adjudication inputs: missing required columns, duplicate conflicting rows, unmatched `atlas_row_id`, mismatched `subject_image_id`, mismatched `cluster_id`, mismatched `original_score`, and mismatched ROI paths.
- [x] 1.5 Implement no-review defaults that produce explicit no-review status artifacts when optional adjudication inputs are absent.

## 2. Atlas Evidence Processing

- [x] 2.1 Extend `src/eq/quantification/embedding_atlas.py` to load validated atlas adjudication exports after cluster assignments, review queues, and representative evidence are available.
- [x] 2.2 Generate `burden_model/embedding_atlas/evidence/atlas_score_corrections.csv` with original scores and adjudicated scores in separate fields.
- [x] 2.3 Generate `burden_model/embedding_atlas/evidence/atlas_recovered_anchor_examples.csv` for row-level recovered anchors without promoting their whole cluster.
- [x] 2.4 Generate `burden_model/embedding_atlas/evidence/atlas_adjudicated_anchor_manifest.csv` for reviewed cluster-level anchor candidates.
- [x] 2.5 Generate `burden_model/embedding_atlas/evidence/atlas_blocked_cluster_manifest.csv` for mixed, source/batch, ROI/mask, or otherwise blocked clusters.
- [x] 2.6 Generate `burden_model/embedding_atlas/evidence/atlas_final_adjudication_outcome.json` and `.md` summarizing score changes, scores kept, recovered anchors, blocked clusters, candidate anchor clusters, source evidence paths, and next implementation action.
- [x] 2.7 Ensure original score fields in atlas metadata, cluster assignments, representatives, review queues, and post hoc diagnostics are never overwritten by adjudicated score evidence.

## 3. Binary Review-Triage Model Contract

- [x] 3.1 Audit current P3, severe-aware, three-band, four-band, embedding, and atlas runtime metrics to document why binary review triage is the primary near-term product and multi-ordinal deployment is not the default claim.
- [x] 3.2 Decide whether binary triage lives inside `burden_model/endotheliosis_grade_model/` or a first-class `burden_model/binary_review_triage_model/` subtree; record the ownership decision in implementation notes and artifact manifests.
- [x] 3.3 Implement the primary binary target: `no_low = score <= 0.5`, `moderate_severe = score >= 1.5`, and `borderline_review = score == 1.0` excluded from primary binary training and primary metrics.
- [x] 3.4 Implement the separate sensitivity target: `no_low_inclusive = score <= 1.0` versus `moderate_severe = score >= 1.5`, reported separately from the primary target.
- [x] 3.5 Generate atlas-derived candidate features for binary triage: selected cluster ID, GMM posterior or distance fields where available, reduced embedding PCA coordinates, nearest reviewed anchor distance/class, blocked-cluster indicator, and recovered-anchor proximity where computable.
- [x] 3.6 Evaluate pure atlas cluster mapping as a simple baseline and prevent blocked clusters from being forced into a binary class.
- [x] 3.7 Evaluate leakage-safe supervised binary candidates using ROI/QC, morphology, learned ROI, embedding PCA, GMM/cluster, anchor-distance, and hybrid feature families with subject-heldout grouped development validation.
- [x] 3.8 Select operating thresholds inside grouped development data only and record whether the operating objective prioritizes moderate/severe sensitivity, review-workload precision/specificity, or a balanced review-triage point.

## 4. Binary Triage Uncertainty And Explanation Outputs

- [x] 4.1 Write binary triage metrics with recall, precision, specificity, balanced accuracy, AUROC, average precision, false-negative count, false-positive count, threshold, row count, subject count, source support, and finite-output status.
- [x] 4.2 Add grouped-resampling or bootstrap confidence intervals for key binary metrics when support permits; explicitly mark intervals that are non-estimable.
- [x] 4.3 Write prediction outputs with predicted probability or score, threshold decision, uncertainty/reliability label, near-threshold flag, source/cohort warning flag, nearest-anchor evidence, and final review route.
- [x] 4.4 Write row-level explanation outputs that summarize model feature contributions or permutation-style evidence and feature-family contributions for ROI/QC, morphology, learned ROI, embedding PCA, GMM/cluster, and anchor-distance evidence where present.
- [x] 4.5 Generate a reviewer-facing binary triage review HTML or queue that shows ROI image, ROI mask, original score, adjudication evidence, predicted route, confidence label, top explanation fields, nearest reviewed anchor, and cluster/source warnings.
- [x] 4.6 Ensure explanation copy states that explanations are model-decision evidence for review prioritization, not causal or mechanistic proof.

## 5. Review HTML Standard

- [x] 5.1 Update `evidence/embedding_atlas_review.html` generation so case-level and cluster-level decisions remain separate and exported with distinct fields.
- [x] 5.2 Add or harden focused flagged-case review HTML generation so selected cases are present as static case cards before JavaScript runs.
- [x] 5.3 Ensure every focused review case shows ROI image and ROI mask beside task-specific score or anchor decision controls.
- [x] 5.4 Ensure review exports include row identity, original decision context, reviewed decision fields, reviewer notes, reviewed timestamp, and ROI path provenance.
- [x] 5.5 Add postflight checks that fail if generated review HTML has zero case cards, missing images, missing dropdowns, missing export controls, or JS-only case rendering.

## 6. Verdicts, Manifests, And Documentation

- [x] 6.1 Update `summary/atlas_verdict.json` to report adjudication status, score-change counts, kept-score counts, recovered-anchor counts, candidate-anchor clusters, blocked clusters, and next action when review evidence is present.
- [x] 6.2 Update `summary/atlas_summary.md` and `INDEX.md` to point reviewers to adjudication outcome, score-correction, recovered-anchor, anchor-manifest, and blocked-cluster artifacts.
- [x] 6.3 Update `summary/artifact_manifest.json` so adjudication artifacts are listed when present and no-review status is explicit when absent.
- [x] 6.4 Update P3/final-product verdicts so binary review triage is reported as review prioritization with explicit current-data gates, not as external validation or autonomous grading.
- [x] 6.5 Keep public/user-facing wording descriptive: reviewed morphology anchors, score-review evidence, binary triage review prioritization, uncertainty, and explanations only; not external validation, causal mechanism, calibrated multi-ordinal probabilities, or automatic replacement of human review.

## 7. Tests

- [x] 7.1 Add unit tests for valid adjudication ingestion using fixture exports modeled on `atlas_adjudication_review_export.csv` and `atlas_flagged_case_decisions.csv`.
- [x] 7.2 Add fail-closed tests for missing columns, duplicate conflicting decisions, unmatched atlas rows, mismatched clusters, mismatched original scores, and mismatched ROI paths.
- [x] 7.3 Add tests proving original scores remain unchanged and adjudicated scores are written only to separate evidence fields.
- [x] 7.4 Add tests for anchor manifests: candidate cluster anchors, blocked clusters, and recovered row-level anchors remain distinct.
- [x] 7.5 Add review HTML tests for static case cards, image tags, dropdown counts, export controls, and nonblank focused review output.
- [x] 7.6 Add binary target-construction tests proving score `1.0` is excluded from the primary target and handled only in the separate inclusive sensitivity target.
- [x] 7.7 Add binary triage model tests for leakage-safe fold preprocessing, atlas-derived feature availability, blocked-cluster routing, threshold selection, confidence interval reporting, prediction uncertainty fields, and explanation fields.
- [x] 7.8 Add run-config integration coverage for `eq run-config --config configs/label_free_roi_embedding_atlas.yaml` with and without optional adjudication evidence.
- [x] 7.9 Add main quantification/P3 integration coverage for binary review-triage outputs when adjudicated atlas evidence is available.

Implementation note: binary review triage is implemented as an atlas-local first-read product under `burden_model/embedding_atlas/binary_review_triage/`. It is not wired into `configs/endotheliosis_quantification.yaml` in this change, so P3/main quantification coverage is satisfied by preserving the existing main workflow and validating the atlas run-config integration path.

## 8. Documentation, Handoff, And Model Artifact Policy

- [x] 8.1 Use the Documentation Wizard workflow to inventory documentation surfaces and compare the live CLI/config/artifact surface against README, onboarding, output-structure, integration, and technical lab documentation.
- [x] 8.2 Update user-facing documentation so the repo direction is clear: binary no/low versus moderate/severe review triage with uncertainty and explanations, not autonomous multi-ordinal grading.
- [x] 8.3 Update technical lab documentation with the evidence chain that led here: existing P3/ordinal/severe limitations, atlas adjudication results, binary target definition, PCA/GMM/anchor feature use, uncertainty metrics, explanation fields, and claim boundaries.
- [x] 8.4 Add a reproducibility guide that starts from environment setup and one-command YAML runs, then lists every artifact needed to reproduce the final binary triage output.
- [x] 8.5 Add math/method documentation covering binary target construction, grouped validation, threshold selection, confidence intervals, GMM posterior/distance features, PCA feature use, anchor distances, and feature-explanation interpretation.
- [x] 8.6 Add a usability-first final-output guide that tells a reviewer exactly which HTML to open, what dropdowns or routes mean, what CSV/JSON/Markdown artifacts to inspect, and how to interpret the triage output.
- [x] 8.7 Add model artifact storage guidance that states generated models stay out of Git by default, and records when Git LFS is appropriate for a final promoted model versus when runtime artifacts should remain under the configured runtime root.
- [x] 8.8 Run a Documentation Wizard report or equivalent repo-local documentation drift check and fix high-impact drift introduced by this change.

## 9. Runtime Regeneration And Postflight

- [x] 9.1 Rerun `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/label_free_roi_embedding_atlas.yaml` after atlas implementation.
- [x] 9.2 Rerun the main quantification/P3 workflow if binary review triage is wired into `configs/endotheliosis_quantification.yaml`.
- [x] 9.3 Verify the regenerated runtime contains `atlas_final_adjudication_outcome.json`, `atlas_score_corrections.csv`, `atlas_recovered_anchor_examples.csv`, `atlas_adjudicated_anchor_manifest.csv`, and `atlas_blocked_cluster_manifest.csv` when the current reviewed evidence is provided.
- [x] 9.4 Verify binary triage runtime artifacts contain primary and sensitivity target support, grouped-development metrics, confidence intervals or non-estimable interval reasons, prediction uncertainty fields, feature explanations, and review-route outputs.
- [x] 9.5 Inspect the generated review HTML, focused review HTML, and binary triage review HTML for visible case cards and image paths.
- [x] 9.6 Run `openspec validate adjudicated-atlas-anchor-and-score-review-contract --strict`.
- [x] 9.7 Run `openspec validate --specs --strict`.
- [x] 9.8 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .`.
- [x] 9.9 Run focused atlas tests: `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_embedding_atlas.py`.
- [x] 9.10 Run focused P3/binary triage tests identified during implementation.
- [x] 9.11 Run full tests: `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.

Postflight note: `configs/endotheliosis_quantification.yaml` is unchanged for this change, so 9.2 is a no-op by condition. The required runtime regeneration is the atlas run-config, which completed successfully after implementation.

## 10. Archive, Sync, And Commit Discipline

- [x] 10.1 After implementation and postflight pass, review the git diff and stage only files changed for this spec.
- [x] 10.2 Commit the implementation as a separate traceable commit before archiving the OpenSpec change.
- [x] 10.3 Archive/sync the OpenSpec change after the implementation commit so the active spec state and archived change agree.
- [x] 10.4 Run post-archive validation with `openspec validate --specs --strict`.
- [x] 10.5 Commit the archive/sync as a separate traceable commit.
- [x] 10.6 Do not stop between implementation tasks unless a fail-closed blocker requires user decision; otherwise proceed autonomously through implementation, postflight, archive/sync, and commits.
