## 1. Reuse Audit And Input Contract

- [ ] 1.1 Inspect `src/eq/quantification/embedding_atlas.py`, `src/eq/quantification/feature_review.py`, and `src/eq/quantification/severe_aware_ordinal_estimator.py` for reusable adjudication loading, validation, HTML, and summary patterns; record whether the implementation stays atlas-local or extracts a shared helper.
- [ ] 1.2 Add optional adjudication input settings to `configs/label_free_roi_embedding_atlas.yaml` without changing the workflow ID or adding a new CLI command.
- [ ] 1.3 Define atlas adjudication input schemas for cluster review exports and flagged-case decisions, including required columns, allowed decision values, row identity fields, original-score fields, reviewed timestamp, reviewer notes, and ROI path provenance.
- [ ] 1.4 Implement fail-closed validation for present adjudication inputs: missing required columns, duplicate conflicting rows, unmatched `atlas_row_id`, mismatched `subject_image_id`, mismatched `cluster_id`, mismatched `original_score`, and mismatched ROI paths.
- [ ] 1.5 Implement no-review defaults that produce explicit no-review status artifacts when optional adjudication inputs are absent.

## 2. Atlas Evidence Processing

- [ ] 2.1 Extend `src/eq/quantification/embedding_atlas.py` to load validated atlas adjudication exports after cluster assignments, review queues, and representative evidence are available.
- [ ] 2.2 Generate `burden_model/embedding_atlas/evidence/atlas_score_corrections.csv` with original scores and adjudicated scores in separate fields.
- [ ] 2.3 Generate `burden_model/embedding_atlas/evidence/atlas_recovered_anchor_examples.csv` for row-level recovered anchors without promoting their whole cluster.
- [ ] 2.4 Generate `burden_model/embedding_atlas/evidence/atlas_adjudicated_anchor_manifest.csv` for reviewed cluster-level anchor candidates.
- [ ] 2.5 Generate `burden_model/embedding_atlas/evidence/atlas_blocked_cluster_manifest.csv` for mixed, source/batch, ROI/mask, or otherwise blocked clusters.
- [ ] 2.6 Generate `burden_model/embedding_atlas/evidence/atlas_final_adjudication_outcome.json` and `.md` summarizing score changes, scores kept, recovered anchors, blocked clusters, candidate anchor clusters, source evidence paths, and next implementation action.
- [ ] 2.7 Ensure original score fields in atlas metadata, cluster assignments, representatives, review queues, and post hoc diagnostics are never overwritten by adjudicated score evidence.

## 3. Binary Review-Triage Model Contract

- [ ] 3.1 Audit current P3, severe-aware, three-band, four-band, embedding, and atlas runtime metrics to document why binary review triage is the primary near-term product and multi-ordinal deployment is not the default claim.
- [ ] 3.2 Decide whether binary triage lives inside `burden_model/endotheliosis_grade_model/` or a first-class `burden_model/binary_review_triage_model/` subtree; record the ownership decision in implementation notes and artifact manifests.
- [ ] 3.3 Implement the primary binary target: `no_low = score <= 0.5`, `moderate_severe = score >= 1.5`, and `borderline_review = score == 1.0` excluded from primary binary training and primary metrics.
- [ ] 3.4 Implement the separate sensitivity target: `no_low_inclusive = score <= 1.0` versus `moderate_severe = score >= 1.5`, reported separately from the primary target.
- [ ] 3.5 Generate atlas-derived candidate features for binary triage: selected cluster ID, GMM posterior or distance fields where available, reduced embedding PCA coordinates, nearest reviewed anchor distance/class, blocked-cluster indicator, and recovered-anchor proximity where computable.
- [ ] 3.6 Evaluate pure atlas cluster mapping as a simple baseline and prevent blocked clusters from being forced into a binary class.
- [ ] 3.7 Evaluate leakage-safe supervised binary candidates using ROI/QC, morphology, learned ROI, embedding PCA, GMM/cluster, anchor-distance, and hybrid feature families with subject-heldout grouped development validation.
- [ ] 3.8 Select operating thresholds inside grouped development data only and record whether the operating objective prioritizes moderate/severe sensitivity, review-workload precision/specificity, or a balanced review-triage point.

## 4. Binary Triage Uncertainty And Explanation Outputs

- [ ] 4.1 Write binary triage metrics with recall, precision, specificity, balanced accuracy, AUROC, average precision, false-negative count, false-positive count, threshold, row count, subject count, source support, and finite-output status.
- [ ] 4.2 Add grouped-resampling or bootstrap confidence intervals for key binary metrics when support permits; explicitly mark intervals that are non-estimable.
- [ ] 4.3 Write prediction outputs with predicted probability or score, threshold decision, uncertainty/reliability label, near-threshold flag, source/cohort warning flag, nearest-anchor evidence, and final review route.
- [ ] 4.4 Write row-level explanation outputs that summarize model feature contributions or permutation-style evidence and feature-family contributions for ROI/QC, morphology, learned ROI, embedding PCA, GMM/cluster, and anchor-distance evidence where present.
- [ ] 4.5 Generate a reviewer-facing binary triage review HTML or queue that shows ROI image, ROI mask, original score, adjudication evidence, predicted route, confidence label, top explanation fields, nearest reviewed anchor, and cluster/source warnings.
- [ ] 4.6 Ensure explanation copy states that explanations are model-decision evidence for review prioritization, not causal or mechanistic proof.

## 5. Review HTML Standard

- [ ] 5.1 Update `evidence/embedding_atlas_review.html` generation so case-level and cluster-level decisions remain separate and exported with distinct fields.
- [ ] 5.2 Add or harden focused flagged-case review HTML generation so selected cases are present as static case cards before JavaScript runs.
- [ ] 5.3 Ensure every focused review case shows ROI image and ROI mask beside task-specific score or anchor decision controls.
- [ ] 5.4 Ensure review exports include row identity, original decision context, reviewed decision fields, reviewer notes, reviewed timestamp, and ROI path provenance.
- [ ] 5.5 Add postflight checks that fail if generated review HTML has zero case cards, missing images, missing dropdowns, missing export controls, or JS-only case rendering.

## 6. Verdicts, Manifests, And Documentation

- [ ] 6.1 Update `summary/atlas_verdict.json` to report adjudication status, score-change counts, kept-score counts, recovered-anchor counts, candidate-anchor clusters, blocked clusters, and next action when review evidence is present.
- [ ] 6.2 Update `summary/atlas_summary.md` and `INDEX.md` to point reviewers to adjudication outcome, score-correction, recovered-anchor, anchor-manifest, and blocked-cluster artifacts.
- [ ] 6.3 Update `summary/artifact_manifest.json` so adjudication artifacts are listed when present and no-review status is explicit when absent.
- [ ] 6.4 Update P3/final-product verdicts so binary review triage is reported as review prioritization with explicit current-data gates, not as external validation or autonomous grading.
- [ ] 6.5 Keep public/user-facing wording descriptive: reviewed morphology anchors, score-review evidence, binary triage review prioritization, uncertainty, and explanations only; not external validation, causal mechanism, calibrated multi-ordinal probabilities, or automatic replacement of human review.

## 7. Tests

- [ ] 7.1 Add unit tests for valid adjudication ingestion using fixture exports modeled on `atlas_adjudication_review_export.csv` and `atlas_flagged_case_decisions.csv`.
- [ ] 7.2 Add fail-closed tests for missing columns, duplicate conflicting decisions, unmatched atlas rows, mismatched clusters, mismatched original scores, and mismatched ROI paths.
- [ ] 7.3 Add tests proving original scores remain unchanged and adjudicated scores are written only to separate evidence fields.
- [ ] 7.4 Add tests for anchor manifests: candidate cluster anchors, blocked clusters, and recovered row-level anchors remain distinct.
- [ ] 7.5 Add review HTML tests for static case cards, image tags, dropdown counts, export controls, and nonblank focused review output.
- [ ] 7.6 Add binary target-construction tests proving score `1.0` is excluded from the primary target and handled only in the separate inclusive sensitivity target.
- [ ] 7.7 Add binary triage model tests for leakage-safe fold preprocessing, atlas-derived feature availability, blocked-cluster routing, threshold selection, confidence interval reporting, prediction uncertainty fields, and explanation fields.
- [ ] 7.8 Add run-config integration coverage for `eq run-config --config configs/label_free_roi_embedding_atlas.yaml` with and without optional adjudication evidence.
- [ ] 7.9 Add main quantification/P3 integration coverage for binary review-triage outputs when adjudicated atlas evidence is available.

## 8. Runtime Regeneration And Postflight

- [ ] 8.1 Rerun `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/label_free_roi_embedding_atlas.yaml` after atlas implementation.
- [ ] 8.2 Rerun the main quantification/P3 workflow if binary review triage is wired into `configs/endotheliosis_quantification.yaml`.
- [ ] 8.3 Verify the regenerated runtime contains `atlas_final_adjudication_outcome.json`, `atlas_score_corrections.csv`, `atlas_recovered_anchor_examples.csv`, `atlas_adjudicated_anchor_manifest.csv`, and `atlas_blocked_cluster_manifest.csv` when the current reviewed evidence is provided.
- [ ] 8.4 Verify binary triage runtime artifacts contain primary and sensitivity target support, grouped-development metrics, confidence intervals or non-estimable interval reasons, prediction uncertainty fields, feature explanations, and review-route outputs.
- [ ] 8.5 Inspect the generated review HTML, focused review HTML, and binary triage review HTML for visible case cards and image paths.
- [ ] 8.6 Run `openspec validate adjudicated-atlas-anchor-and-score-review-contract --strict`.
- [ ] 8.7 Run `openspec validate --specs --strict`.
- [ ] 8.8 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .`.
- [ ] 8.9 Run focused atlas tests: `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_embedding_atlas.py`.
- [ ] 8.10 Run focused P3/binary triage tests identified during implementation.
- [ ] 8.11 Run full tests: `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.

## 9. Archive, Sync, And Commit Discipline

- [ ] 9.1 After implementation and postflight pass, review the git diff and stage only files changed for this spec.
- [ ] 9.2 Commit the implementation as a separate traceable commit before archiving the OpenSpec change.
- [ ] 9.3 Archive/sync the OpenSpec change after the implementation commit so the active spec state and archived change agree.
- [ ] 9.4 Run post-archive validation with `openspec validate --specs --strict`.
- [ ] 9.5 Commit the archive/sync as a separate traceable commit.
- [ ] 9.6 Do not stop between implementation tasks unless a fail-closed blocker requires user decision; otherwise proceed autonomously through implementation, postflight, archive/sync, and commits.
