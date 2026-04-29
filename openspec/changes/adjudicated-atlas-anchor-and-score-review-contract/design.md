## Context

The existing `label_free_roi_embedding_atlas` workflow writes a label-blinded atlas under `burden_model/embedding_atlas/`, generates review queues, and explicitly avoids label overrides. After interactive review of the current atlas, the runtime evidence now includes cluster-level decisions, flagged-case decisions, score corrections, recovered low-score anchor examples, and a final adjudication outcome. These files are scientifically useful but currently live outside a formal ingestion contract.

The broader model-product state argues for a narrower and more honest target. Current runtime outputs show that the P3 final product is `diagnostic_only_current_data_model`, three/four-band ordinal candidates have weak balanced accuracy, severe-aware candidates underpredict severe cases, and the prior severe triage candidate was rejected after Dox overcall review. In contrast, the atlas review produced interpretable no/low and moderate/severe anchor evidence. The next deployable shape should therefore be a binary review-triage system, not a multi-ordinal grader.

Existing reuse points checked:

- `src/eq/quantification/embedding_atlas.py`: owns atlas path layout, review queue generation, `embedding_atlas_review.html`, `atlas_verdict.json`, and `INDEX.md`.
- `src/eq/quantification/feature_review.py`: owns morphology-feature review bundles and plug-and-play operator adjudication templates.
- `src/eq/quantification/severe_aware_ordinal_estimator.py`: owns reviewer adjudication ingestion for severe false negatives, summary artifacts, and contained adjudicated rerun outputs.
- `src/eq/quantification/endotheliosis_grade_model.py`: owns P3 product selection, first-class model-family subtrees, binary severe triage candidates, ordinal candidate evaluation, and current final-product verdicts.
- `src/eq/quantification/modeling_contracts.py`: owns shared preprocessing, threshold, metric, finite-output, and JSON/modeling helper contracts that should be reused for any new binary candidate.
- `configs/label_free_roi_embedding_atlas.yaml`: owns the atlas run-config surface.
- `openspec/specs/label-free-roi-embedding-atlas/spec.md`: already defines atlas review queues without relabeling.
- `openspec/specs/morphology-aware-quantification-features/spec.md`: already defines operator adjudication as a first-class rerun input.

The change should extend the atlas owner for adjudication evidence and extend the P3 grade-model owner for binary triage evaluation. It should borrow the severe-aware pattern of preserving reviewer-provided evidence and writing explicit summaries, but it should not mutate canonical labels or pretend that a multi-ordinal grader is ready.

## Goals / Non-Goals

**Goals:**

- Make adjudicated atlas decisions a first-class evidence input for `eq run-config --config configs/label_free_roi_embedding_atlas.yaml`.
- Validate adjudication rows against atlas row identity, original scores, cluster assignments, and ROI path provenance.
- Preserve original scores and write adjudicated score suggestions as separate evidence.
- Produce explicit anchor manifests, blocked-cluster manifests, score-correction manifests, human-readable summaries, and verdict updates.
- Standardize review HTML behavior for image-based adjudication workflows: visible cases, image/mask assets next to controls, exportable decisions, and postflight coverage that catches blank review pages.
- Evaluate a binary no/low versus moderate/severe review-triage candidate that uses current scored ROI evidence plus adjudicated atlas anchors.
- Include reduced embedding PCA and Gaussian-mixture/posterior or cluster-distance features as candidate feature families, while comparing them against ROI/QC, morphology, learned ROI, and hybrid baselines.
- Report uncertainty, confidence intervals for grouped-development metrics where estimable, source sensitivity, and feature explanations for reviewer-facing triage outputs.
- Keep all generated adjudication artifacts under the runtime output root, not the repository checkout.

**Non-Goals:**

- Do not silently replace Label Studio scores or canonical quantification labels.
- Do not retrain, recalibrate, or redeploy the supervised ordinal or burden model from adjudicated atlas scores.
- Do not claim explicit six-bin, four-band, or three-band multi-ordinal deployment unless the existing P3 gates pass; current evidence does not support making that the primary product.
- Do not force borderline score `1.0` into the primary binary target; handle it as borderline review unless a separate sensitivity analysis is being reported.
- Do not claim external validity, causal mechanism, clinical deployment readiness, or calibrated severity probabilities.
- Do not add a new direct `eq` CLI command; the entrypoint remains run-config.
- Do not require new external pathology labels before the atlas can run.

## Decisions

### Decision 1: Keep `embedding_atlas.py` as the atlas adjudication owner for this change

`src/eq/quantification/embedding_atlas.py` already owns atlas row identity, cluster selection, review queues, review HTML, and atlas verdicts. The first implementation should add adjudication ingestion and manifest writing there instead of creating a new module or workflow.

Alternative considered: centralize all review HTML/export code in a new shared helper immediately. Rejected for this change because the current review surfaces have different schemas and the first priority is to make the atlas evidence contract correct. The new `adjudication-review-workflow` spec still defines shared behavior so future refactoring has a testable target.

### Decision 2: Use explicit evidence files, not label overrides

Adjudicated scores SHALL be represented as review evidence with both `original_score` and `adjudicated_score`. Anchor eligibility SHALL be represented as row-level evidence with `anchor_decision`, cluster context, ROI provenance, and reviewer notes.

Alternative considered: write a label override file and let downstream quantification consume it as labels. Rejected because the atlas is descriptive and review-prioritizing; label mutation would blur original labels, review evidence, and model-training targets.

### Decision 3: Treat cluster anchors and row-level recovered anchors separately

Cluster 1 and Cluster 2 can be summarized as candidate anchor clusters when reviewed cases support coherent morphology. Cluster 0 remains blocked as a cluster-level anchor even though specific rows from it may be recovered as low-score RBC-confounded examples.

Alternative considered: allow any accepted row to promote its whole cluster. Rejected because the current evidence shows a problem cluster can contain useful individual examples while still being invalid as a cluster-level anchor.

### Decision 4: Add config-driven optional adjudication paths

The atlas config should support optional adjudication input paths under `review:` or a dedicated `adjudication:` block. Relative paths resolve under the atlas output root or quantification output root exactly as specified in the design implementation notes; missing optional files produce empty/no-review summaries, not failure. Present but invalid files fail closed.

Alternative considered: hardcode the current runtime filenames. Rejected because future adjudication exports need the same contract without editing code.

### Decision 5: Review HTML must have static visible content

Reviewer-facing HTML SHALL include static case markup for selected cases and shall not depend entirely on JavaScript rendering. JavaScript can support autosave/export, but a script error must not produce a blank page.

Alternative considered: keep all cases in JSON and render with JavaScript. Rejected because the current blank-page failure showed that JS-only rendering is too brittle for review workflows.

### Decision 6: The model target is binary review triage, not multi-ordinal deployment

The primary target SHALL be `no_low` versus `moderate_severe`, with `score <= 0.5` as no/low and `score >= 1.5` as moderate/severe. Score `1.0` SHALL be treated as `borderline_review` and excluded from the primary training/evaluation target. A sensitivity target may include score `1.0` with no/low, but it must be reported separately.

Alternative considered: force all six rubric values into an ordinal model. Rejected for this deployment plan because current three/four-band ordinal metrics are not strong enough for a user-facing grader, and the repo already contains evidence of severe underprediction and source-sensitive behavior.

### Decision 7: PCA/GMM atlas outputs are features and anchors, not the whole classifier by themselves

Gaussian mixture/PCA outputs can support the binary triage model through feature vectors, cluster posterior probabilities, distances to reviewed anchor clusters, nearest-anchor distances, and blocked-cluster indicators. They should be evaluated as candidate features inside grouped out-of-fold validation. A pure GMM cluster mapping can be reported as a simple baseline, but it should not become the product unless it beats supervised/current-data baselines and passes source-sensitivity gates.

Alternative considered: map GMM clusters directly to no/low or moderate/severe labels. Rejected as the primary plan because the current atlas contains a blocked problem cluster and source-sensitive clusters; the reviewed clusters are valuable but insufficient as a standalone classifier.

### Decision 8: Confidence and explanation are required review aids

The triage product SHALL include a confidence/uncertainty surface for every prediction: predicted probability or score, uncertainty band or reliability label, nearest reviewed anchor evidence, source/cohort warning status, and whether the case is near the operating threshold. It SHALL also include feature explanations for review: coefficient or permutation-style feature contributions for simple models, feature-family contribution summaries, and anchor-distance/GMM evidence. These explanations support triage review; they are not causal explanations.

## Artifact Flow

```
label_free_roi_embedding_atlas
        │
        ▼
burden_model/embedding_atlas/
        │
        ├── review_queue/atlas_adjudication_queue.csv
        ├── evidence/embedding_atlas_review.html
        │
        ├── optional adjudication inputs
        │   ├── evidence/atlas_adjudication_review_export.csv
        │   └── evidence/atlas_flagged_case_decisions.csv
        │
        ▼
validated adjudication evidence
        │
        ├── evidence/atlas_score_corrections.csv
        ├── evidence/atlas_recovered_anchor_examples.csv
        ├── evidence/atlas_adjudicated_anchor_manifest.csv
        ├── evidence/atlas_blocked_cluster_manifest.csv
        ├── evidence/atlas_final_adjudication_outcome.json
        └── evidence/atlas_final_adjudication_outcome.md
        │
        ▼
summary/atlas_verdict.json and INDEX.md next-action update
        │
        ▼
binary review-triage evaluation
        │
        ├── no_low target: score <= 0.5
        ├── moderate_severe target: score >= 1.5
        ├── borderline_review: score == 1.0
        ├── candidate features: ROI/QC, morphology, learned ROI,
        │   embedding PCA, GMM/posterior, anchor distances, hybrids
        └── outputs: predictions, uncertainty, metric CIs,
            feature explanations, review queue, verdict
```

## Validation Strategy

- Validate required adjudication columns before using an input file.
- Match every adjudication row to exactly one atlas row by `atlas_row_id`.
- Verify any provided `subject_image_id`, `cluster_id`, `original_score`, `roi_image_path`, and `roi_mask_path` agree with current atlas artifacts.
- Reject duplicate adjudication rows with conflicting decisions.
- Preserve original score columns in existing atlas outputs.
- Write empty adjudication summary artifacts when no optional adjudication file is present.
- Add tests proving review HTML contains visible case cards, image tags, dropdowns, export controls, and the expected case count.
- For binary triage, use subject-heldout grouped development validation and report recall, precision, specificity, balanced accuracy, AUROC, average precision, false-negative count, false-positive count, threshold, finite-output status, source sensitivity, and confidence intervals where support permits.
- Evaluate the primary target and the score-1.0-inclusive sensitivity target separately.
- Require explanation artifacts for selected or finalist candidates.

## Risks / Trade-offs

- [Risk] Reviewer exports can drift from the atlas run they were generated from. → Mitigation: validate row identity, cluster ID, original score, and ROI paths against the current atlas artifacts and fail closed on mismatch.
- [Risk] Adjudicated score suggestions may be mistaken for canonical labels. → Mitigation: use `adjudicated_score` and `score_decision` fields only in evidence outputs; never overwrite `score` or `original_score`.
- [Risk] A mixed cluster can contain useful individual anchors. → Mitigation: write both blocked cluster manifests and recovered row-level anchor manifests.
- [Risk] Review HTML can appear blank because of JavaScript errors. → Mitigation: write static case markup and add postflight tests for case-card/image/dropdown counts.
- [Risk] Source/batch concentration remains a confounder for Cluster 1 and Cluster 2. → Mitigation: include source/cohort warnings and keep anchor outputs as review-supported descriptive evidence, not external validation.
- [Risk] Binary triage could become another overclaimed model after prior severe triage rejection. → Mitigation: keep the public claim to review prioritization, require Dox/source-sensitivity checks, and block README-facing deployment if reviewed non-severe overcalls recur.
- [Risk] Confidence intervals may be unstable with small or source-confounded support. → Mitigation: report them as grouped-development uncertainty only, include row/subject counts, and suppress or flag non-estimable intervals.
- [Risk] Feature explanations could be interpreted as biological mechanism. → Mitigation: label them as model-decision evidence and keep morphology/anchor explanations descriptive.

## Migration Plan

1. Add optional adjudication input configuration to `configs/label_free_roi_embedding_atlas.yaml`.
2. Extend `embedding_atlas.py` to load no-review defaults when adjudication inputs are absent.
3. Add validation and evidence summarization for present adjudication inputs.
4. Write anchor, score-correction, blocked-cluster, JSON, and Markdown outputs.
5. Update `atlas_verdict.json`, `atlas_summary.md`, and `INDEX.md` to point to adjudication outcomes when present.
6. Add the binary review-triage candidate evaluation to the existing P3 owner or a first-class P3 family subtree after the implementation audit resolves ownership.
7. Generate binary triage metrics, confidence intervals, prediction uncertainty, explanation artifacts, and reviewer-facing triage queues.
8. Regenerate the atlas and the main quantification/P3 outputs with the current reviewed evidence.
9. Run strict OpenSpec validation, focused atlas/P3 tests, `ruff check .`, and full `pytest -q`.

Rollback is straightforward: remove or omit the optional adjudication input paths and rerun the atlas. The workflow should still produce baseline atlas outputs and empty/no-review adjudication summaries without changing supervised quantification outputs.

## Explicit Decisions

- Workflow ID: `label_free_roi_embedding_atlas`.
- Config file: `configs/label_free_roi_embedding_atlas.yaml`.
- Primary module: `src/eq/quantification/embedding_atlas.py`.
- Runtime output subtree: `burden_model/embedding_atlas/`.
- Review HTML standard applies to `evidence/embedding_atlas_review.html` and any focused atlas review pages generated by this workflow.
- Required postflight artifact after adjudication is present: `burden_model/embedding_atlas/evidence/atlas_final_adjudication_outcome.json`.
- Original score fields are immutable evidence in this workflow.
- Primary binary triage target: `score <= 0.5` versus `score >= 1.5`.
- Borderline score handling: `score == 1.0` is `borderline_review` for the primary target.
- Required sensitivity target: `score <= 1.0` versus `score >= 1.5`.
- Required prediction explanation types: model feature contribution summary, feature-family contribution summary, nearest reviewed anchor evidence, GMM/PCA cluster evidence, and source/cohort warning status.
- Confidence intervals are grouped-development uncertainty evidence, not external validation.

## Open Questions

- [audit_first_then_decide] Whether the first implementation should extract a shared HTML review helper from `embedding_atlas.py`, `feature_review.py`, and `severe_aware_ordinal_estimator.py`. Decide after inspecting duplication during implementation; if extraction would broaden blast radius, keep this change atlas-local and leave shared helper extraction for a later spec.
- [audit_first_then_decide] Whether binary triage should live as a new first-class `burden_model/binary_review_triage_model/` subtree or as a candidate family inside `burden_model/endotheliosis_grade_model/`. Decide by inspecting the current P3 family-subtree conventions and minimizing duplicate model-selection plumbing.
- [defer_ok] Whether adjudicated atlas anchors later feed a full supervised ordinal retraining or calibration workflow. This change only produces validated evidence manifests and binary review triage.
