## Why

The label-free ROI embedding atlas has now produced reviewer-confirmed anchor candidates, score-correction evidence, recovered low-score RBC-confounded examples, and a blocked problem cluster, but those outputs currently exist as ad hoc evidence files outside a durable workflow contract. The current supervised ordinal/severe-model evidence is not strong enough to sell this as automated multi-grade classification, so the practical next step is to turn adjudicated atlas evidence into a binary review-triage product that groups glomeruli into no/low versus moderate/severe when support is adequate and routes uncertain or borderline cases back to human review.

## What Changes

- Add a reusable adjudication review contract for HTML review bundles that show the relevant image assets beside task-specific controls, persist reviewer choices, and export machine-readable decisions.
- Extend the `label_free_roi_embedding_atlas` workflow to accept explicit adjudicated atlas evidence from `burden_model/embedding_atlas/evidence/` and validate it against atlas row identity, ROI paths, clusters, and original scores.
- Add atlas outputs that separate original scores from adjudicated review evidence:
  - `burden_model/embedding_atlas/evidence/atlas_score_corrections.csv`
  - `burden_model/embedding_atlas/evidence/atlas_recovered_anchor_examples.csv`
  - `burden_model/embedding_atlas/evidence/atlas_final_adjudication_outcome.json`
  - `burden_model/embedding_atlas/evidence/atlas_final_adjudication_outcome.md`
  - `burden_model/embedding_atlas/evidence/atlas_adjudicated_anchor_manifest.csv`
  - `burden_model/embedding_atlas/evidence/atlas_blocked_cluster_manifest.csv`
- Update atlas verdict and index artifacts so adjudication evidence changes the next action and anchor readiness status, not the canonical label table.
- Add a review-triage modeling contract that evaluates binary no/low versus moderate/severe grouping from current ROI/QC, morphology, learned ROI, embedding PCA, Gaussian-mixture/posterior, and atlas-anchor distance features.
- Treat score `1.0` as a borderline/intermediate review stratum by default rather than forcing it into the primary binary training target; report a sensitivity analysis that includes score `1.0` with no/low only if the evidence supports it.
- Require triage outputs to include uncertainty/confidence fields, bootstrap or grouped-resampling confidence intervals for metrics where support permits, and reviewer-facing feature explanations that say which feature families influenced the triage flag.
- Frame the product as a triage/review-prioritization system that speeds grading review, not as an autonomous pathologist, external validation, or final multi-ordinal grader.
- Add postflight checks that fail if an adjudication import cannot be matched to the generated atlas row IDs, if original scores are mutated in place, if exported review HTML is blank or image-less, or if anchor manifests include blocked cluster-level anchors.
- Do not deploy adjudicated scores as replacement labels in the supervised quantification model during this change.

## Capabilities

### New Capabilities

- `adjudication-review-workflow`: Defines reusable reviewer-facing HTML and exported-decision requirements for image-based adjudication workflows.

### Modified Capabilities

- `label-free-roi-embedding-atlas`: Adds adjudicated atlas evidence ingestion, anchor/score-review manifests, blocked-cluster handling, and verdict updates while preserving the atlas claim boundary and original-score provenance.
- `endotheliosis-grade-model`: Adds a support-gated binary no/low versus moderate/severe review-triage candidate that can use atlas/PCA/GMM-derived evidence, uncertainty, confidence intervals, and feature explanations without claiming multi-ordinal deployment.

## Impact

- Affected implementation:
  - `src/eq/quantification/embedding_atlas.py`
  - `src/eq/quantification/endotheliosis_grade_model.py`
  - `src/eq/quantification/modeling_contracts.py`
  - `configs/label_free_roi_embedding_atlas.yaml`
  - `configs/endotheliosis_quantification.yaml` if the binary triage candidate is wired into the main P3 product selector.
  - `src/eq/run_config.py` only if config schema dispatch support needs a new option; no new workflow ID is expected.
- Affected tests:
  - `tests/unit/test_quantification_embedding_atlas.py`
  - tests covering P3/endotheliosis-grade-model candidate selection, metrics, and review outputs.
  - any existing config-dispatch fixture that validates `eq run-config --config configs/label_free_roi_embedding_atlas.yaml`.
- Affected runtime artifacts:
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/embedding_atlas/`
- Data contract changes:
  - Adds explicit adjudication input/output schemas under `burden_model/embedding_atlas/evidence/`.
  - Preserves original human scores in atlas metadata and exports.
  - Records adjudicated score suggestions, anchor eligibility, recovered anchors, blocked clusters, reviewer decisions, and provenance as separate evidence.
  - Adds binary triage target fields that preserve original scores and make the borderline score `1.0` handling explicit.
- Scientific claim impact:
  - Descriptive: supports reviewed morphology-anchor and score-review evidence.
  - Predictive: may support a current-data binary review-triage candidate if grouped out-of-fold metrics, uncertainty, source-sensitivity, and Dox smoke/review gates pass.
  - Associational: may summarize how reviewed anchors, PCA/GMM clusters, and feature explanations relate to original or adjudicated scores.
  - Not causal, not externally validated, not a calibrated multi-ordinal grader, and not automatic replacement of human review.

## Explicit Decisions

- Change name: `adjudicated-atlas-anchor-and-score-review-contract`.
- Existing workflow ID remains `label_free_roi_embedding_atlas`.
- Existing config path remains `configs/label_free_roi_embedding_atlas.yaml`.
- Primary implementation owner remains `src/eq/quantification/embedding_atlas.py`.
- Runtime output subtree remains `burden_model/embedding_atlas/`.
- Adjudication evidence is consumed from and written to `burden_model/embedding_atlas/evidence/`.
- Existing original-score columns SHALL NOT be overwritten by adjudicated scores.
- Primary binary triage target: `no_low = score <= 0.5`; `moderate_severe = score >= 1.5`; score `1.0` defaults to `borderline_review` and is excluded from the primary binary training/evaluation target.
- Required sensitivity target: `no_low_inclusive = score <= 1.0` versus `moderate_severe = score >= 1.5`, reported separately and never used to hide primary-target ambiguity.
- Candidate feature families for binary triage may include ROI/QC, morphology, learned ROI features, reduced embedding PCA features, GMM cluster posterior/distance features, atlas anchor distances, and hybrid combinations.
- Confidence intervals, uncertainty flags, and feature explanations are required review evidence when the binary triage candidate is evaluated.
- The current reviewed evidence files that motivated the contract are:
  - `evidence/atlas_adjudication_review_export.csv`
  - `evidence/atlas_flagged_case_decisions.csv`
  - `evidence/atlas_score_corrections.csv`
  - `evidence/atlas_recovered_anchor_examples.csv`
  - `evidence/atlas_final_adjudication_outcome.json`
  - `evidence/atlas_final_adjudication_outcome.md`

## Open Questions

- [audit_first_then_decide] Whether a shared helper should live inside `src/eq/quantification/embedding_atlas.py` for this change or be centralized for future morphology/severe-review surfaces. The deciding evidence is the existing review code in `src/eq/quantification/embedding_atlas.py`, `src/eq/quantification/feature_review.py`, and severe-aware adjudication code paths.
- [audit_first_then_decide] Whether binary triage should be implemented inside the existing P3 `endotheliosis_grade_model` selector only, or whether a first-class `burden_model/binary_review_triage_model/` subtree should be added. The deciding evidence is the current P3 family-subtree pattern and whether the outputs need a stable first-read surface independent of diagnostic ordinal artifacts.
- [defer_ok] Whether adjudicated atlas evidence should later feed a supervised ordinal retraining or calibration workflow. This change will stop at binary review triage and evidence manifests; full multi-ordinal deployment remains out of scope unless gates unexpectedly pass.
