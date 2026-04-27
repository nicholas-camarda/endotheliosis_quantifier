## Context

The archived burden-index work establishes the current quantification contract: six-bin image-level grades, manifest-owned `subject_id`/`sample_id`/`image_id`, union ROI extraction, frozen segmentation-backbone embeddings, grouped cumulative-threshold burden modeling, subject-heldout validation, and combined review artifacts. The current result is `exploratory_not_ready` because per-image prediction sets are broad, prediction-set coverage is slightly below nominal, and backend matrix warnings remain recorded.

The strongest next evidence is biological: endotheliosis grading depends on the relative amount of open versus collapsed capillary/arteriole lumina within the glomerulus. The current feature surface does not explicitly encode that. It uses frozen segmentation encoder embeddings plus coarse ROI scalar features such as area, fill fraction, mean intensity, and a bright-pixel openness heuristic. Those features are not enough to distinguish a truly collapsed slit-like lumen from a patent lumen filled with erythrocytes.

## Goals / Non-Goals

**Goals:**

- Organize future `burden_model/` outputs into explicit groups: primary model, validation, calibration, summaries, evidence, candidates, diagnostics, and feature sets.
- Extract morphology-aware ROI features aligned to grading: open-lumen features, collapsed/slit/ridge features, erythrocyte-filled patent-lumen confounder features, and quality/orientation features.
- Generate visual feature QA panels before trusting the features.
- Provide a plug-and-play operator review loop that lets the user adjudicate difficult feature cases with a simple CSV template.
- Evaluate morphology features at both image level and subject level using subject-heldout validation.
- Keep subject/cohort burden summaries separate from per-image prediction readiness.
- Record exactly what improved, what did not, and whether subject/cohort summaries are ready for README/docs.

**Non-Goals:**

- This change does not retrain glomeruli segmentation.
- This change does not make a causal mechanistic claim about endotheliosis.
- This change does not claim RBC-filled lumen detection is perfect.
- This change does not add a new user-facing CLI.
- This change does not preserve old flat `burden_model/*` outputs through compatibility shims for new runs.
- This change does not implement a learned morphology encoder unless explicit features and QA artifacts justify that later.

## Decisions

### Decision: Reorganize burden outputs by semantic role

Future runs SHALL write grouped `burden_model/` subfolders:

```text
burden_model/
  primary_model/
  validation/
  calibration/
  summaries/
  evidence/
  candidates/
  diagnostics/
  feature_sets/
```

This is a deliberate contract change. Historical flat outputs remain historical runtime artifacts; the implementation should not add alias files or compatibility shims that make old and new layouts both appear current.

### Decision: Extract explicit morphology features before more deep learning

The first implementation SHALL create `src/eq/quantification/morphology_features.py` and write `burden_model/feature_sets/morphology_features.csv`. Feature families SHALL include:

- open-lumen features: pale-lumen area fraction, lumen candidate count, lumen area distribution, circularity, eccentricity, and open-space density;
- collapsed/slit/ridge features: ridge response, line density, skeleton length per mask area, slit-like object count, and ridge-to-lumen ratio;
- erythrocyte-confounder features: RBC-like color burden, dark/red filled round-lumen candidates, RBC-filled-lumen area fraction, and shape/color separation from collapsed slits;
- quality/orientation features: blur/focus, stain/intensity range, section/orientation ambiguity, and lumen-detectability score.

Alternative considered: jump directly to a fine-tuned CNN. That would be harder to debug and would not answer whether the biological feature target is correctly captured. Explicit features come first.

### Decision: Treat RBC-filled patent lumina as a named confounder

Dark or red-filled lumina SHALL NOT be counted as collapsed solely because they are not pale/empty. The feature extractor must produce evidence fields that separate shape and color:

```text
round or tubular + RBC-like color  -> possible patent lumen with erythrocytes
narrow slit or line-like + collapsed geometry -> possible collapsed lumen
```

The feature QA report SHALL include examples flagged as likely RBC-confounded so the user can inspect whether the detector is making the biologically correct distinction.

### Decision: Treat mesangial/nuclear false slits as a named confounder

The adjudication-driven slit revision made red slit markings visible, but the user identified a new failure mode: mesangial cells and dark compact nuclei can be overcalled as closed slits. The morphology feature layer SHALL therefore separate:

```text
compact dark purple cellular/nuclear structures -> mesangial/nuclear confounder
thin elongated lumen-like spaces               -> collapsed/slit-like candidate
```

The slit detector SHALL exclude nuclear/mesangial-like masks before computing slit area and review overlays. The feature table SHALL include nuclear/mesangial confounder features so downstream candidate models can learn that these are not equivalent to collapsed lumina. The review overlay SHALL render the confounder in purple so the user can distinguish false red slit calls from plausible collapsed lumen calls.

### Decision: Make user adjudication plug-and-play

The workflow SHALL write:

- `burden_model/evidence/morphology_feature_review/feature_review.html`
- `burden_model/evidence/morphology_feature_review/feature_review_cases.csv`
- `burden_model/evidence/morphology_feature_review/operator_adjudication_template.csv`
- `burden_model/evidence/morphology_feature_review/assets/`

The HTML page SHALL show each selected ROI with overlays for pale/open lumen candidates, RBC-filled lumen candidates, collapsed/slit-like candidates, and feature values. The CSV template SHALL use simple columns:

- `case_id`
- `subject_id`
- `sample_id`
- `image_id`
- `score`
- `open_empty_lumen_present`
- `open_rbc_filled_lumen_present`
- `collapsed_slit_like_lumen_present`
- `mesangial_or_nuclear_false_slit_present`
- `poor_orientation_or_quality`
- `feature_detection_problem`
- `preferred_label_if_detection_wrong`
- `notes`

The user workflow should be:

1. Run the YAML.
2. Open `feature_review.html`.
3. Fill the template CSV for selected difficult cases.
4. Rerun the same YAML.
5. The pipeline reads the adjudication CSV if present and reports feature QA agreement/disagreement.

No manual code edits or custom scripts should be required.

### Decision: Evaluate morphology features on image and subject tracks

Image-level candidates answer whether morphology features improve individual ROI score prediction and uncertainty. They SHALL use subject-heldout folds and report stage-index MAE, grade-scale MAE, prediction-set coverage, average prediction-set size, cohort behavior, and numerical diagnostics.

Subject-level candidates answer whether morphology features improve subject/cohort burden summaries. They SHALL aggregate morphology features by `subject_id`, validate by held-out subjects, and report subject-level MAE, grouped bootstrap confidence intervals, cohort stability, and treatment/cohort summaries.

### Decision: Keep candidate screens separate from deployed models

Candidate outputs SHALL live under `burden_model/candidates/` and report `candidate_status`, `target_level`, `intended_use`, and `readiness_status`. Candidate screens are evidence, not deployed models.

## Risks / Trade-offs

- [Risk] Threshold-based lumen/RBC features may be stain-sensitive. -> Mitigation: write stain/intensity diagnostics, QA overlays, and operator adjudication agreement.
- [Risk] RBC-filled patent lumina may be mistaken for collapse. -> Mitigation: include shape plus color features and a dedicated review label.
- [Risk] Output reorganization could confuse users reading old runs. -> Mitigation: docs state that historical flat outputs remain historical and new runs use grouped folders.
- [Risk] Subject-level performance could be mistaken for per-image readiness. -> Mitigation: every candidate row carries `target_level` and report text keeps the tracks separate.
- [Risk] Feature engineering could overfit the current cohort. -> Mitigation: require subject-heldout validation and cohort-stratified summaries before readiness claims.

## Migration Plan

1. Add grouped `burden_model/` output path helpers inside quantification code.
2. Move future burden artifacts into grouped subfolders and update report links/tests.
3. Add morphology feature extraction from existing ROI image/mask crops.
4. Add feature diagnostics and QA panel generation.
5. Add operator adjudication template ingestion and agreement reporting.
6. Add image-level and subject-level morphology candidate screens.
7. Rerun `configs/endotheliosis_quantification.yaml` on the full cohort.
8. Record results in `audit-results.md` with exact artifact links and readiness decision.

Rollback is not a compatibility branch. If morphology features are weak, retain the organized output contract and report that morphology features did not improve readiness.

## Explicit Decisions

- Feature module: `src/eq/quantification/morphology_features.py`.
- Feature review module: `src/eq/quantification/feature_review.py`.
- Workflow config: `configs/endotheliosis_quantification.yaml`.
- Output root: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated`.
- Operator template path: `burden_model/evidence/morphology_feature_review/operator_adjudication_template.csv`.
- New candidate summary path: `burden_model/candidates/morphology_candidate_summary.json`.

## Open Questions

- [audit_first_then_decide] Pale-lumen, RBC-color, and collapsed-line thresholds SHALL be selected after inspecting `feature_review.html` and feature diagnostics from the current scored cohort.
- [audit_first_then_decide] README/docs readiness SHALL be decided from `morphology_candidate_summary.json`, subject-heldout metrics, and cohort stability after the full workflow rerun.
- [defer_ok] Learned encoders and mitochondria-transfer morphology representation tests are deferred until explicit morphology features establish a reliable review baseline.
