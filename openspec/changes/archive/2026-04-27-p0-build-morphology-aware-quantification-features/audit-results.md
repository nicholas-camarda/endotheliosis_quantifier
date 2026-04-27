# Audit Results

## Runtime

- Run command: `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`
- Run time: April 27, 2026 09:11-09:16 local time.
- Output root: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated`
- Status: completed successfully.

## Verification

- Changed-file lint: passed.
- Focused quantification tests: `14 passed`.
- Full test suite: `205 passed, 3 skipped`.
- OpenSpec strict validation: passed.
- OpenSpec explicitness check: passed.

## Output Contract Result

New full-cohort runs now write grouped burden artifacts under:

- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/primary_model`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/validation`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/calibration`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/summaries`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/evidence`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/candidates`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/diagnostics`
- `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/feature_sets`

No flat legacy files remain directly under the current `burden_model/` root after the rerun.

## Primary Quantification Result

- Examples: `707`
- Subjects: `60`
- Support gate: `passed`
- Numerical stability: `backend_warnings_outputs_finite`
- Stage-index MAE: `22.617`
- Grade-scale MAE: `0.629`
- Prediction-set coverage: `0.898` versus nominal `0.900`
- Average prediction-set size: `5.308 / 6`
- Burden interval empirical coverage: `0.911`
- Operational status in combined review: `exploratory_not_ready`
- README/docs-ready: `False`

This is not ready as an operational per-image model claim because prediction-set coverage remains just below nominal, prediction sets remain broad, and backend matrix warnings remain recorded despite finite outputs.

## Morphology Feature Result

- Feature rows: `707`
- Subjects: `60`
- Feature count: `22`
- Feature status counts: `ok=707`
- Feature diagnostics: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/diagnostics/morphology_feature_diagnostics.json`
- Feature table: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/feature_sets/morphology_features.csv`
- Subject feature table: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/feature_sets/subject_morphology_features.csv`

Candidate screen:

| Candidate | Target | Stage-index MAE | Grade-scale MAE | Status |
| --- | --- | ---: | ---: | --- |
| `image_morphology_only_ridge` | image | `16.924` | `0.508` | finite with backend warnings |
| `image_embedding_plus_morphology_ridge` | image | `29.328` | `0.880` | finite with backend warnings |
| `subject_morphology_only_ridge` | subject | `10.765` | `0.323` | finite with backend warnings |
| `subject_embedding_plus_morphology_ridge` | subject | `13.070` | `0.392` | finite with backend warnings |

Interpretation:

- The morphology-only screens are materially better than the current primary image burden MAE and the prior subject ROI scalar candidate, especially at subject level.
- The embedding-plus-morphology candidates are worse than morphology-only here, so adding frozen embeddings is not automatically helpful for the burden target.
- Backend warnings are still present for morphology candidates, so these screens remain evidence for the next modeling direction, not deployed models.

## Feature Biology Caution

The first feature audit does not justify a simple statement that pale-lumen area alone explains the score. Score correlations were strongest for pale-lumen eccentricity, lumen detectability, RBC-like burden, dark filled-lumen evidence, pale-lumen area, orientation ambiguity, and ridge-to-lumen ratio. Mean pale-lumen area fraction was narrow across score bins, roughly `0.208-0.211`, while RBC-like burden decreased from about `0.082` at score `0` to `0.042` at score `3`.

This means the candidate improvement is coming from a multifeature morphology signal, not from a single obvious open-area percentage feature. The operator review is required before making a biological interpretation claim.

## Operator Review

- Feature review HTML: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/evidence/morphology_feature_review/feature_review.html`
- Selected cases: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/evidence/morphology_feature_review/feature_review_cases.csv`
- Adjudication template: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/evidence/morphology_feature_review/operator_adjudication_template.csv`
- Agreement summary: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/evidence/morphology_feature_review/operator_adjudication_agreement.json`

Current adjudication status after user review and rerun: `completed`.

Adjudication summary:

- Completed rows: `46 / 46`
- Feature detection problem rows: `30 / 46`
- `collapsed_slit_missed` was the dominant failure mode.
- The user reported that the review overlays showed essentially no red closed-slit markings and that closed slit-like structures were often being treated as open lumina.
- RBC-confounded narrowed lumina were also observed and labeled as mixed cases rather than pure open or pure collapsed structures.

User workflow:

1. Open `feature_review.html`.
2. Fill `operator_adjudication_template.csv`.
3. Save the CSV in the same review directory.
4. Rerun `configs/endotheliosis_quantification.yaml`.
5. Inspect `operator_adjudication_agreement.json` and the combined quantification review.

## Cohort Stability

Final full-cohort fitted summaries were stable against subject-heldout validation summaries:

- `lauren_preeclampsia`: absolute difference `0.586` burden points, gate `passed`
- `vegfri_dox`: absolute difference `2.003` burden points, gate `passed`

## What Worked

- The grouped output contract is implemented and verified on the real full-cohort output root.
- Morphology feature extraction ran over every admitted ROI row.
- Feature review artifacts are generated without adding a new CLI.
- Morphology-only candidate screens substantially improved held-out MAE relative to the current primary burden model.
- Cohort stability artifacts passed for final versus validation subject-weighted burden summaries.

## What Did Not Fully Work

- The current primary burden model remains `exploratory_not_ready`.
- Prediction sets remain too broad for confident per-image operational use.
- Prediction-set coverage is still slightly below nominal.
- Backend matrix warnings remain recorded for model screens, although outputs are finite.
- The explicit pale-lumen area feature is not by itself a clean severity axis in the current implementation.
- Operator adjudication found a major feature-detection failure: closed/slit-like structures are not being detected reliably in the visual overlays.
- RBC-like features also overcall some open capillaries as RBC-confounded, based on user notes.
- The morphology candidate MAE improvement should not be promoted as biology-ready because the closed-slit feature family failed visual adjudication.

## Next Decision

The best next move is not more frozen-embedding work. The evidence favors:

1. Fix closed/slit-like lumen detection and overlay rendering before promoting morphology features.
2. Add explicit adjudication-derived diagnostics: false-negative closed slits, open lumina mislabeled as RBC-like, and RBC-filled narrowed lumina.
3. Rerun the same operator review set after the slit detector is revised.
4. Only after the revised overlays are visually plausible should `subject_morphology_only_ridge` be considered for a subject/cohort burden candidate with calibrated intervals.

## Adjudication-Driven Slit Detector Revision

The first revised detector was interrupted because its connected-component implementation was too slow on the full cohort. The component classifier was optimized and the full workflow was rerun successfully at April 27, 2026 13:10-13:20 local time.

Revised detector changes:

- open/pale-lumen threshold is stricter than the first pass;
- elongated pale components are removed from open-lumen features and treated as slit-like candidates;
- dark elongated components use a lower dark threshold and connected-component shape criteria;
- RBC-like candidates are excluded from pale/slit masks and use stricter red-dominance and saturation criteria;
- overlay priority was changed so ridge is drawn faintly first, while slit-like candidates are drawn last in high-opacity red.

Revised adjudication summary:

- Completed rows: `46 / 46`
- Feature detection problem rows from user adjudication: `30 / 46`
- User-labeled closed/slit-positive rows: `28`
- Revised slit feature detected nonzero slit area in `28 / 28` user-labeled slit-positive rows.
- Agreement counts now record `collapsed_slit_missed=22`, `open_mislabeled_as_rbc=11`, and `rbc_filled_narrowed_lumen=3` from the user's preferred labels.

Revised feature distribution:

- Mean pale-lumen area fraction changed from about `0.210` to `0.063`.
- Mean slit-like area fraction changed from about `0.003` to `0.103`.
- Mean RBC-like color burden changed from about `0.067` to `0.024`.

Revised candidate screen:

| Candidate | Target | Stage-index MAE | Grade-scale MAE | Status |
| --- | --- | ---: | ---: | --- |
| `image_morphology_only_ridge` | image | `20.173` | `0.605` | finite with backend warnings |
| `image_embedding_plus_morphology_ridge` | image | `29.874` | `0.896` | finite with backend warnings |
| `subject_morphology_only_ridge` | subject | `12.731` | `0.382` | finite with backend warnings |
| `subject_embedding_plus_morphology_ridge` | subject | `13.749` | `0.412` | finite with backend warnings |

Interpretation of the revision:

- The visual false-negative problem was corrected enough that red slit markings now appear in the previously missed high-score examples.
- The revised detector is likely over-sensitive: slit-like area is nonzero across the cohort, and candidate MAE worsened relative to the first morphology pass.
- This is progress for review visibility, not final biological calibration. The next pass should tune slit specificity against the adjudicated cases and add a reviewer-facing derived status such as `slit_detected_when_user_says_yes`, `slit_overcalled_when_user_says_no`, and `rbc_overcalled_on_open_lumen`.

Verification after the revision:

- Focused morphology/burden tests: `6 passed`.
- Full test suite: `205 passed, 3 skipped`.
- OpenSpec strict validation: passed.

## Mesangial/Nuclear False-Slit Revision

The user identified that some red slit calls were actually mesangial cells or compact nuclei. The spec and implementation were amended to treat mesangial/nuclear false slits as a named confounder rather than a threshold-tuning nuisance.

Implementation changes:

- Added nuclear/mesangial confounder feature columns:
  - `morph_nuclear_mesangial_confounder_area_fraction`
  - `morph_nuclear_mesangial_confounder_count`
  - `morph_slit_excluded_nuclear_overlap_fraction`
- Added a `nuclear_mesangial` mask to the morphology overlay rendered in purple.
- Excluded nuclear/mesangial-like pixels from slit-like masks before computing slit area/count.
- Extended `operator_adjudication_template.csv` with `mesangial_or_nuclear_false_slit_present` without overwriting the user's existing review responses.
- Added adjudication summary counts for the new label.

Full workflow rerun:

- Command: `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`
- Runtime: April 27, 2026 13:54-14:06 local time.
- Status: completed successfully.

Revised feature distribution after nuclear/mesangial exclusion:

- Mean slit-like area fraction decreased from `0.103` to `0.063`.
- Mean nuclear/mesangial confounder area fraction is `0.147`.
- Mean slit-excluded nuclear overlap fraction is `0.040`.
- User-labeled slit-positive cases still have nonzero slit area in `28 / 28` reviewed rows.
- The reviewed high-score example `morphology_case_621` now shows red slit candidates plus purple nuclear/mesangial confounder candidates in the same panel.

Revised candidate screen:

| Candidate | Target | Stage-index MAE | Grade-scale MAE | Status |
| --- | --- | ---: | ---: | --- |
| `image_morphology_only_ridge` | image | `19.706` | `0.591` | finite with backend warnings |
| `image_embedding_plus_morphology_ridge` | image | `30.225` | `0.907` | finite with backend warnings |
| `subject_morphology_only_ridge` | subject | `14.520` | `0.436` | finite with backend warnings |
| `subject_embedding_plus_morphology_ridge` | subject | `13.716` | `0.411` | finite with backend warnings |

Interpretation:

- The purple confounder layer materially reduces red slit overcall and makes mesangial/nuclear false positives reviewable.
- The detector is still not biology-ready: nuclear/mesangial burden is high across the cohort, and subject-level morphology-only performance worsened relative to the first morphology pass.
- The next review should use the new `mesangial_or_nuclear_false_slit_present` column to distinguish three failure modes: true slits missed, mesangial/nuclear structures still red, and true slits incorrectly converted to purple.

## Runtime Logging Improvement

The user identified that the quantification run appeared stalled because the terminal only showed model-load and embedding messages before a long morphology-processing gap. The workflow now logs the major stage boundaries and long-running inner loops directly.

Implementation changes:

- `src/eq/quantification/pipeline.py` logs manifest root, output root, `stop_after`, scored-example rows, ROI rows, embedding rows/columns, ordinal comparator start/finish, burden evaluation start/finish, combined review start/finish, and final artifact count.
- `src/eq/quantification/burden.py` logs grouped burden output directories, subject-grouped validation counts, morphology feature/review paths, fold sizes, candidate-screen paths, burden metric status, final-prediction path, and serialized model path.
- `src/eq/quantification/morphology_features.py` logs morphology extraction every 25 ROI rows plus first/final rows, including elapsed seconds and rows/minute.
- `src/eq/quantification/feature_review.py` logs selected review cases, rendered asset progress, adjudication summary path, and review HTML path.
- The adjudication summary yes-count path no longer emits the pandas object-downcast `FutureWarning`.

Full workflow verification before the final 25-row interval tightening:

- Command: `PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml`
- Runtime: April 27, 2026 14:59-15:11 local time.
- Status: completed successfully.
- The terminal showed the top-level workflow parameters, scored examples `707`, ROI rows `707`, embedding rows `707`, embedding columns `553`, subject-grouped validation with `60` subjects and `707` samples, fold sizes, candidate paths, metric paths, and final artifact count `51`.
- The prior pandas `FutureWarning` from `feature_review.py` did not reappear during this full run.

Verification after the final 25-row interval tightening:

- Changed-file format: passed.
- Changed-file lint: passed.
- Focused quantification tests: `14 passed`.

Remaining note:

- The FastAI `load_learner` pickle warning still appears when loading the trusted local segmentation model. That warning is expected from FastAI model loading and is separate from this workflow's own logging.

## Border And False-Slit Readiness Blocker

The user identified that essentially all review images had mesangial/nuclear false-slit issues and that many red slit calls were on the outer glomerular border. Direct visual review confirmed this concern.

Visual evidence:

- Contact sheet inspected: `/tmp/morphology_boundary_failure_contact_sheet.png`
- Revised contact sheet after border separation: `/tmp/morphology_boundary_revision_contact_sheet.png`
- Current review HTML: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/evidence/morphology_feature_review/feature_review.html`

Pre-revision boundary check on the 46 selected review cases:

- Mean fraction of red slit pixels within 8 px of the glomerular boundary: about `0.240`.
- Several review cases had boundary-contact fractions above `0.40`.
- This means the previous red slit feature was partly measuring glomerular capsule, mask boundary, or crop-edge texture rather than collapsed capillary lumina.

Implementation changes:

- Added border false-slit features:
  - `morph_border_false_slit_area_fraction`
  - `morph_border_false_slit_object_count`
  - `morph_slit_boundary_overlap_fraction`
- Excluded boundary-adjacent slit-like candidates from accepted `morph_slit_like_area_fraction` and `morph_slit_like_object_count`.
- Rendered rejected boundary-adjacent slit candidates in cyan in the feature review overlay.
- Extended `operator_adjudication_template.csv` with `border_false_slit_present` without requiring a new CLI.
- Added `feature_readiness` to `burden_model/candidates/morphology_candidate_summary.json`.

Post-revision feature-readiness result:

- `feature_readiness.status`: `failed_visual_feature_readiness`
- `selection_status`: `blocked_by_visual_feature_readiness`
- `feature_readiness.decision`: `do_not_promote_morphology_candidates_or_use_slit_features_for_biology_claims`

Recorded blockers:

- `accepted_slit_signal_is_common_in_score_0_images`
- `accepted_slit_signal_is_nearly_ubiquitous`
- `slit_signal_has_high_boundary_overlap`
- `nuclear_mesangial_confounder_burden_is_high`

Post-revision quantitative checks from the regenerated full-cohort morphology table:

- Overall accepted slit-positive fraction: `1.000`
- Score-0 accepted slit-positive fraction: `1.000`
- Mean slit boundary-overlap fraction: `0.283`
- Mean border false-slit area fraction: `0.019`
- Mean nuclear/mesangial confounder area fraction: `0.147`

Interpretation:

- The border split is useful because it makes one failure mode explicit and prevents boundary-adjacent slit candidates from being counted as accepted slit area.
- It does not rescue the current deterministic slit detector. Accepted red slit remains ubiquitous, including in all score-0 images.
- The morphology candidate models must remain blocked from promotion. The next scientific path should not be more candidate-model churn on these deterministic slit features; it should be either a stronger supervised morphology-labeling task or a simpler quantification path that does not claim closed-lumen mechanistic evidence from this detector.

Verification:

- Changed-file format: passed.
- Changed-file lint: passed.
- Focused quantification tests: `14 passed`.
