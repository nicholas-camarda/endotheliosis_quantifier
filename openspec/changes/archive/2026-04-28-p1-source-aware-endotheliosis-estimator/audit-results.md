# Audit Results

## Starting Point

Created on 2026-04-28 after archiving `p0-learned-roi-endotheliosis-quantification`.

P0 established:

- The learned ROI workflow runs end-to-end and writes grouped artifacts.
- The current P0 estimator is not README/docs-ready under its promotion gates.
- P0 readiness blockers included average prediction-set size above `4.0`, observed score-2 coverage around `0.642`, source/cohort predictability, and candidate numerical-warning blockers.
- The strongest P0 fitted signal came from simple ROI/QC-style features rather than the current glomeruli encoder.

## P1 Reframing

P1 does not treat all P0 readiness failures as reasons to stop. The practical goal is to build the best defensible grade-calibrated estimator for the fixed current MR TIFF/ROI data.

Carry-forward rules:

- Broken joins, unsupported labels, nonfinite selected predictions, validation leakage, missing provenance, and overclaims remain hard blockers.
- Score-2 ambiguity, broad prediction sets, source sensitivity, small strata, leave-source-out degradation, and nonfatal numerical warnings are scope limiters unless they expose a hard blocker.
- Source/cohort behavior should be modeled and reported explicitly rather than treated as automatic global disqualification.
- Experimental code and artifacts must be contained under one module surface and one indexed output subtree.
- Upstream MR TIFF-to-ROI adequacy must be visible in the estimator verdict; a downstream estimator cannot imply that segmentation, masks, or ROI extraction were adequate unless the adequacy artifact supports that scope.
- Unknown or missing source context must receive explicit `unknown_source` reliability labeling rather than being silently treated as a known Lauren or VEGFRi/Dox source.
- Reportability must be scoped separately for image-level, subject-level, and aggregate-current-data use.
- The source-aware output tree must be capped by an artifact manifest so new files cannot become undocumented interface surfaces.

## Pending Audit Decisions

- Decide whether source-adjusted candidates use explicit `cohort_id` indicators, cohort-specific calibration, or both after inspecting P0 runtime artifacts.
- Decide whether any P1 result can enter `readme_results_snippet.md` after the full P1 runtime verdict exists.
- Decide whether upstream ROI adequacy supports image-level, subject-level, aggregate-current-data, or no reportable scope after inspecting the current quantification runtime outputs.

## P0 Runtime Audit

Inspected the current P0 runtime output under:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated
```

Observed current-data support:

- ROI rows: `707`.
- ROI status counts: `ok = 707`.
- Independent `subject_id` groups: `60`.
- Cohort rows: `vegfri_dox = 619`, `lauren_preeclampsia = 88`.
- Cohort subjects: `vegfri_dox = 52`, `lauren_preeclampsia = 8`.
- Score counts: `0 = 204`, `0.5 = 194`, `1 = 113`, `1.5 = 93`, `2 = 81`, `3 = 22`.
- Raw image paths present: `707`.
- Raw mask paths present: `707`.

Learned ROI P0 evidence:

- Best image candidate: `image_simple_roi_qc`.
- Best image candidate stage-index MAE: `22.432`.
- Best image candidate grade-scale MAE: `0.568`.
- Best image candidate prediction-set coverage: `0.911`.
- Best image candidate average prediction-set size: `4.320`.
- Best subject candidate: `subject_simple_roi_qc`.
- Best subject candidate stage-index MAE: `11.733`.
- Best subject candidate grade-scale MAE: `0.352`.
- P0 README/docs-ready: `False`.
- P0 blockers: broad prediction sets, score-2 undercoverage, numerical warning scope issues, and cohort/source predictability.

## Audit Decisions For Implementation

- Source-adjusted P1 candidates SHALL include explicit `cohort_id` indicator candidates and a within-source calibrated hybrid candidate. The current data show strong source structure, and P1 needs both adjustment and calibration surfaces to quantify whether source handling improves current-data estimation.
- P1 source-aware estimator outputs SHALL remain excluded from `readme_results_snippet.md` by default in this change. Runtime reports and the indexed estimator subtree are the review surfaces. README eligibility can only become true if the final verdict explicitly supports a reportable scope, but the implementation default is `readme_snippet_eligible = false`.
- Current upstream ROI adequacy supports running image-level, subject-level, and aggregate-current-data estimator diagnostics because all `707` rows have `roi_status = ok`. That adequacy does not by itself validate endotheliosis quantification; it only establishes that downstream estimator inputs are present.

## P1 Final Runtime Verdict

Final validation command:

```text
PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml
```

Final runtime output root:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated
```

Source-aware estimator root:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated/burden_model/source_aware_estimator
```

Runtime result:

- Full workflow completed successfully on 2026-04-28 in `688.49s`.
- Quantification artifact count reported by the workflow: `87`.
- Source-aware manifest artifact count: `20`.
- Source-aware manifest completeness: `true`.
- Missing source-aware manifest artifacts: `none`.
- Flat duplicate source-aware aliases under `burden_model/*`: `none observed by manifest containment test`.

Final source-aware verdict:

- Overall status: `limited_current_data_estimator`.
- Selected image candidate: `pooled_roi_qc`.
- Selected subject candidate: `subject_source_adjusted_hybrid`.
- Hard blockers: `none`.
- Scope limiters: `wide_uncertainty`, `current_data_leave_source_out_sensitivity_only`, `numerical_warning_scope_limiter`.
- README snippet eligible: `false`.
- Testing status: `testing_not_available_current_data_sensitivity`.
- Next action: `review_current_data_source_aware_estimator_outputs`.
- Claim boundary: predictive grade-equivalent endotheliosis burden for current scored MR TIFF/ROI data; not tissue percent, closed-capillary percent, causal evidence, or external validation.

Upstream ROI adequacy:

- Status: `adequate_current_data`.
- Total input rows: `707`.
- Scored rows: `707`.
- Usable ROI rows: `707`.
- Failed ROI rows: `0`.
- ROI status counts: `ok = 707`.
- Cohort rows: `vegfri_dox = 619`, `lauren_preeclampsia = 88`.
- Cohort subjects: `vegfri_dox = 52`, `lauren_preeclampsia = 8`.
- ROI image paths present: `707`.
- ROI mask paths present: `707`.
- Raw image paths present: `707`.
- Raw mask paths present: `707`.
- Segmentation provenance status: `present`.
- Reportable scope support from upstream adequacy: `image_level = true`, `subject_level = true`, `aggregate_current_data = true`.

Metrics by split:

- `pooled_roi_qc` training/apparent image MAE: stage-index `21.709`, grade-scale `0.547`, coverage `0.900`, average prediction-set width `4.093`, warning count `3`, eligible for model selection `false`.
- `pooled_roi_qc` subject-heldout validation image MAE: stage-index `22.432`, grade-scale `0.568`, coverage `0.900`, average prediction-set width `4.197`, warning count `3`, eligible for model selection `true`.
- `pooled_roi_qc` testing row: `testing_not_available_current_data_sensitivity`; no independent test partition exists.
- `pooled_learned_roi` subject-heldout validation image MAE: stage-index `34.084`, grade-scale `0.929`, coverage `0.900`, average prediction-set width `4.993`.
- `pooled_hybrid` subject-heldout validation image MAE: stage-index `33.021`, grade-scale `0.900`, coverage `0.900`, average prediction-set width `4.953`.
- `source_adjusted_roi_qc` subject-heldout validation image MAE: stage-index `22.568`, grade-scale `0.572`, coverage `0.900`, average prediction-set width `4.202`.
- `source_adjusted_hybrid` subject-heldout validation image MAE: stage-index `32.986`, grade-scale `0.888`, coverage `0.900`, average prediction-set width `4.955`.
- `within_source_calibrated_hybrid` subject-heldout validation image MAE: stage-index `33.009`, grade-scale `0.900`, coverage `0.900`, average prediction-set width `4.953`.
- `subject_source_adjusted_hybrid` subject-heldout subject-level MAE: stage-index `11.550`, grade-scale `0.324`, interval width `47.996`, warning count `3`.

Source sensitivity:

- Lauren/preeclampsia selected-candidate rows: `88`; subjects: `8`; score support: `0 = 41`, `0.5 = 22`, `1 = 18`, `1.5 = 7`; stage-index MAE `16.545`; grade-scale MAE `0.398`; coverage `0.977`; average prediction-set size `3.716`.
- VEGFRi/Dox selected-candidate rows: `619`; subjects: `52`; score support: `0 = 163`, `0.5 = 172`, `1 = 95`, `1.5 = 86`, `2 = 81`, `3 = 22`; stage-index MAE `23.268`; grade-scale MAE `0.592`; coverage `0.889`; average prediction-set size `4.265`.
- Leave-Lauren-out sensitivity estimated Lauren/preeclampsia with stage-index MAE `17.430`, grade-scale MAE `0.409`, warning count `3`.
- Leave-VEGFRi/Dox-out sensitivity estimated VEGFRi/Dox with stage-index MAE `32.219`, grade-scale MAE `0.823`, warning count `3`.
- Interpretation remains current-data source sensitivity, not external validation.

Summary figures generated:

- `summary/figures/metrics_by_split.png`
- `summary/figures/predicted_vs_observed.png`
- `summary/figures/calibration_by_score.png`
- `summary/figures/source_performance.png`
- `summary/figures/uncertainty_width_distribution.png`
- `summary/figures/reliability_label_counts.png`

Combined review integration:

- `quantification_review/results_summary.md` includes `## Source-Aware Estimator`.
- `quantification_review/quantification_review.html` links `../burden_model/source_aware_estimator/INDEX.md` and `../burden_model/source_aware_estimator/summary/metrics_by_split.csv`.
- `quantification_review/readme_results_snippet.md` does not include source-aware result language because `readme_snippet_eligible = false`.

Validation commands run:

- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_source_aware_estimator.py` passed: `6 passed`.
- Focused quantification tests covering source-aware, learned ROI, burden, and pipeline integration passed: `14 passed`.
- Full test suite passed: `214 passed, 3 skipped`.
- `ruff check` on changed source/test files passed.
- `ruff format --check` on changed source/test files passed.
- `openspec validate p1-source-aware-endotheliosis-estimator --strict` passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python scripts/check_openspec_explicitness.py openspec/changes/p1-source-aware-endotheliosis-estimator` passed.
