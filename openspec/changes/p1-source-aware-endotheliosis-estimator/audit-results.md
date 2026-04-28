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
