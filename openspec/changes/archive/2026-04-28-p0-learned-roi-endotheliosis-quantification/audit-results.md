# Audit Results

## Starting Evidence From Archived Morphology Change

Source archive:

- `openspec/changes/archive/2026-04-27-p0-build-morphology-aware-quantification-features/audit-results.md`

Key evidence carried forward:

- Deterministic slit features failed visual feature-readiness.
- Accepted slit signal was present in all score-0 images and all images overall.
- Mean slit boundary-overlap fraction after border separation was `0.283`.
- Mean nuclear/mesangial confounder area fraction was `0.147`.
- `morphology_candidate_summary.json` reported `selection_status = blocked_by_visual_feature_readiness`.
- The correct next direction is not further hand-tuning deterministic slit thresholds; it is learned ROI representation modeling with strict validation and claim boundaries.

## Planned Claim Boundary

This change targets predictive grade-equivalent endotheliosis burden from ROI pixels and masks. It does not target:

- causal inference;
- true tissue-area percent endotheliosis;
- validated closed-capillary percent;
- mechanistic proof of endotheliosis from saliency or nearest-neighbor evidence.

## Pre-Apply Ramification Review

Reviewed on 2026-04-27 before implementation.

Current runtime baseline:

- Scored rows: `707`.
- Independent `subject_id` groups: `60`.
- Cohorts: `lauren_preeclampsia` with `88` rows and `8` subjects; `vegfri_dox` with `619` rows and `52` subjects.
- Current image-level burden uncertainty baseline: overall prediction-set coverage `0.898` at nominal `0.90`, average prediction-set size `5.308` out of 6 score classes.
- Current ordinal embedding baseline: grade-scale MAE `0.726`, accuracy `0.269`, quadratic weighted kappa `0.017`, with numerical-instability warnings recorded.

Ramification decisions added to the change:

- Phase-1 fitted candidates are capped to `current_glomeruli_encoder`, `simple_roi_qc`, and `current_glomeruli_encoder_plus_simple_roi_qc` at both image and subject levels.
- Optional `torchvision`, `timm`, DINO/ViT, UNI, CONCH, and other foundation/backbone providers are audit-only in this change. This avoids broad provider shopping on about 60 independent subjects.
- Image-track README/docs readiness now requires overall coverage `>=0.88`, observed-score coverage `>=0.80` for strata with at least 30 rows, cohort coverage `>=0.80` for cohorts with at least 30 rows, average prediction-set size `<=4.0`, finite outputs, no unresolved numerical-instability warnings, and no cohort-confounding blocker.
- Subject/cohort-track README/docs readiness now requires one row per `subject_id`, subject-heldout validation, grouped bootstrap intervals, finite outputs, no unresolved numerical-instability warnings, no cohort-confounding blocker, and explicit no-per-image-precision claim text.
- Cohort-confounding diagnostics are required because learned features may encode cohort, stain, acquisition, or mask-source differences rather than endotheliosis burden.
- Candidate selection must report ordinal/grade-scale metrics alongside 0-100 stage-index metrics and must not promote a model solely because the arbitrary stage-index recoding improves.

## Initial Implementation Verdict

Implemented and runtime-validated on 2026-04-27.

Runtime command:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml
```

The full YAML workflow completed successfully and wrote quantification artifacts under:

```text
/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated
```

The learned ROI branch was regenerated after the final numerical-warning capture/stability fixes using the validated runtime embedding table and existing burden/ordinal artifacts.

Key learned ROI artifacts:

- Provider audit: `burden_model/learned_roi/diagnostics/provider_audit.json`
- Learned feature table: `burden_model/learned_roi/feature_sets/learned_roi_features.csv`
- Feature diagnostics: `burden_model/learned_roi/diagnostics/learned_roi_feature_diagnostics.json`
- Candidate metrics: `burden_model/learned_roi/candidates/learned_roi_candidate_metrics.csv`
- Image predictions: `burden_model/learned_roi/validation/learned_roi_predictions.csv`
- Subject predictions: `burden_model/learned_roi/validation/learned_roi_subject_predictions.csv`
- Calibration: `burden_model/learned_roi/calibration/learned_roi_calibration.json`
- Cohort diagnostics: `burden_model/learned_roi/diagnostics/cohort_confounding_diagnostics.json`
- Evidence review: `burden_model/learned_roi/evidence/learned_roi_review.html`
- Combined review: `quantification_review/quantification_review.html`

Provider availability:

- `current_glomeruli_encoder`: `available_fit_allowed`
- `simple_roi_qc`: `available_fit_allowed`
- `torchvision_resnet18_imagenet`: `available_audit_only` with `torchvision` version `0.26.0`
- `timm_dino_or_vit`: `unavailable`

Fitted phase-1 candidates:

- `image_current_glomeruli_encoder`
- `image_simple_roi_qc`
- `image_current_glomeruli_encoder_plus_simple_roi_qc`
- `subject_current_glomeruli_encoder`
- `subject_simple_roi_qc`
- `subject_current_glomeruli_encoder_plus_simple_roi_qc`

No optional backbone or foundation provider produced fitted candidate rows.

Runtime verdict:

- README/docs-ready: `False`
- Best image candidate: `image_simple_roi_qc`
- Best image candidate stage-index MAE: `22.432`
- Best image candidate grade-scale MAE: `0.568`
- Best image candidate prediction-set coverage: `0.911`
- Best image candidate average prediction-set size: `4.320`
- Best subject/cohort candidate: `subject_simple_roi_qc`
- Best subject/cohort candidate stage-index MAE: `11.733`
- Best subject/cohort candidate grade-scale MAE: `0.352`

Readiness blockers:

- Image-track average prediction-set size was `4.320`, above the explicit `<=4.0` gate.
- Observed score `2` coverage was `0.642`, below the explicit `>=0.80` gate for strata with at least 30 rows.
- Selected image candidate retained finite-output numerical-instability warnings.
- Cohort diagnostics showed selected features predicted cohort with balanced accuracy `1.0`.
- Cohort diagnostics retained numerical-instability warnings for the cohort-predictability and leave-one-cohort diagnostics.

Interpretation:

The learned ROI workflow now executes and produces complete failure evidence. The result improves overall coverage relative to the nominal target but does not satisfy the explicit uncertainty, numerical-stability, and cohort-confounding gates. The learned ROI branch remains exploratory and SHALL NOT be promoted as a README/docs-ready quantification claim.

## Closeout Lessons for the Next Change

Recorded on 2026-04-28 before archive.

The readiness failure above should not be interpreted as a reason to abandon learned or source-aware endotheliosis estimation. It means the phase-0 gate design was deliberately conservative and blocked promotion of a general learned ROI claim. The next change should preserve that evidence while reframing fixed-data limitations into a practical estimator contract.

Key carry-forward decisions:

- The long-term practical goal remains MR TIFF application: segment or identify usable glomerular/ROI tissue, quantify grade-calibrated endotheliosis burden inside those ROIs, and produce image-level and subject-level outputs with uncertainty and reliability labels.
- Score-specific undercoverage, especially the observed score-2 undercoverage, is not by itself a reason to block all estimator generation when it reflects an inherent transitional or underpowered region of the current data. It should become an uncertainty/scope limiter that widens intervals, lowers reliability labels, and prevents overinterpretation of single-image values.
- Cohort/source sensitivity is not automatically disqualifying for a practical estimator built from the current fixed data. It should be modeled and reported explicitly as source-aware behavior, source-adjusted behavior, within-source behavior, and leave-source-out sensitivity.
- Hard blockers in the next change should be limited to conditions that make the estimator invalid or misleading regardless of disclosure: broken joins, unsupported labels, nonfinite predictions, unverifiable provenance, held-out leakage, untraceable artifacts, or claims that exceed the evidence.
- Scope limiters in the next change should include broad intervals, weak score-specific coverage, source sensitivity, small strata, leave-source-out degradation, and low single-image reliability. These should produce explicit reliability flags rather than silently suppressing the estimator.
- Experimental code and artifacts must be contained. The next implementation should avoid codebase bloat by using one bounded orchestrator surface, one output subtree, one top-level artifact index, no duplicate flat aliases, and a clear split between human-facing summaries, diagnostics, predictions, evidence, and internal candidate files.

The next OpenSpec change should therefore target a source-aware, grade-calibrated practical estimator for the current MR TIFF workflow rather than trying to make the phase-0 learned ROI readiness gates pass by weakening them.
