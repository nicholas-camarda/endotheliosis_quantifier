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

## Initial Implementation Verdict

Not started. This file SHALL be updated during apply with provider audit results, learned ROI candidate metrics, calibration results, evidence artifact links, readiness verdict, and remaining blockers.
