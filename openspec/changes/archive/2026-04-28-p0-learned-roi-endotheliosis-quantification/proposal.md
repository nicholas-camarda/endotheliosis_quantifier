## Why

The deterministic morphology-feature pass showed that hand-engineered closed-slit detection is not biologically reliable: accepted slit signal was ubiquitous, present in all score-0 images, boundary-contaminated, and confounded by mesangial/nuclear structures. The next quantification step should pivot from brittle threshold rules to learned ROI representations with calibrated subject-heldout validation and reviewer-visible evidence.

## What Changes

- Add a learned ROI representation workflow that extracts one or more learned feature sets from the existing union ROI image/mask crops and evaluates them for endotheliosis score and burden prediction.
- Keep deterministic morphology features as QC/evidence only; they SHALL NOT be promoted as closed-lumen biology while `morphology_candidate_summary.json` reports `blocked_by_visual_feature_readiness`.
- Add a capped phase-1 learned-feature candidate screen that fits only:
  - `current_glomeruli_encoder` embeddings from `embeddings/roi_embeddings.csv`;
  - `simple_roi_qc` low-dimensional non-mechanistic ROI/mask/stain/quality features;
  - `current_glomeruli_encoder_plus_simple_roi_qc`.
- Audit optional local backbone/foundation providers, but keep them audit-only in phase 1 unless a later explicit OpenSpec decision promotes them to fitted candidates.
- Add calibrated ordinal/stage-index/burden heads on learned ROI embeddings using subject-heldout validation and grouped conformal uncertainty.
- Add subject/cohort summary artifacts that clearly separate subject/cohort burden summaries from per-image operational predictions.
- Add cohort-confounding diagnostics so learned ROI results cannot be promoted when they mainly separate cohort/stain/acquisition lanes rather than score/burden signal.
- Add learned-model evidence artifacts: nearest neighbors, high-error examples, uncertainty examples, embedding-space diagnostics, and optional saliency/attention visualizations labeled as heuristic model-support evidence.
- Add documentation and report gating so README/docs-ready claims are allowed only if the selected track passes validation, uncertainty, numerical, and claim-boundary gates.
- No compatibility shim: existing deterministic morphology artifacts remain historical/QC artifacts; the new workflow writes a new grouped output branch under `burden_model/learned_roi/`.

## Capabilities

### New Capabilities
- `learned-roi-quantification`: Defines learned ROI representation extraction, candidate evaluation, calibration, evidence artifacts, and readiness gates for endotheliosis quantification.

### Modified Capabilities
- `endotheliosis-burden-index`: Adds the learned ROI output branch, selected-track reporting, and burden/README readiness rules for learned ROI candidates.
- `morphology-aware-quantification-features`: Clarifies that deterministic morphology features are QC/evidence only while visual feature-readiness is blocked, and that they must not be used for biological closed-lumen claims.

## Impact

- Affected modules:
  - `src/eq/quantification/pipeline.py`
  - `src/eq/quantification/burden.py`
  - `src/eq/quantification/embeddings.py`
  - new `src/eq/quantification/learned_roi.py`
  - new `src/eq/quantification/learned_roi_review.py`
- Affected CLI/config surfaces:
  - `eq run-config --config configs/endotheliosis_quantification.yaml`
  - `eq quant-endo`
  - `configs/endotheliosis_quantification.yaml`
- Affected runtime output root:
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated`
- New learned ROI artifacts under:
  - `burden_model/learned_roi/feature_sets/`
  - `burden_model/learned_roi/candidates/`
  - `burden_model/learned_roi/validation/`
  - `burden_model/learned_roi/calibration/`
  - `burden_model/learned_roi/evidence/`
  - `burden_model/learned_roi/diagnostics/`
  - `burden_model/learned_roi/primary_model/`
- Tests:
  - unit tests for learned feature schema, finite outputs, grouped validation, calibration artifacts, readiness gating, and evidence artifact links;
  - integration test proving the YAML workflow writes learned ROI artifacts without flat compatibility aliases.
- Dependencies:
  - use already-installed `eq-mac` packages first;
  - optional foundation-model providers must be capability-detected and recorded as unavailable rather than silently substituted.

## Explicit Decisions

- Change name: `p0-learned-roi-endotheliosis-quantification`.
- Primary runtime front door remains `eq run-config --config configs/endotheliosis_quantification.yaml`.
- Direct CLI remains `eq quant-endo`; no new user-facing quantification CLI is introduced for the first implementation.
- Implementation module paths are fixed as `src/eq/quantification/learned_roi.py` and `src/eq/quantification/learned_roi_review.py`.
- Learned ROI output branch is fixed as `burden_model/learned_roi/`.
- Existing deterministic morphology features remain in `burden_model/feature_sets/` and `burden_model/evidence/morphology_feature_review/` as QC/evidence, not as promoted biological closed-lumen features.
- The first selected target remains predictive grade-equivalent endotheliosis burden, not causal/mechanistic inference and not true tissue percent endotheliosis.
- Phase-1 fitted candidates are fixed to `current_glomeruli_encoder`, `simple_roi_qc`, and `current_glomeruli_encoder_plus_simple_roi_qc`; optional `torchvision`, `timm`, DINO/ViT, UNI, CONCH, or other foundation providers are audit-only in this change.
- Image-track README/docs readiness requires all of: empirical prediction-set coverage `>= 0.88` overall, coverage `>= 0.80` for each observed-score stratum with at least 30 rows, average prediction-set size `<= 4.0`, no cohort-specific coverage below `0.80`, no nonfinite predictions, no unresolved numerical-instability warnings, and no cohort-confounding blocker.
- Subject/cohort-track README/docs readiness requires all of: one row per `subject_id`, subject-heldout validation, grouped bootstrap intervals, no nonfinite predictions, no unresolved numerical-instability warnings, no cohort-confounding blocker, and explicit wording that subject/cohort readiness does not imply operationally precise per-image predictions.
- Candidate selection SHALL report both ordinal/grade-scale metrics and stage-index metrics; it SHALL NOT select a candidate solely from the 0-100 stage-index recoding.
- Implementation SHALL iterate within the fixed phase-1 scope until the learned ROI workflow either satisfies the explicit readiness gates or produces complete failure evidence showing which gates remain unmet.
- Iteration SHALL be limited to implementation defects, calibration/reporting corrections, and required artifact completeness. It SHALL NOT expand fitted providers, weaken validation gates, add compatibility paths, introduce new biological claims, or tune against held-out results without a new explicit OpenSpec decision.

## Open Questions

- [audit_first_then_decide] Which learned feature providers are available in the current `eq-mac` environment: current glomeruli encoder, torchvision/ImageNet backbones, `timm`/DINO-style models, or local pathology foundation embeddings? The implementation SHALL audit installed packages and record provider availability before fitting candidates.
- [audit_first_then_decide] Whether an external pathology foundation model can be used locally without network downloads, licensing friction, or heavyweight setup SHALL be decided from environment/package inspection before implementation chooses it.
- [defer_ok] A supervised open-lumen/RBC-filled-lumen/collapsed-slit annotation workflow is deferred unless learned ROI candidates fail or explanation needs require a mechanistic detector.
