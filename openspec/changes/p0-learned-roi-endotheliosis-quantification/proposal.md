## Why

The deterministic morphology-feature pass showed that hand-engineered closed-slit detection is not biologically reliable: accepted slit signal was ubiquitous, present in all score-0 images, boundary-contaminated, and confounded by mesangial/nuclear structures. The next quantification step should pivot from brittle threshold rules to learned ROI representations with calibrated subject-heldout validation and reviewer-visible evidence.

## What Changes

- Add a learned ROI representation workflow that extracts one or more learned feature sets from the existing union ROI image/mask crops and evaluates them for endotheliosis score and burden prediction.
- Keep deterministic morphology features as QC/evidence only; they SHALL NOT be promoted as closed-lumen biology while `morphology_candidate_summary.json` reports `blocked_by_visual_feature_readiness`.
- Add learned-feature candidate screens that compare:
  - current glomeruli encoder embeddings from `src/eq/quantification/pipeline.py`;
  - frozen pathology/image foundation embeddings when available in the `eq-mac` environment;
  - simple low-dimensional QC/coarse ROI features as a sanity baseline;
  - optional embedding-plus-QC hybrids.
- Add calibrated ordinal/stage-index/burden heads on learned ROI embeddings using subject-heldout validation and grouped conformal uncertainty.
- Add subject/cohort summary artifacts that clearly separate subject/cohort burden summaries from per-image operational predictions.
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

## Open Questions

- [audit_first_then_decide] Which learned feature providers are available in the current `eq-mac` environment: current glomeruli encoder, torchvision/ImageNet backbones, `timm`/DINO-style models, or local pathology foundation embeddings? The implementation SHALL audit installed packages and record provider availability before fitting candidates.
- [audit_first_then_decide] Whether an external pathology foundation model can be used locally without network downloads, licensing friction, or heavyweight setup SHALL be decided from environment/package inspection before implementation chooses it.
- [defer_ok] A supervised open-lumen/RBC-filled-lumen/collapsed-slit annotation workflow is deferred unless learned ROI candidates fail or explanation needs require a mechanistic detector.
