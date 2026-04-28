## Context

P1 changed the question from "can every promotion gate pass?" to "what can we estimate honestly with the fixed current data?" That was the right reframing. It produced a contained `burden_model/source_aware_estimator/` subtree, a verdict, source diagnostics, training/validation/testing-status metrics, prediction artifacts, and a capped figure set. The final P1 verdict was `limited_current_data_estimator` with no hard blockers and three scope limiters: broad uncertainty, current-data source sensitivity only, and nonfatal numerical warnings.

The P1 result also made the next failure mode concrete. The selected image candidate was `pooled_roi_qc`, not learned embeddings. Its training/apparent and subject-heldout validation grade MAE were similar (`0.547` versus `0.568`), while learned and hybrid candidates fit the training data much better but generalized poorly (`0.250` training to `0.929` validation for learned; `0.229` training to `0.900` validation for hybrid). That is classic high-dimensional overfit in the current sample shape, not a reason to add more learned feature providers.

The remaining scientific problem is severe-end compression. The selected image estimator predicts low-to-mid burden more reasonably than score `2` or `3` disease. Runtime evidence showed score `2` stage-index MAE `42.15` and score `3` stage-index MAE `62.68`; score `3` rows had mean predicted stage index around `37.3` despite the observed score mapping to `100`. Severe examples also live in the source-confounded part of the data: Lauren/preeclampsia has scores up to `1.5`, while VEGFRi/Dox contains all observed score `2` and `3` rows.

External methodological guidance supports the design direction without changing the claim boundary. TRIPOD+AI emphasizes transparent prediction-model reporting, validation, calibration, and clear distinction between development/internal validation and external validation. Ordinal regression methods such as CORAL motivate cumulative threshold formulations for ordered labels. Ordinal conformal prediction work in medical imaging supports reporting severity prediction sets rather than pretending a single scalar is precise when uncertainty is large.

The design should therefore target this shape:

```text
Current scored MR TIFF/ROI data
        │
        ▼
Feature audit: ROI/QC, morphology, learned ROI, embeddings
        │
        ▼
Can severe cases be separated at all?
        │
        ├── no  → report non-separability and preserve limited P1 estimator
        │
        └── yes → fit severe-threshold / ordinal / two-stage candidates
                    │
                    ▼
             report scalar burden, severe risk, ordinal set, or scoped combination
```

## Goals / Non-Goals

**Goals:**

- Build a P2 estimator track that directly evaluates severe-endotheliosis behavior instead of optimizing only overall MAE.
- Audit severe-case separability before fitting final severe-aware candidates.
- Evaluate threshold targets `score >= 1.5`, `score >= 2`, and where support permits `score >= 3`.
- Evaluate ordinal/cumulative-threshold candidates that preserve the ordered score rubric.
- Evaluate a two-stage candidate where severe-risk detection gates or calibrates downstream burden estimation.
- Keep validation grouped by `subject_id`.
- Preserve P1 split labels: `training_apparent`, `validation_subject_heldout`, `testing_explicit_heldout`, and `testing_not_available_current_data_sensitivity`.
- Report severe false negatives explicitly, including example review.
- Keep source behavior visible and scoped as current-data sensitivity, not external validation.
- Keep the artifact tree bounded, indexed, and manifest-governed.
- Decide from evidence whether the honest P2 output is a scalar burden, severe-risk label, ordinal prediction set, subject-level aggregate, or limited/non-reportable finding.

**Non-Goals:**

- Do not add new fitted learned ROI providers such as `torchvision`, `timm`, DINO/ViT, UNI, CONCH, or other foundation models.
- Do not claim external validation.
- Do not claim tissue-area percent, closed-capillary percent, causal mechanism, or histologic truth.
- Do not weaken validation gates to make severe prediction look better.
- Do not create a second active quantification config unless the code audit shows the existing `endotheliosis_quantification` workflow cannot safely host the estimator.
- Do not solve segmentation promotion, mask provenance, or glomeruli model scientific promotion in this change.

## Decisions

### Decision: Add a contained P2 estimator subtree

P2 should write under:

```text
burden_model/severe_aware_ordinal_estimator/
```

Proposed role layout:

```text
burden_model/severe_aware_ordinal_estimator/
├── INDEX.md
├── summary/
│   ├── estimator_verdict.json
│   ├── estimator_verdict.md
│   ├── metrics_by_split.csv
│   ├── metrics_by_split.json
│   ├── severe_threshold_metrics.csv
│   ├── artifact_manifest.json
│   └── figures/
├── predictions/
│   ├── image_predictions.csv
│   └── subject_predictions.csv
├── diagnostics/
│   ├── severe_separability_audit.json
│   ├── threshold_support.json
│   ├── source_severe_sensitivity.json
│   └── reliability_labels.json
├── evidence/
│   └── severe_false_negative_review.html
└── internal/
    ├── candidate_metrics.csv
    └── candidate_summary.json
```

Rationale: P1 showed that artifact organization matters. The P2 artifact tree should remain one subtree with one first-read index and a manifest. Severe false-negative evidence belongs under `evidence/`, not scattered across candidate folders.

Alternative considered: place P2 outputs inside `source_aware_estimator/`. Rejected because P2 has a distinct target: severe threshold and ordinal behavior. Mixing it into P1 would obscure the difference between source-aware calibration and severe-end failure analysis.

### Decision: Audit severe separability before candidate promotion

The first P2 stage should answer whether score `2/3` rows are visibly separable in current features.

Minimum separability audit:

- row and subject support for `score >= 1.5`, `score >= 2`, and `score >= 3`
- support by `cohort_id`
- feature-family availability and feature counts
- severe/non-severe distribution summaries for ROI/QC, morphology, learned ROI, and embedding-derived feature families
- simple subject-heldout threshold screens where support permits
- feature-warning diagnostics, including nonfinite, zero variance, near-zero variance, rank or singular-value issues where feasible

Rationale: If severe cases are not separable in the measured feature space, no modeling family can honestly recover them. If they are separable but regression shrinks them downward, threshold-aware modeling is justified.

Alternative considered: immediately fit more complex ordinal or nonlinear candidates. Rejected for first pass because the data are sparse and source-confounded; a separability audit prevents blind model shopping.

### Decision: Treat severe false negatives as a primary failure mode

P2 model selection should not be driven solely by overall grade MAE or stage-index MAE. The key severe-specific metrics should include:

- false-negative rate for `score >= 2`
- false-negative count for `score >= 2`
- severe recall/sensitivity for `score >= 2`
- severe precision/positive predictive value where estimable
- threshold calibration where estimable
- severe prediction-set coverage
- severe examples predicted below `1.5` or below `2`

Rationale: P1's average metrics hid the clinically/scientifically important failure: high-grade rows were pulled into the low/mid range. P2 must make that failure impossible to miss.

Alternative considered: optimize average prediction-set size and coverage only. Rejected because prediction sets can have acceptable overall coverage while still being unhelpful for severe disease.

### Decision: Use ordinal/threshold candidates before adding more embeddings

The first P2 candidate families should be low-capacity, explicit, and severe-aware:

- severe-threshold logistic or regularized linear classifier for `score >= 1.5`
- severe-threshold logistic or regularized linear classifier for `score >= 2`
- `score >= 3` threshold only if support permits
- cumulative-threshold ordinal model for score thresholds
- two-stage model: severe-risk gate plus calibrated burden or ordinal prediction
- subject-level severe-aware aggregation candidate

Candidate feature families should be decided after the audit. Safe first-pass candidates are expected to include ROI/QC and morphology; learned/embedding features should be included only as constrained comparators if the audit shows they do not dominate warnings or overfit.

Rationale: The ordered score rubric already encodes threshold structure. Modeling thresholds directly is more aligned with the label than unconstrained scalar regression.

Alternative considered: add a deep ordinal model. Rejected for P2 because there are only 60 independent subjects and all severe labels are source-confounded.

### Decision: Keep split semantics strict

P2 must keep:

```text
training_apparent                      optimistic diagnostic only
validation_subject_heldout             primary current-data selection split
testing_explicit_heldout               only if predeclared independent heldout exists
testing_not_available_current_data_sensitivity
                                       required when no explicit heldout test exists
```

Rationale: P1 established this language because it prevents accidental overclaiming. P2 should inherit it unchanged.

Alternative considered: call leave-source-out testing. Rejected. Leave-source-out is useful, but with two confounded sources it is current-data sensitivity, not independent external validation.

### Decision: Use current workflow and config

P2 should run from:

```text
PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/endotheliosis_quantification.yaml
```

The evaluator should be integrated after burden, learned ROI, and source-aware artifacts are available.

Rationale: The user needs one runnable control surface. P1 already integrated into the existing workflow. P2 should not create a parallel manual workflow unless an audit proves the current integration point is unsafe.

Alternative considered: create `configs/severe_aware_ordinal_quantification.yaml`. Deferred. A separate config may be useful later for experiments, but the current spec should keep the main quantification command complete.

### Decision: Gate manual annotation and segmentation-backbone escalation

P2 should not assume that new Label Studio patch/mask annotation or a stronger segmentation backbone is required. Those are plausible future directions, but they answer a different question than severe-aware grading:

```text
Failure source?                         Appropriate response
────────────────────────────────────────────────────────────────────
Bad/missing glomerulus ROI extraction   segmentation audit/backbone comparison
Good ROI but weak severe signal          severe-aware ordinal/threshold modeling
Good ROI and weak label separability     target collapse / uncertainty / aggregation
MR TIFF-specific ROI failure             targeted MR patch review before annotation
```

Better segmenters can be tested without changing masks by training or evaluating alternative architectures/backbones on the existing mask contract. Plausible families include a better-configured U-Net/nnU-Net-style baseline, DeepLabV3+-style semantic segmentation, Mask2Former-style mask classification, and prompt/foundation segmenters such as SAM or MedSAM used as audit or fine-tuning candidates.

The nuance is that better segmentation may improve grading if it exposes severity-correlated glomerular structure that the current ROI/mask representation blurs, clips, merges, fragments, or measures poorly. A better mask or component decomposition could change lumen openness, collapsed-capillary morphology, component topology, boundary irregularity, pale/open-space measurements, or other severity-correlated features. In that setting, segmentation improvement is not merely an upstream convenience; it changes the feature space available to the severe-aware estimator.

The limit is narrower: a better segmenter will not fix score `2/3` underprediction if severe grades remain non-separable after correct glomerulus localization and after extracting the relevant morphology from those ROIs. P2 should therefore test whether segmentation-derived feature changes improve severe separability and severe false negatives rather than assuming either direction.

MedSAM is especially interesting because it is promptable and medically adapted. But the original MedSAM framing is not "type glomerulus and get glomeruli." It is primarily visual-prompt driven, especially bounding-box prompted. That makes it testable with current data:

```text
Existing image + manual mask
        │
        ├── oracle box from manual mask
        │       └── MedSAM mask = upper-bound segmentation with answer-shaped prompt
        │
        └── automatic box from current glomerulus proposals
                └── MedSAM mask = realistic deployable prompt path
```

The oracle-box result answers: "If the glomerulus is localized correctly, can MedSAM trace the boundary better than the current mask/model?" The automatic-prompt result answers the operational question: "Can we generate prompts from the current pipeline or MR TIFF tiling well enough for MedSAM to improve ROI extraction without manual boxes?"

The MedSAM/backbone audit should therefore measure at least four layers:

- prompt generation quality: proposal recall, proposal precision, missed glomeruli, duplicate boxes
- segmentation quality against existing masks: Dice, IoU, boundary error, component count, area ratio, and failure category
- ROI-feature stability and signal: changes in ROI area, fill fraction, openness, morphology features, component topology, candidate closed/open-lumen features, and embedding features
- downstream severe behavior: whether changed masks reduce severe false negatives or only change geometry without improving grading

Current local feasibility snapshot on `eq-mac`:

```text
Package / family                  Current availability
────────────────────────────────────────────────────────────
torch                             installed
torchvision                       installed
fastai                            installed
opencv                            installed
albumentations                    installed
timm                              not installed
MONAI                             not installed
segmentation_models_pytorch       not installed
nnunetv2                          not installed
transformers                      not installed
detectron2                        not installed
segment_anything                  not installed
medsam                            not installed
```

This means upstream segmentation comparison must start with a feasibility inventory, not training. The comparison families should be triaged as:

```text
Family                  How it would actually get done
─────────────────────────────────────────────────────────────────────────────
Current FastAI U-Net     Already available; rerun or tune current candidate workflow.
Torchvision DeepLabV3    Available through torchvision; implement adapter/training loop
                         if selected. DeepLabV3+ would require extra dependency or custom
                         implementation.
nnU-Net v2               Requires installing nnunetv2 and converting current image/mask
                         data to nnU-Net raw/preprocessed/results layout.
Mask2Former-style        Requires transformers or detectron2 stack; likely heavier and
                         should be feasibility-only unless clear upside exists.
MedSAM/SAM               Requires installing MedSAM/SAM code/weights; run oracle-box and
                         automatic-box inference before considering fine-tuning.
```

The feasibility inventory should record exact install candidates, whether they are compatible with macOS/MPS or require WSL/CUDA, whether network weight downloads are required, where isolated environments would live, and how outputs would be compared without polluting the main codebase or active runtime tree.

The spec therefore treats upstream segmentation escalation as a future decision gate:

- first inspect severe false negatives and MR TIFF ROI failures
- determine whether errors localize to bad ROI extraction, missed glomeruli, bad masks, or ambiguous severe labels
- only then decide whether to create a separate segmentation-backbone comparison or a targeted Label Studio annotation batch

Rationale: dense patch/mask annotation is expensive and risks solving the wrong problem. The smallest useful annotation round would be targeted active-learning review of severe false negatives and uncertain MR TIFF ROIs, not broad manual masking of large TIFFs.

Alternative considered: start a new manual patch/mask dataset now. Rejected for P2 because current P1 data have `707/707` usable ROI rows, and the severe failure appears at the grading/feature/model target rather than a proven ROI extraction failure.

## Risks / Trade-offs

- [Risk] Severe labels are too sparse or too source-confounded to estimate `score >= 3`.
  - Mitigation: make `score >= 3` support an audit-first decision. If underpowered, report it as non-estimable rather than fitting a misleading classifier.

- [Risk] Optimizing severe recall could inflate false positives and make low/mid estimates worse.
  - Mitigation: report severe recall and severe precision together; require split-specific metrics; separate severe-risk output from scalar burden if needed.

- [Risk] Learned embeddings may again dominate apparent training metrics and mislead model selection.
  - Mitigation: require subject-heldout metrics as primary; require apparent metrics to be ineligible for model selection; require warning and overfit diagnostics by feature family.

- [Risk] Source-aware adjustment could hide that severe support comes only from VEGFRi/Dox.
  - Mitigation: report threshold support and severe false-negative metrics by source; label leave-source-out as current-data sensitivity only.

- [Risk] Artifact bloat could reappear.
  - Mitigation: cap first-pass artifacts through `summary/artifact_manifest.json`; put exhaustive tables in `internal/`; require `INDEX.md` as the first-read artifact.

- [Risk] A two-stage model can become difficult to explain.
  - Mitigation: verdict must say whether the reportable output is scalar burden, severe-risk label, ordinal prediction set, subject-level aggregate, or limited/non-reportable.

- [Risk] A stronger segmenter may improve ROI extraction but not severe-grade prediction.
  - Mitigation: require severe false-negative review, ROI-failure localization, and testing of whether alternative masks improve severity-correlated feature extraction before starting a segmentation-backbone or annotation change.

- [Risk] Manual patch/mask annotation could consume substantial effort without changing the limiting signal.
  - Mitigation: require a small targeted annotation pilot only after P2 identifies annotation as the likely bottleneck.

## Migration Plan

1. Archive and sync P1 before starting P2. Completed before this proposal: `p1-source-aware-endotheliosis-estimator` archived as `openspec/changes/archive/2026-04-28-p1-source-aware-endotheliosis-estimator/`.
2. Implement P2 as additive. Do not remove P1 source-aware artifacts or change their meaning.
3. Add the P2 evaluator behind the existing `endotheliosis_quantification` workflow.
4. Write P2 artifacts under `burden_model/severe_aware_ordinal_estimator/`.
5. Integrate P2 verdict links into `quantification_review/` without adding README-snippet claims unless the final verdict explicitly permits them.
6. Validate with focused unit tests, full `pytest`, strict OpenSpec validation, explicitness check, and the full `eq run-config` workflow.

Rollback strategy: because P2 is additive and contained under a new output subtree, rollback means removing the P2 pipeline call and generated combined-review references while leaving P1 source-aware outputs intact.

## Explicit Decisions

- Change ID: `p2-severe-aware-ordinal-endotheliosis-estimator`.
- Module: `src/eq/quantification/severe_aware_ordinal_estimator.py`.
- Public evaluator: `evaluate_severe_aware_ordinal_endotheliosis_estimator`.
- Output root: `burden_model/severe_aware_ordinal_estimator/`.
- Main workflow: `endotheliosis_quantification`.
- Main config: `configs/endotheliosis_quantification.yaml`.
- Primary validation grouping key: `subject_id`.
- Primary source/context field: `cohort_id`.
- Primary severe threshold: `score >= 2`.
- Secondary threshold: `score >= 1.5`.
- Tail threshold: `score >= 3` only if support permits.
- Primary severe failure metric: `score >= 2` false-negative rate and count.
- First-read artifact: `burden_model/severe_aware_ordinal_estimator/INDEX.md`.
- Verdict artifact: `burden_model/severe_aware_ordinal_estimator/summary/estimator_verdict.json`.
- README snippet eligibility default: `false`.

## Open Questions

- [audit_first_then_decide] Should implementation reuse current cumulative-threshold code in `src/eq/quantification/burden.py` or keep P2 candidates fully contained in `src/eq/quantification/severe_aware_ordinal_estimator.py`? Decide after auditing existing threshold helpers, warning sources, and whether reuse would create confusing dual semantics.
- [audit_first_then_decide] Is `score >= 3` estimable as a separate threshold? Decide from `threshold_support.json` row count, subject count, source support, and positive/negative support.
- [audit_first_then_decide] Which feature families are allowed into final P2 candidate selection? Decide from `diagnostics/severe_separability_audit.json`, feature warning diagnostics, and subject-heldout overfit comparisons.
- [audit_first_then_decide] Is upstream segmentation or MR TIFF ROI extraction limiting severe estimation enough to justify a separate segmentation-backbone comparison or Label Studio annotation batch? Decide from `evidence/severe_false_negative_review.html`, `diagnostics/severe_separability_audit.json`, ROI adequacy diagnostics, and MR TIFF ROI extraction evidence.
- [audit_first_then_decide] Is a MedSAM or promptable-SAM audit warranted as a separate upstream comparison? Decide from oracle-prompt versus automatic-prompt feasibility, expected compute/dependency burden, existing mask comparability, and whether promptable masks would change severe false-negative behavior rather than only mask geometry.
- [audit_first_then_decide] Which upstream segmentation families are feasible enough to compare in a later segmentation-specific OpenSpec change? Decide from a local dependency inventory, install/runtime constraints, dataset-conversion requirements, expected compute cost, and whether each family can consume existing masks without additional annotation.
- [defer_ok] Whether P2 results become README-snippet eligible is deferred until `summary/estimator_verdict.json` exists after the full runtime workflow.

## Literature Notes

- TRIPOD+AI supports the reporting stance: prediction models need transparent validation, calibration, and clear limits around internal versus external validation.
- CORAL-style ordinal regression supports the design idea that ordered severity labels should be modeled with rank/threshold structure rather than unconstrained nominal classes or scalar-only regression.
- Ordinal conformal prediction in medical imaging supports reporting severity prediction sets when single-label severity predictions are uncertain.
- nnU-Net supports the idea that biomedical segmentation gains often come from robust configuration and validation rather than architectural novelty alone.
- SAM and MedSAM support promptable/foundation segmentation as possible audit or adaptation tools, but they do not remove the need for task-specific validation on this MR TIFF/glomerulus setting.
- Torchvision exposes DeepLabV3 models in the installed stack, making it the lowest-friction non-FastAI architecture family to prototype if a segmentation comparison is justified.
- nnU-Net v2, Mask2Former-style models, and MedSAM/SAM are not current project dependencies; they require explicit feasibility and isolation before use.
