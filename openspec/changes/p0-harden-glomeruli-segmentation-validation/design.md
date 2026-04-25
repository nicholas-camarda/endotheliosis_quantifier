## Context

The current glomeruli workflow has the right high-level shape but insufficient evidence hygiene. Training uses `src/eq/data_management/datablock_loader.py` for full-image dynamic patching, `src/eq/training/train_glomeruli.py` and `src/eq/training/transfer_learning.py` for transfer/no-base candidate training, and `src/eq/training/compare_glomeruli_candidates.py` plus `src/eq/training/promotion_gates.py` for promotion evidence. The full workflow is controlled by `configs/full_segmentation_retrain.yaml` and writes comparison evidence under `$EQ_RUNTIME_ROOT/output/segmentation_evaluation/glomeruli_candidate_comparison/latest_run/`. Promotion-facing documentation must not cite that report unless the regenerated artifacts pass the hardened audit gates.

The immediate problem is not just one metric. It is the chain from data admission through DataBlock sampling, synchronized crop/augmentation behavior, preprocessing parity, candidate training, deterministic evaluation, promotion gates, and documentation claims.

```
raw_data/cohorts/manifest.csv
        |
        v
admitted image/mask rows
        |
        +--> training split manifest -----> DataBlock dynamic patches -----> model artifact
        |                                      |                                |
        |                                      v                                v
        |                               sampler/augmentation audit       provenance sidecar
        |
        +--> held-out evaluation manifest --> deterministic panels -----> promotion report
                                                |
                                                v
                                      README/docs claim audit
```

The Research Partner review stance for this change is:

- implementation audit: verify code implements the intended split, sampler, transform, and report contracts
- statistical validity: verify evaluation is held-out, not foreground-biased, and not a trivial-overcoverage win
- scientific interpretation: prevent internal Dice from being described as external validity or scientific promotion
- robustness tests: create regression tests for silent leakage, misaligned transforms, threshold drift, and degenerate masks
- documentation consistency: ensure README/onboarding claims cite only eligible evidence

Audit findings and production decisions are tracked in `audit-results.md`. Chat discussion is not authoritative unless the result is recorded there or in the capability specs.

## Goals / Non-Goals

**Goals:**

- Create pytest-backed validation checks for glomeruli segmentation decision gates.
- Make promotion evidence held-out-only or explicitly not promotion-eligible.
- Audit the actual DataBlock item set, train/validation split, positive-aware crop sampling, augmentation alignment, resizing/preprocessing, prediction thresholding, and per-category metrics.
- Add gates that catch the failure mode where broad foreground masks score well on foreground-heavy crops.
- Treat crop-to-network resizing as an explicit validation target rather than assuming `512 -> 256` helps.
- Audit the mitochondria pretraining base because it is part of the transfer-learning treatment, even if mitochondria segmentation is not the downstream prediction target.
- Require exported glomeruli artifacts and comparison reports to expose split and sampler provenance.
- Prevent README/onboarding performance tables from citing compatibility-only, research-use-only, partly in-sample, not-promotion-eligible, or audit-failed evidence as current model performance.

**Non-Goals:**

- Do not add a new segmentation architecture.
- Do not tune model hyperparameters or decide that transfer/no-base should win.
- Do not implement curated negative-crop supervision; that remains in `p2-add-negative-glomeruli-crop-supervision`.
- Do not resurrect static patch datasets or add compatibility shims for legacy FastAI pickle artifacts.
- Do not claim external validation from internal held-out panels.

## Decisions

1. **Make the audit a pytest contract, not a user-facing workflow.**
   - Decision: add `src/eq/training/segmentation_validation_audit.py` plus tests in `tests/test_segmentation_validation_audit.py` and `tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py`.
   - Rationale: this is validation of correctness, not a research workflow users should choose from the README. The normal interface for “is the loader/split/promotion contract valid?” should be `pytest`.
   - Run triggers: the focused tests should run during implementation and before merge. The optional runtime integration test should be run after training new scratch/transfer candidates, changing `raw_data/cohorts/manifest.csv` or lane admission, modifying `src/eq/data_management/datablock_loader.py`, modifying augmentation/preprocessing/threshold code, changing candidate-comparison gates, before using a segmentation artifact as an upstream dependency for downstream quantification, and before adding or refreshing README/onboarding current-performance claims.
   - Alternative rejected: add `configs/glomeruli_segmentation_validation_audit.yaml`. That would make an internal validation gate look like a user-facing research workflow and dilute the role of configs as workflow convenience utilities.
   - Alternative rejected: only patch `compare_glomeruli_candidates.py`. That would improve one path but would not audit DataBlock, training provenance, or documentation claims.

2. **Keep audit helpers separate from the runner.**
   - Decision: implement reusable audit functions in `src/eq/training/segmentation_validation_audit.py`.
   - Rationale: tests can exercise split overlap, crop-distribution, transform-alignment, preprocessing, and prediction-shape logic without running an entire MPS training workflow.
   - Alternative rejected: embed audit logic in a CLI runner. That would make regression tests brittle and hide reusable checks from candidate comparison.

3. **Reproduce the failing visual evidence before classifying performance.**
   - Decision: the audit must start from the exact current failing mitochondria, glomeruli transfer, and glomeruli no-base/scratch prediction panels or their underlying artifact paths, then reconstruct the model artifact, source image, source mask, crop box, preprocessing, resize policy, threshold, prediction tensor shape, and overlay-generation path for each displayed example.
   - Rationale: the observed failure could come from true model weakness, wrong image/mask pairing, crop misalignment, class-channel/threshold mistakes, resize/threshold artifacts, or an invalid comparison panel. The spec must make the first output a traceable reproduction dossier rather than a guess.
   - Required output: `failure_reproduction_audit.csv` or equivalent structured payload with one row per panel example and a root-cause status for each candidate family.
   - Alternative rejected: retrain immediately from the current panels. Retraining before reproducing the failure can hide transform or evaluation bugs and waste the limited training budget.

4. **Promotion panels must be held-out-only.**
   - Decision: candidate comparison must build deterministic evaluation manifests from explicit held-out image paths, not from the full admitted `raw_data/cohorts` item set.
   - Rationale: current reports can include training images; no promotion-facing metric should mix training and evaluation examples.
   - Alternative rejected: report overlap but allow promotion. That still permits inflated front-page metrics.

5. **Split provenance becomes part of artifact support.**
   - Decision: glomeruli model sidecars and split manifests must record `train_images`, `valid_images`, `split_seed`, `splitter_name`, `data_root`, `manifest_rows`, lane/cohort counts, and enough image identifiers to compare against later promotion manifests.
   - Rationale: promotion gates cannot detect leakage unless split provenance is machine-readable.
   - Alternative rejected: infer split membership later from current filesystem state. That is not stable after cohort or manifest changes.

6. **Foreground-heavy evidence is audited, not normalized away.**
   - Decision: reports must disclose foreground-fraction distributions by category and mark promotion-facing claims as not eligible when the deterministic panel lacks meaningful background/edge/interior balance.
   - Rationale: glomeruli crops are naturally foreground-rich when centered on a glomerulus; the report must make this limitation explicit and prevent all-foreground-like wins.
   - Alternative rejected: reweight Dice silently. Weighting can be useful later, but it does not replace transparent evidence composition.

7. **Add explicit oversegmentation and background false-positive gates.**
   - Decision: promotion review must include per-category false-positive burden, prediction/ground-truth foreground-ratio summaries, connected-component/shape plausibility summaries, and background crop foreground limits.
   - Rationale: the observed high-recall/high-prediction-fraction pattern can pass permissive whole-glomerulus gates while still being scientifically weak.
   - Alternative rejected: rely on aggregate Dice/Jaccard and visual panels. Aggregate metrics can obscure category-specific failure modes.

8. **Audit preprocessing and augmentation parity.**
   - Decision: the audit must verify that deterministic evaluation uses learner-consistent preprocessing, that image/mask transforms stay aligned, and that crop/resize semantics match the training provenance.
   - Rationale: a technically valid metric requires the same geometric and normalization assumptions used by the learner.
   - Alternative rejected: treat a successful `learn.predict` call as enough. Runtime success does not prove label alignment or statistical validity.

9. **Make resizing a sensitivity audit, not a hidden convenience.**
   - Decision: the audit must record and test the active resize policy: source image and mask dimensions, `crop_size`, `output_size`, crop-to-output resize ratio, aspect-ratio handling, resize method, interpolation semantics for images and masks, mask binarization after resize, prediction resize-back behavior, and threshold/resize ordering. For the current glomeruli workflow, the primary policy to audit is `crop_size=512`, `output_size=256`, `ResizeMethod.Squish`, followed by prediction resize-back to the deterministic crop shape.
   - Rationale: resizing can change the effective biological scale, boundary thickness, foreground fraction, and small/background crop behavior. It can also make oversegmentation look better or worse depending on the interpolation and threshold path. A high Dice value after downsampling and resize-back is not enough to prove the resolution choice is doing useful work.
   - Required evidence: held-out deterministic evaluation must report metrics, prediction-shape summaries, and source-resolution distributions by resize policy when a candidate is proposed for promotion or README-facing performance claims. At minimum, the current policy must be compared against a no-downsample or less-downsample sensitivity when local memory permits; if not feasible, the report must mark resize benefit as unproven and keep promotion evidence below `promotion_eligible`.
   - Alternative rejected: only record `image_size` and `crop_size` in provenance. That proves reproducibility, but it does not answer whether the resizing choice is improving the model or masking a failure mode.
   - Alternative rejected: infer resize quality from training curves. Training-time validation crops are stochastic and can be foreground-heavy, so they cannot answer whether the resolution policy supports promotion-facing claims.

10. **Classify the root cause before allowing remediation training.**
   - Decision: the validation report must assign each poor-performance finding to one or more explicit root-cause classes before any fresh candidate training is treated as the solution: `image_mask_pairing_error`, `transform_alignment_error`, `mask_binarization_error`, `class_channel_or_threshold_error`, `resize_policy_artifact`, `split_or_panel_bias`, `training_signal_insufficient`, `mitochondria_base_defect`, `negative_background_supervision_missing`, or `true_model_underfit`.
   - Rationale: the fix is different for each class. Pairing, transform, threshold, or resize bugs require code correction and reevaluation; training-signal or negative-background deficits may require P2; true underfit may require retraining; mitochondria-base defects require base retraining or an all-available-pretraining decision.
   - Remediation rule: candidate retraining is not an acceptable completion step until the report identifies which class is active and records the intended remediation path.
   - Alternative rejected: one broad "poor performance" status. That does not tell the operator whether to fix code, data, evaluation, supervision, or training.

11. **Documentation claims are gated artifacts.**
   - Decision: add `documentation_claim_audit.md` to the validation-audit output and require README/onboarding performance tables to cite only reports that pass the audit.
   - Rationale: the public front door should not present internal, biased, or not-promotion-eligible evidence as a current performance claim.
   - Alternative rejected: add more caveat prose around the existing table. Caveats are not enough when the cited evidence is structurally biased.

12. **Audit mitochondria pretraining as transfer provenance.**
   - Decision: the validation audit must inspect the mitochondria base artifact used for glomeruli transfer and record its training data scope, split policy, resize/preprocessing policy, training command, artifact path, and whether the physical Lucchi `testing/` root was preserved as held-out or included as representation-pretraining data.
   - Rationale: the transfer candidate is not just "glomeruli training"; its initialization depends on the mitochondria base. If the base artifact is weak, stale, trained with different preprocessing, or trained with a different data-scope contract than reported, the transfer comparison is not interpretable.
   - Transfer-pretraining rule: using all available mitochondria `training/` and `testing/` image/mask pairs may be allowed for a representation base when the mitochondria model is not used for mitochondria inference, not used for a mitochondria performance claim, and not selected by peeking at a held-out mitochondria metric. In that mode, the artifact must be labeled `mitochondria_training_scope=all_available_pretraining`, and any mitochondria held-out metric must be marked `not_applicable_for_inference_claim`.
   - Held-out-mito rule: preserving `raw_data/mitochondria_data/testing` as held-out remains required whenever the report claims mitochondria segmentation performance, uses mitochondria testing metrics for model selection, or describes the base artifact as validated on mitochondria.
   - Alternative rejected: silently fold the physical mitochondria testing split into training. That would be statistically acceptable only for representation pretraining with the correct claim boundary; without explicit labeling it would look like test leakage.

13. **Use two status axes instead of implying unusability.**
   - Decision: reports should distinguish `runtime_use_status` from `promotion_evidence_status`. A model can be `available_research_use` while its evidence is `not_promotion_eligible`, `audit_missing`, or `insufficient_evidence_for_promotion`.
   - Rationale: scratch and transfer are currently the only available glomeruli candidates, so failing a promotion gate should not imply that the artifacts are unusable for exploratory work or pipeline development.
   - Alternative rejected: use `blocked` as the only status. That is too blunt and conflates scientific promotion with ordinary research/runtime use.

## Risks / Trade-offs

- [Risk] The current admitted cohort may not support a strict subject-held-out positive/boundary/background panel with enough examples. -> Mitigation: fail with `insufficient_evidence_for_promotion_validation` and report the limiting category rather than relaxing the gate silently.
- [Risk] Running full real-model audits on macOS/MPS may require unsandboxed execution. -> Mitigation: keep pure artifact/schema checks sandbox-testable and document the exact unsandboxed `eq-mac` command for real model validation.
- [Risk] Resize sensitivity may require expensive reruns at larger output sizes. -> Mitigation: make the full runtime comparison optional for exploratory use but mandatory before `promotion_eligible` or README-facing performance claims; if memory limits prevent a fair comparison, report `resize_benefit_unproven` rather than inventing a conclusion.
- [Risk] Using all mitochondria data for base pretraining could be mistaken for a validated mitochondria model. -> Mitigation: add explicit `mitochondria_training_scope` and `mitochondria_inference_claim_status` fields and block any mitochondria performance claim when the physical testing root was included in training.
- [Risk] Adding audit gates could temporarily remove the README performance table. -> Mitigation: prefer no front-page performance claim over a misleading one; docs can still link to research-use-only or not-promotion-eligible audit evidence as limitations.
- [Risk] More report artifacts increase maintenance burden. -> Mitigation: use one schema-owned audit payload and derive Markdown/HTML/CSV views from it.
- [Risk] Split manifests from older artifacts may be incomplete. -> Mitigation: classify those artifacts as compatibility-only for promotion until rerun through the hardened provenance contract.

## Migration Plan

1. Add audit helper functions and pytest fixtures for synthetic and fixture-backed validation.
2. Add current-failure reproduction audit outputs for the mitochondria, glomeruli transfer, and glomeruli no-base/scratch panels.
3. Extend mitochondria base sidecars with training-scope, split-policy, resize/preprocessing, and inference-claim provenance.
4. Extend glomeruli training sidecars and split manifests with split, sampler, DataBlock, crop, resize, augmentation, interpolation, prediction resize-back, transfer-base, and preprocessing provenance.
5. Update candidate comparison to consume held-out split manifests and to mark promotion evidence as not eligible when candidate artifacts lack sufficient split or transfer-base provenance.
6. Add resize-policy sensitivity summaries, prediction-shape gates, foreground-burden gates, root-cause classification, and mitochondria-base provenance sections to promotion reports.
7. Add documentation-claim tests and update README/onboarding wording to cite only audit-passing held-out evidence.
8. Run focused tests, the optional runtime integration test when local artifacts matter, and OpenSpec validation. Real MPS model validation should be executed unsandboxed with `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python` when Metal access matters.

## Explicit Decisions

- No new workflow ID, committed audit YAML, `eq run-config` dispatch, or user-facing audit CLI is introduced.
- Durable audit findings and production-level decisions for this change are recorded in `openspec/changes/p0-harden-glomeruli-segmentation-validation/audit-results.md`.
- Helper module: `src/eq/training/segmentation_validation_audit.py`.
- Primary regression tests: `tests/test_segmentation_validation_audit.py`.
- Optional local-runtime integration tests: `tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py`.
- Pytest-generated report payloads use pytest temporary directories. Persistent promotion evidence remains in candidate-comparison artifacts, not a separate audit workflow output tree.
- Status fields: `runtime_use_status` records whether an artifact is available for research/runtime use; `promotion_evidence_status` records whether the evidence is `promotion_eligible`, `not_promotion_eligible`, `audit_missing`, or `insufficient_evidence_for_promotion`.
- Mitochondria training-scope fields: `mitochondria_training_scope` is either `heldout_test_preserved` or `all_available_pretraining`; `mitochondria_inference_claim_status` is either `heldout_evaluable`, `not_applicable_for_inference_claim`, or `audit_missing`.
- Resize policy fields: `source_image_size`, `source_mask_size`, `crop_size`, `output_size`, `crop_to_output_resize_ratio`, `aspect_ratio_policy`, `resize_method`, `image_interpolation`, `mask_interpolation`, `mask_binarization_after_resize`, `prediction_resize_back_method`, and `threshold_resize_order`.
- Root-cause classes: `image_mask_pairing_error`, `transform_alignment_error`, `mask_binarization_error`, `class_channel_or_threshold_error`, `resize_policy_artifact`, `split_or_panel_bias`, `training_signal_insufficient`, `mitochondria_base_defect`, `negative_background_supervision_missing`, and `true_model_underfit`.
- Required resize sensitivity comparison for promotion-facing evidence: current `512 -> 256` glomeruli policy versus a no-downsample or less-downsample held-out sensitivity when feasible; otherwise the report records `resize_benefit_unproven`.
- Hard promotion-ineligibility triggers: any train/evaluation image overlap, any train/evaluation subject overlap when subject IDs are reliable, missing split provenance, missing category support, failure to beat trivial baselines, overcoverage on positive-like crops, excessive foreground on background crops, preprocessing mismatch, resize-policy mismatch, non-binary transformed masks, materially different train/held-out source-resolution distributions, unproven resize benefit when resize-dependent claims are made, and audit-failed documentation claims.

## Open Questions

- [audit_first_then_decide] Whether the current admitted masked cohort supports subject-held-out validation with all required categories. Resolve from `tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py` when run against the current ProjectsRuntime artifacts.
- [audit_first_then_decide] Whether the mitochondria base should preserve the physical `raw_data/mitochondria_data/testing` split or use all available `training/` plus `testing/` pairs for representation pretraining. Resolve by inspecting whether any current report, README table, or workflow depends on mitochondria held-out performance; if none does, `all_available_pretraining` is allowed with mitochondria inference claims disabled.
- [defer_ok] Whether future curated negative-crop supervision should become mandatory before scientific promotion. This change records the limitation and remains compatible with `p2-add-negative-glomeruli-crop-supervision`.
