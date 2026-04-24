## Context

The current glomeruli workflow has the right high-level shape but insufficient evidence hygiene. Training uses `src/eq/data_management/datablock_loader.py` for full-image dynamic patching, `src/eq/training/train_glomeruli.py` and `src/eq/training/transfer_learning.py` for transfer/no-base candidate training, and `src/eq/training/compare_glomeruli_candidates.py` plus `src/eq/training/promotion_gates.py` for promotion evidence. The latest README-facing table cites a report under `$EQ_RUNTIME_ROOT/output/segmentation_evaluation/glomeruli_candidate_comparison/all_manual_mask_glomeruli_seed42/`, but the deterministic panel can include images used during training and is foreground-heavy.

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

## Goals / Non-Goals

**Goals:**

- Create pytest-backed validation checks for glomeruli segmentation decision gates.
- Make promotion evidence held-out-only or explicitly not promotion-eligible.
- Audit the actual DataBlock item set, train/validation split, positive-aware crop sampling, augmentation alignment, resizing/preprocessing, prediction thresholding, and per-category metrics.
- Add gates that catch the failure mode where broad foreground masks score well on foreground-heavy crops.
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

3. **Promotion panels must be held-out-only.**
   - Decision: candidate comparison must build deterministic evaluation manifests from explicit held-out image paths, not from the full admitted `raw_data/cohorts` item set.
   - Rationale: current reports can include training images; no promotion-facing metric should mix training and evaluation examples.
   - Alternative rejected: report overlap but allow promotion. That still permits inflated front-page metrics.

4. **Split provenance becomes part of artifact support.**
   - Decision: glomeruli model sidecars and split manifests must record `train_images`, `valid_images`, `split_seed`, `splitter_name`, `data_root`, `manifest_rows`, lane/cohort counts, and enough image identifiers to compare against later promotion manifests.
   - Rationale: promotion gates cannot detect leakage unless split provenance is machine-readable.
   - Alternative rejected: infer split membership later from current filesystem state. That is not stable after cohort or manifest changes.

5. **Foreground-heavy evidence is audited, not normalized away.**
   - Decision: reports must disclose foreground-fraction distributions by category and mark promotion-facing claims as not eligible when the deterministic panel lacks meaningful background/edge/interior balance.
   - Rationale: glomeruli crops are naturally foreground-rich when centered on a glomerulus; the report must make this limitation explicit and prevent all-foreground-like wins.
   - Alternative rejected: reweight Dice silently. Weighting can be useful later, but it does not replace transparent evidence composition.

6. **Add explicit oversegmentation and background false-positive gates.**
   - Decision: promotion review must include per-category false-positive burden, prediction/ground-truth foreground-ratio summaries, connected-component/shape plausibility summaries, and background crop foreground limits.
   - Rationale: the observed high-recall/high-prediction-fraction pattern can pass permissive whole-glomerulus gates while still being scientifically weak.
   - Alternative rejected: rely on aggregate Dice/Jaccard and visual panels. Aggregate metrics can obscure category-specific failure modes.

7. **Audit preprocessing and augmentation parity.**
   - Decision: the audit must verify that deterministic evaluation uses learner-consistent preprocessing, that image/mask transforms stay aligned, and that crop/resize semantics match the training provenance.
   - Rationale: a technically valid metric requires the same geometric and normalization assumptions used by the learner.
   - Alternative rejected: treat a successful `learn.predict` call as enough. Runtime success does not prove label alignment or statistical validity.

8. **Documentation claims are gated artifacts.**
   - Decision: add `documentation_claim_audit.md` to the validation-audit output and require README/onboarding performance tables to cite only reports that pass the audit.
   - Rationale: the public front door should not present internal, biased, or not-promotion-eligible evidence as a current performance claim.
   - Alternative rejected: add more caveat prose around the existing table. Caveats are not enough when the cited evidence is structurally biased.

9. **Use two status axes instead of implying unusability.**
   - Decision: reports should distinguish `runtime_use_status` from `promotion_evidence_status`. A model can be `available_research_use` while its evidence is `not_promotion_eligible`, `audit_missing`, or `insufficient_evidence_for_promotion`.
   - Rationale: scratch and transfer are currently the only available glomeruli candidates, so failing a promotion gate should not imply that the artifacts are unusable for exploratory work or pipeline development.
   - Alternative rejected: use `blocked` as the only status. That is too blunt and conflates scientific promotion with ordinary research/runtime use.

## Risks / Trade-offs

- [Risk] The current admitted cohort may not support a strict subject-held-out positive/boundary/background panel with enough examples. -> Mitigation: fail with `insufficient_evidence_for_promotion_validation` and report the limiting category rather than relaxing the gate silently.
- [Risk] Running full real-model audits on macOS/MPS may require unsandboxed execution. -> Mitigation: keep pure artifact/schema checks sandbox-testable and document the exact unsandboxed `eq-mac` command for real model validation.
- [Risk] Adding audit gates could temporarily remove the README performance table. -> Mitigation: prefer no front-page performance claim over a misleading one; docs can still link to research-use-only or not-promotion-eligible audit evidence as limitations.
- [Risk] More report artifacts increase maintenance burden. -> Mitigation: use one schema-owned audit payload and derive Markdown/HTML/CSV views from it.
- [Risk] Split manifests from older artifacts may be incomplete. -> Mitigation: classify those artifacts as compatibility-only for promotion until rerun through the hardened provenance contract.

## Migration Plan

1. Add audit helper functions and pytest fixtures for synthetic and fixture-backed validation.
2. Extend glomeruli training sidecars and split manifests with split, sampler, DataBlock, crop, resize, augmentation, and preprocessing provenance.
3. Update candidate comparison to consume held-out split manifests and to mark promotion evidence as not eligible when candidate artifacts lack sufficient split provenance.
4. Add prediction-shape and foreground-burden gates to promotion reports.
5. Add documentation-claim tests and update README/onboarding wording to cite only audit-passing held-out evidence.
6. Run focused tests, the optional runtime integration test when local artifacts matter, and OpenSpec validation. Real MPS model validation should be executed unsandboxed with `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python` when Metal access matters.

## Explicit Decisions

- No new workflow ID, committed audit YAML, `eq run-config` dispatch, or user-facing audit CLI is introduced.
- Helper module: `src/eq/training/segmentation_validation_audit.py`.
- Primary regression tests: `tests/test_segmentation_validation_audit.py`.
- Optional local-runtime integration tests: `tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py`.
- Pytest-generated report payloads use pytest temporary directories. Persistent promotion evidence remains in candidate-comparison artifacts, not a separate audit workflow output tree.
- Status fields: `runtime_use_status` records whether an artifact is available for research/runtime use; `promotion_evidence_status` records whether the evidence is `promotion_eligible`, `not_promotion_eligible`, `audit_missing`, or `insufficient_evidence_for_promotion`.
- Hard promotion-ineligibility triggers: any train/evaluation image overlap, any train/evaluation subject overlap when subject IDs are reliable, missing split provenance, missing category support, failure to beat trivial baselines, overcoverage on positive-like crops, excessive foreground on background crops, preprocessing mismatch, and audit-failed documentation claims.

## Open Questions

- [audit_first_then_decide] Whether the current admitted masked cohort supports subject-held-out validation with all required categories. Resolve from `tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py` when run against the current ProjectsRuntime artifacts.
- [defer_ok] Whether future curated negative-crop supervision should become mandatory before scientific promotion. This change records the limitation and remains compatible with `p2-add-negative-glomeruli-crop-supervision`.
