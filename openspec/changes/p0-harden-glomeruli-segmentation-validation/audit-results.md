## Audit Results And Production Decision Log

This file is part of the P0 audit contract. Production-level decisions from this change must be recorded here or in the capability specs before they are treated as settled.

## Current Audit Snapshot

- Snapshot date: 2026-04-24
- Change: `p0-harden-glomeruli-segmentation-validation`
- Current OpenSpec progress at last audit update: `41/41` tasks complete
- Scope audited in this snapshot: schema/provenance/report-contract implementation, pytest audit surface, documentation-claim gate, subject-held-out shared split contract for future fresh comparisons, mitochondria transfer-base claim boundaries, resize-policy decision state with concrete current-artifact infeasibility evidence, validation-panel trace sidecars, old cached artifact limitations, regenerated existing-artifact subject-held-out audit, and runtime integration checks
- Scope not yet audited in this snapshot: promotion-eligible fresh retraining under the new subject-held-out split contract, real no-downsample or less-downsample resize sensitivity from newly trained artifacts, direct-run logging for non-`run-config` comparison commands, and negative/background supervision remediation

## Production Decisions Recorded

| Decision | Current Result | Production Implication | Evidence Source |
|---|---|---|---|
| Full retraining is not required to prove the audit contract | Accepted | Contract/schema/report behavior must be proven with deterministic unit and fixture tests before expensive training is used for evidence generation | `tests/test_segmentation_validation_audit.py`, `tests/test_glomeruli_candidate_comparison.py`, `tests/test_training_entrypoint_contract.py` |
| Current mitochondria base workflow preserves physical testing split | `heldout_test_preserved` | Current full workflow trains mitochondria on `raw_data/mitochondria_data/training` only; physical `testing/` remains held out unless the YAML is intentionally changed | `configs/full_segmentation_retrain.yaml` |
| Mitochondria held-out inference claim status for the current full workflow | `heldout_evaluable` | A mitochondria held-out claim is allowed only if the physical testing root was not included in fitting and the held-out evaluation is actually run | `configs/full_segmentation_retrain.yaml`, `build_mitochondria_training_provenance` |
| Current mitochondria `256x256` policy | `resize_benefit_unproven` | The policy is reproducible, not proven optimal; it must not be described as improving performance without sensitivity evidence | `configs/full_segmentation_retrain.yaml` |
| Current glomeruli `512 -> 256` policy | `resize_benefit_unproven` | Promotion-facing claims cannot imply the resize policy helps; the current artifacts have no no-downsample or less-downsample comparator artifact available | `resize_policy_audit.csv`, `resize_sensitivity_infeasibility_reason=no_no_downsample_or_less_downsample_candidate_artifact_available` |
| Old April 24 promotion report with missing candidate split provenance | `audit_missing` for promotion | It cannot support README-facing current performance claims; artifacts may remain research-use if loadable | runtime integration failure against old `promotion_report.json` |
| Cached fixedloader models and archived reports | Legacy diagnostic evidence only | Cached artifacts cannot answer whether current training/data/provenance fixes improve model performance; they may only demonstrate why old reports were audit-missing | archived runtime tree under `ProjectsRuntime/endotheliosis_quantifier/models/segmentation/.../archive` and `output/segmentation_evaluation/.../archive` |
| Regenerated current-code glomeruli candidate evidence | `available_research_use`, insufficient for promotion | The transfer and scratch artifacts are current-code research-use candidates, but the current split cannot produce a subject-held-out deterministic promotion panel, so neither can support model promotion or README-facing performance metrics | `promotion_report.json` from `output/segmentation_evaluation/glomeruli_candidate_comparison/latest_run` |
| Regenerated promotion decision | `insufficient_evidence_for_promotion` | The report should not be interpreted as "the models are unusable"; it means the current artifacts were trained under an image-level split with no validation-only subjects available for deterministic promotion evidence | `decision_state=insufficient_evidence`, `decision_reason=insufficient_heldout_category_support` |
| Fresh current-code full workflow | Partially completed outside `eq run-config` | Mitochondria training completed in the killed wrapper log; glomeruli transfer and scratch comparison were restarted directly on MPS and generated current comparison artifacts, but that direct restart did not produce a wrapper log | dry-run evidence, wrapper log, direct-run artifact mtimes, promotion report |
| Documentation current-performance claim | No front-page performance metric claim from old internal panel | README/onboarding must not cite aggregate Dice/Jaccard from audit-missing or not-promotion-eligible reports | `README.md`, `docs/ONBOARDING_GUIDE.md`, `documentation_claim_audit.md` contract |

## Implementation Audit Findings

### Split And Training Provenance

- Finding: candidate comparison now writes `shared_candidate_training_split.json` before launching fresh transfer/scratch training.
- Finding: fresh shared splits now use `splitter_name=explicit_shared_subject_split` and record `train_subjects` and `valid_subjects`, so future training cannot recreate the previous image-level split where every validation subject also appeared in training.
- Finding: `train_glomeruli.py` accepts `--split-manifest` and uses that explicit split through `fixed_splitter_from_manifest`.
- Finding: `transfer_learning.py` accepts the same explicit split surface through its transfer dataloader construction.
- Finding: glomeruli training provenance now records `train_images`, `valid_images`, `split_seed`, `splitter_name`, `manifest_rows`, lane/cohort counts, source image and mask size summaries, crop/resize policy, positive sampling settings, learner preprocessing, candidate family, transfer-base path/scope, package versions, and code state.
- Production implication: new candidate artifacts can be audited without sidecar compatibility shims or inferred filesystem reconstruction.

### Candidate Comparison Reports

- Finding: `promotion_report.json`, `promotion_report.md`, and `promotion_report.html` now expose transfer-base artifact path, mitochondria training scope, mitochondria inference-claim status, physical training/testing counts, fitted image/mask counts, and base resize policy.
- Finding: `metric_by_category.csv` now receives cohort and lane fields when deterministic manifest rows can be matched to `raw_data/cohorts/manifest.csv`.
- Finding: `documentation_claim_audit.md` is generated by candidate comparison rather than existing only as an in-memory pytest helper.
- Finding: missing source size summaries or missing transfer-base inference-claim provenance now classify the artifact as `promotion_evidence_status=audit_missing`.
- Production implication: report generation can fail closed on missing promotion-critical fields before any full training run is used to justify performance.

### Resize And Failure Reproduction

- Finding: resize policy fields are recorded and tested for parity, but real resize sensitivity has not been run for the current artifacts.
- Finding: candidate-comparison reports write `failure_reproduction_audit.csv` rows for generated review panels.
- Finding: the regenerated existing-artifact report writes concrete resize infeasibility rows to `resize_policy_audit.csv`: the current `512 -> 256` policy remains `resize_benefit_unproven` because no no-downsample or less-downsample candidate artifact is available for a fair held-out comparison.
- Finding: `failure_reproduction_audit.csv` now includes the current transfer, scratch/no-base, and mitochondria validation-panel trace sidecars when present. These rows preserve artifact path, image path, mask path, resize policy, threshold, prediction tensor shape, overlay path, and trace source.
- Finding: training validation-prediction PNG generation now writes adjacent `*_validation_prediction_trace.csv` sidecars for future mitochondria, transfer, and no-base/scratch panels. These sidecars include artifact path, image path, mask path, resize policy, threshold, prediction tensor shape, and overlay path. The current DataBlock crop coordinates used inside those old validation PNGs were not recorded at generation time, so old PNGs remain visually identifiable but not fully reconstructable.
- Finding: the archived April 24 report contains `candidate_predictions.csv` and `deterministic_validation_manifest.json`, but `promotion_report.json` points at the pre-archive manifest path and exposes no `candidate_manifest`. Its candidate provenance also lacks `train_images`, `valid_images`, and prediction tensor shape fields.
- Production implication: no promotion-facing claim may state that `512 -> 256` improves glomeruli performance; no remediation training should be justified from the old screenshots as if they were current-code evidence.

### Regenerated Current-Code Report

- Artifact root: `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/latest_run`.
- Decision: `insufficient_evidence`, with reason `insufficient_heldout_category_support`.
- Transfer candidate: `runtime_use_status=available_research_use`, `promotion_evidence_status=not_promotion_eligible`; aggregate held-out Dice/Jaccard are not reported because no subject-held-out deterministic panel can be formed from the current image-level split.
- Scratch candidate: `runtime_use_status=available_research_use`, `promotion_evidence_status=not_promotion_eligible`; aggregate held-out Dice/Jaccard are not reported because no subject-held-out deterministic panel can be formed from the current image-level split.
- Shared gate failures for both candidates: `insufficient_heldout_category_support` and `resize_benefit_unproven`.
- Split integrity: the old current artifacts recorded `566` train images and `141` validation images, but `0` validation-only subjects. The regenerated subject-held-out audit therefore writes an empty deterministic promotion manifest and reports `insufficient_evidence_for_promotion` rather than scoring a subject-overlapping panel.
- Previous prediction behavior from the direct MPS report remains diagnostic only: both candidates showed high recall and foreground overprediction on the earlier image-held-out deterministic panel. That panel was overwritten by the subject-held-out audit because it was not promotion-eligible.
- Root-cause classification for the current existing-artifact report: insufficient subject-held-out evidence plus unproven resize benefit. The previous foreground-overprediction finding remains a supervision concern for the next fresh training run.
- Current remediation path: retrain fresh candidates under the new subject-held-out split contract, then evaluate whether negative/background supervision is still required before promotion.

### Documentation Claims

- Finding: README and onboarding language no longer cite the old internal deterministic panel as current model performance.
- Finding: documentation-claim auditing blocks metric claims when cited evidence is `audit_missing`, `not_promotion_eligible`, or otherwise not promotion-eligible.
- Production implication: front-facing docs can describe research-use artifacts and audit status, but cannot publish segmentation performance metrics until a regenerated report passes the hardened gates.

## Validation Evidence

| Command | Result | Notes |
|---|---|---|
| `python3 -m py_compile src/eq/data_management/datablock_loader.py src/eq/training/segmentation_validation_audit.py src/eq/training/train_glomeruli.py src/eq/training/transfer_learning.py src/eq/training/compare_glomeruli_candidates.py` | Passed | Syntax/import compilation only |
| `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/test_glomeruli_candidate_comparison.py tests/test_segmentation_validation_audit.py tests/test_segmentation_training_contract.py tests/test_training_entrypoint_contract.py tests/test_training_smoke_v2.py` | Passed, `67 passed` | Deterministic contract, schema, and fixture tests; no full training |
| `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/test_glomeruli_candidate_comparison.py tests/test_segmentation_validation_audit.py tests/test_segmentation_training_contract.py tests/test_training_entrypoint_contract.py tests/test_training_smoke_v2.py` | Passed, `69 passed` | Current focused contract suite after validation-panel trace and split-sidecar provenance changes; no full training |
| `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/full_segmentation_retrain.yaml --dry-run` | Passed | Dry-run resolved the fresh current-code workflow commands: cohort manifest refresh, fresh mitochondria base training, fresh transfer candidate training, fresh no-base candidate training, and candidate comparison |
| `env OPENSPEC_TELEMETRY=0 openspec validate p0-harden-glomeruli-segmentation-validation --strict` | Passed | OpenSpec artifact validation |
| `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check ...` | Failed | Remaining findings are pre-existing FastAI star-import `F405` and bare `except` in training modules; not used as completion evidence for this P0 contract |
| runtime integration test against old April 24 report | Failed previously | Old report lacks candidate `train_images`/`valid_images` in `promotion_report.json`; this failure is expected and keeps tasks 6.6/6.7 open until regenerated artifacts exist |
| `EQ_GLOMERULI_PROMOTION_REPORT=/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/archive/all_manual_mask_glomeruli_seed42/promotion_report.json /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py` | Failed | Archived report does not expose a deterministic manifest for audit because `promotion_report.json` points to the pre-archive manifest path and lacks inline `candidate_manifest` |
| `env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/full_segmentation_retrain.yaml` | Failed before training started | Unsandboxed Codex launch did not resolve the relative config path; rerun with an absolute config path was requested and rejected at the approval step, so no fresh training artifacts were generated in this session |
| direct `eq.training.compare_glomeruli_candidates` MPS restart with `--device mps` | Passed | Generated current transfer and scratch artifacts plus `promotion_report.json`; direct command did not create a `logs/run_config` wrapper log |
| `EQ_GLOMERULI_PROMOTION_REPORT=/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/glomeruli_candidate_comparison/latest_run/promotion_report.json /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py` | Failed | Regenerated report exposes auditable manifest/provenance, but the runtime integration test correctly fails because split audit status is `not_promotion_eligible` due subject overlap |
| `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq.training.compare_glomeruli_candidates --run-id latest_run --transfer-model-path ... --scratch-model-path ...` | Passed | Regenerated `latest_run` as an existing-artifact subject-held-out audit without retraining; the report now records `insufficient_heldout_category_support` because current artifacts have no validation-only subjects |
| `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/test_glomeruli_candidate_comparison.py tests/test_segmentation_validation_audit.py tests/test_segmentation_training_contract.py tests/test_training_entrypoint_contract.py tests/test_training_smoke_v2.py tests/integration/test_cli.py tests/unit/test_config_paths.py tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py` | Passed, `88 passed` | Focused contract suite after subject-held-out split, resize infeasibility, failure trace, and runtime integration updates |
| `env PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/integration/test_glomeruli_segmentation_validation_audit_runtime.py` | Passed, `1 passed` | Unsandboxed Mac runtime integration check; this report exits before model inference because promotion evidence is insufficient |
| `env OPENSPEC_TELEMETRY=0 openspec validate p0-harden-glomeruli-segmentation-validation --strict` | Passed | OpenSpec artifact validation after final audit/task updates |

## Remaining Production Blockers

- Promotion-eligible deterministic evaluation remains a future evidence requirement: current artifacts cannot satisfy it because they were trained under the old image-level split; future fresh comparison runs now use a subject-held-out shared split.
- Negative/background supervision remains open: the previous direct MPS report showed foreground overprediction on background crops, so the next method-development change should be `p2-add-negative-glomeruli-crop-supervision` or an equivalent explicit supervision fix before promotion is expected.
- Real resize sensitivity remains open for future promoted artifacts: current artifacts record concrete infeasibility because no no-downsample or less-downsample comparator artifact exists.
- Direct-run logging is open: direct comparison commands should either be run only through the canonical `eq run-config` logging surface or gain one canonical durable log contract; the current direct MPS restart has artifact evidence but no single wrapper log.

## Current Production Decision

The repository can generate auditable candidate evidence through the hardened contract, and the current transfer and scratch artifacts are available for research/runtime use. It is not ready to promote a glomeruli segmentation model or publish README-facing performance metrics.

The evidence-design bug is fixed for future fresh comparisons: shared training splits are subject-held-out, and runtime integration accepts explicit insufficient-evidence reports when no subject-held-out panel exists. The next implementation work should be a fresh subject-held-out retraining run and then negative/background supervision if foreground overprediction persists, not interpretation of the old image-level split as promotion evidence.
