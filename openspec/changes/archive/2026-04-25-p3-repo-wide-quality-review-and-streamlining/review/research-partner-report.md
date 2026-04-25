# Research Partner Report

## Scope

Repo-wide review of segmentation training, glomeruli candidate comparison, negative-crop supervision, transport/concordance workflow boundaries, and image-level endotheliosis quantification.

## Direct Evidence Reviewed

- `AGENTS.md`
- `analysis_registry.yaml`
- `src/eq/data_management/datablock_loader.py`
- `src/eq/training/compare_glomeruli_candidates.py`
- `src/eq/training/promotion_gates.py`
- `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`
- `src/eq/evaluation/run_glomeruli_transport_audit_workflow.py`
- `src/eq/evaluation/run_highres_glomeruli_concordance_workflow.py`
- `src/eq/quantification/pipeline.py`
- `src/eq/quantification/cohorts.py`
- `configs/*.yaml`
- `tests/test_glomeruli_candidate_comparison.py`
- `tests/test_segmentation_training_contract.py`
- `tests/test_segmentation_validation_audit.py`
- `tests/unit/test_quantification_cohorts.py`
- Runtime evidence summarized in `p0`, `p1`, and `p2` review artifacts.

## Bottom Line

The repo is now substantially clearer about workflow boundaries than it was before p2: candidate comparison, standard transport audit, high-resolution concordance, and quantification have separate YAML surfaces. The main remaining scientific blocker is not workflow naming; it is the absence of supported true-negative crop supervision for glomeruli false-positive control. That is explicitly documented by p1 and should become the next method-development implementation change before expecting another full training run to solve the observed over-segmentation behavior.

## Claim Classification

- Segmentation training completion: descriptive runtime claim only.
- Candidate comparison metrics: predictive/internal validation evidence, not external validity.
- Glomeruli promotion: scientific/predictive readiness claim; requires held-out evidence, non-degenerate prediction review, and baseline comparisons.
- MR concordance: external concordance/evaluation claim, not training admission.
- Quantification baseline: predictive audit baseline from image-level labels; not causal and not clinically validated.

## Statistical Review

Direct evidence: `p0` hardened validation outputs and `p1` negative-crop inventory.

- Current promotion evidence is correctly classified as insufficient when held-out category support is absent.
- The observed failure mode is false-positive/over-foreground behavior, so additional epochs alone are not the right scientific fix.
- Negative/background supervision must be crop-level reviewed evidence, not unlabeled large-image mining.
- Quantification uses image-level scores; per-glomerulus biological interpretation would require a different label structure.

## Scientific Review

Direct evidence: `docs/TECHNICAL_LAB_NOTEBOOK.md`, `README.md`, `src/eq/quantification/pipeline.py`, and OpenSpec p0-p2.

- Current docs mostly avoid claiming promoted segmentation performance.
- The current quantification framing is appropriately a baseline/audit workflow rather than a validated scoring system.
- MR is correctly treated as concordance/evaluation in phase 1, not as training expansion.

## Implementation Audit

Direct evidence: p2 implementation and CLI/config dry-runs.

- `eq run-config` now dispatches explicit stage-specific workflow IDs.
- Downstream workflow configs require explicit upstream artifacts and do not auto-discover or retrain.
- `quant-endo` and `prepare-quant-contract` are retained as direct CLI entrypoints into the canonical quantification workflow module.
- No compatibility alias was kept for the retired mixed segmentation workflow.

## Robustness Tests Required

- Keep tests that reject retired workflow IDs.
- Keep tests that fail downstream workflows when required upstream artifact references are missing.
- Keep segmentation contract tests that reject static patch roots.
- Add follow-on tests when negative-crop manifest validation and sampler integration are implemented.
- Treat the p3 5-epoch run as runtime smoke evidence only; it is not promotion evidence.

## Literature And Practical Support

No new external literature claim is introduced by p3. Practical support for the current method choices is internal and code-backed: dynamic full-image patching, explicit held-out testing for mitochondria, deterministic glomeruli review manifests, and separated workflow gates. External method/literature support should be revisited in the follow-on negative-crop curation/sampler implementation because that change will affect the training distribution.

## Direct Evidence vs Inference

- Direct evidence: file paths, config dry-runs, test surfaces, runtime manifest counts, OpenSpec artifacts.
- Inference: the most likely next scientific remediation is negative-crop curation because p0 review artifacts show foreground overprediction and p1 found no supported negative-crop annotations.

## Recommended Actions

- Accept: keep workflow split, fail-closed downstream inputs, and p3 quick 5-epoch smoke validation.
- Accept: document `docs/PIPELINE_INTEGRATION_PLAN.md` as historical/non-current because it contains stale commands.
- Defer: implement negative-crop curation, manifest validation, sampler integration, and post-curation candidate comparison in a follow-on change.
- Defer: external literature/method review for negative-crop sampling until that implementation is scoped.
