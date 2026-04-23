## Context

The current quantification pipeline contains two ordinal-model implementations: one embedded in `src/eq/quantification/pipeline.py` and another in `src/eq/quantification/ordinal.py`. The real local runtime pipeline currently uses the implementation in `pipeline.py`, where grouped subject-level cross-validation on the local embedding table completes but emits repeated `sklearn` overflow, divide-by-zero, and invalid-value warnings during logistic fitting. The observed runtime cohort is small relative to feature width and class sparsity: `88` examples, `512` embedding features, `8` subject groups, and only `7` examples in the highest observed score class. That shape makes the current cumulative-threshold logistic path numerically fragile, especially in grouped folds where upper-threshold positive counts drop to `4-6`.

The change needs to stabilize the ordinal stage without silently changing the contract-first quantification artifact schema. It also needs to avoid treating numerical convergence or warning-free execution as scientific validation. The goal is a stable predictive workflow with honest limits, not a stronger scientific claim than the data supports.

## Goals / Non-Goals

**Goals:**
- Define one canonical ordinal quantification implementation used by the contract-first pipeline and any shared evaluation helpers.
- Remove or retire duplicate ordinal-model logic so future changes do not diverge between two estimator paths.
- Enforce a numerical-stability contract on the real grouped-CV quantification workflow, including regression coverage against the local runtime cohort shape or a faithful fixture approximation.
- Allow replacement of the current cumulative-threshold logistic stack if the current estimator family cannot satisfy the stability contract under grouped splits and sparse classes.
- Preserve current output artifact names and overall schema unless a deliberate spec-level change is required.
- Record cohort-shape and external-validity limits in the design so stable execution is not conflated with strong predictive evidence.

**Non-Goals:**
- Expanding the cohort with new scored-only or newly masked data.
- Promoting the ordinal model scientifically beyond its current descriptive/predictive role.
- Changing the upstream ROI, embedding, Label Studio, or segmentation contracts.
- Solving the glomeruli segmentation promotion question in this change.

## Decisions

1. **Use `src/eq/quantification/ordinal.py` as the single canonical ordinal implementation.**
   - Rationale: the repository currently carries two implementations with similar intent but different behavior and test surfaces. The estimator belongs in the dedicated ordinal module, while `pipeline.py` should orchestrate data preparation, artifact generation, and report writing. Making `ordinal.py` canonical removes the duplicate estimator path that let the real pipeline drift away from the shared ordinal surface.
   - Alternatives considered:
     - Keep both and document which one is canonical. Rejected because the duplicate code would still drift.
     - Keep the pipeline-local implementation as canonical and leave `ordinal.py` as a side helper. Rejected because it preserves the same split-brain failure mode in a less explicit form.

2. **Treat warning-free grouped evaluation as a contract requirement, not a best-effort aspiration.**
   - Rationale: the current runtime warnings are a real defect because they indicate numerical instability on the actual cohort used for local validation.
   - Alternatives considered:
     - Accept the warnings because the pipeline still completes. Rejected because the failure mode is silent enough to be misread as benign.
     - Suppress warnings in tests only. Rejected because it hides the defect instead of constraining the estimator.

3. **Prefer a strongly regularized penalized multiclass logistic estimator if the current cumulative-threshold family remains unstable.**
   - Rationale: the repository has `88 x 512` embeddings with sparse upper classes, and the current threshold-by-threshold binary fits are exactly where rare-event instability appears. A penalized multiclass linear classifier is a defensible replacement because it is not only a sparse-class rescue tactic; it is also a reasonable long-run choice if later specs improve class representation. It remains interpretable, regularizes directly in the high-dimensional embedding space, and avoids multiplying instability across several upper-threshold binary problems.
   - Alternatives considered:
     - Force the current estimator family to remain in place. Rejected because the evidence so far does not justify treating that family as a hard requirement.
     - Collapse the score space by default. Rejected because it changes the predictive target and should only happen in a separate target-definition change.
     - Adopt a niche ordinal-specific dependency immediately. Rejected because this change needs one supported estimator family that is easy to audit, available in the current stack, and defensible on both sparse and moderately richer class distributions.

4. **Keep artifact schemas stable while making stability observable.**
   - Rationale: downstream review flows already consume `ordinal_predictions.csv`, `ordinal_metrics.json`, confusion matrices, and HTML review output. This change should preserve those surfaces where possible, but it can add metadata that makes estimator choice, class coverage, and warning checks explicit.
   - Alternatives considered:
     - Redesign the whole ordinal output package. Rejected because that would mix pipeline-contract churn with numerical stabilization.

5. **Tie validation to grouped subject-level splits on the real cohort shape.**
   - Rationale: this repository's predictive target is image-level scoring joined to grouped subject structure. Stability on IID toy folds would not prove the real use case is healthy.
   - Alternatives considered:
     - Validate only on synthetic fixtures. Rejected because they may miss the sparse-fold behavior already seen locally.
     - Validate only on the full local runtime pipeline. Rejected because the pipeline run is valuable but too expensive and indirect to be the only stability test.

6. **Report incomplete target support when the available cohort does not represent the full seven-bin target.**
   - Rationale: silent class collapse or opportunistic target simplification would change the scientific question. If the current score granularity is unsupported by the available data, the honest outcome is to keep the pipeline running, report the support gap explicitly, and leave target simplification as a separate later decision rather than treating the run as a hard failure.
   - Alternatives considered:
     - Auto-collapse sparse upper classes inside this change. Rejected because it would silently redefine the prediction target.
     - Fail the pipeline whenever upper score bins are absent. Rejected because missing class support is a reporting limitation, not an execution defect.

## Risks / Trade-offs

- [Risk] A more stable estimator could reduce apparent flexibility or alter score probabilities compared with the current logistic stack. → Mitigation: preserve output schemas, record estimator provenance, and compare old vs new metric surfaces explicitly.
- [Risk] Tightening warning-free requirements on the real cohort may reveal that the current score granularity is too ambitious for the available data. → Mitigation: make that possibility explicit in the design, report incomplete target support when necessary, and treat target simplification as a conscious follow-up decision rather than an accidental side effect.
- [Risk] Removing duplicate implementations may break tests or helper imports that currently depend on private ordinal-model classes. → Mitigation: identify all import sites first and move callers onto one canonical surface before deleting the duplicate path.
- [Risk] The local runtime cohort may be too small to support stable grouped threshold models even after estimator cleanup. → Mitigation: allow estimator-family replacement and require the final design to report residual cohort limitations honestly.
- [Risk] A penalized multiclass replacement could later be misread as a temporary imbalance-specific workaround. → Mitigation: document that the replacement is acceptable both under the present sparse-class regime and under later cohort expansions unless a future evidence-backed change selects a different supported estimator.
- [Risk] Warning-free execution may still produce a weak predictor. → Mitigation: separate numerical stability from predictive quality and document the latter as a remaining scientific limitation.
