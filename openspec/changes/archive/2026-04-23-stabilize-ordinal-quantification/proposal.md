## Why

The current ordinal quantification stage completes, but it emits repeated `sklearn` overflow, divide-by-zero, and invalid-value warnings during grouped cross-validation on the real local runtime cohort. That means the current image-level predictor is executable but not numerically well-behaved, and the repository now needs a clean, canonical ordinal modeling contract before further cohort expansion or scientific interpretation.

## What Changes

- Define one canonical ordinal quantification implementation for grouped image-level endotheliosis prediction and remove duplicate ordinal-model logic paths.
- Add a numerical-stability contract for grouped ordinal evaluation, including real-cohort regression coverage and a zero-unresolved-warning gate for model fitting on the supported cohort shape.
- Permit replacing the current cumulative-threshold logistic stack if it cannot satisfy the stability contract on the actual embedding table shape and class distribution, with strongly regularized penalized multiclass logistic as the preferred replacement family.
- Preserve current quantification artifact schemas unless a deliberate spec-level output change is justified and recorded.
- Record the statistical and cohort-shape limits of the stabilized ordinal workflow so runtime success is not confused with strong predictive evidence, and report incomplete target support rather than silently collapsing targets if the current seven-bin contract is not represented in the available cohort.

## Capabilities

### New Capabilities
- `ordinal-quantification-stability`: Defines the canonical ordinal estimator path, grouped evaluation contract, and numerical-stability requirements for frozen-embedding image-level quantification.

### Modified Capabilities
- `macos-mps-local-development`: Tightens the local-runtime quantification certification requirement so the contract-first pipeline is not treated as fully healthy while the ordinal stage still emits unresolved numerical-instability warnings.

## Impact

- Affected code: `src/eq/quantification/pipeline.py`, `src/eq/quantification/ordinal.py`, quantification report generation, and any shared ordinal helper code.
- Affected tests: ordinal unit tests, local-runtime integration tests, and regression coverage around grouped subject splits, class sparsity, and emitted warnings.
- Affected artifacts: `ordinal_predictions.csv`, `ordinal_metrics.json`, `ordinal_confusion_matrix.csv`, `ordinal_embedding_model.pkl`, and review-report outputs must remain current and interpretable.
- Dependencies/systems: `scikit-learn`, embedding-table preprocessing, grouped CV behavior, and any future cohort-expansion workflow that depends on stable image-level score modeling.
