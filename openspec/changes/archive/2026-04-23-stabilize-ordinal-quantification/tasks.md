## 1. Ordinal Architecture Audit

- [x] 1.1 Inventory every repository entrypoint, helper, and test that currently uses `src/eq/quantification/pipeline.py` ordinal logic versus `src/eq/quantification/ordinal.py`.
- [x] 1.2 Move the supported estimator surface into `src/eq/quantification/ordinal.py` and document `pipeline.py` as an orchestration caller rather than a second estimator implementation.
- [x] 1.3 Capture the current local runtime cohort shape and fold-level class sparsity assumptions used to justify the stability contract.

## 2. Canonical Estimator Implementation

- [x] 2.1 Refactor the quantification pipeline to call one canonical ordinal estimator surface.
- [x] 2.2 Remove or retire duplicate ordinal-model code that remains as a second supported execution path.
- [x] 2.3 Ensure estimator provenance is recorded in ordinal run outputs without changing the existing artifact schema unexpectedly.

## 3. Numerical Stability Hardening

- [x] 3.1 Reproduce the current grouped-CV warning pattern in a focused regression test or harness tied to the supported cohort shape.
- [x] 3.2 Stabilize the canonical estimator so grouped evaluation no longer emits unresolved overflow, divide-by-zero, or invalid-value warnings.
- [x] 3.3 If the current cumulative-threshold logistic family cannot satisfy the contract, replace it with a strongly regularized penalized multiclass logistic estimator or another explicitly approved supported estimator and document the rationale in code-facing outputs or metadata.
- [x] 3.4 If the available cohort does not represent the zero-warning seven-bin contract, report incomplete target support in the outputs instead of failing the pipeline or collapsing classes inside this change.

## 4. Validation And Regression Coverage

- [x] 4.1 Add or update unit tests for canonical ordinal fitting, probability-simplex behavior, grouped subject splits, sparse-threshold edge cases, and zero-warning enforcement on the pinned regression cohort or faithful fixture.
- [x] 4.2 Update the local-runtime quantification integration test so ordinal-stage health is treated as incomplete while unresolved numerical-instability warnings remain.
- [x] 4.3 Verify ordinal output artifacts keep their expected names and schemas after the estimator stabilization work.

## 5. Documentation And Spec Validation

- [x] 5.1 Update relevant docs or inline notes to describe the canonical ordinal estimator path and its remaining cohort limitations without overstating predictive validity.
- [x] 5.2 Run targeted ordinal and local-runtime quantification tests in `eq-mac`.
- [x] 5.3 Run `env OPENSPEC_TELEMETRY=0 openspec validate stabilize-ordinal-quantification --strict`.
