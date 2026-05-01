## 1. Spec alignment audit

- [x] 1.1 Audit existing cohort/build + quant writers for lineage fields (`grep`/read paths in `src/eq/quantification/`) and map gaps versus new **`scored-only-quantification-cohort`** scenarios.

## 2. Documentation

- [x] 2.1 Update quant onboarding docs (prefer extending existing cohort/quant README sections or `docs/` guidance referenced by `workflow-config-entrypoints`) with the explicit **grade → cohort → quant rerun** cadence language consistent with **`label-studio-glomerulus-grading`**.

## 3. Implementation (lineage / audits)

- [x] 3.1 Implement missing lineage persistence identified in 1.1 (additive JSON/CSV audit fields; avoid silent stale-score reuse consistent with existing fail-closed posture).
- [x] 3.2 Implement explicit per-glomerulus scored-example builder and contract-first branching when instance-level Label Studio exports are present (no silent fallback to legacy image-level aggregation).

## 4. Verification

- [x] 4.1 Extend or add fixture-oriented tests proving two synthetic grading snapshots yield distinguishable lineage in cohort or burden summaries where feasible without heavyweight LS fixtures.
- [ ] 4.2 Run `openspec validate endotheliosis-grading-quant-learning-loop --strict`.
