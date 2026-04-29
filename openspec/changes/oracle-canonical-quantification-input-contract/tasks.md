## 1. Reuse-First Input Contract Audit

- [ ] 1.1 Inventory current score/cohort/input resolution in `src/eq/quantification/cohorts.py`, `src/eq/quantification/labelstudio_scores.py`, `src/eq/quantification/pipeline.py`, `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, `src/eq/run_config.py`, and `src/eq/__main__.py`.
- [ ] 1.2 Record which existing owner will hold the canonical resolved quantification input contract and why existing owners can or cannot absorb the behavior.
- [ ] 1.3 Identify every touched caller that currently loads scores, mapping files, annotation sources, segmentation model paths, output roots, or reviewed overrides outside the chosen owner.
- [ ] 1.4 Confirm no new standalone script is needed; if one is unavoidable, document why existing pytest/helpers/config runner surfaces cannot express the requirement.

## 2. Stable Reviewed Override Input

- [ ] 2.1 Verify the current reviewed override artifact row count, required columns, accepted score set, and content hash.
- [ ] 2.2 Copy or regenerate the reviewed override file under `derived_data/quantification_inputs/reviewed_label_overrides/endotheliosis_grade_model/rubric_label_overrides.csv` in the runtime root.
- [ ] 2.3 Update `configs/endotheliosis_quantification.yaml` so `inputs.label_overrides` references the stable runtime-derived path instead of `output/quantification_results/...`.
- [ ] 2.4 Add config/path validation that rejects committed `inputs.label_overrides` paths under `output/quantification_results/`.
- [ ] 2.5 Add regression coverage proving a missing stable override path fails before modeling.

## 3. Canonical Resolver Implementation

- [ ] 3.1 Implement or extend the chosen `src/eq/quantification/` owner so it returns a resolved contract containing data root, score source, annotation source, mapping file, label override path, segmentation artifact, output root, and target-definition metadata.
- [ ] 3.2 Reuse existing path helpers from `src/eq/utils/paths.py` and existing score/cohort parsing helpers; do not duplicate path resolution or label parsing.
- [ ] 3.3 Enforce reviewed override validation for required columns, unique `subject_image_id`, recognized scores, row matches, and nonnumeric values before downstream label consumption.
- [ ] 3.4 Write label-contract provenance into existing `scored_examples/score_label_overrides_audit.csv` and `scored_examples/score_label_overrides_summary.json`.
- [ ] 3.5 Include validation grouping identity in the resolved contract: `subject_id`, row identity, `subject_image_id` uniqueness, grouping-key derivation, and subject/image counts.
- [ ] 3.6 Record content hashes for all target-defining inputs, including the base scored table or manifest, annotation source when file-backed, mapping file when present, reviewed override file when present, and segmentation artifact metadata reference.
- [ ] 3.7 Ensure no optional feature-table merge can overwrite the canonical resolved label columns with stale score columns.

## 4. Entrypoint Migration

- [ ] 4.1 Refactor YAML workflow execution to call the canonical resolver before ROI extraction, learned ROI modeling, source-aware modeling, severe-aware modeling, and P3 grade-model fitting.
- [ ] 4.2 Add `--label-overrides` to `eq quant-endo` only if the direct command can fully route through the same resolver.
- [ ] 4.3 Add `--label-overrides` to `eq prepare-quant-contract` only if the direct command prepares label-dependent artifacts through the same resolver.
- [ ] 4.4 Make direct CLI commands fail before label loading when they cannot satisfy the full current label contract.
- [ ] 4.5 Update direct CLI help text so `eq run-config --config configs/endotheliosis_quantification.yaml` remains the primary documented workflow.

## 5. P3 Artifact Provenance

- [ ] 5.1 Thread the resolved label contract into P3 candidate evaluation and final selector code.
- [ ] 5.2 Write label-contract references into `burden_model/endotheliosis_grade_model/summary/final_product_verdict.json`.
- [ ] 5.3 Write label-contract references into `burden_model/endotheliosis_grade_model/model/final_model_metadata.json` when a final model is written.
- [ ] 5.4 Ensure P3 comparison tables distinguish runs with different override hashes or target-definition versions.

## 6. Tests

- [ ] 6.1 Add unit coverage proving YAML and direct CLI resolution produce identical effective label contracts when given equivalent inputs.
- [ ] 6.2 Add unit coverage proving direct CLI fails closed when reviewed overrides are required but omitted.
- [ ] 6.3 Add unit coverage proving `output/quantification_results/...` override paths are rejected in committed config validation.
- [ ] 6.4 Add unit coverage for duplicate, unmatched, nonnumeric, and unrecognized reviewed override rows.
- [ ] 6.5 Add unit coverage proving P3 metadata and final verdict include the resolved label contract.
- [ ] 6.6 Add parser-level regression coverage proving `eq quant-endo` and `eq prepare-quant-contract` cannot reach label-dependent quantification execution unless they expose and pass the full resolved label contract or fail before label loading.
- [ ] 6.7 Add regression coverage proving grouping identity and target-defining input hashes are recorded and that changed hashes/target-definition versions invalidate direct metric comparability.

## 7. Documentation And Validation

- [ ] 7.1 Update README/docs/config comments only where current behavior changes; keep docs current-state and YAML-first.
- [ ] 7.2 Run focused tests for quantification input contract, label overrides, direct CLI parsing, and P3 metadata.
- [ ] 7.3 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [ ] 7.4 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .`.
- [ ] 7.5 Run `OPENSPEC_TELEMETRY=0 openspec validate oracle-canonical-quantification-input-contract --strict`.
- [ ] 7.6 Run `python3 scripts/check_openspec_explicitness.py oracle-canonical-quantification-input-contract`.
- [ ] 7.7 Record final validation results, migrated callers, any rejected duplicate surfaces, and residual risks in the implementation closeout.

## 8. Postflight And Archive Lifecycle

- [ ] 8.1 Complete the per-change postflight required by `openspec/changes/ACTIVE_EXECUTION_ORDER.md`, including spec-to-diff review, completed-task evidence review, `git diff --check`, `git diff --stat`, and unrelated-edit inspection.
- [ ] 8.2 Commit the implementation as `implement oracle-canonical-quantification-input-contract`.
- [ ] 8.3 Archive/sync with `openspec archive oracle-canonical-quantification-input-contract --yes`.
- [ ] 8.4 Run `openspec validate --specs --strict` after archive/sync.
- [ ] 8.5 Revalidate every remaining active change with `openspec validate <remaining-change> --strict` and `python3 scripts/check_openspec_explicitness.py <remaining-change>`.
- [ ] 8.6 Commit the archive/sync as `archive oracle-canonical-quantification-input-contract`.
