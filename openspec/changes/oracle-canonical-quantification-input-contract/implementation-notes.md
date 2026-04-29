## Implementation Notes

Implemented on 2026-04-29.

## Reuse-First Audit

- Added the canonical owner as `src/eq/quantification/input_contract.py` because the contract crosses YAML config, direct CLI, pipeline score loading, stable reviewed overrides, grouping identity, segmentation artifact provenance, and P3 metadata. Keeping it only in `pipeline.py` would deepen the execution module and leave direct entrypoints without a reusable resolver.
- Reused existing score/cohort owners:
  - `src/eq/quantification/cohorts.py` remains the manifest/admission/source-audit owner.
  - `src/eq/quantification/labelstudio_scores.py` remains the Label Studio score recovery owner.
  - `src/eq/quantification/pipeline.py::_apply_score_label_overrides` remains the reviewed override parser/applicator and audit writer.
  - `src/eq/quantification/run_endotheliosis_quantification_workflow.py` remains the YAML workflow boundary and now calls the shared resolver before label loading.
- No standalone scripts were added. Regression coverage was added through pytest.

## Stable Reviewed Override Input

- Copied the reviewed override artifact into the stable runtime-derived input path:
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/quantification_inputs/reviewed_label_overrides/endotheliosis_grade_model/rubric_label_overrides.csv`
- Source and stable copy SHA-256 matched:
  - `687779688605907fdbb6a319844e53c3f38d4ddc940457821828cbf9948b98d9`
- Stable override CSV evidence:
  - rows: `169`
  - columns: `subject_image_id, original_score, rubric_score, reviewer_confidence_1_5, accepted_teaching, review_flags, review_notes, review_source, grade_delta_vs_base, severe_changed`
  - accepted rubric scores: `0.0, 0.5, 1.0, 1.5, 2.0, 3.0`
- Updated `configs/endotheliosis_quantification.yaml` to reference the stable runtime-derived path.

## Code Changes

- Added `ResolvedQuantificationInputContract`, path validation, content hashing, grouping-identity validation, and label-contract reference generation in `src/eq/quantification/input_contract.py`.
- `run_endotheliosis_quantification_workflow.py` now validates committed `inputs.label_overrides`, rejects divergent `inputs.score_source` versus `options.score_source`, and resolves the canonical contract before calling `run_contract_first_quantification`.
- `run_contract_first_quantification` and manifest/non-manifest branches now pass a resolved contract and label-contract reference into override audits and model evaluation.
- `_apply_score_label_overrides` now writes explicit `label_overrides: none` summaries when no override is supplied and writes target-definition provenance when overrides are applied.
- `evaluate_embedding_table` threads label-contract references into P3.
- P3 writes the label-contract reference into `summary/final_product_verdict.json` and `model/final_model_metadata.json` when final metadata is written.
- Direct `eq quant-endo` and `eq prepare-quant-contract` expose `--label-overrides` and fail before label loading when omitted, pointing to the YAML workflow.

## Tests

- Added `tests/unit/test_quantification_input_contract.py` for stable override path rejection, YAML/direct resolver equivalence, grouping identity, and direct CLI forwarding.
- Extended `tests/unit/test_quantification_pipeline.py` for explicit no-override provenance and duplicate/unmatched/nonnumeric/unrecognized override failures.
- Extended `tests/unit/test_quantification_endotheliosis_grade_model.py` for P3 verdict and final metadata label-contract provenance.
- Extended `tests/integration/test_cli.py` for direct CLI `--label-overrides` help and fail-closed behavior when omitted.

## Validation

- Focused tests passed: `37 passed, 4 warnings`.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q` passed: `268 passed, 3 skipped, 8 warnings`.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .` passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help` passed.
- `OPENSPEC_TELEMETRY=0 openspec validate oracle-canonical-quantification-input-contract --strict` passed.
- `python3 scripts/check_openspec_explicitness.py oracle-canonical-quantification-input-contract` passed.

## Residual Risk

- This change standardizes target-definition provenance and direct/YAML behavior; it does not change the reviewed rubric itself.
- Existing test warnings are dependency deprecations and pre-existing quantification warning cases unrelated to the input-contract change.
