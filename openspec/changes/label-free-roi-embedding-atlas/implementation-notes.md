# Implementation Notes

## Reuse Decisions

- `src/eq/run_config.py` is the workflow dispatcher. The current file already lists `label_free_roi_embedding_atlas` in `SUPPORTED_WORKFLOWS` and dispatches to `eq.quantification.embedding_atlas.run_label_free_roi_embedding_atlas(config_path, dry_run=dry_run)`.
- `src/eq/__main__.py` owns the `eq run-config --config <yaml> [--dry-run]` CLI surface. No direct atlas CLI alias is present or needed.
- `src/eq/utils/paths.py` owns runtime-root resolution through `EQ_RUNTIME_ROOT`, `get_runtime_quantification_results_root()`, and `get_runtime_quantification_result_path()`. Atlas configs should keep runtime-root-relative paths.
- `src/eq/utils/execution_logging.py` owns durable logging through `runtime_root_environment()`, `make_execution_log_context()`, and `execution_log_context()`. `eq run-config` logs to `logs/run_config/<run_id>/<timestamp>.log` and records workflow ID, config path, runtime root, command, Python, dry-run status, and elapsed status.
- `src/eq/quantification/modeling_contracts.py` owns shared JSON and artifact helpers through `save_json()`, `build_artifact_manifest()`, and warning/blocker payload helpers. `to_finite_numeric_matrix()` currently coerces missing or nonfinite selected values to `0.0`, so atlas code should use it only when that preprocessing is explicitly recorded or should add stricter finite-matrix validation before clustering.
- `src/eq/quantification/embeddings.py` owns frozen encoder embedding extraction and writes ROI embedding provenance fields such as `embedding_status`, `embedding_summary.json`, pooling, preprocessing, expected size, and source model path when that extractor is used.
- `src/eq/quantification/learned_roi.py` owns learned ROI feature table conventions: `burden_model/learned_roi/feature_sets/learned_roi_features.csv`, `learned_current_glomeruli_encoder_` columns, `learned_simple_roi_qc_` columns, provider audit patterns, and same-subject-excluded nearest-neighbor evidence logic.
- `src/eq/quantification/feature_review.py` and `src/eq/quantification/learned_roi_review.py` provide useful HTML and asset-copying patterns, but their report schemas are morphology-feature and supervised learned-ROI specific. Atlas HTML should use a small atlas-specific renderer in `embedding_atlas.py` while reusing their conventions for copied ROI assets, compact cards, provenance text, and missing-asset handling.

## Runtime Input Shape

Inspected active quantification root:

`/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/quantification_results/endotheliosis_quantification_full_cohort_transfer_p0_adjudicated`

- `embeddings/roi_embeddings.csv`: 707 rows, 555 columns, including 512 `embedding_` columns. Identity/provenance columns include `subject_image_id`, `subject_id`, `sample_id`, `image_id`, `glomerulus_id`, `cohort_id`, `score`, `roi_image_path`, `roi_mask_path`, `raw_image_path`, and `raw_mask_path`.
- `roi_crops/roi_scored_examples.csv`: 707 rows, 43 columns. ROI/QC columns include `roi_status`, bounding-box columns, `roi_area`, `roi_fill_fraction`, `roi_mean_intensity`, `roi_openness_score`, `roi_component_count`, `roi_component_selection`, `roi_union_bbox_width`, `roi_union_bbox_height`, and `roi_largest_component_area_fraction`.
- `burden_model/learned_roi/feature_sets/learned_roi_features.csv`: 707 rows, 535 columns, including 512 `learned_current_glomeruli_encoder_` columns and 12 `learned_simple_roi_qc_` columns.

The current inspected tables carry basic ROI status and component-selection fields. Atlas provenance gates should still check for the hardened ROI geometry, preprocessing, threshold, ROI status, and artifact-provenance columns required by the spec and fail before clustering when those fields are absent.

## Runtime Artifact Shape

The atlas output root is:

`output/quantification_results/<run_id>/burden_model/embedding_atlas/`

Expected first-read and review artifacts:

- `INDEX.md`
- `summary/atlas_verdict.json`
- `summary/atlas_summary.md`
- `summary/artifact_manifest.json`
- `feature_space/feature_space_manifest.json`
- `clusters/cluster_assignments.csv`
- `stability/cluster_stability.json`
- `diagnostics/label_blinding_audit.json`
- `diagnostics/method_availability.json`
- `diagnostics/cluster_posthoc_diagnostics.json`
- `evidence/embedding_atlas_review.html`
- `review_queue/atlas_adjudication_queue.csv`

The reviewer claim boundary is descriptive morphology clustering and review prioritization. Human score, cohort, source, treatment, reviewer, path, and target fields are metadata for post hoc diagnostics only and must not enter clustering feature matrices.

## Postflight

Implementation surfaces were checked against `proposal.md`, `design.md`, `tasks.md`, and the spec deltas on 2026-04-29. The change is additive to the workflow registry and quantification artifact tree: `eq run-config --config configs/label_free_roi_embedding_atlas.yaml` is the only supported atlas entrypoint, and no direct CLI alias was added.

The atlas runner reuses `src/eq/run_config.py` durable logging, `eq.quantification.modeling_contracts.save_json()`, and `to_finite_numeric_matrix()`. It keeps atlas-specific HTML rendering in `src/eq/quantification/embedding_atlas.py` because the existing review renderers are tied to supervised learned-ROI and morphology-feature schemas.

Validation evidence:

- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_embedding_atlas.py`: 17 passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q tests/unit/test_quantification_learned_roi.py tests/unit/test_quantification_embedding_atlas.py`: 19 passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`: 311 passed, 3 skipped.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/label_free_roi_embedding_atlas.yaml --dry-run`: passed and wrote durable run-config log metadata.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/label_free_roi_embedding_atlas.yaml`: command completed; the current configured runtime quantification root failed closed before clustering because its ROI/embedding tables do not yet carry the hardened provenance columns required by this atlas contract.
- `OPENSPEC_TELEMETRY=0 openspec validate label-free-roi-embedding-atlas --strict`: passed.
- `OPENSPEC_TELEMETRY=0 openspec validate --specs --strict`: 21 specs passed.
- `python3 scripts/check_openspec_explicitness.py label-free-roi-embedding-atlas`: passed.
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check .`: passed.
- `git diff --check`: passed.

Postflight finding: no fallback clustering path, direct CLI alias, generated repo-root atlas artifact, label override writer, or duplicated quantification path owner was introduced. The configured full-cohort runtime output must be regenerated or otherwise refreshed through the hardened upstream quantification contract before the production atlas can complete rather than emit a fail-closed stale-provenance verdict.
