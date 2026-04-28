## Execution Surface Inventory

This inventory records the logging contract for live top-level execution surfaces as of this change. Runtime logs must resolve through `src/eq/utils/paths.py::get_logs_path()` and must not create repository-root runtime directories.

## Classification Values

- `entrypoint_capture`: the entrypoint attaches an automatic durable runtime log.
- `global_log_file_only`: the command remains captured only when the operator passes `eq --log-file`.
- `subprocess_worker`: the module is normally launched by a parent workflow and its stdout/stderr is captured by the parent; direct execution may also attach its own durable log when supported.
- `high_level_function_events_only`: imported functions emit logger events but do not create files or configure handlers.
- `low_level_helper_no_durable_logging`: helper or diagnostic surface; logging is caller-owned.
- `retired_or_unsupported`: not a supported active execution surface.

## `eq` CLI Subcommands

| Surface | Module/function | Classification | Current behavior before change | Target behavior | Planned tests |
| --- | --- | --- | --- | --- | --- |
| `eq run-config` | `src/eq/__main__.py::run_config_command`, `src/eq/run_config.py::run_config` | `entrypoint_capture` | dispatched YAML workflows; only some workflows had durable tee logs | every supported workflow writes `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/<timestamp>.log` and captures worker output | all six committed config dry-runs |
| `eq cohort-manifest` | `src/eq/__main__.py::cohort_manifest_command` | `global_log_file_only` | logger events through global CLI setup | remains `eq --log-file` capture unless invoked as a subprocess worker by run-config | run-config candidate dry-run captures command |
| `eq prepare-quant-contract` | `src/eq/__main__.py::prepare_quant_contract_command` | `global_log_file_only` | logger events through global CLI setup | remains explicit `--log-file` only | governance inventory check |
| `eq quant-endo` | `src/eq/__main__.py::quant_endo_command` | `global_log_file_only` | logger events through global CLI setup | remains explicit `--log-file` only; YAML quantification uses direct/run-config logs | docs distinguish `eq --log-file` |
| `eq glomeruli-overcoverage-audit` | `src/eq/__main__.py::glomeruli_overcoverage_audit_command` | `global_log_file_only` | logger events through global CLI setup | remains explicit `--log-file` only for this change | governance inventory check |
| `eq pipeline` | `src/eq/__main__.py::pipeline_orchestrator_command` | `retired_or_unsupported` | legacy orchestration surface | no new durable runtime contract in this change | governance inventory check |
| `eq extract-images` | `src/eq/__main__.py::extract_images_command` | `global_log_file_only` | logger events through global CLI setup | explicit `--log-file` only | governance inventory check |
| `eq organize-lucchi` | `src/eq/__main__.py::organize_lucchi_command` | `global_log_file_only` | logger events through global CLI setup | explicit `--log-file` only | governance inventory check |
| `eq validate-naming` | `src/eq/__main__.py::validate_naming_command` | `global_log_file_only` | logger events through global CLI setup | explicit `--log-file` only | governance inventory check |
| `eq dox-mask-quality-audit` | `src/eq/__main__.py::dox_mask_quality_audit_command` | `global_log_file_only` | logger events through global CLI setup | explicit `--log-file` only | governance inventory check |
| `eq backup-project-data` | `src/eq/__main__.py::backup_project_data_command` | `global_log_file_only` | logger events through global CLI setup | explicit `--log-file` only | governance inventory check |
| `eq metadata-process` | `src/eq/__main__.py::metadata_process_command` | `global_log_file_only` | logger events through global CLI setup | explicit `--log-file` only | governance inventory check |
| `eq capabilities` | `src/eq/__main__.py::capabilities_command` | `low_level_helper_no_durable_logging` | status output | no automatic runtime log | governance inventory check |
| `eq mode` | `src/eq/__main__.py::mode_command` | `low_level_helper_no_durable_logging` | status/setting output | no automatic runtime log | governance inventory check |
| `eq visualize` | `src/eq/__main__.py::visualize_command` | `global_log_file_only` | logger events through global CLI setup | explicit `--log-file` only | governance inventory check |

## Direct Module Entrypoints

| Surface | Module/function | Classification | Current behavior before change | Target behavior | Planned tests |
| --- | --- | --- | --- | --- | --- |
| `python -m eq.run_config` | `src/eq/run_config.py::main` | `entrypoint_capture` | no shared durable handler | writes `logs/run_config/<run_id>/<timestamp>.log` | config dry-run tests |
| `python -m eq.training.run_glomeruli_candidate_comparison_workflow` | `run_glomeruli_candidate_comparison_workflow` | `entrypoint_capture` | custom `_emit(..., log_handle)` under `logs/run_config` | direct invocation writes `logs/direct/glomeruli_candidate_comparison/<run_id>/...`; run-config invocation reuses parent | direct workflow dry-run and resize failure tests |
| `python -m eq.evaluation.run_glomeruli_transport_audit_workflow` | `run_glomeruli_transport_audit_workflow` | `entrypoint_capture` | print-only status | direct invocation writes `logs/direct/glomeruli_transport_audit/<run_id>/...`; run-config invocation reuses parent | direct workflow dry-run tests |
| `python -m eq.evaluation.run_highres_glomeruli_concordance_workflow` | `run_highres_glomeruli_concordance_workflow` | `entrypoint_capture` | print-only status | direct invocation writes `logs/direct/highres_glomeruli_concordance/<run_id>/...`; run-config invocation reuses parent | direct workflow dry-run tests |
| `python -m eq.quantification.run_endotheliosis_quantification_workflow` | `run_endotheliosis_quantification_workflow` | `entrypoint_capture` | print-only status | direct invocation writes `logs/direct/endotheliosis_quantification/<run_id>/...`; run-config invocation reuses parent | direct workflow dry-run tests |
| `python -m eq.training.train_mitochondria` | `train_mitochondria.py::main` | `subprocess_worker` | direct CLI configured console logging only | direct invocation writes `logs/direct/train_mitochondria/<model_name>/...`; parent run-config captures stdout/stderr | monkeypatched direct training test |
| `python -m eq.training.train_glomeruli` | `train_glomeruli.py::main` | `subprocess_worker` | direct CLI configured console logging only | direct invocation writes `logs/direct/train_glomeruli/<model_name>/...`; parent workflows capture stdout/stderr | monkeypatched direct training test |
| `python -m eq.training.compare_glomeruli_candidates` | `compare_glomeruli_candidates.py::main` | `entrypoint_capture` | direct CLI emitted function logs but no durable runtime file | direct invocation writes `logs/direct/compare_glomeruli_candidates/<run_id>/...` and nonfatal candidate unavailability is logged and artifact-recorded | monkeypatched direct comparison and unavailability tests |
| `python -m eq.training.glomeruli_overcoverage_audit` | `glomeruli_overcoverage_audit.py::main` | `global_log_file_only` | module-level audit CLI | explicit log-file/global caller capture only in this change | governance inventory check |
| `python -m eq.data_management.metadata_processor` | `metadata_processor.py::main` | `global_log_file_only` | data utility module | explicit log-file/global caller capture only | governance inventory check |
| `python -m eq.data_management.negative_glomeruli_crops` | `negative_glomeruli_crops.py::main` | `low_level_helper_no_durable_logging` | curation helper | caller-owned logging | governance inventory check |
| `python -m eq.data_management.organize_lucchi_dataset` | `organize_lucchi_dataset.py::main` | `global_log_file_only` | data utility module | explicit log-file/global caller capture only | governance inventory check |
| `python -m eq.inference.gpu_inference` | `gpu_inference.py::main` | `low_level_helper_no_durable_logging` | diagnostic inference helper | caller-owned logging | governance inventory check |
| `python -m eq.training.mitochondria_validation_examples` | `mitochondria_validation_examples.py::main` | `low_level_helper_no_durable_logging` | validation example utility | caller-owned logging | governance inventory check |
| `python -m eq.utils.hardware_detection` | `hardware_detection.py::main` | `low_level_helper_no_durable_logging` | diagnostics | no durable runtime log | governance inventory check |
| `python -m eq.utils.image_mask_vis` | `image_mask_vis.py::main` | `low_level_helper_no_durable_logging` | visualization helper | caller-owned logging | governance inventory check |

## Imported High-Level Functions

| Surface | Module/function | Classification | Current behavior before change | Target behavior | Planned tests |
| --- | --- | --- | --- | --- | --- |
| Quantification manifest workflow | `src/eq/quantification/pipeline.py::run_manifest_quantification` | `high_level_function_events_only` | emitted logger milestones | continue emitting start/input/output/count events without file handlers | `caplog` imported-function test |
| Quantification contract workflow | `src/eq/quantification/pipeline.py::run_contract_first_quantification` | `high_level_function_events_only` | emitted partial logger milestones | emit start/input/output/count/failure-adjacent events without file handlers | `caplog` imported-function test |
| Mitochondria training function | `train_mitochondria_with_datablock` | `high_level_function_events_only` | emitted function logger events | no file handler attachment when imported | monkeypatched imported-function test |
| Glomeruli training functions | `train_glomeruli_with_datablock`, `train_glomeruli_with_transfer_learning` | `high_level_function_events_only` | emitted function logger events; some imported helpers configured logging | emit function events only; imported helpers must not call `setup_logging(...)` | setup-order regression test |
| Transfer helper | `transfer_learn_glomeruli` | `high_level_function_events_only` | called `setup_logging(...)` internally | use `get_logger(...)` only | setup-order regression test |

## Artifact Boundary

No implementation in this change may create repository-root `logs/`, `output/`, `models/`, `raw_data/`, or `derived_data/`. Automatic durable logs are runtime artifacts beneath the canonical logs root:

- `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/<timestamp>.log`
- `$EQ_RUNTIME_ROOT/logs/direct/<surface>/<run_id>/<timestamp>.log`

Generic `eq --log-file` remains an explicit operator-selected log file path and is separate from automatic runtime logging.
