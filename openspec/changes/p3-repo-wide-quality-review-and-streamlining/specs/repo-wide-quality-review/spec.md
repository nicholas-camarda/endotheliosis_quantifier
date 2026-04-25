## ADDED Requirements

### Requirement: Repo-wide review dossier
The repository SHALL preserve a durable review dossier for the repo-wide quality review under `openspec/changes/p3-repo-wide-quality-review-and-streamlining/review/`.

#### Scenario: Dossier is initialized before implementation edits
- **WHEN** the repo-wide quality review change is applied
- **THEN** the implementation SHALL create `review/preflight-path-and-artifact-map.md`, `review/documentation-wizard-report.md`, `review/workspace-governor-report.md`, `review/research-partner-report.md`, `review/repo-wide-quality-synthesis.md`, and `review/action-register.tsv`
- **AND** code, docs, config, or workspace cleanup edits SHALL NOT begin before the preflight map, lane reports, synthesis, and action register exist

#### Scenario: Dossier records direct evidence
- **WHEN** a lane report makes a finding about a command, config, path, data contract, artifact, method, or document
- **THEN** the report SHALL cite the concrete file, command output, runtime artifact, or source path used as evidence
- **AND** the report SHALL distinguish direct evidence from inference

### Requirement: Documentation lane
The review SHALL run a documentation lane that compares public docs, internal docs, OpenSpec artifacts, CLI help, configs, and code-backed interfaces.

#### Scenario: Documentation drift is reviewed
- **WHEN** the documentation lane runs
- **THEN** it SHALL inspect `README.md`, `docs/`, `AGENTS.md`, `analysis_registry.yaml`, active OpenSpec artifacts, committed workflow configs, `src/eq/__main__.py`, and `src/eq/run_config.py`
- **AND** it SHALL write findings to `review/documentation-wizard-report.md`

#### Scenario: Public and internal documentation boundaries are checked
- **WHEN** public documentation mentions local runtime, cloud, maintainer-specific, or machine-specific paths
- **THEN** the documentation lane SHALL classify whether the content belongs in portable public docs, `AGENTS.md`, another internal doc, or no tracked doc
- **AND** it SHALL record the smallest correct patch direction

#### Scenario: CLI and config claims are checked against live surfaces
- **WHEN** docs describe commands, flags, workflow IDs, config keys, or output roots
- **THEN** the documentation lane SHALL compare those claims against parser/help output, `SUPPORTED_WORKFLOWS` in `src/eq/run_config.py`, committed YAML configs, and path helpers before recommending edits

### Requirement: Workspace lane
The review SHALL run a workspace lane that evaluates canonical repo/runtime/cloud layout, tracked clutter, generated artifacts, publish-preview boundaries, and public/private path hygiene.

#### Scenario: Workspace assessment runs non-mutating first
- **WHEN** the workspace lane runs
- **THEN** it SHALL perform a non-mutating assessment of `/Users/ncamarda/Projects/endotheliosis_quantifier`
- **AND** it SHALL write classification, rewrite candidates, move candidates, publish-preview findings, and unresolved questions to `review/workspace-governor-report.md`

#### Scenario: Clutter classification precedes cleanup
- **WHEN** generated files, caches, `.history`, `.agent-os`, old docs, retired artifacts, raw data, derived data, models, logs, or output artifacts are found in or near the repo checkout
- **THEN** the workspace lane SHALL classify each relevant surface before any move, local exclude, tracked delete, archive, or doc rewrite is performed

#### Scenario: Source, runtime, and cloud roots remain separate
- **WHEN** the workspace lane reviews path ownership
- **THEN** it SHALL treat `/Users/ncamarda/Projects/endotheliosis_quantifier` as source code, `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier` as runtime working state, and `/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/SideProjects/endotheliosis_quantifier/Analysis` as cloud publish output
- **AND** it SHALL flag any code, config, docs, or artifacts that create a competing active storage root

### Requirement: Research review lane
The review SHALL run a research lane that evaluates the repository's scientific, statistical, implementation, robustness, and literature-support quality using actual project data, runtime artifacts, code, tests, docs, and outputs before generic methodology.

#### Scenario: Research preflight reconstructs objectives and artifact flow
- **WHEN** the research lane starts
- **THEN** it SHALL identify major objectives for segmentation training, glomeruli promotion, quantification, scored-cohort expansion, Mac/WSL execution, and publication or publish-preview artifacts
- **AND** it SHALL map the code, config, data, runtime, and output surfaces used by each objective

#### Scenario: Methodological claims are classified
- **WHEN** the research lane evaluates repository claims
- **THEN** it SHALL classify each major claim as descriptive, associational, causal, or predictive/prognostic
- **AND** it SHALL flag any claim that conflates code execution, model loading, convergence, statistical significance, calibration, discrimination, internal validation, external validity, or scientific promotion

#### Scenario: Robustness tests are proposed from failure modes
- **WHEN** the research lane identifies a failure mode that could silently change labels, masks, ROI geometry, embeddings, metrics, artifact provenance, or review outputs
- **THEN** it SHALL propose a concrete regression test or validation command and record it in `review/research-partner-report.md`

### Requirement: Review synthesis and action register
The change SHALL synthesize lane outputs into one deduplicated action register before implementation decisions are made.

#### Scenario: Lane findings are deduplicated
- **WHEN** all lane reports exist
- **THEN** the implementation SHALL write `review/repo-wide-quality-synthesis.md` with the bottom line, ranked findings, cross-lane conflicts, direct evidence, uncertainty, and recommended actions

#### Scenario: Action register controls implementation
- **WHEN** the synthesis identifies accepted work for this change
- **THEN** each accepted action SHALL be recorded in `review/action-register.tsv` with `action_id`, `lane`, `surface`, `finding`, `evidence`, `decision`, `implementation_target`, `validation`, `risk_level`, and `status`
- **AND** implementation SHALL only proceed for accepted rows that name an implementation target and validation check

#### Scenario: Deferred findings remain visible
- **WHEN** a finding is evidence-backed but outside the safe scope of this change
- **THEN** it SHALL remain in `review/action-register.tsv` with `status=deferred`
- **AND** the decision field SHALL state why it is deferred rather than silently dropping the finding

### Requirement: Evidence-backed streamlining
The change SHALL streamline CLI, docs, configs, tests, and workspace surfaces only when the action register records supporting evidence.

#### Scenario: CLI surface is streamlined
- **WHEN** the action register accepts a CLI or workflow streamlining action
- **THEN** the implementation SHALL reconcile `eq --help`, `eq run-config`, committed configs, direct module entrypoints, tests, and docs so the supported command surface is current and unambiguous
- **AND** it SHALL NOT add fallback aliases, compatibility branches, or undocumented rescue paths for unsupported commands

#### Scenario: Documentation is made current-state only
- **WHEN** the action register accepts a documentation edit
- **THEN** the implementation SHALL write current behavior directly
- **AND** it SHALL avoid historical comparison, migration framing, or "now supported" language unless the action row explicitly requires historical context

#### Scenario: Reproducibility and integrity gates are strengthened
- **WHEN** the action register accepts a reproducibility or integrity finding
- **THEN** the implementation SHALL add or update a test, validation command, config check, artifact-provenance check, or docs contract that would detect the failure mode in future runs

### Requirement: Final repo-quality validation
The change SHALL finish with explicit validation that covers OpenSpec, CLI behavior, committed configs, tests, and the review dossier.

#### Scenario: Required final validation commands run
- **WHEN** implementation is complete
- **THEN** validation SHALL include `openspec validate p3-repo-wide-quality-review-and-streamlining --strict`
- **AND** validation SHALL include `python3 scripts/check_openspec_explicitness.py openspec/changes/p3-repo-wide-quality-review-and-streamlining`
- **AND** validation SHALL include `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`
- **AND** validation SHALL include `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`
- **AND** validation SHALL include dry-runs for `configs/mito_pretraining_config.yaml`, `configs/glomeruli_finetuning_config.yaml`, and `configs/glomeruli_candidate_comparison.yaml` through `eq run-config`

#### Scenario: Accepted actions are closed out
- **WHEN** final validation has run
- **THEN** every accepted row in `review/action-register.tsv` SHALL have `status=done` or a documented non-green validation result with next action
- **AND** `review/repo-wide-quality-synthesis.md` SHALL summarize residual risk and any deferred findings
