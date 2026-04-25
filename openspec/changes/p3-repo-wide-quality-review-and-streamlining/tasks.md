## 1. Preflight and Review Dossier

- [x] 1.1 Create `openspec/changes/p3-repo-wide-quality-review-and-streamlining/review/` and initialize the six required review artifacts.
- [x] 1.2 Write `review/preflight-path-and-artifact-map.md` with repo path, runtime root, cloud publish root, active OpenSpec changes, committed configs, key docs, CLI entrypoints, direct module entrypoints, tests, and known artifact roots.
- [x] 1.3 Capture baseline command/interface evidence: `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`, `eq run-config --help`, `openspec list --json`, and committed config dry-runs where they do not launch training.
- [x] 1.4 Record current `git status --short`, tracked generated/cache files, untracked clutter classes, and any repo-root runtime/data/output directories or symlinks in the preflight map.

## 2. Parallel Specialist Review Lanes

- [x] 2.1 Run the `documentation-wizard` lane against `/Users/ncamarda/Projects/endotheliosis_quantifier` and write `review/documentation-wizard-report.md` with doc surfaces, interface extraction, drift findings, public/private doc issues, impact, and proposed regression checks.
- [x] 2.2 Run the `workspace-governor` non-mutating assessment against `/Users/ncamarda/Projects/endotheliosis_quantifier` and write `review/workspace-governor-report.md` with classification, rewrite candidates, move candidates, publish-preview findings, and unresolved questions.
- [x] 2.3 Run the `research-partner` repo-wide lane with review-preflight, implementation-auditor, stats-reviewer, scientific-reviewer, robustness-test-designer, literature-support-reviewer, documentation-wizard, and review-synthesizer coverage; write `review/research-partner-report.md`.
- [x] 2.4 Ensure every lane report separates direct evidence from inference and names concrete files, commands, runtime artifacts, or source paths for major findings.

## 3. Synthesis and Action Register

- [x] 3.1 Deduplicate lane findings into `review/repo-wide-quality-synthesis.md` with bottom line, ranked findings, cross-lane conflicts, uncertainty, and recommended action groups.
- [x] 3.2 Create `review/action-register.tsv` with columns `action_id`, `lane`, `surface`, `finding`, `evidence`, `decision`, `implementation_target`, `validation`, `risk_level`, and `status`.
- [x] 3.3 Mark each finding as `accepted`, `deferred`, or `rejected`; require accepted rows to name an implementation target and validation command before implementation begins.
- [x] 3.4 Update proposal/design open-question answers in the synthesis or action register when audit evidence resolves them.

## 4. Evidence-Backed Implementation

- [x] 4.1 Implement accepted documentation actions, keeping public docs portable and current-state only while keeping maintainer-specific path topology in `AGENTS.md` or internal docs.
- [x] 4.2 Implement accepted CLI/workflow streamlining actions across `src/eq/__main__.py`, `src/eq/run_config.py`, committed configs, tests, and docs without adding fallback aliases or compatibility hacks.
- [x] 4.3 Implement accepted path/artifact-boundary actions using existing helpers in `src/eq/utils/paths.py`, `src/eq/utils/run_io.py`, and existing data-contract utilities before adding abstractions.
- [x] 4.4 Implement accepted reproducibility, integrity, and robustness actions as tests, validation commands, config checks, artifact-provenance checks, or docs contracts.
- [x] 4.5 Classify accepted clutter actions before mutating files; use local `.git/info/exclude`, tracked deletion, docs archive, or runtime `_retired/` only when the action register records evidence and validation.
- [x] 4.6 After each implementation group, update `review/action-register.tsv` row statuses and rerun the focused validation named for those rows.

## 5. Documentation and OpenSpec Alignment

- [x] 5.1 Update active OpenSpec specs or change artifacts when accepted actions alter observable requirements for CLI, configs, path contracts, docs governance, or review workflow.
- [x] 5.2 Ensure `README.md`, `docs/README.md`, `docs/TECHNICAL_LAB_NOTEBOOK.md`, `docs/SEGMENTATION_ENGINEERING_GUIDE.md`, `docs/OUTPUT_STRUCTURE.md`, and `analysis_registry.yaml` describe the same current workflow boundaries.
- [x] 5.3 Verify docs do not use stale migration framing or unsupported legacy artifact claims unless explicitly labeled historical.
- [x] 5.4 Verify any direct module entrypoints retained in docs are tested, intentionally user-facing, and consistent with the YAML-first `eq run-config` workflow.

## 6. Final Validation

- [x] 6.1 Run `openspec validate p3-repo-wide-quality-review-and-streamlining --strict`.
- [x] 6.2 Run `python3 scripts/check_openspec_explicitness.py openspec/changes/p3-repo-wide-quality-review-and-streamlining`.
- [x] 6.3 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`.
- [x] 6.4 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/mito_pretraining_config.yaml --dry-run`.
- [x] 6.5 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/glomeruli_finetuning_config.yaml --dry-run`.
- [x] 6.6 Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run`.
- [x] 6.7 Run `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [x] 6.8 Update `review/repo-wide-quality-synthesis.md` and `review/action-register.tsv` with final validation results, residual risks, and deferred findings.
