# Documentation Wizard Report

## Scope

Documentation-drift review for `/Users/ncamarda/Projects/endotheliosis_quantifier`.

## Direct Evidence

- Command run:
  `python3 /Users/ncamarda/.codex/plugins/cache/local-plugins/documentation-wizard/0.2.0/scripts/documentation_wizard.py report --repo /Users/ncamarda/Projects/endotheliosis_quantifier`
- Documentation surfaces scanned:
  `README.md`, `docs/README.md`, `docs/ONBOARDING_GUIDE.md`, `docs/OUTPUT_STRUCTURE.md`, `docs/SEGMENTATION_ENGINEERING_GUIDE.md`, `docs/TECHNICAL_LAB_NOTEBOOK.md`, `docs/HISTORICAL_IMPLEMENTATION_ANALYSIS.md`, `docs/HISTORICAL_NOTES.md`, `docs/INTEGRATION_GUIDE.md`, and `docs/PIPELINE_INTEGRATION_PLAN.md`.
- Source truth candidates included CLI modules under `src/eq/`, workflow runners, `src/eq/run_config.py`, and `scripts/check_openspec_explicitness.py`.
- Raw tool result contained 96 parser findings: 50 broken referenced paths, 43 missing CLI flags, and 3 stale CLI flags.

## Findings

1. **Historical planning docs are the highest-value documentation drift target.**
   Direct evidence: `docs/PIPELINE_INTEGRATION_PLAN.md` contains older command and fallback-oriented examples, including stale flags around the historical validation path.
   Impact: without an explicit historical label, readers can mistake this for current operational guidance.
   Decision: accepted. Label the document as historical and list it under historical notes rather than current operational docs.

2. **README command-help references include parser false positives.**
   Direct evidence: the tool flagged `README.md:54` as a stale `--help` flag, but the line is a valid command example: `python -m eq --help`.
   Impact: this is not a user-facing drift issue.
   Decision: rejected as parser noise.

3. **Broad command and path extraction over-flags prose tokens.**
   Direct evidence: tool output included apparent missing paths derived from backticked prose or command fragments, not actual repository references.
   Impact: this should not drive compatibility shims or extra path aliases.
   Decision: rejected unless a human-readable doc line points to a concrete missing file or unsupported command.

4. **The current operational docs are aligned with the YAML-first front door.**
   Direct evidence: `README.md`, `docs/ONBOARDING_GUIDE.md`, and `docs/TECHNICAL_LAB_NOTEBOOK.md` now point users to `eq run-config` and split workflow configs.
   Impact: no additional public CLI surface should be invented for P3.
   Decision: accepted as current-state contract.

## Public/Private Documentation Boundary

Direct evidence: maintainer-specific runtime path topology remains concentrated in `AGENTS.md`, `analysis_registry.yaml`, and OpenSpec review artifacts. Public docs emphasize package commands, YAML configs, and runtime layout without requiring local user-specific paths.

Inference: this boundary is good enough for P3. Future maintainer-only machine details should stay out of public README copy unless the user explicitly asks for operator notes.

## Regression Checks

- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq --help`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --help`
- `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run`
- `python3 scripts/check_openspec_explicitness.py openspec/changes/p3-repo-wide-quality-review-and-streamlining`

## Direct Evidence vs Inference

- Direct evidence: command output categories, named documentation surfaces, named source truth candidates, specific `docs/PIPELINE_INTEGRATION_PLAN.md` and `README.md` findings.
- Inference: most parser path findings are not actionable unless a named live doc line gives a real unsupported command or path.
