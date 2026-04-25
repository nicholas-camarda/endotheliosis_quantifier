## Why

The repository now has multiple active workflow surfaces, accumulated historical docs, runtime-path conventions, model-promotion gates, and recent cleanup changes that need one evidence-backed alignment pass. A repo-wide review should make the supported objectives, CLI entrypoints, data/runtime contracts, documentation, and quality gates internally consistent before more segmentation or quantification work builds on stale assumptions.

## What Changes

- Add a repo-wide quality review workflow that runs exhaustive specialist review lanes before making cleanup or streamlining edits:
  - `documentation-wizard` lane for README, `docs/`, OpenSpec, config, CLI-help, and public/private documentation drift.
  - `workspace-governor` lane for canonical repo/runtime/cloud layout, tracked clutter, publish-preview boundaries, generated artifact placement, and public/private path hygiene.
  - `research-partner` lane for data-first statistical, scientific, implementation, robustness, and literature-support review across segmentation, quantification, model promotion, and workflow claims.
- Add a durable review dossier under `openspec/changes/p3-repo-wide-quality-review-and-streamlining/review/` so implementation decisions are tied to concrete evidence instead of broad cleanup taste.
- Streamline the supported CLI and workflow surface after audit by reconciling `eq` subcommands, `eq run-config`, committed YAMLs in `configs/`, direct module entrypoints under `src/eq/training/`, and docs examples into one current workflow contract.
- Reduce repo clutter by classifying historical, generated, cache, archive, and unsupported surfaces before moving, excluding, retiring, or documenting them.
- Strengthen reproducibility and integrity gates for path resolution, runtime-output boundaries, config execution, model artifact provenance, segmentation promotion evidence, quantification data joins, and regression tests.
- Update docs and OpenSpec artifacts so objectives are explicit and current-state only: segmentation training/support, glomeruli promotion, quantification, cohort expansion, and local Mac/WSL execution should not be mixed or described with stale migration framing.

## Capabilities

### New Capabilities

- `repo-wide-quality-review`: Defines the required review dossier, specialist lanes, synthesis, evidence-backed action plan, and validation gates for whole-repository quality review and streamlining.

### Modified Capabilities

- `openspec-change-governance`: Adds repository-level expectations for broad audit changes: review artifacts must preserve evidence, distinguish audit findings from implementation decisions, and keep audit-first questions explicitly traceable.

## Impact

- Affected code: `src/eq/__main__.py`, `src/eq/run_config.py`, `src/eq/utils/paths.py`, `src/eq/utils/run_io.py`, `src/eq/data_management/`, `src/eq/training/`, `src/eq/quantification/`, and any stale wrappers or entrypoints identified by the review dossier.
- Affected CLI/API surfaces: `eq --help`, `eq quant-endo`, `eq prepare-quant-contract`, `eq cohort-manifest`, `eq run-config`, `eq capabilities`, `eq mode`, direct module commands `python -m eq.training.train_mitochondria`, `python -m eq.training.train_glomeruli`, `python -m eq.training.run_full_segmentation_retrain`, and direct or documented commands discovered during the audit.
- Affected configs/docs: `configs/mito_pretraining_config.yaml`, `configs/glomeruli_finetuning_config.yaml`, `configs/segmentation_fixedloader_full_retrain.yaml`, `README.md`, `docs/README.md`, `docs/TECHNICAL_LAB_NOTEBOOK.md`, `docs/SEGMENTATION_ENGINEERING_GUIDE.md`, `docs/OUTPUT_STRUCTURE.md`, `analysis_registry.yaml`, `AGENTS.md`, and active OpenSpec changes/specs.
- Affected artifact boundaries: repo checkout, `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier`, and `/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/SideProjects/endotheliosis_quantifier/Analysis`.
- Compatibility risk: cleanup may retire unsupported commands, docs, generated files, or stale historical artifacts. The change must not add fallback compatibility branches for unsupported legacy model artifacts, static patch training inputs, duplicate path systems, or stale CLI aliases.

## Explicit Decisions

- Change name: `p3-repo-wide-quality-review-and-streamlining`.
- Review dossier root: `openspec/changes/p3-repo-wide-quality-review-and-streamlining/review/`.
- Required lane artifact filenames:
  - `review/preflight-path-and-artifact-map.md`
  - `review/documentation-wizard-report.md`
  - `review/workspace-governor-report.md`
  - `review/research-partner-report.md`
  - `review/repo-wide-quality-synthesis.md`
  - `review/action-register.tsv`
- Required non-mutating first-pass commands:
  - `python3 /Users/ncamarda/.codex/plugins/cache/local-plugins/documentation-wizard/0.2.0/scripts/documentation_wizard.py report --repo /Users/ncamarda/Projects/endotheliosis_quantifier`
  - `python3 /Users/ncamarda/.codex/plugins/cache/local-plugins/workspace-governor/0.2.0/scripts/workspace_governor.py assess --repo /Users/ncamarda/Projects/endotheliosis_quantifier`
  - data-first `research-partner` review covering the whole repo with lanes `review-preflight`, `implementation-auditor`, `stats-reviewer`, `scientific-reviewer`, `robustness-test-designer`, `literature-support-reviewer`, `documentation-wizard`, and `review-synthesizer`.
- Review scope includes the repo checkout, active OpenSpec tree, `analysis_registry.yaml`, committed configs, `src/eq`, tests, public/internal docs, the configured runtime root, and the configured cloud publish root.
- Implementation must treat missing critical runtime inputs, ambiguous path ownership, unsupported model artifacts, and stale documented commands as hard failures or retirements, not as warning-only fallbacks.

## Open Questions

- [audit_first_then_decide] Which specific files, commands, docs, tests, and configs should be retired or rewritten? Decide from `review/action-register.tsv`, import/CLI reachability evidence, documentation drift evidence, and workspace-governor clutter classification.
- [audit_first_then_decide] Should direct module training entrypoints remain user-facing after `eq run-config` is the documented YAML control surface? Decide from CLI/help extraction, committed config coverage, tests, and the documentation-wizard report.
- [audit_first_then_decide] Which generated or historical files should move to runtime `_retired/`, local `.git/info/exclude`, tracked docs, or remain untouched? Decide from git status, workspace-governor assessment, file provenance, and public/private docs review.
