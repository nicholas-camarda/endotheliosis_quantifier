# Design

## Retirement boundary

Move stale active files to:

`/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/src_eq_prune_2026-04-24/`

The move must preserve relative source paths and write a manifest with original path, retired path, file size, and move timestamp. Active package paths are then removed from imports, CLI registration, docs, configs, and tests.

## Active package shape

The active package keeps:

- full-image dynamic segmentation dataloaders and validators
- current training modules
- candidate comparison and promotion gates
- current quantification, cohort, path, hardware, and config helpers
- explicit image extraction for full-image inputs

The active package retires:

- broken or no-op CLI command handlers
- stale production pipeline and explicit prediction/evaluation scripts
- static patch creation/loading/audit utilities
- stale helper modules with no current contract
- historical FastAI pickle shim behavior

## Safety checks

The implementation must avoid touching unrelated worktree changes. Because `src/eq/utils/paths.py` is already modified before this change, it is out of scope unless a failing validation proves it must be edited.

Validation must prove active imports still work, removed commands are absent from CLI help, dynamic-patching tests still cover supported training, stale symbols are absent from active `src/eq`, and retired files exist in the runtime `_retired` tree.
