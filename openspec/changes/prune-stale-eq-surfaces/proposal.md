# Prune stale EQ surfaces

## Summary

Retire stale `src/eq` modules, no-op CLI commands, broken production/inference entrypoints, historical model-loading shims, and static-patch utility paths so the active package exposes only current full-image dynamic-patching and quantification workflows.

Retirement means moving stale tracked source/config files out of active package paths into the runtime `_retired/` tree, not copying them, so they cannot be imported or confused with supported code.

## Why

The active CLI still advertises commands that either do nothing, call missing symbols, or expose retired static-patch behavior. The model loader also patches `__main__` for historical pickle compatibility even though the current contract requires current-namespace artifacts without legacy shims.

## Impact

- `eq production`, `eq data-load`, `eq extract-features`, `eq quantify`, `eq process-data`, and `eq audit-derived` are removed from the active CLI.
- Static patch creation/loading/audit code is retired from active `src/eq`.
- Stale helper modules are moved to `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/src_eq_prune_2026-04-24/`.
- Current segmentation training remains full-image dynamic patching.
- Unsupported historical FastAI pickle artifacts fail closed instead of being rescued by `__main__` shims.
