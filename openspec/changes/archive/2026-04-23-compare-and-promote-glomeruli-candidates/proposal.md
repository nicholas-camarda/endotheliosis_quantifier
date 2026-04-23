## Why

The repository now has the right training-data contract and promotion-gate primitives for glomeruli segmentation, but it still does not answer the practical question of which candidate, if any, should be promoted. The next change needs to turn the current abstract promotion rules into an actual candidate-comparison workflow that can test transfer learning against scratch training on the same deterministic evaluation contract and end with a real promotion decision.

Implementation review after the first pass surfaced three concrete gaps that keep this change open:
- the scratch glomeruli path was ignoring the requested `crop_size`, so a nominal `512 -> 256` workflow could silently run as `256 -> 256`
- the comparison workflow still advertised repo-local `output/...` examples instead of the active runtime output root used on this machine
- the initial `eq-mac` comparison report proved only that the workflow could write failure-report artifacts; it did not prove that both candidate families actually executed successfully under the real unsandboxed MPS runtime

## What Changes

- Define a concrete glomeruli candidate-comparison workflow that trains and evaluates at least one transfer candidate and one scratch candidate under the current `raw_data/.../training_pairs` contract.
- Make the dedicated glomeruli training module CLI the canonical comparison surface; YAML files may remain as optional override inputs, but they are not the authoritative promotion-workflow contract.
- Persist a deterministic validation manifest and promotion report artifact that both candidates must use for comparison.
- Require candidate comparison against trivial all-background and all-foreground baselines, plus the existing compatibility artifact when available.
- End the change with an explicit decision outcome: promote one candidate, promote neither, or record insufficient evidence, including an explicit tie path when both candidates are scientifically indistinguishable.
- Keep runtime compatibility and scientific promotion as separate states in model provenance and documentation.
- Clean up stale glomeruli config/module/doc surfaces that still imply patch-era data paths, config-first execution, or automatic transfer-first fallback semantics for promotion comparison.
- Make the candidate-comparison output contract runtime-root aware so default comparison artifacts land under the active runtime output tree on this machine unless the caller explicitly overrides the destination.
- Require final change validation to include a successful unsandboxed `eq-mac` comparison run where both transfer and scratch execute under the real runtime environment rather than only producing diagnostic failure reports.

## Capabilities

### New Capabilities
- `glomeruli-candidate-comparison`: Defines how transfer and scratch glomeruli candidates are trained, evaluated on the same deterministic evidence set, and compared for promotion.

### Modified Capabilities
- `segmentation-training-contract`: Extends the existing promotion-gate requirements from helper-level checks to a concrete candidate-comparison and promotion-decision workflow.

## Impact

- Affected code: `src/eq/training/train_glomeruli.py`, `src/eq/training/transfer_learning.py`, `src/eq/training/promotion_gates.py`, `src/eq/training/compare_glomeruli_candidates.py`, runtime path helpers, provenance/reporting helpers, and any candidate-evaluation CLI or utility surface introduced by the design.
- Cleanup candidates likely include `configs/glomeruli_finetuning_config.yaml`, `src/eq/inference/run_glomeruli_prediction.py`, and any README or engineering-note surfaces that still blur the canonical training/promotion interface.
- Affected tests: glomeruli training-contract tests, deterministic validation-manifest tests, promotion-gate regression tests, crop-size propagation tests, runtime-output default-path tests, and bounded real-data or fixture-based candidate-comparison checks.
- Affected artifacts: glomeruli model exports, run metadata sidecars, validation manifests, baseline comparisons, prediction-review panels, and promotion reports under the active runtime output root or an explicit caller-supplied override.
- Compatibility risks: candidate retraining may produce new current-namespace artifacts without promoting them; existing compatibility artifacts and downstream quantification consumers must not silently treat a new runtime artifact as scientifically promoted unless the report explicitly says so.
