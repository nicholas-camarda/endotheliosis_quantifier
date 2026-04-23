## 1. Candidate Comparison Inputs

- [x] 1.1 Inventory the current supported glomeruli training root, compatibility artifact, and any existing deterministic validation helpers that will feed candidate comparison.
- [x] 1.2 Define the exact canonical transfer and scratch candidate commands, including the explicit seed, output locations, and provenance fields that the comparison workflow will require.
- [x] 1.3 Make the dedicated glomeruli training module CLI the canonical comparison interface, and mark YAML config files as optional non-authoritative overlays rather than the promotion contract.
- [x] 1.4 Decide how the compatibility artifact is included when available without allowing it to be mistaken for a promoted model.
- [x] 1.5 Inventory and clean or retire stale glomeruli config/module/doc surfaces that still imply patch-era paths, config-first execution, or silent transfer/scratch fallback semantics.

## 2. Deterministic Evaluation Contract

- [x] 2.1 Implement or persist a deterministic glomeruli validation manifest covering positive, boundary, and background examples, using mutually exclusive crop categories and image-diverse selection when sufficient coverage exists.
- [x] 2.2 Add report helpers that compute shared-manifest Dice/Jaccard metrics, trivial baselines, manifest coverage summaries, and prediction-degeneracy review on the shared evidence set.
- [x] 2.3 Add explicit prediction-review handling for “reasonable whole-glomerulus approximation” vetoes beyond trivial all-background/all-foreground collapse.
- [x] 2.4 Add regression tests for deterministic manifest reuse and for report behavior when one candidate family is unavailable or blocked.

## 3. Candidate Training Workflow

- [x] 3.1 Implement the bounded transfer-candidate training path under the supported `raw_data/.../training_pairs` contract.
- [x] 3.2 Implement the bounded scratch-candidate training path under the same contract and evaluation setup, ensuring the requested `crop_size` propagates through batch-size sizing, dynamic cropping, and provenance instead of collapsing silently to `image_size`.
- [x] 3.3 Ensure both candidate families emit current-namespace artifacts with provenance that distinguishes runtime support from scientific promotion.
- [x] 3.4 Make the candidate-comparison workflow default its artifacts to the active runtime output root on this machine when `--output-dir` is omitted, while preserving an explicit override path.

## 4. Promotion Decision Workflow

- [x] 4.1 Implement a promotion report artifact that compares transfer, scratch, trivial baselines, and the compatibility artifact when available, and label review panels with clear crop provenance, panel order, and per-panel metrics.
- [x] 4.2 Encode a composite promotion gate that requires baseline clearance, non-degenerate predictions, and shared-manifest visual review rather than a single-metric winner rule.
- [x] 4.3 Encode the explicit final decision states `promoted`, `blocked`, and `insufficient_evidence` without auto-promoting the most recent runtime-compatible model.
- [x] 4.4 Record an explicit tie outcome inside `insufficient_evidence` when transfer and scratch remain scientifically indistinguishable, using an absolute practical tie margin of `0.02` or less on both shared-manifest Dice and Jaccard, and keep both artifacts available as explicit research-use comparators for downstream segmentation and quantification.
- [x] 4.5 Add tests that promotion remains blocked when both candidates fail baselines or prediction-review gates, and that ties do not create two promoted defaults.
- [x] 4.6 Structure promotion report rows and provenance so the initial one-seed-per-family comparison can later expand to repeated-seed comparisons without breaking the artifact contract.

## 5. Validation And Documentation

- [x] 5.1 Run the bounded initial one-seed-per-family candidate comparison validation unsandboxed in `eq-mac`, and require the recorded artifacts to show that both transfer and scratch executed successfully rather than only producing structured failure reports.
- [x] 5.2 Update relevant docs or engineering notes to describe the canonical CLI-based candidate-comparison workflow, the runtime-root output default on this machine, the non-authoritative role of YAML overlays, and the meaning of promoted versus runtime-compatible versus tied research-use artifacts.
- [x] 5.3 Run `env OPENSPEC_TELEMETRY=0 openspec validate compare-and-promote-glomeruli-candidates --strict`.
- [x] 5.4 Add regression tests for scratch crop-size propagation and for the compare workflow's runtime-root default output behavior.
