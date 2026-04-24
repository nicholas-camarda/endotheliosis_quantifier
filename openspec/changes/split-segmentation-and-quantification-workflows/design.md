## Context

The repository already has three distinct workflow domains with different scientific and operational meanings:

1. glomeruli candidate comparison and promotion evidence,
2. external-cohort segmentation transport audit or prediction,
3. downstream endotheliosis quantification and ordinal grading.

The current workflow-config surface does not reflect that split cleanly. `eq run-config` dispatches `segmentation_fixedloader_full_retrain` to `src/eq/training/run_segmentation_fixedloader_full.py`, whose implementation refreshes the cohort manifest, trains a mitochondria base, derives the new base artifact path, and launches glomeruli candidate comparison. The name still encodes stale implementation jargon (`fixedloader_full`) instead of the actual contract. At the same time, quantification still has a separate direct CLI surface (`quant-endo` plus `prepare-quant-contract`), and the scored-only cohort spec already treats transport audit and MR concordance as explicit gates before downstream grading use.

That leaves the repository in an inconsistent state:

- candidate comparison is exposed through a stale mixed-purpose workflow name,
- transport audit and MR concordance are specified as independent gates but not yet reflected in the workflow-config family,
- quantification is operationally separate but not represented as a dedicated YAML workflow family alongside the segmentation-side workflows.

Because these stages produce different artifact classes under different runtime output roots, a mixed runner increases audit risk. Promotion evidence belongs under `output/segmentation_evaluation/`, model-generated masks belong under `output/predictions/`, and quantification artifacts belong under `output/quantification_results/`. One broad run name makes it too easy to overstate what a run actually validated.

## Goals / Non-Goals

**Goals:**

- Define separate supported workflow-config families for candidate comparison, transport audit or MR concordance, and quantification.
- Replace stale workflow naming with names that describe the actual stage and output contract.
- Keep `eq run-config` as the repository-level YAML entrypoint while making each YAML correspond to one stage with one scientific meaning.
- Require downstream stages to consume explicit upstream artifact references rather than silently retraining models or rerunning segmentation.
- Preserve the current runtime storage boundaries: evaluation under `output/segmentation_evaluation/`, prediction assets under `output/predictions/`, and quantification outputs under `output/quantification_results/`.
- Keep the canonical ordinal estimator and quantification artifact schemas intact unless a separate change modifies them.

**Non-Goals:**

- Changing the scientific promotion gate for glomeruli candidates.
- Changing the ordinal estimator family, score semantics, or grouped evaluation rules.
- Admitting MR into segmentation training or predicted-ROI training expansion in phase 1.
- Introducing compatibility aliases or fallback behavior that keep the stale mixed workflow name alive indefinitely.
- Redesigning the low-level training CLIs or the quantification artifact schemas beyond what is needed to support cleaner orchestration.

## Decisions

1. **Keep one repository-level YAML runner, but split the workflow families underneath it.**
   - Rationale: the user already pushed the repository toward directly executable YAMLs. `eq run-config` is the right top-level control surface, but a single mixed workflow ID is not. The split should happen at the workflow-family level, not by abandoning the generic runner.
   - Alternatives considered:
   - Keep direct module CLIs as the only supported surfaces. Rejected because that reintroduces manual shell stitching and weakens reproducible multi-step provenance.
   - Create one bigger umbrella YAML that branches by flags. Rejected because it keeps unrelated stages coupled and makes audit boundaries ambiguous.

2. **Define three stage-specific workflow families with explicit names and contracts.**
   - Rationale: the implementation should expose separate YAMLs and workflow IDs for:
     - candidate comparison or promotion evidence,
     - transport audit or MR concordance,
     - quantification.
     Each workflow name should describe what it proves and what artifacts it writes.
   - Alternatives considered:
   - Split only candidate comparison and keep transport audit embedded in quantification. Rejected because transport audit is an upstream gate on segmentation usability, not a side effect of ordinal grading.
   - Split candidate comparison and MR only, but leave all non-MR quantification coupled to segmentation inference. Rejected because the same separation-of-concerns problem would remain for non-MR external cohorts.

3. **Treat the underlying training, inference, and quantification commands as worker surfaces, not competing top-level contracts.**
   - Rationale: dedicated commands such as `eq.training.compare_glomeruli_candidates` and `quant-endo` can remain implementation entrypoints, but the reproducible repository contract should live in stage-specific YAML workflows. The workflow runner owns orchestration, logging, path resolution, and artifact linking; the worker commands own the stage internals.
   - Alternatives considered:
   - Make YAML only an optional overlay on top of direct CLIs. Rejected because the repo already moved away from that pattern for reproducibility and self-documenting configs.

4. **Require explicit upstream artifact inputs across workflow boundaries.**
   - Rationale: candidate comparison should emit named promoted or non-promoted artifacts; transport audit should accept a specific segmentation artifact and emit reviewable prediction or concordance artifacts; quantification should accept an explicit segmentation artifact or accepted predicted-ROI input surface. No downstream workflow should retrain a segmentation model because a path was omitted.
   - Alternatives considered:
   - Auto-discover "latest" artifacts per stage. Rejected because it is brittle, obscures provenance, and can silently evaluate the wrong model.
   - Allow downstream workflows to fall back to training or prediction when an input is missing. Rejected because that conflates gates and violates the no-fallback rule.

5. **Rename the current mixed candidate-comparison runner to match its true responsibility and retire the stale name.**
   - Rationale: `run_segmentation_fixedloader_full.py` is implementation-era jargon. A name such as `run_segmentation_candidate_comparison_workflow.py` or a narrower `run_glomeruli_candidate_comparison_workflow.py` communicates the actual contract and leaves room for separate transport-audit and quantification workflow modules.
   - Alternatives considered:
   - Keep the module name and only change docs. Rejected because the stale name will keep leaking into logs, stack traces, and imports.
   - Add the new name as an alias while retaining the old one as a permanent supported path. Rejected because it preserves duplicate public surfaces without need.

6. **Align workflow output roots with their stage instead of one run pretending to own multiple output classes.**
   - Rationale: candidate-comparison reports stay under `output/segmentation_evaluation/`; transport audit and MR concordance also stay under `output/segmentation_evaluation/` while model-generated masks and overlays go under `output/predictions/`; quantification remains under `output/quantification_results/`. The workflow spec should make those boundaries explicit so implementation does not create hybrid output trees.
   - Alternatives considered:
   - Use a single workflow-run directory that nests all downstream outputs. Rejected because it obscures which outputs are evaluation evidence versus reusable predicted assets versus grading artifacts.

## Risks / Trade-offs

- [Risk] Renaming workflow IDs and config files breaks existing local habits and tests. → Mitigation: make the change explicitly breaking in the spec, update all committed configs and tests together, and avoid indefinite aliases.
- [Risk] Candidate comparison and transport audit may still duplicate some path-resolution and logging logic. → Mitigation: reuse shared runner helpers for runtime root resolution, subprocess logging, and artifact-path recording instead of copy-pasting workflow plumbing.
- [Risk] Splitting workflows may tempt implementation to auto-chain stages for convenience. → Mitigation: require explicit upstream artifact references and fail closed when a required artifact is not supplied.
- [Risk] Quantification may still contain direct CLI and workflow-config surfaces at once. → Mitigation: document one reproducible YAML contract as the stage-level workflow surface and keep any direct command as a thin worker or convenience wrapper, not a second conflicting orchestration contract.
- [Risk] MR concordance could be treated as proof of general transportability or training readiness. → Mitigation: keep MR phase 1 outputs labeled as external concordance and transport-evaluation artifacts only, consistent with the scored-only cohort contract.

## Migration Plan

1. Add the new workflow-config capability and spec deltas first so the target split is explicit before code changes.
2. Rename the mixed candidate-comparison runner module and replace the stale workflow ID and YAML name.
3. Introduce dedicated transport-audit or MR-concordance workflow IDs and YAMLs that accept explicit segmentation artifact inputs.
4. Introduce or normalize a dedicated quantification workflow YAML that calls the contract-first quantification path without embedding candidate comparison or transport audit behavior.
5. Update dispatch, tests, docs, and committed config examples together so the supported workflow family is internally consistent.
6. Remove the retired mixed workflow names rather than keeping them as a second supported contract.

## Open Questions

- Should the candidate-comparison workflow module name stay broad (`run_segmentation_candidate_comparison_workflow.py`) or become glomeruli-specific (`run_glomeruli_candidate_comparison_workflow.py`)? The latter is semantically tighter if this workflow will remain glomeruli-only.
- Should the quantification workflow YAML wrap the existing `quant-endo` semantics directly, or should `quant-endo` become a thin caller into the new workflow runner so there is only one orchestration implementation?
- Should MR concordance be a general transport-audit workflow mode or its own separate workflow family with stricter TIFF-tiling inputs? The answer depends on whether future external cohorts will share MR-like preprocessing constraints.
