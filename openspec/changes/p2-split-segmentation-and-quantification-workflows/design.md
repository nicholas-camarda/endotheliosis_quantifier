## Context

The repository already has three distinct workflow domains with different scientific and operational meanings:

1. glomeruli candidate comparison and promotion evidence,
2. external-cohort segmentation transport audit on standard cohort image regimes,
3. high-resolution concordance on large-field microscope images such as the MR TIFF lane,
4. downstream endotheliosis quantification and ordinal grading.

The current workflow-config surface does not reflect that split cleanly. `eq run-config` dispatches `full_segmentation_retrain` to `src/eq/training/run_full_segmentation_retrain.py`, whose implementation refreshes the cohort manifest, trains a mitochondria base, derives the new base artifact path, and launches glomeruli candidate comparison. The name now matches the committed full-retraining config, but the runner still mixes stages that should become separate workflow contracts. At the same time, quantification still has a separate direct CLI surface (`quant-endo` plus `prepare-quant-contract`), and the scored-only cohort spec already treats transport audit and MR concordance as explicit gates before downstream grading use.

That leaves the repository in an inconsistent state:

- candidate comparison is exposed through a stale mixed-purpose workflow name,
- transport audit and MR concordance are specified as independent gates but not yet reflected in distinct workflow-config families,
- quantification is operationally separate but not represented as a dedicated YAML workflow family alongside the segmentation-side workflows.

Because these stages produce different artifact classes under different runtime output roots, a mixed runner increases audit risk. Promotion evidence belongs under `output/segmentation_evaluation/`, model-generated masks belong under `output/predictions/`, and quantification artifacts belong under `output/quantification_results/`. One broad run name makes it too easy to overstate what a run actually validated.

## Goals / Non-Goals

**Goals:**

- Define separate supported workflow-config families for candidate comparison, standard transport audit, high-resolution concordance, and quantification.
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

## Explicit Decisions

1. **Keep one repository-level YAML runner, but split the workflow families underneath it.**
   - Rationale: the user already pushed the repository toward directly executable YAMLs. `eq run-config` is the right top-level control surface, but a single mixed workflow ID is not. The split should happen at the workflow-family level, not by abandoning the generic runner.
   - Alternatives considered:
   - Keep direct module CLIs as the only supported surfaces. Rejected because that reintroduces manual shell stitching and weakens reproducible multi-step provenance.
   - Create one bigger umbrella YAML that branches by flags. Rejected because it keeps unrelated stages coupled and makes audit boundaries ambiguous.

2. **Define stage-specific workflow families with exact names and exact owning modules, including a separate high-resolution concordance family.**
   - Rationale: the implementation should expose separate YAMLs and workflow IDs for:
     - candidate comparison or promotion evidence via `workflow: glomeruli_candidate_comparison`, implemented by `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`, configured by `configs/glomeruli_candidate_comparison.yaml`
     - standard transport audit via `workflow: glomeruli_transport_audit`, implemented by `src/eq/evaluation/run_glomeruli_transport_audit_workflow.py`, configured by `configs/glomeruli_transport_audit.yaml`
     - high-resolution concordance for large-field microscope images such as MR via `workflow: highres_glomeruli_concordance`, implemented by `src/eq/evaluation/run_highres_glomeruli_concordance_workflow.py`, configured by `configs/highres_glomeruli_concordance.yaml`
     - quantification via `workflow: endotheliosis_quantification`, implemented by `src/eq/quantification/run_endotheliosis_quantification_workflow.py`, configured by `configs/endotheliosis_quantification.yaml`
     Each workflow name should describe what it proves and what artifacts it writes.
   - Alternatives considered:
   - Fold MR into the standard transport-audit family. Rejected because giant TIFF or future large-field microscope regimes introduce materially different tiling, preprocessing, runtime cost, and concordance logic than standard cohort images.
   - Split only candidate comparison and keep transport audit embedded in quantification. Rejected because transport audit is an upstream gate on segmentation usability, not a side effect of ordinal grading.

3. **Treat the underlying training, inference, and quantification commands as worker surfaces, not competing top-level contracts.**
   - Rationale: dedicated commands such as `eq.training.compare_glomeruli_candidates` and `quant-endo` can remain implementation entrypoints, but the reproducible repository contract should live in stage-specific YAML workflows. The workflow runner owns orchestration, logging, path resolution, and artifact linking; the worker commands own the stage internals.
   - Alternatives considered:
   - Make YAML only an optional overlay on top of direct CLIs. Rejected because the repo already moved away from that pattern for reproducibility and self-documenting configs.

4. **Require explicit upstream artifact inputs across workflow boundaries.**
   - Rationale: candidate comparison should emit named promoted or non-promoted artifacts; transport audit should accept a specific segmentation artifact and emit reviewable prediction or concordance artifacts; quantification should accept an explicit segmentation artifact or accepted predicted-ROI input surface. No downstream workflow should retrain a segmentation model because a path was omitted.
   - Alternatives considered:
   - Auto-discover "latest" artifacts per stage. Rejected because it is brittle, obscures provenance, and can silently evaluate the wrong model.
   - Allow downstream workflows to fall back to training or prediction when an input is missing. Rejected because that conflates gates and violates the no-fallback rule.

5. **Rename the current mixed candidate-comparison runner to one exact glomeruli-specific surface and retire the stale name.**
   - Rationale: this repository is specifically about glomerular segmentation and endotheliosis quantification, so the split workflow name should be precise rather than a broad full-retraining label. The exact dedicated candidate-comparison surface is:
     - module: `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`
     - function: `run_glomeruli_candidate_comparison_workflow(...)`
     - workflow ID: `glomeruli_candidate_comparison`
     - config: `configs/glomeruli_candidate_comparison.yaml`
     This communicates the actual contract and leaves room for separate transport-audit, high-resolution concordance, and quantification workflow modules.
   - Alternatives considered:
   - Keep the module name and only change docs. Rejected because the stale name will keep leaking into logs, stack traces, and imports.
   - Add the new name as an alias while retaining the old one as a permanent supported path. Rejected because it preserves duplicate public surfaces without need.

6. **Audit `quant-endo` rigorously against explicit retention criteria before deciding whether it survives as a thin caller or is retired.**
   - Rationale: the live CLI wrapper in `src/eq/__main__.py` currently delegates to `run_contract_first_quantification(...)`, so it is not an entirely separate quantification implementation. But the wrapper still carries stale training-era framing, unused or misleading knobs such as `--batch-size` and `--epochs`, and overlapping orchestration with `prepare-quant-contract`. The repo should not bless it blindly; the implementation needs an explicit salvage-or-retire audit.
   - Retention criteria:
     - `quant-endo` may survive only as a thin compatibility caller into `src/eq/quantification/run_endotheliosis_quantification_workflow.py`
     - the surviving wrapper may keep only arguments that are semantically consumed by the canonical workflow: `--data-dir`, `--segmentation-model`, `--score-source`, `--annotation-source`, `--mapping-file`, `--output-dir`, `--apply-migration`, and `--stop-after`
     - the wrapper SHALL NOT keep dead or misleading arguments such as `--batch-size` and `--epochs` unless the canonical workflow is explicitly changed to consume them
     - the wrapper SHALL NOT mutate runtime behavior beyond argument forwarding, standard CLI logging, and mode validation
     - if those conditions cannot be met cleanly, `quant-endo` SHALL be retired rather than preserved as a second orchestration contract
   - `prepare-quant-contract` may survive only as a thin compatibility caller that forwards into the same canonical workflow with `stop_after=contract`; otherwise it too should be retired
   - Alternatives considered:
   - Declare `quant-endo` the canonical workflow surface immediately. Rejected because the wrapper has not been reviewed recently and may still expose stale behavior or dead arguments.
   - Retire `quant-endo` immediately and force a new YAML runner. Rejected because the underlying call path may already be salvageable as a thin compatibility wrapper once its stale surface is cleaned up.

7. **Align workflow output roots with their stage instead of one run pretending to own multiple output classes.**
   - Rationale: candidate-comparison reports stay under `output/segmentation_evaluation/`; standard transport audit and high-resolution concordance also write evaluation artifacts under `output/segmentation_evaluation/` while reusable prediction assets go under `output/predictions/`; quantification remains under `output/quantification_results/`. The workflow spec should make those boundaries explicit so implementation does not create hybrid output trees.
   - Alternatives considered:
   - Use a single workflow-run directory that nests all downstream outputs. Rejected because it obscures which outputs are evaluation evidence versus reusable predicted assets versus grading artifacts.

## Risks / Trade-offs

- [Risk] Renaming workflow IDs and config files breaks existing local habits and tests. → Mitigation: make the change explicitly breaking in the spec, update all committed configs and tests together, and avoid indefinite aliases.
- [Risk] Candidate comparison and transport audit may still duplicate some path-resolution and logging logic. → Mitigation: reuse shared runner helpers for runtime root resolution, subprocess logging, and artifact-path recording instead of copy-pasting workflow plumbing.
- [Risk] Splitting workflows may tempt implementation to auto-chain stages for convenience. → Mitigation: require explicit upstream artifact references and fail closed when a required artifact is not supplied.
- [Risk] Quantification may still contain direct CLI and workflow-config surfaces at once. → Mitigation: audit `quant-endo` first; if salvageable, keep it as a thin caller into the canonical workflow runner, otherwise retire it instead of letting both orchestration paths drift.
- [Risk] Large-field microscope support could accrete ad hoc MR-only hacks. → Mitigation: make it a separate high-resolution workflow family with explicit tiling and concordance inputs so future microscope regimes can reuse the same contract without pretending they are standard transport audits.

## Migration Plan

1. Add the new workflow-config capability and spec deltas first so the target split is explicit before code changes.
2. Rename the mixed candidate-comparison runner module and replace the stale workflow ID and YAML name.
3. Introduce a dedicated standard transport-audit workflow ID and YAML that accepts explicit segmentation artifact inputs.
4. Introduce a separate high-resolution concordance workflow ID and YAML for MR-like large-field image regimes.
5. Audit `quant-endo` and `prepare-quant-contract` against the explicit retention criteria, then either retire them or normalize them into thin callers over `src/eq/quantification/run_endotheliosis_quantification_workflow.py`.
6. Introduce or normalize the dedicated quantification workflow YAML `configs/endotheliosis_quantification.yaml` so it calls the canonical workflow runner without embedding candidate comparison, transport audit, or high-resolution concordance behavior.
7. Update dispatch, tests, docs, and committed config examples together so the supported workflow family is internally consistent.
8. Remove the retired mixed workflow names rather than keeping them as a second supported contract.

## Open Questions

- [audit_first_then_decide] Inspect `quant-endo`, `prepare-quant-contract`, `prepare_quantification_contract(...)`, `run_endotheliosis_scoring_pipeline(...)`, and `run_contract_first_quantification(...)` to decide whether the retained canonical quantification runner needs a second worker helper beyond `run_contract_first_quantification(...)`, or whether that function itself is the correct canonical engine to wrap directly.
