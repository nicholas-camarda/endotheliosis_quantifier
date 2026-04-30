## Context

The completed `medsam_manual_glomeruli_comparison` workflow established that MedSAM with oracle manual-mask component boxes has much higher agreement with admitted manual glomeruli masks than the current tiled segmenters: MedSAM oracle mean Dice `0.922948` and Jaccard `0.857703`, compared with current transfer Dice `0.708759`/Jaccard `0.567467` and current scratch Dice `0.710214`/Jaccard `0.571440`. That result is encouraging but not deployable because the boxes came from manual masks.

Automatic MedSAM prompting needs a separate localization test. If current segmenter probabilities can provide high-recall rough boxes, MedSAM may be a better mask generator. If they miss glomeruli entirely, MedSAM cannot recover them. The automatic-prompt workflow must therefore measure both proposal recall and final MedSAM mask agreement before any broad replacement candidate is generated.

## Goals / Non-Goals

**Goals:**

- Implement a deterministic pilot workflow named `medsam_automatic_glomeruli_prompts`.
- Generate proposal boxes from tiled current-segmenter probability maps without whole-field single-pass resize.
- Sweep proposal thresholds on the same 20 admitted manual-mask pilot rows and report proposal recall against manual components.
- Run MedSAM from automatic boxes and compare automatic MedSAM masks against manual masks and oracle MedSAM evidence.
- Write all generated artifacts under the active runtime output root.
- Add gates and provenance that make it clear when broad derived-mask generation is allowed.
- Define the transition path for MedSAM automatic masks to become the primary generated glomeruli segmentation source if the pilot passes gates.
- Define the decision path for MedSAM/SAM fine-tuning if prompt engineering is not enough.

**Non-Goals:**

- Do not overwrite manual masks or raw cohort masks.
- Do not claim MedSAM automatic prompts are scientifically promoted from the 20-row pilot alone.
- Do not fine-tune MedSAM or add MedSAM to the project package dependencies.
- Do not use manual masks when generating automatic boxes, except for evaluation of proposal recall and metrics.
- Do not make downstream quantification consume MedSAM automatic masks by default in this change.
- Do not retire current glomeruli segmenter artifacts until MedSAM automatic masks have passed pilot gates and downstream compatibility checks.
- Do not implement MedSAM fine-tuning in this change; this change records when a separate fine-tuning proposal is warranted.

## Decisions

### Decision: Add a separate automatic-prompt workflow

The new workflow SHALL be `medsam_automatic_glomeruli_prompts`, configured by `configs/medsam_automatic_glomeruli_prompts.yaml` and run through `eq run-config`. The worker SHALL live at `src/eq/evaluation/run_medsam_automatic_glomeruli_prompts_workflow.py`.

Alternatives considered:

- Extend `medsam_manual_glomeruli_comparison`: rejected because oracle prompt performance and automatic localization performance are different claims and should have separate artifact roots.
- Immediately regenerate all cohort masks: rejected because automatic box recall is unknown.
- Use a notebook or shell script: rejected because raw-data isolation, provenance, and repeatability matter.

### Decision: Reuse and centralize manual-pilot helpers

The implementation SHALL inspect `src/eq/evaluation/run_medsam_manual_glomeruli_comparison_workflow.py` first and factor shared behavior when needed instead of copying selection, output-path, metric, MedSAM batch, and overlay helpers into a parallel workflow. A likely shared owner is `src/eq/evaluation/medsam_glomeruli_workflow.py` or another focused module under `src/eq/evaluation/`.

Alternatives considered:

- Copy helpers into the automatic workflow: rejected because it would create parallel MedSAM contracts and likely drift in output schemas.
- Move helpers into quantification modules: rejected because these are evaluation and derived-mask generation concerns, not grading behavior.

### Decision: Generate automatic boxes from high-recall current-segmenter probabilities

The initial proposal source SHALL be tiled full-field probability maps from current glomeruli candidates. The workflow SHALL threshold probabilities at configured values, derive connected components, filter implausibly small or large components, pad boxes, merge overlapping boxes, and cap boxes per image. The pilot SHALL record proposal recall against manual components for every candidate and threshold.

The default threshold sweep SHALL be `0.20`, `0.35`, and `0.50`. The workflow SHALL select a default automatic prompt setting only after evaluating recall and guardrails, rather than assuming the same `0.75` threshold used for current mask comparison.

Alternatives considered:

- Use current binary masks at the deployed comparison threshold only: rejected because automatic prompting needs high recall more than clean boundaries.
- Use the entire image as one box: rejected because MedSAM box prompts become too nonspecific and can merge unrelated structures.
- Use manual masks for boxes: rejected for automatic deployment; retained only as oracle evidence from the previous pilot.

### Decision: Keep broad mask replacement as derived-only and gated

If the 20-row automatic-prompt pilot passes configured gates, the workflow MAY support a second mode that writes derived MedSAM automatic masks for a configured image set under `output/derived_masks/medsam_automatic_glomeruli/<run_id>/`. This mode SHALL remain opt-in and SHALL not write into `raw_data/cohorts/**/masks`.

Alternatives considered:

- Replace raw masks in place: rejected because it destroys provenance and mixes manual, MedSAM, and current-segmenter semantics.
- Skip broad generation support: rejected because the user goal is to evaluate replacing generated masks if automatic prompts work.

### Decision: Treat MedSAM automatic masks as a candidate primary generated segmenter after gates pass

If the automatic-prompt pilot passes gates, the transition target SHALL be a derived mask family named `medsam_automatic_glomeruli`. Documentation and config defaults MAY then identify it as the preferred generated-mask candidate for glomeruli segmentation workflows, while preserving manual masks as reference labels and preserving current segmenter artifacts as fallback/comparator evidence.

The transition SHALL be staged:

1. Pilot gate: automatic-prompt metrics, proposal recall, prompt failures, and overlays support use.
2. Derived generation gate: broad derived masks are generated under `output/derived_masks/medsam_automatic_glomeruli/<run_id>/` with manifest provenance.
3. Documentation gate: `docs/TECHNICAL_LAB_NOTEBOOK.md` and workflow config docs describe MedSAM automatic masks as the current preferred generated-mask candidate.
4. Downstream gate: quantification or grading configs opt into MedSAM-derived masks explicitly and record mask source in provenance.

Alternatives considered:

- Immediately change all downstream defaults: rejected because downstream feature stability and generated-mask transport need evidence.
- Keep MedSAM as audit-only forever: rejected if automatic prompts close enough of the oracle gap and overlays are acceptable.
- Rename raw masks in place: rejected because raw/manual/derived semantics must remain separate.

### Decision: Use pilot failure mode to decide between better prompts and fine-tuning

Fine-tuning SHALL not be part of this implementation. The workflow SHALL instead classify failure modes in `summary.json`:

- Poor proposal recall with good oracle MedSAM performance means the box proposer needs improvement before MedSAM fine-tuning.
- Good proposal recall with poor automatic MedSAM boundary metrics or overlays means a separate MedSAM/SAM fine-tuning proposal is justified.
- Good proposal recall and good automatic MedSAM metrics means prompt-based MedSAM is sufficient for the next derived-mask generation step.

Alternatives considered:

- Start fine-tuning immediately: rejected because the oracle pilot suggests MedSAM boundaries are already strong when prompted well.
- Never fine-tune: rejected because automatic prompts may expose domain-specific boundary or texture failures not visible in oracle boxes.

### Decision: Report descriptive evidence and explicit failure modes

The pilot SHALL report proposal recall, prompt failures, final mask metrics, overlay panels, and guardrail decisions. It SHALL identify missed manual components, too-many-box images, empty proposal images, and area-ratio outliers. It SHALL not update README-facing model claims or silently promote MedSAM automatic masks into quantification defaults.

## Risks / Trade-offs

- Current segmenter misses a glomerulus entirely -> MedSAM cannot segment missing prompts, so proposal recall is a required first-class metric.
- Low thresholds create too many false-positive boxes -> The workflow caps boxes, records overflow, and reports area-ratio/proposal-count guardrails.
- Automatic prompts over-segment non-glomerular tissue -> Metrics and overlays are reviewed before broad generation.
- Reusing current segmenter as a box proposer may bake in localization bias -> The report separates localization recall from MedSAM boundary quality.
- Broad derived masks could be mistaken for manual truth -> Artifact paths and summary wording label them as MedSAM automatic derived masks, never raw/manual masks.
- Premature primary-segmenter switch could change downstream quantification silently -> Configs must opt in explicitly and provenance must record `mask_source=medsam_automatic_glomeruli`.
- Fine-tuning could add dependency and training complexity before it is needed -> Fine-tuning requires a separate OpenSpec change after failure-mode evidence.

## Migration Plan

1. Factor reusable MedSAM/manual workflow helpers if needed to avoid duplicate contracts.
2. Add automatic proposal-box generation and proposal-recall metrics on the 20-row admitted manual pilot.
3. Add MedSAM automatic-mask generation, metrics, overlays, and summary gates.
4. Run focused unit tests, import/CLI smoke checks, and `openspec validate pilot-medsam-automatic-glomeruli-prompts --strict`.
5. Run a dry-run config and then the real 20-row automatic-prompt pilot.
6. Review metrics and overlays before enabling any broad derived-mask generation.
7. If pilot gates pass, generate a broad derived-mask candidate under `output/derived_masks/medsam_automatic_glomeruli/<run_id>/`.
8. If broad derived masks pass review, update docs and opt-in configs to describe/use `medsam_automatic_glomeruli` as the preferred generated-mask candidate.
9. If automatic boxes fail by localization, create a box-proposer improvement change; if they fail by boundary quality despite good localization, create a MedSAM/SAM fine-tuning change.

Rollback is to remove the new output directory. No raw data or existing masks are modified.

## Explicit Decisions

- Workflow ID: `medsam_automatic_glomeruli_prompts`.
- Config path: `configs/medsam_automatic_glomeruli_prompts.yaml`.
- Worker module: `src/eq/evaluation/run_medsam_automatic_glomeruli_prompts_workflow.py`.
- Shared-helper candidate module: `src/eq/evaluation/medsam_glomeruli_workflow.py`.
- Default output root: `output/segmentation_evaluation/medsam_automatic_glomeruli_prompts/<run_id>/`.
- Default run ID: `pilot_medsam_automatic_glomeruli_prompts`.
- Pilot sample size: 20 admitted manual-mask rows using the same deterministic selection contract as `medsam_manual_glomeruli_comparison`.
- Proposal thresholds: `0.20`, `0.35`, `0.50`.
- Proposal box minimum component area: `2000`.
- Proposal box maximum component area: `750000`.
- Proposal padding: `16` pixels.
- Proposal merge IoU: `0.25`.
- Maximum boxes per image: `20`.
- Broad derived output root, if enabled: `output/derived_masks/medsam_automatic_glomeruli/<run_id>/`.
- Primary generated mask source label after gates pass: `medsam_automatic_glomeruli`.
- Required provenance field for downstream opt-in: `mask_source=medsam_automatic_glomeruli`.
- Documentation files to update after gates pass: `docs/TECHNICAL_LAB_NOTEBOOK.md` and the relevant workflow config notes for glomeruli segmentation inputs.

## Open Questions

- [audit_first_then_decide] The default proposal candidate and threshold will be selected from the pilot's `proposal_recall.csv`, `metrics.csv`, `prompt_failures.csv`, and overlays.
- [audit_first_then_decide] The first broad derived-mask image set will be chosen only after the 20-row automatic-prompt pilot is reviewed.
- [audit_first_then_decide] The first downstream config to opt into `medsam_automatic_glomeruli` will be selected after broad derived masks and overlay review.
- [audit_first_then_decide] Fine-tuning direction will be decided only if pilot failure-mode evidence shows boundary quality is the limiting factor after localization is adequate.
- [defer_ok] Human review/adjudication of MedSAM automatic masks can be added later if the pilot supports using these masks downstream.