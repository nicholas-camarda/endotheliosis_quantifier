## Pilot Run 2026-04-30

## Command

```bash
/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/medsam_automatic_glomeruli_prompts.yaml
```

## Output Path

`/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/medsam_automatic_glomeruli_prompts/pilot_medsam_automatic_glomeruli_prompts`

## Key Results

- Selected inputs: 20 admitted manual-mask rows.
- Selected proposal source: scratch current segmenter.
- Selected proposal threshold: 0.20.
- Proposal recall mean: 1.000000.
- Prompt failures: 0.
- MedSAM automatic mean Dice: 0.784805.
- MedSAM automatic mean Jaccard: 0.654146.
- MedSAM automatic mean precision: 0.737846.
- MedSAM automatic mean recall: 0.878500.
- Current transfer mean Dice: 0.708759.
- Current transfer mean Jaccard: 0.567467.
- Current scratch mean Dice: 0.710214.
- Current scratch mean Jaccard: 0.571440.
- Prior oracle MedSAM mean Dice: 0.922948.
- Prior oracle MedSAM mean Jaccard: 0.857703.

## Gate Decision

- Gates passed: false.
- Failure mode: `medsam_boundary_quality`.
- Primary generated-mask transition status: `blocked`.
- Recommended generated mask source: none.
- Fine-tuning recommendation: `open_medsam_sam_fine_tuning_change`.

## Decision

Automatic MedSAM improves over the current segmenters on this pilot, but it does not close enough of the oracle gap to become the primary generated glomeruli segmentation source yet. Documentation and downstream configs should not switch to `medsam_automatic_glomeruli` from this run. Because proposal recall was perfect and prompt failures were zero, the next technical question is whether prompt geometry can be improved enough, or whether a separate MedSAM/SAM fine-tuning change is needed for boundary quality.
