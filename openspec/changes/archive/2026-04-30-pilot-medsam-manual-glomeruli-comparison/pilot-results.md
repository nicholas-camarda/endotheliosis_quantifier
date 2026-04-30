## Pilot Run 2026-04-30

## Command

```bash
/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/medsam_manual_glomeruli_comparison.yaml
```

## Output Path

`/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/medsam_manual_glomeruli_comparison/pilot_medsam_manual_glomeruli_comparison`

## Key Results

- Selected inputs: 20 admitted manual-mask rows.
- Metric rows: 60 rows, covering MedSAM oracle, current transfer, and current scratch.
- Prompt failures: 0.
- MedSAM oracle mean Dice: 0.922948.
- MedSAM oracle mean Jaccard: 0.857703.
- Current transfer mean Dice: 0.708759.
- Current transfer mean Jaccard: 0.567467.
- Current scratch mean Dice: 0.710214.
- Current scratch mean Jaccard: 0.571440.

## Decision

The oracle-box MedSAM pilot supports reviewing MedSAM as a stronger boundary-quality candidate than the current segmenters on this admitted manual-mask subset. This remains upper-bound prompt evidence only. The next decision should be based on overlay review before adding automatic MedSAM prompts from current segmenter proposals.
