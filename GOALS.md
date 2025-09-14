## Project Goal

**Research Hypothesis:**

* Determine whether **pretraining a ResNet34 encoder on mitochondria EM images** improves **glomeruli histology segmentation** performance compared to:

  1. ImageNet pretraining
  2. Random initialization (scratch)

---

## Spec Goal

**Experimental Specification for Testing the Hypothesis:**

1. **Experimental Design**

   * Fix encoder: ResNet34
   * Fix decoder: U-Net
   * Compare three encoder initialization sources: scratch, ImageNet, EM-pretrained

2. **Validation Protocol**

   * Use a **fixed validation set** (precomputed tiles, no randomness)
   * Evaluate **per-slide**, not per-tile
   * Disable augmentations during validation
   * Freeze BatchNorm stats during training/eval

3. **Training Procedure**

   * Train up to 30-50 epochs with **early stopping** and **checkpointing** on validation Dice
   * Use appropriate loss:

     * Binary: BCE + Dice
     * Multiclass: CE + softDice
   * Add gradient clipping, weight decay, and ReduceLROnPlateau

4. **Reproducibility**

   * Run each condition with ≥3 random seeds
   * Report mean ± standard deviation of validation metrics

5. **Secondary Experiments**

   * Test effect of crop size (256 vs 512 vs 768)
   * Test freeze/unfreeze schedules (frozen early layers vs gradual unfreeze)

6. **Reporting**

   * Provide comparison table (Dice/IoU) for scratch vs ImageNet vs EM-pretrain
   * Include learning curves (validation Dice, smoothed for display)
   * Highlight both convergence speed and final accuracy

