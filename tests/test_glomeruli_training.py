#!/usr/bin/env python3
"""Simple glomeruli training test to prove the pipeline works."""

from pathlib import Path

import matplotlib.pyplot as plt
from fastai.vision.all import Dice, resnet34, unet_learner

from eq.data_management.datablock_loader import build_segmentation_dls_dynamic_patching
from eq.utils.paths import get_runtime_cohort_path


def main():
    """Train glomeruli model from scratch to prove pipeline works."""
    print("🚀 Starting glomeruli training from scratch...")
    
    data_dir = get_runtime_cohort_path("lauren_preeclampsia")
    from eq.core.constants import DEFAULT_GLOMERULI_MODEL_DIR
    output_dir = f"{DEFAULT_GLOMERULI_MODEL_DIR}_test"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Loading data from: {data_dir}")
    
    # Build DataLoaders using the supported dynamic full-image approach
    dls = build_segmentation_dls_dynamic_patching(data_dir, bs=4, num_workers=0)
    
    print(f"✅ Data loaded: {len(dls.train_ds)} train, {len(dls.valid_ds)} val samples")
    
    # Create learner - use FastAI v2 best practices for binary segmentation
    learn = unet_learner(dls, resnet34, n_out=2, metrics=Dice)
    # FastAI automatically sets CrossEntropyLossFlat for n_out=2
    
    print("🏗️  U-Net model created")
    
    # Train for a few epochs
    print("🚀 Training for 3 epochs...")
    learn.fit_one_cycle(3, 1e-3)
    
    print("✅ Training completed!")
    
    # Save the model
    model_path = Path(output_dir) / "glomeruli_test_model.pkl"
    learn.export(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Show some results
    print("📊 Showing sample results...")
    learn.show_results(max_n=4, figsize=(8, 8))
    plt.savefig(Path(output_dir) / "sample_results.png")
    plt.close()
    print(f"📈 Results saved to: {Path(output_dir) / 'sample_results.png'}")
    
    print("🎉 Glomeruli training pipeline completed successfully!")

if __name__ == "__main__":
    main()
