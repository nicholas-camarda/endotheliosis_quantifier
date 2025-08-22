#!/usr/bin/env python3
"""Production pipeline that generates actual output files, models, and visualizations."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from eq.segmentation.fastai_segmenter import FastaiSegmenter, SegmentationConfig
from eq.utils.output_manager import OutputManager


def run_pipeline(epochs: int = 10, run_type: str = "production"):
    """Run the pipeline with configurable epochs.
    
    Args:
        epochs: Number of training epochs
        run_type: Type of run ('production' or 'development')
    """
    
    # Set environment for MPS compatibility
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    print(f"🚀 === {run_type.upper()} PIPELINE === 🚀")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output manager and directories
    output_manager = OutputManager()
    data_source = output_manager.get_data_source_name("data/preeclampsia_data")
    output_dirs = output_manager.create_output_directory(data_source, run_type)
    
    print(f"📁 Output directory: {output_dirs['main']}")
    
    # Create configuration based on run type
    if run_type == "development":
        config = SegmentationConfig(
            epochs=epochs,
            batch_size=4,
            device_mode='development',
            model_arch='resnet18',  # Smaller model for development
            learning_rate=0.001,
            model_save_path=output_dirs['models'] / f"{run_type}_segmenter.pkl",
            results_save_path=output_dirs['results']
        )
    else:  # production
        config = SegmentationConfig(
            epochs=epochs,
            batch_size=4,
            device_mode='development',
            model_arch='resnet50',  # Larger model for production
            learning_rate=0.001,
            model_save_path=output_dirs['models'] / f"{run_type}_segmenter.pkl",
            results_save_path=output_dirs['results']
        )
    
    print("⚙️ Configuration:")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Architecture: {config.model_arch}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Device mode: {config.device_mode}")
    print(f"   Run type: {run_type}")
    
    # Create segmenter
    segmenter = FastaiSegmenter(config)
    print("✅ Segmenter created")
    
    try:
        # 1. Prepare data
        print("\n📊 === DATA PREPARATION ===")
        cache_dir = Path('data/preeclampsia_data/cache')
        segmenter.prepare_data_from_cache(cache_dir, 'glomeruli')
        
        # Save data statistics
        data_stats = {
            'training_samples': len(segmenter.dls.train_ds),
            'validation_samples': len(segmenter.dls.valid_ds),
            'batch_size': segmenter.config.batch_size,
            'image_size': segmenter.config.image_size
        }
        
        print("✅ Data prepared:")
        for key, value in data_stats.items():
            print(f"   {key}: {value}")
        
        # 2. Create model
        print("\n🧠 === MODEL CREATION ===")
        segmenter.create_model('glomeruli')
        
        model_stats = {
            'architecture': segmenter.config.model_arch,
            'parameters': sum(p.numel() for p in segmenter.learn.model.parameters()),
            'device': str(segmenter.device)
        }
        
        print("✅ Model created:")
        for key, value in model_stats.items():
            print(f"   {key}: {value}")
        
        # 3. Train model
        print(f"\n🏋️ === MODEL TRAINING ({epochs} epochs) ===")
        print("Starting training... This will generate real training curves!")
        
        training_result = segmenter.train(epochs=config.epochs, learning_rate=config.learning_rate)
        
        print("✅ Training completed:")
        print(f"   Final training loss: {training_result.get('train_loss', 'N/A')}")
        print(f"   Final validation loss: {training_result.get('valid_loss', 'N/A')}")
        print(f"   Final dice score: {training_result.get('dice', 'N/A')}")
        
        # 4. Save model
        print("\n💾 === SAVING MODEL ===")
        try:
            model_path = segmenter.save_model(output_dirs['models'] / f"{run_type}_glomerulus_segmenter")
            print(f"✅ Model saved to: {model_path}")
        except Exception as e:
            print(f"⚠️ Model saving failed: {e}")
            print("   Continuing with visualization and results generation...")
            model_path = "Model saving failed"
        
        # 5. Generate training plots
        print("\n📈 === GENERATING PLOTS ===")
        generate_training_plots(segmenter, output_dirs['plots'], training_result)
        
        # 6. Run inference examples
        print("\n🔍 === RUNNING INFERENCE ===")
        run_inference_examples(segmenter, output_dirs['results'])
        
        # 7. Generate summary report
        print("\n📋 === GENERATING REPORT ===")
        run_info = {
            'data_source': data_source,
            'run_type': run_type,
            'config': {
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'architecture': config.model_arch,
                'learning_rate': config.learning_rate,
                'device_mode': config.device_mode
            },
            'data_stats': data_stats,
            'model_stats': model_stats,
            'results': training_result,
            'model_path': str(model_path)
        }
        
        output_manager.create_run_summary(output_dirs, run_info)
        
        print(f"\n🎉 === {run_type.upper()} PIPELINE COMPLETE ===")
        print(f"📁 All outputs saved to: {output_dirs['main']}")
        print("\n📋 Generated files:")
        for file_path in output_dirs['main'].rglob("*"):
            if file_path.is_file():
                print(f"   {file_path.relative_to(output_dirs['main'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_training_plots(segmenter, plots_dir, training_result):
    """Generate training visualization plots."""
    
    # Get training history from learner
    history = segmenter.learn.recorder
    
    # Create training loss plot
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history.losses, label='Training Loss', color='blue')
    plt.plot(history.val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Learning rate
    plt.subplot(1, 2, 2)
    plt.plot(history.lrs, label='Learning Rate', color='green')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    loss_plot_path = plots_dir / "training_curves.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves saved: {loss_plot_path}")
    
    # Create architecture visualization
    plt.figure(figsize=(10, 6))
    model_summary = str(segmenter.learn.model)
    plt.text(0.1, 0.5, model_summary[:1000] + "...", 
             fontsize=8, fontfamily='monospace',
             transform=plt.gca().transAxes)
    plt.title('Model Architecture Summary')
    plt.axis('off')
    
    arch_plot_path = plots_dir / "model_architecture.png"
    plt.savefig(arch_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Architecture plot saved: {arch_plot_path}")

def run_inference_examples(segmenter, results_dir):
    """Run inference on sample data and save results."""
    
    # Get a batch of validation data
    batch = segmenter.dls.valid.one_batch()
    images, masks = batch
    
    # Run inference
    with segmenter.learn.no_bar():
        predictions = segmenter.learn.model(images)
    
    # Convert to numpy for visualization
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = predictions.cpu().numpy()
    
    # Create inference visualization
    n_samples = min(4, len(images_np))
    
    plt.figure(figsize=(15, 5 * n_samples))
    
    for i in range(n_samples):
        # Original image
        plt.subplot(n_samples, 3, i*3 + 1)
        plt.imshow(images_np[i].transpose(1, 2, 0))
        plt.title(f'Sample {i+1}: Original Image')
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(n_samples, 3, i*3 + 2)
        plt.imshow(masks_np[i].squeeze(), cmap='gray')
        plt.title(f'Sample {i+1}: Ground Truth Mask')
        plt.axis('off')
        
        # Prediction
        plt.subplot(n_samples, 3, i*3 + 3)
        pred_mask = preds_np[i].argmax(axis=0)  # Get class predictions
        plt.imshow(pred_mask, cmap='gray')
        plt.title(f'Sample {i+1}: Predicted Mask')
        plt.axis('off')
    
    plt.tight_layout()
    inference_path = results_dir / "inference_examples.png"
    plt.savefig(inference_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Inference examples saved: {inference_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fastai segmentation pipeline.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--run-type", choices=['production', 'development'], default='production',
                       help="Type of run (production or development)")
    args = parser.parse_args()

    success = run_pipeline(epochs=args.epochs, run_type=args.run_type)
    if success:
        print(f"\n🎉 {args.run_type.upper()} PIPELINE SUCCESSFUL!")
        print(f"Check the output directory for all results!")
    else:
        print(f"\n❌ {args.run_type.upper()} PIPELINE FAILED!")
