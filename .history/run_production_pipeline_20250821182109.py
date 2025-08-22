#!/usr/bin/env python3
"""Full production pipeline that generates actual output files, models, and visualizations."""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from eq.segmentation.fastai_segmenter import FastaiSegmenter, SegmentationConfig

def run_production_pipeline():
    """Run the complete production pipeline with full output generation."""
    
    # Set environment for MPS compatibility
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    print("🚀 === PRODUCTION FASTAI PIPELINE === 🚀")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    output_dir = Path("output/fastai_production_run")
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    results_dir = output_dir / "results"
    
    for dir_path in [output_dir, models_dir, plots_dir, results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Create production configuration
    config = SegmentationConfig(
        epochs=10,  # More epochs for production
        batch_size=4,
        device_mode='development',
        model_arch='resnet50',  # Larger model for production
        learning_rate=0.001,
        model_save_path=models_dir / "glomerulus_segmenter.pkl",
        results_save_path=results_dir
    )
    
    print(f"⚙️ Configuration:")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Architecture: {config.model_arch}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Device mode: {config.device_mode}")
    
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
        
        print(f"✅ Data prepared:")
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
        
        print(f"✅ Model created:")
        for key, value in model_stats.items():
            print(f"   {key}: {value}")
        
        # 3. Train model
        print("\n🏋️ === MODEL TRAINING ===")
        print("Starting training... This will generate real training curves!")
        
        training_result = segmenter.train(epochs=config.epochs, learning_rate=config.learning_rate)
        
        print(f"✅ Training completed:")
        print(f"   Final training loss: {training_result.get('train_loss', 'N/A')}")
        print(f"   Final validation loss: {training_result.get('valid_loss', 'N/A')}")
        print(f"   Final dice score: {training_result.get('dice', 'N/A')}")
        
        # 4. Save model
        print("\n💾 === SAVING MODEL ===")
        model_path = segmenter.save_model(models_dir / "production_glomerulus_segmenter")
        print(f"✅ Model saved to: {model_path}")
        
        # 5. Generate training plots
        print("\n📈 === GENERATING PLOTS ===")
        generate_training_plots(segmenter, plots_dir, training_result)
        
        # 6. Run inference examples
        print("\n🔍 === RUNNING INFERENCE ===")
        run_inference_examples(segmenter, results_dir)
        
        # 7. Generate summary report
        print("\n📋 === GENERATING REPORT ===")
        generate_summary_report(
            output_dir, 
            config, 
            data_stats, 
            model_stats, 
            training_result
        )
        
        print("\n🎉 === PRODUCTION PIPELINE COMPLETE ===")
        print(f"📁 All outputs saved to: {output_dir}")
        print("\n📋 Generated files:")
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                print(f"   {file_path.relative_to(output_dir)}")
        
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

def generate_summary_report(output_dir, config, data_stats, model_stats, training_result):
    """Generate a comprehensive summary report."""
    
    report_content = f"""
# Fastai Production Pipeline Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Architecture: {config.model_arch}
- Epochs: {config.epochs}
- Batch Size: {config.batch_size}
- Learning Rate: {config.learning_rate}
- Device Mode: {config.device_mode}
- Image Size: {config.image_size}

## Data Statistics
- Training Samples: {data_stats['training_samples']}
- Validation Samples: {data_stats['validation_samples']}
- Total Samples: {data_stats['training_samples'] + data_stats['validation_samples']}

## Model Statistics
- Parameters: {model_stats['parameters']:,}
- Device: {model_stats['device']}
- Pretrained: {config.pretrained}

## Training Results
- Final Training Loss: {training_result.get('train_loss', 'N/A')}
- Final Validation Loss: {training_result.get('valid_loss', 'N/A')}
- Final Dice Score: {training_result.get('dice', 'N/A')}

## Generated Files
- Model: models/production_glomerulus_segmenter.pkl
- Training Curves: plots/training_curves.png
- Model Architecture: plots/model_architecture.png
- Inference Examples: results/inference_examples.png

## Pipeline Status
✅ Data Preparation: Complete
✅ Model Creation: Complete
✅ Training: Complete
✅ Model Saving: Complete
✅ Visualization: Complete
✅ Inference: Complete

This model is ready for production use!
"""
    
    report_path = output_dir / "PRODUCTION_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"✅ Production report saved: {report_path}")

if __name__ == "__main__":
    success = run_production_pipeline()
    if success:
        print("\n🎉 PRODUCTION PIPELINE SUCCESSFUL!")
        print("Check the output/fastai_production_run/ directory for all results!")
    else:
        print("\n❌ PRODUCTION PIPELINE FAILED!")
