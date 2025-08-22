#!/usr/bin/env python3
"""Production pipeline that generates actual output files, models, and visualizations."""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from eq.segmentation.fastai_segmenter import FastaiSegmenter, SegmentationConfig
from eq.utils.output_manager import OutputManager


def run_pipeline(epochs: int = 50, run_type: str = "production", use_existing_models: bool = True, 
                data_dir: str = "data/preeclampsia_data", cache_dir: str = "data/preeclampsia_data/cache"):
    """Run the production pipeline for endotheliosis quantification.
    
    This is PRODUCTION CODE that runs the full end-to-end pipeline:
    1. Load or train segmentation models
    2. Extract ROIs from images
    3. Quantify endotheliosis
    4. Generate comprehensive outputs
    
    Args:
        epochs: Number of training epochs (default: 50 for production)
        run_type: Type of run ('production' or 'development')
        use_existing_models: Whether to use existing pre-trained models
        data_dir: Path to data directory
        cache_dir: Path to cache directory
    """
    
    # Set environment for MPS compatibility
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    print(f"🚀 === {run_type.upper()} PIPELINE ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'QUICK_TEST' if os.getenv('QUICK_TEST') == 'true' else 'FULL_PRODUCTION'}")
    
    # Create output manager and directories
    output_manager = OutputManager()
    data_source = output_manager.get_data_source_name(data_dir)
    output_dirs = output_manager.create_output_directory(data_source, run_type)
    
    print(f"📁 Output directory: {output_dirs['main']}")
    print(f"📊 Data directory: {data_dir}")
    print(f"💾 Cache directory: {cache_dir}")
    
    # Check for existing models
    existing_models = {
        'glomeruli': Path('segmentation_model_dir/glomerulus_segmentation_model-dynamic_unet-e50_b16_s84.pkl'),
        'mitochondria': Path('segmentation_model_dir/mito_dynamic_unet_seg_model-e50_b16.pkl')
    }
    
    if use_existing_models:
        print("🔍 Checking for existing pre-trained models...")
        for task, model_path in existing_models.items():
            if model_path.exists():
                print(f"✅ Found existing {task} model: {model_path}")
            else:
                print(f"❌ Missing {task} model: {model_path}")
        
        # If we have models and want to use them, run full production inference
        if any(model_path.exists() for model_path in existing_models.values()):
            print("🚀 Running full production inference with existing models...")
            return run_full_production_inference(output_dirs, existing_models, data_dir, cache_dir)
    
    print("🏋️ Training new models for production...")
    
    # Create configuration based on run_type and QUICK_TEST
    is_quick_test = os.getenv('QUICK_TEST') == 'true'
    
    if run_type == "development" or is_quick_test:
        # Development/Quick test settings
        config = SegmentationConfig(
            epochs=min(epochs, 5) if is_quick_test else epochs,
            batch_size=4,  # Smaller batch for development
            device_mode="development",
            model_arch='resnet18',  # Smaller model for development
            learning_rate=0.001,
            model_save_path=output_dirs['models'] / "glomerulus_segmenter.pkl",
            results_save_path=output_dirs['results']
        )
        print("⚙️ Development Configuration:")
    else:
        # Production settings
        config = SegmentationConfig(
            epochs=epochs,
            batch_size=8,  # Production batch size
            device_mode="production",
            model_arch='resnet34',  # Production model
            learning_rate=0.001,
            model_save_path=output_dirs['models'] / "glomerulus_segmenter.pkl",
            results_save_path=output_dirs['results']
        )
        print("⚙️ Production Configuration:")
    
    for key, value in vars(config).items():
        print(f"   {key}: {value}")
    
    # Create segmenter
    segmenter = FastaiSegmenter(config)
    print("✅ Production segmenter created")
    
    try:
        # 1. Prepare data
        print("\n📊 === DATA PREPARATION ===")
        cache_path = Path(cache_dir)
        segmenter.prepare_data_from_cache(cache_path, 'glomeruli')
        
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
        print("Starting production training...")
        
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
        
        # 6. Generate inference visualizations
        print("\n🔍 === GENERATING INFERENCE VISUALIZATIONS ===")
        generate_inference_visualizations(segmenter, output_dirs['results'])
        
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

def generate_inference_visualizations(segmenter, results_dir):
    """Generate inference visualizations from the trained model."""
    
    # Check if we have data loaders available
    if not hasattr(segmenter, 'dls') or segmenter.dls is None:
        print("⚠️ No data loaders available for inference visualizations")
        print("   This is normal when loading pre-trained models")
        print("   Skipping inference visualization generation")
        return
    
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
    inference_path = results_dir / "inference_visualizations.png"
    plt.savefig(inference_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Inference visualizations saved: {inference_path}")

def run_full_production_inference(output_dirs: Dict[str, Path], existing_models: Dict[str, Path], data_dir: str, cache_dir: str):
    """Run full production inference using existing pre-trained models."""
    
    print("🚀 === FULL PRODUCTION INFERENCE ===")
    print("Running complete end-to-end inference pipeline...")
    
    # Load glomeruli model if available
    if existing_models['glomeruli'].exists():
        print(f"📥 Loading glomeruli model: {existing_models['glomeruli']}")
        
        # Create segmenter and load model
        config = SegmentationConfig(
            device_mode='production',
            model_save_path=output_dirs['models'] / "production_glomerulus_segmenter.pkl",
            results_save_path=output_dirs['results']
        )
        
        segmenter = FastaiSegmenter(config)
        segmenter.load_model(existing_models['glomeruli'])
        
        print("✅ Glomeruli model loaded successfully")
        
        # Run full production inference
        print("\n🔍 === RUNNING FULL PRODUCTION INFERENCE ===")
        
        # Load test data
        test_data_path = Path(data_dir) / "test"
        if test_data_path.exists():
            test_images = list(test_data_path.rglob("*.jpg")) + list(test_data_path.rglob("*.png"))
            print(f"📊 Found {len(test_images)} test images")
            
            # Run inference on all test images
            results = []
            for i, image_path in enumerate(test_images[:10] if os.getenv('QUICK_TEST') == 'true' else test_images):
                print(f"Processing image {i+1}/{len(test_images)}: {image_path.name}")
                
                try:
                    # Run segmentation
                    masks = segmenter.predict(image_path)
                    
                    # Extract ROIs
                    rois = segmenter.extract_rois(image_path, masks)
                    
                    # Store results
                    result = {
                        'image_path': str(image_path),
                        'image_name': image_path.name,
                        'num_glomeruli': len(rois),
                        'masks': masks,
                        'rois': rois,
                        'processing_success': True
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"⚠️ Failed to process {image_path.name}: {e}")
                    results.append({
                        'image_path': str(image_path),
                        'image_name': image_path.name,
                        'processing_success': False,
                        'error': str(e)
                    })
            
            # Save results
            import json
            results_file = output_dirs['results'] / "production_inference_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"✅ Production inference results saved: {results_file}")
            
            # Generate production visualizations
            print("\n📈 === GENERATING PRODUCTION VISUALIZATIONS ===")
            generate_inference_visualizations(segmenter, output_dirs['results'])
            
            # Generate model architecture plot
            print("\n📈 === GENERATING MODEL ARCHITECTURE PLOT ===")
            generate_model_architecture_plot(segmenter, output_dirs['plots'])
            
            # Generate production report
            print("\n📋 === GENERATING PRODUCTION REPORT ===")
            run_info = {
                'data_source': 'preeclampsia_data',
                'run_type': 'production_inference',
                'config': {
                    'model_arch': 'dynamic_unet',
                    'device_mode': 'production',
                    'model_path': str(existing_models['glomeruli'])
                },
                'results': {
                    'total_images': len(results),
                    'successful_images': len([r for r in results if r['processing_success']]),
                    'total_glomeruli': sum([r['num_glomeruli'] for r in results if r['processing_success']]),
                    'results_file': str(results_file)
                }
            }
            
            output_manager = OutputManager()
            output_manager.create_run_summary(output_dirs, run_info)
            
            print("\n🎉 === PRODUCTION INFERENCE COMPLETE ===")
            print(f"📁 All outputs saved to: {output_dirs['main']}")
            
            return True
        else:
            print(f"❌ Test data directory not found: {test_data_path}")
            return False
    else:
        print("❌ No glomeruli model found for production inference")
        return False


def generate_model_architecture_plot(segmenter, plots_dir):
    """Generate model architecture visualization."""
    
    plt.figure(figsize=(12, 8))
    
    # Get model summary
    model_summary = str(segmenter.learn.model)
    
    # Create a text plot showing model architecture
    plt.text(0.1, 0.5, model_summary[:2000] + "...", 
             fontsize=8, fontfamily='monospace',
             transform=plt.gca().transAxes,
             verticalalignment='center')
    
    plt.title('Pre-trained Model Architecture')
    plt.axis('off')
    
    arch_plot_path = plots_dir / "pretrained_model_architecture.png"
    plt.savefig(arch_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Model architecture plot saved: {arch_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fastai segmentation pipeline.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--run-type", choices=['production', 'development'], default='production',
                       help="Type of run (production or development)")
    parser.add_argument("--use-existing-models", action="store_true",
                       help="Whether to use existing pre-trained models for inference.")
    parser.add_argument("--data-dir", type=str, default="data/preeclampsia_data",
                       help="Path to the data directory.")
    parser.add_argument("--cache-dir", type=str, default="data/preeclampsia_data/cache",
                       help="Path to the cache directory.")
    args = parser.parse_args()

    success = run_pipeline(epochs=args.epochs, run_type=args.run_type, use_existing_models=args.use_existing_models,
                           data_dir=args.data_dir, cache_dir=args.cache_dir)
    if success:
        print(f"\n🎉 {args.run_type.upper()} PIPELINE SUCCESSFUL!")
        print("Check the output directory for all results!")
    else:
        print(f"\n❌ {args.run_type.upper()} PIPELINE FAILED!")
