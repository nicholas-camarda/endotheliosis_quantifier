"""
FastAI-based segmentation model training for endotheliosis quantification.

This module provides a fastai implementation of the segmentation training pipeline,
replacing the TensorFlow-based train_segmenter.py with modern fastai/PyTorch functionality.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from eq.segmentation.fastai_segmenter import (
    FastaiSegmenter,
    SegmentationConfig,
    create_glomeruli_segmenter,
)
from eq.utils.common import load_pickled_data, plot_history


def _local_plot_history(history, output_dir, file_name):
    """Plot training history and save to output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract training history from fastai recorder
    if hasattr(history, 'values') and len(history['values']) > 0:
        # fastai recorder format
        values = history['values']
        if len(values) > 0 and len(values[0]) >= 4:
            loss = [v[0] for v in values]
            val_loss = [v[1] for v in values]
            dice = [v[2] for v in values] if len(values[0]) > 2 else []
            epochs = range(1, len(loss) + 1)
        else:
            # Fallback to simple loss plotting
            loss = [v[0] for v in values] if values else []
            val_loss = [v[1] for v in values] if values and len(values[0]) > 1 else []
            epochs = range(1, len(loss) + 1)
    else:
        # Fallback for other history formats
        loss = history.get('loss', [])
        val_loss = history.get('val_loss', [])
        epochs = range(1, len(loss) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(epochs, loss, 'y', label='Training loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Dice score if available
    if dice:
        plt.subplot(132)
        plt.plot(epochs, dice, 'g', label='Dice score')
        plt.title('Dice score over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.legend()
    
    # Plot learning rate if available
    if hasattr(history, 'lrs') and history['lrs']:
        plt.subplot(133)
        plt.plot(epochs, history['lrs'], 'b', label='Learning rate')
        plt.title('Learning rate over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_training_history.png"))
    plt.close()


def check_model_performance(segmenter: FastaiSegmenter, X_test: np.ndarray, y_test: np.ndarray, 
                           final_plots_dir: str, threshold: float = 0.5) -> None:
    """
    Check model performance on test data.
    
    Args:
        segmenter: Trained FastaiSegmenter instance
        X_test: Test images
        y_test: Test masks
        final_plots_dir: Directory to save performance plots
        threshold: Threshold for binary segmentation
    """
    # Create predictions output directory
    predictions_output_dir = os.path.join(final_plots_dir, 'predictions')
    if not os.path.exists(predictions_output_dir):
        os.makedirs(predictions_output_dir)

    # Generate and save predictions for test images
    for test_img_number in range(X_test.shape[0]):
        # Get test image and ground truth
        test_img = X_test[test_img_number]
        ground_truth = y_test[test_img_number]
        
        # Create temporary image file for prediction
        temp_img_path = f"/tmp/temp_test_img_{test_img_number}.jpg"
        # Save numpy array as image (you might need to implement this based on your data format)
        # For now, we'll use a placeholder approach
        
        try:
            # Predict using the segmenter
            # Note: This requires the segmenter to have a predict method that works with numpy arrays
            # You may need to adapt this based on your specific fastai implementation
            prediction = segmenter.predict_batch([temp_img_path])[0]
            
            # Threshold the prediction
            prediction_thresholded = (prediction > threshold).astype(np.uint8)
            
            # Create file name based on index of image in dataset
            file_name_prediction = f"prediction_{test_img_number}"
            
            # Plot and save image
            plt.figure(figsize=(16, 8))
            plt.subplot(231)
            plt.title('Testing Image')
            plt.imshow(test_img[:, :, 0] if test_img.ndim == 3 else test_img, cmap='gray')
            plt.subplot(232)
            plt.title('Testing Label')
            plt.imshow(ground_truth[:, :, 0] if ground_truth.ndim == 3 else ground_truth, cmap='gray')
            plt.subplot(233)
            plt.title('Prediction on test image')
            plt.imshow(prediction_thresholded, cmap='gray')
            plt.savefig(os.path.join(predictions_output_dir, f"{file_name_prediction}.png"))
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not process test image {test_img_number}: {e}")
            continue
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)


def train_segmentation_model_fastai(base_model_path: Optional[str], cache_dir: str, output_dir: str, 
                                   model_name: str = 'glomerulus_segmenter', batch_size: int = 8, 
                                   epochs: int = 50, image_size: int = 256, learning_rate: float = 1e-3,
                                   device_mode: str = "auto") -> FastaiSegmenter:
    """
    Train segmentation model using fastai.
    
    Args:
        base_model_path: Path to base model (optional, for transfer learning)
        cache_dir: Directory containing cached data
        output_dir: Output directory for model and results
        model_name: Name for the model
        batch_size: Training batch size
        epochs: Number of training epochs
        image_size: Input image size
        learning_rate: Learning rate for training
        device_mode: Device mode (auto, development, production)
        
    Returns:
        Trained FastaiSegmenter instance
    """
    # Set up paths
    top_output_directory = output_dir
    cache_dir_path = cache_dir
    
    # Create output directory
    final_output_path = os.path.join(top_output_directory, 'glomerulus_segmentation', model_name)
    os.makedirs(final_output_path, exist_ok=True)
    final_plots_dir = os.path.join(final_output_path, 'plots')
    os.makedirs(final_plots_dir, exist_ok=True)

    # Create model save path
    model_save_path = os.path.join(final_output_path, f"{model_name}.pkl")
    
    # Load the data
    print("Loading training data...")
    X_train = load_pickled_data(os.path.join(cache_dir_path, 'train_images.pickle'))
    y_train = load_pickled_data(os.path.join(cache_dir_path, 'train_masks.pickle'))
    X_val = load_pickled_data(os.path.join(cache_dir_path, 'val_images.pickle'))
    y_val = load_pickled_data(os.path.join(cache_dir_path, 'val_masks.pickle'))
    X_test = load_pickled_data(os.path.join(cache_dir_path, 'test_images.pickle'))

    print(f'Training images shape: {X_train.shape}')
    print(f'Training masks shape: {y_train.shape}')
    print(f'Validation images shape: {X_val.shape}')
    print(f'Validation masks shape: {y_val.shape}')
    print(f'Testing images shape: {X_test.shape}')

    # Create fastai segmenter configuration
    config = SegmentationConfig(
        image_size=image_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        device_mode=device_mode,
        model_save_path=Path(model_save_path),
        results_save_path=Path(final_plots_dir)
    )
    
    # Create segmenter
    segmenter = create_glomeruli_segmenter(config)
    
    # Prepare data (this would need to be adapted based on your data format)
    # For now, we'll create a simplified approach
    print("Preparing data for fastai...")
    
    # Note: The data preparation would need to be adapted based on your specific data format
    # and how fastai expects the data to be structured
    # This is a placeholder for the actual data preparation logic
    
    # Create and train the model
    print("Creating U-Net model...")
    segmenter.create_model("glomeruli")
    
    # Find optimal learning rate
    print("Finding optimal learning rate...")
    lr_min, lr_steep, lr_valley, lr_slide = segmenter.find_learning_rate()
    
    # Use the steep learning rate for training
    optimal_lr = lr_steep
    
    # Train the model
    print(f"Training for {epochs} epochs with learning rate {optimal_lr:.2e}")
    training_history = segmenter.train(epochs=epochs, learning_rate=optimal_lr)
    
    # Plot training history
    plot_history(training_history, final_plots_dir, model_name)
    
    # Save the model
    segmenter.save_model()
    
    # Evaluate on validation data (using validation as test for now)
    print("Evaluating model performance...")
    X_test_eval = X_val
    y_test_eval = y_val
    
    check_model_performance(
        segmenter=segmenter,
        X_test=X_test_eval, 
        y_test=y_test_eval,
        final_plots_dir=final_plots_dir,
        threshold=0.5
    )
    
    print(f"Training completed. Model saved to {model_save_path}")
    print(f"Results saved to {final_plots_dir}")
    
    return segmenter


def main():
    """Main function for command-line execution."""
    # Configuration
    top_data_directory = 'data/preeclampsia_data'
    cache_dir_path = os.path.join(top_data_directory, 'cache')
    top_output_directory = 'output/segmentation_models'
    
    # Get the current date
    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y-%m-%d')
    
    # Model parameters
    square_size = 256
    n_epochs = 75
    n_batch_size = 16
    
    # Base model path (optional for transfer learning)
    base_model_path = os.path.join(
        top_output_directory, 
        'unet_binseg_50epoch_3960images_8batchsize',
        'unet_binseg_50epoch_3960images_8batchsize.hdf5'
    )
    
    # Model name
    file_name_with_ext = f'{formatted_date}-glom_unet_fastai_seg_model-epochs{n_epochs}_batch{n_batch_size}'
    file_name = os.path.splitext(file_name_with_ext)[0]
    
    # Create output directories
    final_output_path = os.path.join(top_output_directory, 'glomerulus_segmentation', file_name)
    os.makedirs(final_output_path, exist_ok=True)
    final_plots_dir = os.path.join(final_output_path, 'plots')
    os.makedirs(final_plots_dir, exist_ok=True)
    
    # Train the model
    segmenter = train_segmentation_model_fastai(
        base_model_path=base_model_path,
        cache_dir=cache_dir_path,
        output_dir=top_output_directory,
        model_name=file_name,
        batch_size=n_batch_size,
        epochs=n_epochs,
        image_size=square_size,
        learning_rate=1e-3,
        device_mode="auto"
    )
    
    print("Training completed successfully!")
    print(f"Model saved to: {final_output_path}")
    print(f"Results saved to: {final_plots_dir}")


if __name__ == '__main__':
    main()
