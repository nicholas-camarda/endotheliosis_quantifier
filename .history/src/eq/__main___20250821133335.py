#!/usr/bin/env python3
"""Main CLI entry point for the endotheliosis quantifier package."""

import argparse
import sys

from eq.features.data_loader import (
    create_train_val_test_lists,
    generate_binary_masks,
    generate_final_dataset,
    get_scores_from_annotations,
    load_annotations_from_json,
    organize_data_into_subdirs,
)
from eq.models.feature_extractor import run_feature_extraction
from eq.pipeline.quantify_endotheliosis import run_endotheliosis_quantification
from eq.segmentation.train_segmenter import train_segmentation_model


def data_load_command(args):
    """Load and preprocess data for the pipeline."""
    print("Loading and preprocessing data...")
    
    # Generate binary masks if annotation file provided
    if args.annotation_file:
        print(f"Generating binary masks from {args.annotation_file}")
        generate_binary_masks(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir
        )
    
    # Organize test data
    print("Organizing test data...")
    test_images_1_paths, test_data_dict_1 = organize_data_into_subdirs(
        data_dir=args.test_data_dir
    )
    
    # Create train/val/test lists
    print("Creating train/val/test splits...")
    train_images_paths, train_masks_paths, test_images_2_paths, train_data_dict, test_data_dict_2 = create_train_val_test_lists(
        data_dir=args.data_dir
    )
    
    # Combine test images
    import numpy as np
    test_images_paths = np.concatenate((test_images_1_paths, test_images_2_paths))
    test_data_dict = dict(sorted({**test_data_dict_1, **test_data_dict_2}.items()))
    
    # Generate final dataset
    print("Generating final dataset...")
    generate_final_dataset(
        train_images_paths, train_masks_paths, test_images_paths,
        train_data_dict, test_data_dict, 
        size=args.image_size, cache_dir=args.cache_dir
    )
    
    # Process scores if annotation file provided
    if args.annotation_file:
        print("Processing scores...")
        annotations = load_annotations_from_json(args.annotation_file)
        scores = get_scores_from_annotations(annotations, cache_dir=args.cache_dir)
    
    print("Data loading complete!")


def train_segmenter_command(args):
    """Train a segmentation model."""
    print("Training segmentation model...")
    train_segmentation_model(
        base_model_path=args.base_model_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    print("Segmentation training complete!")


def extract_features_command(args):
    """Extract features from images."""
    print("Extracting features...")
    run_feature_extraction(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        model_path=args.model_path
    )
    print("Feature extraction complete!")


def quantify_command(args):
    """Run endotheliosis quantification."""
    print("Running endotheliosis quantification...")
    run_endotheliosis_quantification(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        model_path=args.model_path
    )
    print("Quantification complete!")


def pipeline_command(args):
    """Run the complete pipeline end-to-end."""
    print("Running complete endotheliosis quantification pipeline...")
    
    # Step 1: Load data
    print("\n=== Step 1: Loading Data ===")
    data_load_command(args)
    
    # Step 2: Train segmenter
    print("\n=== Step 2: Training Segmentation Model ===")
    train_segmenter_command(args)
    
    # Step 3: Extract features
    print("\n=== Step 3: Extracting Features ===")
    extract_features_command(args)
    
    # Step 4: Quantify
    print("\n=== Step 4: Quantifying Endotheliosis ===")
    quantify_command(args)
    
    print("\nPipeline complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Endotheliosis Quantifier Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  eq data-load --data-dir data/train --test-data-dir data/test
  eq train-segmenter --cache-dir data/cache --output-dir output
  eq extract-features --cache-dir data/cache --output-dir output
  eq quantify --cache-dir data/cache --output-dir output
  eq pipeline --data-dir data/train --test-data-dir data/test --output-dir output
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data loading command
    data_parser = subparsers.add_parser('data-load', help='Load and preprocess data')
    data_parser.add_argument('--data-dir', required=True, help='Training data directory')
    data_parser.add_argument('--test-data-dir', required=True, help='Test data directory')
    data_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    data_parser.add_argument('--annotation-file', help='Annotation JSON file')
    data_parser.add_argument('--image-size', type=int, default=256, help='Image size for processing')
    data_parser.set_defaults(func=data_load_command)
    
    # Training command
    train_parser = subparsers.add_parser('train-segmenter', help='Train segmentation model')
    train_parser.add_argument('--base-model-path', required=True, help='Path to base model')
    train_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    train_parser.add_argument('--output-dir', required=True, help='Output directory')
    train_parser.add_argument('--model-name', default='glomerulus_segmenter', help='Model name')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.set_defaults(func=train_segmenter_command)
    
    # Feature extraction command
    features_parser = subparsers.add_parser('extract-features', help='Extract features')
    features_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    features_parser.add_argument('--output-dir', required=True, help='Output directory')
    features_parser.add_argument('--model-path', required=True, help='Path to trained model')
    features_parser.set_defaults(func=extract_features_command)
    
    # Quantification command
    quant_parser = subparsers.add_parser('quantify', help='Run endotheliosis quantification')
    quant_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    quant_parser.add_argument('--output-dir', required=True, help='Output directory')
    quant_parser.add_argument('--model-path', required=True, help='Path to trained model')
    quant_parser.set_defaults(func=quantify_command)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--data-dir', required=True, help='Training data directory')
    pipeline_parser.add_argument('--test-data-dir', required=True, help='Test data directory')
    pipeline_parser.add_argument('--cache-dir', required=True, help='Cache directory')
    pipeline_parser.add_argument('--output-dir', required=True, help='Output directory')
    pipeline_parser.add_argument('--annotation-file', help='Annotation JSON file')
    pipeline_parser.add_argument('--base-model-path', required=True, help='Path to base model')
    pipeline_parser.add_argument('--image-size', type=int, default=256, help='Image size for processing')
    pipeline_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    pipeline_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    pipeline_parser.set_defaults(func=pipeline_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
