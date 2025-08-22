#!/usr/bin/env python3
"""Simple pipeline runner - makes it obvious how to run the fastai segmentation pipeline."""

import sys
from pathlib import Path

def main():
    """Main pipeline runner."""
    
    print("🚀 === FASTAI SEGMENTATION PIPELINE RUNNER ===")
    print()
    print("Available pipelines:")
    print("  1. Quick Production (2 epochs) - Fast testing")
    print("  2. Full Production (10 epochs) - Complete training")
    print("  3. Smoke Test - Basic functionality check")
    print()
    
    choice = input("Select pipeline (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Running Quick Production Pipeline...")
        print("Command: python src/eq/pipeline/run_quick_production.py")
        print()
        
        # Import and run the quick production pipeline
        sys.path.insert(0, 'src')
        from eq.pipeline.run_quick_production import run_quick_production
        run_quick_production()
        
    elif choice == "2":
        print("\n🚀 Running Full Production Pipeline...")
        print("Command: python src/eq/pipeline/run_production_pipeline.py")
        print()
        
        # Import and run the full production pipeline
        sys.path.insert(0, 'src')
        from eq.pipeline.run_production_pipeline import run_production_pipeline
        run_production_pipeline()
        
    elif choice == "3":
        print("\n🚀 Running Smoke Test...")
        print("Command: python src/eq/pipeline/smoke_test_pipeline.py")
        print()
        
        # Import and run the smoke test
        sys.path.insert(0, 'src')
        from eq.pipeline.smoke_test_pipeline import main as run_smoke_test
        run_smoke_test()
        
    else:
        print("❌ Invalid choice. Please run one of these commands directly:")
        print()
        print("  # Quick test (2 epochs)")
        print("  python src/eq/pipeline/run_quick_production.py")
        print()
        print("  # Full production (10 epochs)")
        print("  python src/eq/pipeline/run_production_pipeline.py")
        print()
        print("  # Basic functionality check")
        print("  python src/eq/pipeline/smoke_test_pipeline.py")

if __name__ == "__main__":
    main()
