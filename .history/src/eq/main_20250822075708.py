#!/usr/bin/env python3
"""Main pipeline orchestrator for the endotheliosis quantifier package."""

import sys
from pathlib import Path
from datetime import datetime

def main():
    """Main pipeline orchestrator."""
    
    print("🚀 === ENDOTHELIOSIS QUANTIFIER PIPELINE ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Add src to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    print("Available pipeline modes:")
    print("  1. Quick Test (2 epochs) - Fast validation")
    print("  2. Production (10 epochs) - Full training")
    print("  3. Visualization - Create enhanced visualizations")
    print("  4. Smoke Test - Basic functionality check")
    print()
    
    choice = input("Select mode (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Running Quick Test Pipeline...")
        from pipeline.run_production_pipeline import run_quick_test
        run_quick_test()
        
    elif choice == "2":
        print("\n🚀 Running Production Pipeline...")
        from pipeline.run_production_pipeline import run_production_pipeline
        run_production_pipeline()
        
    elif choice == "3":
        print("\n🎨 Running Visualization Pipeline...")
        from visualization.create_better_visualization import create_enhanced_visualization
        create_enhanced_visualization()
        
    elif choice == "4":
        print("\n🧪 Running Smoke Test...")
        print("Note: This is now a test script. Run with: python -m pytest tests/test_smoke_pipeline.py")
        print("Or run directly: python tests/test_smoke_pipeline.py")
        
    else:
        print("❌ Invalid choice. Please select 1-4.")
        print()
        print("You can also run specific components directly:")
        print("  # Quick test")
        print("  python src/eq/pipeline/run_production_pipeline.py --quick")
        print()
        print("  # Full production")
        print("  python src/eq/pipeline/run_production_pipeline.py")
        print()
        print("  # Visualization")
        print("  python src/eq/visualization/create_better_visualization.py")
        print()
        print("  # Smoke test")
        print("  python tests/test_smoke_pipeline.py")

if __name__ == "__main__":
    main()
