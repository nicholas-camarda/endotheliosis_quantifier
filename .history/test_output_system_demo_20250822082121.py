#!/usr/bin/env python3
"""Demo script to test the output directory system."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eq.utils.output_manager import OutputManager, create_output_directories


def demo_output_directory_system():
    """Demonstrate the output directory system functionality."""
    
    print("🚀 === OUTPUT DIRECTORY SYSTEM DEMO ===")
    
    # Create output manager
    output_manager = OutputManager()
    
    # Demo 1: Basic output directory creation
    print("\n📁 Demo 1: Basic output directory creation")
    data_source = "preeclampsia_data"
    run_type = "production"
    
    output_dirs = output_manager.create_output_directory(data_source, run_type)
    print(f"✅ Created output directory: {output_dirs['main']}")
    print(f"   Models: {output_dirs['models']}")
    print(f"   Plots: {output_dirs['plots']}")
    print(f"   Results: {output_dirs['results']}")
    print(f"   Reports: {output_dirs['reports']}")
    print(f"   Logs: {output_dirs['logs']}")
    print(f"   Cache: {output_dirs['cache']}")
    
    # Demo 2: Different run types
    print("\n📁 Demo 2: Different run types")
    run_types = ["quick", "production", "smoke", "development"]
    
    for run_type in run_types:
        output_dirs = output_manager.create_output_directory(data_source, run_type)
        print(f"✅ {run_type}: {output_dirs['main'].name}")
    
    # Demo 3: Data source name extraction
    print("\n📁 Demo 3: Data source name extraction")
    test_paths = [
        "data/preeclampsia_data",
        "data/experiment_data", 
        "data/my_study_data",
        "data/clinical_trial_data"
    ]
    
    for path in test_paths:
        extracted_name = output_manager.get_data_source_name(path)
        print(f"   {path} → {extracted_name}")
    
    # Demo 4: Run summary creation
    print("\n📁 Demo 4: Run summary creation")
    output_dirs = output_manager.create_output_directory("demo_data", "test")
    
    # Create some dummy files
    (output_dirs['models'] / "model.pkl").touch()
    (output_dirs['plots'] / "training_curves.png").touch()
    (output_dirs['results'] / "results.json").touch()
    
    run_info = {
        'data_source': 'demo_data',
        'run_type': 'test',
        'config': {
            'epochs': 5,
            'batch_size': 8,
            'learning_rate': 0.001
        },
        'results': {
            'accuracy': 0.92,
            'loss': 0.08,
            'training_time': '45m'
        }
    }
    
    output_manager.create_run_summary(output_dirs, run_info)
    print(f"✅ Created run summary: {output_dirs['reports'] / 'run_summary.md'}")
    
    # Demo 5: Convenience function
    print("\n📁 Demo 5: Convenience function")
    output_dirs = create_output_directories(
        data_source_name="convenience_demo",
        run_type="quick",
        custom_suffix="v1"
    )
    print(f"✅ Created with convenience function: {output_dirs['main'].name}")
    
    print("\n🎉 === DEMO COMPLETE ===")
    print("Check the 'output' directory to see all created directories!")


if __name__ == "__main__":
    demo_output_directory_system()
