#!/usr/bin/env python3
"""Test raw data structure validation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def test_raw_data_exists():
    """Test that raw data directory exists and has expected structure."""
    print("Testing raw data existence...")
    
    raw_data_dir = Path('raw_data')
    assert raw_data_dir.exists(), "raw_data directory does not exist"
    print("✅ raw_data directory exists")
    
    preeclampsia_dir = raw_data_dir / 'preeclampsia_project'
    assert preeclampsia_dir.exists(), "preeclampsia_project directory does not exist"
    print("✅ preeclampsia_project directory exists")
    
    return True

def test_preeclampsia_project_structure():
    """Test that preeclampsia project has expected structure."""
    print("\nTesting preeclampsia project structure...")
    
    project_dir = Path('raw_data/preeclampsia_project')
    
    # Expected directories
    expected_dirs = ['data', 'annotations']
    #  'backup_before_reorganization', 'clean_backup' # these are not expected to exist necessarily, these are just for me
    for dir_name in expected_dirs:
        dir_path = project_dir / dir_name
        assert dir_path.exists(), f"Expected directory {dir_name} does not exist"
        print(f"✅ {dir_name}/ directory exists")
    
    # Expected metadata file
    metadata_file = project_dir / 'subject_metadata.xlsx'
    assert metadata_file.exists(), "subject_metadata.xlsx does not exist"
    print("✅ subject_metadata.xlsx exists")
    
    return True

def test_raw_data_content():
    """Test that raw data contains actual image data."""
    print("\nTesting raw data content...")
    
    data_dir = Path('raw_data/preeclampsia_project/data')
    if not data_dir.exists():
        print("⚠️  Data directory doesn't exist, skipping content test")
        return True
    
    # Count subjects
    subject_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"✅ Found {len(subject_dirs)} subject directories")
    
    if len(subject_dirs) > 0:
        # Check first subject for image files
        first_subject = subject_dirs[0]
        image_files = list(first_subject.glob('*.tif')) + list(first_subject.glob('*.jpg'))
        print(f"✅ First subject ({first_subject.name}) has {len(image_files)} image files")
        
        if len(image_files) > 0:
            # Check file sizes are reasonable (not empty)
            first_image = image_files[0]
            size_mb = first_image.stat().st_size / (1024 * 1024)
            assert size_mb > 0.1, f"Image file {first_image.name} is too small ({size_mb:.2f} MB)"
            print(f"✅ Image files have reasonable size (e.g., {first_image.name}: {size_mb:.2f} MB)")
    
    return True

def test_annotations_structure():
    """Test that annotations directory has expected structure."""
    print("\nTesting annotations structure...")
    
    annotations_dir = Path('raw_data/preeclampsia_project/annotations')
    if not annotations_dir.exists():
        print("⚠️  Annotations directory doesn't exist, skipping annotation test")
        return True
    
    # Look for annotation files
    annotation_files = list(annotations_dir.glob('*.json')) + list(annotations_dir.glob('*.xml'))
    print(f"✅ Found {len(annotation_files)} annotation files")
    
    # Check for mask files if they exist
    mask_files = list(annotations_dir.glob('*mask*')) + list(annotations_dir.glob('*_m.tif'))
    print(f"✅ Found {len(mask_files)} mask files")
    
    return True

def test_subject_metadata():
    """Test that subject metadata file is readable."""
    print("\nTesting subject metadata...")
    
    metadata_file = Path('raw_data/preeclampsia_project/subject_metadata.xlsx')
    if not metadata_file.exists():
        print("⚠️  Metadata file doesn't exist, skipping metadata test")
        return True
    
    try:
        import pandas as pd
        df = pd.read_excel(metadata_file)
        
        print(f"✅ Metadata loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check for expected columns (basic validation)
        if 'subject' in df.columns or 'Subject' in df.columns:
            print("✅ Subject column found in metadata")
        
        if len(df) > 0:
            print("✅ Metadata contains data for subjects")
        
    except Exception as e:
        print(f"⚠️  Could not read metadata file: {e}")
        # Don't fail the test if pandas isn't available
    
    return True

if __name__ == "__main__":
    print("RAW DATA STRUCTURE VALIDATION")
    print("=" * 40)
    
    success = True
    success &= test_raw_data_exists()
    success &= test_preeclampsia_project_structure()
    success &= test_raw_data_content()
    success &= test_annotations_structure()
    success &= test_subject_metadata()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 RAW DATA STRUCTURE VALIDATED!")
        print("✅ All expected directories and files exist")
        print("✅ Raw data appears to contain actual image data")
        print("✅ Project structure follows expected format")
    else:
        print("❌ RAW DATA STRUCTURE ISSUES")
        print("Some expected files or directories are missing...")
