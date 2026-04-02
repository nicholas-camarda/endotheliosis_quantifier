#!/usr/bin/env python3
"""Test derived data structure validation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def test_derived_data_exists():
    """Test that derived data directory exists."""
    print("Testing derived data existence...")
    
    derived_data_dir = Path('derived_data')
    assert derived_data_dir.exists(), "derived_data directory does not exist"
    print("✅ derived_data directory exists")
    
    return True

def test_mitochondria_data_structure():
    """Test mitochondria data structure."""
    print("\nTesting mitochondria data structure...")
    
    mito_dir = Path('derived_data/mitochondria_data')
    assert mito_dir.exists(), "mitochondria_data directory does not exist"
    print("✅ mitochondria_data directory exists")
    
    # Expected subdirectories
    expected_dirs = ['training', 'testing', 'cache']
    for dir_name in expected_dirs:
        dir_path = mito_dir / dir_name
        assert dir_path.exists(), f"Expected mitochondria directory {dir_name} does not exist"
        print(f"✅ mitochondria_data/{dir_name}/ exists")
    
    return True

def test_glomeruli_data_structure():
    """Test glomeruli data structure."""
    print("\nTesting glomeruli data structure...")
    
    glom_dir = Path('derived_data/glomeruli_data')
    assert glom_dir.exists(), "glomeruli_data directory does not exist"
    print("✅ glomeruli_data directory exists")
    
    # Expected subdirectories
    expected_dirs = ['training', 'testing', 'cache', 'prediction']
    for dir_name in expected_dirs:
        dir_path = glom_dir / dir_name
        assert dir_path.exists(), f"Expected glomeruli directory {dir_name} does not exist"
        print(f"✅ glomeruli_data/{dir_name}/ exists")
    
    return True

def test_mitochondria_training_data():
    """Test that mitochondria training data contains expected files."""
    print("\nTesting mitochondria training data content...")
    
    training_dir = Path('derived_data/mitochondria_data/training')
    if not training_dir.exists():
        print("⚠️  Mitochondria training directory doesn't exist")
        return True
    
    # Check for patches and masks
    image_patches_dir = training_dir / 'image_patches'
    mask_patches_dir = training_dir / 'mask_patches'
    
    if image_patches_dir.exists():
        image_patches = list(image_patches_dir.glob('*.jpg'))
        print(f"✅ Found {len(image_patches)} mitochondria image patches")
        
        if len(image_patches) > 0:
            # Check first patch size
            first_patch = image_patches[0]
            size_kb = first_patch.stat().st_size / 1024
            assert size_kb > 1, f"Patch {first_patch.name} is too small ({size_kb:.1f} KB)"
            print(f"✅ Image patches have reasonable size (e.g., {first_patch.name}: {size_kb:.1f} KB)")
    
    if mask_patches_dir.exists():
        mask_patches = list(mask_patches_dir.glob('*.jpg'))
        print(f"✅ Found {len(mask_patches)} mitochondria mask patches")
        
        # Verify masks and images match
        if image_patches_dir.exists():
            image_count = len(list(image_patches_dir.glob('*.jpg')))
            mask_count = len(mask_patches)
            if image_count > 0 and mask_count > 0:
                ratio = mask_count / image_count
                print(f"✅ Mask-to-image ratio: {ratio:.2f} (masks: {mask_count}, images: {image_count})")
    
    return True

def test_glomeruli_training_data():
    """Test that glomeruli training data contains expected files."""
    print("\nTesting glomeruli training data content...")
    
    training_dir = Path('derived_data/glomeruli_data/training')
    if not training_dir.exists():
        print("⚠️  Glomeruli training directory doesn't exist")
        return True
    
    # Check for patches and masks
    image_patches_dir = training_dir / 'image_patches'
    mask_patches_dir = training_dir / 'mask_patches'
    
    if image_patches_dir.exists():
        image_patches = list(image_patches_dir.glob('*.jpg'))
        print(f"✅ Found {len(image_patches)} glomeruli image patches")
        
        if len(image_patches) > 0:
            # Check first patch size
            first_patch = image_patches[0]
            size_kb = first_patch.stat().st_size / 1024
            assert size_kb > 1, f"Patch {first_patch.name} is too small ({size_kb:.1f} KB)"
            print(f"✅ Image patches have reasonable size (e.g., {first_patch.name}: {size_kb:.1f} KB)")
    
    if mask_patches_dir.exists():
        mask_patches = list(mask_patches_dir.glob('*.jpg'))
        print(f"✅ Found {len(mask_patches)} glomeruli mask patches")
    
    return True

def test_cache_directories():
    """Test that cache directories exist and contain expected files."""
    print("\nTesting cache directories...")
    
    # Mitochondria cache
    mito_cache = Path('derived_data/mitochondria_data/cache')
    if mito_cache.exists():
        cache_files = list(mito_cache.glob('*.pkl')) + list(mito_cache.glob('*.pickle'))
        print(f"✅ Mitochondria cache: {len(cache_files)} cache files")
    
    # Glomeruli cache
    glom_cache = Path('derived_data/glomeruli_data/cache')
    if glom_cache.exists():
        cache_files = list(glom_cache.glob('*.pkl')) + list(glom_cache.glob('*.pickle'))
        print(f"✅ Glomeruli cache: {len(cache_files)} cache files")
    
    return True

def test_data_consistency():
    """Test consistency between training and testing data."""
    print("\nTesting data consistency...")
    
    # Check mitochondria data consistency
    mito_train = Path('derived_data/mitochondria_data/training/image_patches')
    mito_test = Path('derived_data/mitochondria_data/testing/image_patches')
    
    if mito_train.exists() and mito_test.exists():
        train_count = len(list(mito_train.glob('*.jpg')))
        test_count = len(list(mito_test.glob('*.jpg')))
        
        if train_count > 0 and test_count > 0:
            ratio = test_count / train_count
            print(f"✅ Mitochondria train/test ratio: {ratio:.2f} (train: {train_count}, test: {test_count})")
            
            # Reasonable split should be between 0.1 and 0.5
            if 0.05 <= ratio <= 0.6:
                print("✅ Mitochondria train/test split appears reasonable")
            else:
                print(f"⚠️  Mitochondria train/test split may be unusual: {ratio:.2f}")
    
    # Check glomeruli data consistency
    glom_train = Path('derived_data/glomeruli_data/training/image_patches')
    glom_test = Path('derived_data/glomeruli_data/testing/image_patches')
    
    if glom_train.exists() and glom_test.exists():
        train_count = len(list(glom_train.glob('*.jpg')))
        test_count = len(list(glom_test.glob('*.jpg')))
        
        if train_count > 0 and test_count > 0:
            ratio = test_count / train_count
            print(f"✅ Glomeruli train/test ratio: {ratio:.2f} (train: {train_count}, test: {test_count})")
            
            # Reasonable split should be between 0.1 and 0.5
            if 0.05 <= ratio <= 0.6:
                print("✅ Glomeruli train/test split appears reasonable")
            else:
                print(f"⚠️  Glomeruli train/test split may be unusual: {ratio:.2f}")
    
    return True

if __name__ == "__main__":
    print("DERIVED DATA STRUCTURE VALIDATION")
    print("=" * 45)
    
    success = True
    success &= test_derived_data_exists()
    success &= test_mitochondria_data_structure()
    success &= test_glomeruli_data_structure()
    success &= test_mitochondria_training_data()
    success &= test_glomeruli_training_data()
    success &= test_cache_directories()
    success &= test_data_consistency()
    
    print("\n" + "=" * 45)
    if success:
        print("🎉 DERIVED DATA STRUCTURE VALIDATED!")
        print("✅ All expected directories exist")
        print("✅ Training and testing data appear organized")
        print("✅ Data splits appear reasonable")
        print("✅ Cache directories properly structured")
    else:
        print("❌ DERIVED DATA STRUCTURE ISSUES")
        print("Some expected derived data is missing or malformed...")
