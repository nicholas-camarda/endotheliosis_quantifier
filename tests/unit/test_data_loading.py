#!/usr/bin/env python3
"""
Test the complete data loading pipeline to verify FastAI v2 compatibility.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from eq.data_management.loaders import build_segmentation_dls
from eq.utils.logger import get_logger

logger = get_logger("test_data_loading")

def test_mitochondria_data_loading():
    """Test mitochondria data loading with FastAI v2 DataBlock."""
    logger.info("🧪 Testing mitochondria data loading...")
    
    # Test with existing derived data
    data_root = Path("derived_data/mito/training")
    
    if not data_root.exists():
        logger.error(f"❌ Data root does not exist: {data_root}")
        return False
    
    try:
        # Create DataLoaders
        logger.info("📊 Creating DataLoaders...")
        dls = build_segmentation_dls(data_root, bs=4, num_workers=0)
        
        logger.info(f"✅ Created DataLoaders: {len(dls.train_ds)} train, {len(dls.valid_ds)} val")
        
        # Test getting a batch
        logger.info("🔄 Testing batch creation...")
        batch = dls.one_batch()
        
        if batch is None:
            logger.error("❌ Failed to create batch")
            return False
            
        images, masks = batch
        logger.info(f"✅ Batch created: images {images.shape}, masks {masks.shape}")
        
        # Test showing batch
        logger.info("👀 Testing show_batch...")
        dls.show_batch(max_n=2, figsize=(6, 6))
        
        logger.info("✅ Data loading test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mitochondria_data_loading()
    if success:
        print("🎉 Data loading test PASSED")
    else:
        print("💥 Data loading test FAILED")
        sys.exit(1)

