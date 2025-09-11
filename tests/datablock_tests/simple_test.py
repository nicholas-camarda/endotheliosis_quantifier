#!/usr/bin/env python3
"""
Simple test to debug the datablock issues.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("Starting simple test...")

try:
    from eq.data_management.datablock_loader import get_items_full_images
    print("✅ Imported get_items_full_images")
    
    data_root = "/home/ncamarda/endotheliosis_quantifier/data/derived_data/mito"
    print(f"Testing with data root: {data_root}")
    
    items = get_items_full_images(Path(data_root))
    print(f"✅ Found {len(items)} items")
    
    if items:
        print(f"First item: {items[0]}")
        
        # Test get_y function
        from eq.data_management.standard_getters import get_y
        mask = get_y(items[0])
        print(f"✅ Loaded mask: {type(mask)}")
        
        import numpy as np
        mask_array = np.array(mask)
        print(f"Mask shape: {mask_array.shape}")
        print(f"Mask dtype: {mask_array.dtype}")
        print(f"Mask unique values: {np.unique(mask_array)}")
        print(f"Positive pixels: {(mask_array > 0).sum()}/{mask_array.size}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")
