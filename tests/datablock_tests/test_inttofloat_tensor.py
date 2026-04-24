#!/usr/bin/env python3
"""
Test IntToFloatTensor behavior with masks.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("Testing IntToFloatTensor behavior...")

try:
    from fastai.vision.all import IntToFloatTensor, PILMask
    from eq.data_management.standard_getters import get_y
    import numpy as np
    import torch
    
    data_root = str(Path.home() / "ProjectsRuntime" / "endotheliosis_quantifier" / "raw_data" / "mitochondria_data" / "training")
    from eq.data_management.datablock_loader import get_items_full_images
    
    items = get_items_full_images(Path(data_root))
    if items:
        # Get a mask
        mask = get_y(items[0])
        print(f"Original mask type: {type(mask)}")
        
        mask_array = np.array(mask)
        print(f"Original mask array dtype: {mask_array.dtype}")
        print(f"Original mask array unique: {np.unique(mask_array)}")
        
        # Convert to tensor
        mask_tensor = torch.tensor(mask_array)
        print(f"Tensor dtype: {mask_tensor.dtype}")
        print(f"Tensor unique: {mask_tensor.unique()}")
        
        # Apply IntToFloatTensor
        int_to_float = IntToFloatTensor()
        result = int_to_float.encodes(mask_tensor)
        print(f"After IntToFloatTensor dtype: {result.dtype}")
        print(f"After IntToFloatTensor unique: {result.unique()}")
        
        # Test with different tensor types
        print(f"\n--- Testing different tensor types ---")
        
        # Float32 tensor
        float_tensor = mask_tensor.float()
        print(f"Float32 tensor dtype: {float_tensor.dtype}")
        result_float = int_to_float.encodes(float_tensor)
        print(f"Float32 after IntToFloatTensor dtype: {result_float.dtype}")
        
        # Int64 tensor
        int_tensor = mask_tensor.long()
        print(f"Int64 tensor dtype: {int_tensor.dtype}")
        result_int = int_to_float.encodes(int_tensor)
        print(f"Int64 after IntToFloatTensor dtype: {result_int.dtype}")
        
        # Test batch processing
        print(f"\n--- Testing batch processing ---")
        batch_tensor = torch.stack([mask_tensor, mask_tensor])
        print(f"Batch tensor dtype: {batch_tensor.dtype}")
        result_batch = int_to_float.encodes(batch_tensor)
        print(f"Batch after IntToFloatTensor dtype: {result_batch.dtype}")
        
        # Check if the issue is in the DataBlock itself
        print(f"\n--- Testing DataBlock without IntToFloatTensor ---")
        from fastai.data.block import DataBlock
        from fastai.data.transforms import RandomSplitter
        from fastai.vision.all import ImageBlock, MaskBlock
        
        block_no_transform = DataBlock(
            blocks=[ImageBlock, MaskBlock(codes=[0, 1])],
            get_items=get_items_full_images,
            get_y=get_y,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            item_tfms=[],
            batch_tfms=[],  # No transforms
        )
        
        dls_no_transform = block_no_transform.dataloaders(Path(data_root), bs=2, num_workers=0)
        batch_no_transform = dls_no_transform.train.one_batch()
        if len(batch_no_transform) == 2:
            images_nt, masks_nt = batch_no_transform
            print(f"No transform - Masks dtype: {masks_nt.dtype}")
            print(f"No transform - Masks unique: {masks_nt.unique()}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("IntToFloatTensor testing complete.")
