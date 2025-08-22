#!/usr/bin/env python3
"""Test script to check if MPS linalg solve issue is fixed in PyTorch 2.6.0."""

import os
import torch
import torch.nn.functional as F

def test_mps_linalg_solve():
    """Test if the MPS linalg solve issue is fixed."""
    
    print("🧪 Testing MPS linalg solve fix in PyTorch 2.6.0")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available, cannot test")
        return False
    
    # Set device to MPS
    device = torch.device("mps")
    print(f"Using device: {device}")
    
    try:
        # Test 1: Basic tensor operations that might trigger linalg solve
        print("\n🔍 Test 1: Basic tensor operations...")
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        
        # This might trigger internal linalg operations
        result = torch.mm(x, y)
        print("✅ Basic matrix multiplication: PASSED")
        
        # Test 2: Try to trigger the specific linalg solve operation
        print("\n🔍 Test 2: Testing linalg solve operations...")
        
        # Create a batch of matrices
        A = torch.randn(3, 3, 3, device=device)  # Batch of 3x3 matrices
        b = torch.randn(3, 3, device=device)     # Batch of 3x1 vectors
        
        # This should use the linalg solve internally
        try:
            # Try to solve Ax = b for each batch
            x_solved = torch.linalg.solve(A, b.unsqueeze(-1))
            print("✅ torch.linalg.solve: PASSED")
        except Exception as e:
            print(f"❌ torch.linalg.solve failed: {e}")
            return False
        
        # Test 3: Test the specific operation that was failing
        print("\n🔍 Test 3: Testing the previously failing operation...")
        
        # Create some data that might trigger the internal linalg solve
        data = torch.randn(5, 5, device=device)
        
        # Apply some operations that might use linalg solve internally
        try:
            # This might trigger the aten::_linalg_solve_ex.result internally
            result = torch.linalg.inv(data)
            print("✅ torch.linalg.inv: PASSED")
        except Exception as e:
            print(f"❌ torch.linalg.inv failed: {e}")
            return False
        
        # Test 4: Test fastai-style augmentations that were causing issues
        print("\n🔍 Test 4: Testing fastai-style augmentations...")
        
        # Simulate the kind of operations that fastai augmentations do
        try:
            # Create a batch of images
            batch = torch.randn(4, 3, 224, 224, device=device)
            
            # Apply some transformations that might use linalg operations
            # This simulates what fastai's augment_transforms might do internally
            transformed = F.interpolate(batch, scale_factor=1.1, mode='bilinear', align_corners=False)
            transformed = F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0).expand(4, -1, -1), 
                                      transformed.shape, align_corners=False)
            
            print("✅ Fastai-style transformations: PASSED")
        except Exception as e:
            print(f"❌ Fastai-style transformations failed: {e}")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The MPS linalg solve issue appears to be FIXED in PyTorch 2.6.0!")
        print("✅ You should be able to remove PYTORCH_ENABLE_MPS_FALLBACK=1")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("❌ The MPS linalg solve issue is still present")
        print("❌ You still need PYTORCH_ENABLE_MPS_FALLBACK=1")
        return False

if __name__ == "__main__":
    # Don't set the fallback environment variable for this test
    if 'PYTORCH_ENABLE_MPS_FALLBACK' in os.environ:
        del os.environ['PYTORCH_ENABLE_MPS_FALLBACK']
        print("🔧 Removed PYTORCH_ENABLE_MPS_FALLBACK for testing")
    
    success = test_mps_linalg_solve()
    
    if success:
        print("\n🚀 RECOMMENDATION: Try running your fastai pipeline without PYTORCH_ENABLE_MPS_FALLBACK=1")
    else:
        print("\n⚠️ RECOMMENDATION: Keep using PYTORCH_ENABLE_MPS_FALLBACK=1 for now")
