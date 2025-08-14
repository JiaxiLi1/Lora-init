"""
Debug script to find the indexing issue.
"""

import torch
import sys
sys.path.append('/home/rtx3090/code_jiaxi/LORO-main_temp')

from fused_sparsity_ops import sparsity_tracker
from optimized_split_gemm_reorder import ReorderedSplitGEMM

def debug_indexing():
    # Create test data
    hidden_size = 768
    device = 'cuda'
    
    # Create sparse mask
    sparse_mask = torch.rand(hidden_size, device=device) < 0.95
    print(f"Sparse mask shape: {sparse_mask.shape}")
    print(f"Sparse count: {sparse_mask.sum().item()}")
    print(f"Dense count: {(~sparse_mask).sum().item()}")
    
    # Get indices
    sparse_indices = torch.where(sparse_mask)[0]
    dense_indices = torch.where(~sparse_mask)[0]
    
    print(f"\nSparse indices: min={sparse_indices.min().item()}, max={sparse_indices.max().item()}, count={len(sparse_indices)}")
    print(f"Dense indices: min={dense_indices.min().item() if len(dense_indices) > 0 else 'N/A'}, max={dense_indices.max().item() if len(dense_indices) > 0 else 'N/A'}, count={len(dense_indices)}")
    
    # Create permutation
    perm = torch.cat([sparse_indices, dense_indices])
    print(f"\nPermutation shape: {perm.shape}")
    print(f"Permutation min: {perm.min().item()}, max: {perm.max().item()}")
    
    # Check if permutation is valid
    unique = torch.unique(perm)
    print(f"Unique values in perm: {len(unique)} (should be {hidden_size})")
    
    # Test indexing
    test_array = torch.randn(2048, hidden_size, device=device, dtype=torch.float16)
    print(f"\nTest array shape: {test_array.shape}")
    
    try:
        # This should work
        reordered = test_array[:, perm]
        print(f"Reordered shape: {reordered.shape}")
        print("✓ Forward reordering works")
    except Exception as e:
        print(f"✗ Forward reordering failed: {e}")
    
    # Test with ReorderedSplitGEMM
    reordered_gemm = ReorderedSplitGEMM()
    layer_id = "test_layer"
    
    # Store in tracker
    col_sparsity = torch.rand(hidden_size, device=device)
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
    
    # Get permutation
    perm2, inv_perm, num_sparse = reordered_gemm.get_or_create_permutation(layer_id, sparse_mask)
    
    print(f"\nFrom ReorderedSplitGEMM:")
    print(f"  perm shape: {perm2.shape}")
    print(f"  inv_perm shape: {inv_perm.shape}")
    print(f"  num_sparse: {num_sparse}")
    
    # Check inv_perm validity
    print(f"  inv_perm min: {inv_perm.min().item()}, max: {inv_perm.max().item()}")
    
    # Test the problematic operation from apply_split_gemm_to_dy1_reordered
    dy1 = torch.randn(2048, hidden_size, device=device, dtype=torch.float16)
    dy1_reordered = dy1[:, perm2]
    
    print(f"\ndy1_reordered shape: {dy1_reordered.shape}")
    
    # Try to reorder back using different methods
    print("\nTesting reorder back methods:")
    
    # Method 1: Using scatter
    try:
        result1 = torch.zeros_like(dy1)
        result1[:, perm2] = dy1_reordered
        print("✓ Method 1 (scatter) works")
    except Exception as e:
        print(f"✗ Method 1 failed: {e}")
    
    # Method 2: Using inverse permutation (problematic)
    try:
        result2 = dy1_reordered[:, inv_perm]
        print("✓ Method 2 (inv_perm) works")
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")
    
    print("\n" + "="*60)
    print("Debug completed")


if __name__ == "__main__":
    debug_indexing()