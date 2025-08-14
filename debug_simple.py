"""
Simple debug to isolate the issue.
"""

import torch
import sys
sys.path.append('/home/rtx3090/code_jiaxi/LORO-main_temp')


def test_original_vs_optimized():
    """Compare original and optimized implementations."""
    print("Comparing original vs optimized split-GEMM...")
    print("="*60)
    
    # Small test case
    M, K, N = 128, 64, 32
    device = 'cuda'
    dtype = torch.float16
    
    # Create test data
    dy1 = torch.randn(M, K, device=device, dtype=dtype)
    weight = torch.randn(K, N, device=device, dtype=dtype)
    
    # Create sparse mask
    sparse_mask = torch.rand(K, device=device) < 0.8  # 80% sparse
    
    print(f"Test configuration:")
    print(f"  dy1: [{M}, {K}]")
    print(f"  weight: [{K}, {N}]")
    print(f"  Sparse ratio: {sparse_mask.float().mean()*100:.1f}%")
    
    # Test 1: Original implementation (manual)
    print("\n1. Testing original implementation:")
    try:
        dense_mask = ~sparse_mask
        result_orig = torch.zeros(M, N, device=device, dtype=dtype)
        
        # Sparse part
        if sparse_mask.any():
            dy1_sparse = dy1[:, sparse_mask]
            weight_sparse = weight[sparse_mask, :]
            # Skip 2:4 sparsity for this test
            result_orig += torch.mm(dy1_sparse, weight_sparse)
        
        # Dense part
        if dense_mask.any():
            dy1_dense = dy1[:, dense_mask]
            weight_dense = weight[dense_mask, :]
            result_orig += torch.mm(dy1_dense, weight_dense)
        
        print(f"  ✓ Success! Result shape: {result_orig.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 2: Optimized reordering
    print("\n2. Testing optimized reordering:")
    try:
        # Get indices
        sparse_indices = torch.where(sparse_mask)[0]
        dense_indices = torch.where(~sparse_mask)[0]
        
        # Create permutation
        perm = torch.cat([sparse_indices, dense_indices])
        num_sparse = len(sparse_indices)
        
        print(f"  Permutation shape: {perm.shape}")
        print(f"  Num sparse: {num_sparse}")
        
        # Reorder
        dy1_reordered = dy1[:, perm]
        weight_reordered = weight[perm, :]
        
        print(f"  dy1_reordered shape: {dy1_reordered.shape}")
        print(f"  weight_reordered shape: {weight_reordered.shape}")
        
        # Split using views
        dy1_sparse_view = dy1_reordered[:, :num_sparse]
        dy1_dense_view = dy1_reordered[:, num_sparse:]
        
        weight_sparse_view = weight_reordered[:num_sparse, :]
        weight_dense_view = weight_reordered[num_sparse:, :]
        
        print(f"  Sparse view shapes: {dy1_sparse_view.shape}, {weight_sparse_view.shape}")
        print(f"  Dense view shapes: {dy1_dense_view.shape}, {weight_dense_view.shape}")
        
        # Compute
        result_opt = torch.zeros(M, N, device=device, dtype=dtype)
        
        if num_sparse > 0:
            result_opt += torch.mm(dy1_sparse_view, weight_sparse_view)
        
        if dy1_dense_view.shape[1] > 0:
            result_opt += torch.mm(dy1_dense_view, weight_dense_view)
        
        print(f"  ✓ Success! Result shape: {result_opt.shape}")
        
        # Compare results
        diff = (result_orig - result_opt).abs().max()
        print(f"  Max difference: {diff:.6f}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test completed")


if __name__ == "__main__":
    test_original_vs_optimized()