"""
Test with actual 2:4 sparsity functions.
"""

import torch
import sys
sys.path.append('/home/rtx3090/code_jiaxi/LORO-main_temp')

from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
from sparse_fullrank_linear import fake_fp8_mm


def test_with_2to4_sparsity():
    """Test with actual 2:4 sparsity operations."""
    print("Testing with 2:4 sparsity...")
    print("="*60)
    
    # Test configuration - use multiples of 4 for 2:4 sparsity
    M, K, N = 128, 64, 32
    device = 'cuda'
    dtype = torch.float16
    
    # Create test data
    dy1 = torch.randn(M, K, device=device, dtype=dtype)
    weight = torch.randn(K, N, device=device, dtype=dtype)
    
    # Create sparse mask
    sparse_mask = torch.rand(K, device=device) < 0.8
    
    print(f"Configuration:")
    print(f"  dy1: [{M}, {K}]")
    print(f"  weight: [{K}, {N}]")
    print(f"  Sparse ratio: {sparse_mask.float().mean()*100:.1f}%")
    
    # Get indices
    sparse_indices = torch.where(sparse_mask)[0]
    dense_indices = torch.where(~sparse_mask)[0]
    num_sparse = len(sparse_indices)
    
    print(f"\nNum sparse: {num_sparse}, Num dense: {len(dense_indices)}")
    
    # Test 1: Apply 2:4 sparsity to a subset
    print("\n1. Testing apply_naive_2to4_sparsity_featurewise:")
    try:
        dy1_sparse = dy1[:, sparse_indices]
        print(f"  dy1_sparse shape: {dy1_sparse.shape}")
        
        dy1_sparse_t = dy1_sparse.t()
        print(f"  dy1_sparse_t shape: {dy1_sparse_t.shape}")
        
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_t)
        print(f"  dy1_sparse_2to4_t shape: {dy1_sparse_2to4_t.shape}")
        
        dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
        print(f"  dy1_sparse_2to4 shape: {dy1_sparse_2to4.shape}")
        
        print("  ✓ 2:4 sparsity applied successfully")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Use fake_fp8_mm
    print("\n2. Testing fake_fp8_mm:")
    try:
        weight_sparse = weight[sparse_indices, :]
        print(f"  weight_sparse shape: {weight_sparse.shape}")
        
        result = fake_fp8_mm(dy1_sparse_2to4, weight_sparse, torch.float8_e4m3fn)
        print(f"  ✓ fake_fp8_mm successful! Result shape: {result.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Full optimized implementation
    print("\n3. Testing full optimized implementation:")
    try:
        # Create permutation
        perm = torch.cat([sparse_indices, dense_indices])
        
        # Reorder
        dy1_reordered = dy1[:, perm]
        weight_reordered = weight[perm, :]
        
        # Initialize result
        result_full = torch.zeros(M, N, device=device, dtype=dtype)
        
        # Sparse part with 2:4
        if num_sparse > 0:
            dy1_sparse_view = dy1_reordered[:, :num_sparse]
            weight_sparse_view = weight_reordered[:num_sparse, :]
            
            # Apply 2:4
            dy1_sparse_t = dy1_sparse_view.t()
            dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_t)
            dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
            
            # Compute with fake_fp8_mm
            result_full += fake_fp8_mm(dy1_sparse_2to4, weight_sparse_view, torch.float8_e4m3fn)
        
        # Dense part
        if len(dense_indices) > 0:
            dy1_dense_view = dy1_reordered[:, num_sparse:]
            weight_dense_view = weight_reordered[num_sparse:, :]
            result_full += torch.mm(dy1_dense_view, weight_dense_view)
        
        print(f"  ✓ Full implementation successful! Result shape: {result_full.shape}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test completed")


if __name__ == "__main__":
    test_with_2to4_sparsity()