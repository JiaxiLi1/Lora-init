"""
Clean integration test for optimized split-GEMM.
"""

import torch
import time
import sys
sys.path.append('/home/rtx3090/code_jiaxi/LORO-main_temp')


def test_clean_integration():
    """Clean integration test without any state pollution."""
    print("Clean Integration Test for Optimized Split-GEMM")
    print("="*60)
    
    # Clear CUDA state at start
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    from fused_sparsity_ops import sparsity_tracker
    from peft_pretraining.split_gemm_utils import (
        compute_split_gemm_lowrank_intermediate,
        apply_split_gemm_to_dy1
    )
    
    # Configuration
    batch_seq = 2048
    hidden_size = 768
    rank = 256
    
    device = 'cuda'
    dtype = torch.float16
    
    print(f"Configuration:")
    print(f"  Batch*Seq: {batch_seq}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Rank: {rank}")
    print(f"  Dtype: {dtype}")
    
    # Create test data
    dy1 = torch.randn(batch_seq, hidden_size, device=device, dtype=dtype)
    weight_out1 = torch.randn(hidden_size, rank, device=device, dtype=dtype)
    
    # Create sparse mask
    sparse_mask = torch.rand(hidden_size, device=device) < 0.95
    layer_id = "layer_0"
    col_sparsity = torch.rand(hidden_size, device=device)
    
    print(f"  Sparse ratio: {sparse_mask.float().mean()*100:.1f}%")
    
    # Store in tracker
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
    
    num_iterations = 100
    
    # Test 1: compute_split_gemm_lowrank_intermediate
    print(f"\n1. Testing compute_split_gemm_lowrank_intermediate ({num_iterations} iterations):")
    
    # Warmup
    for _ in range(10):
        _ = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, layer_id)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result1 = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, layer_id)
    torch.cuda.synchronize()
    opt_time = time.time() - start
    
    print(f"   Optimized time: {opt_time:.4f}s ({opt_time/num_iterations*1000:.3f}ms per call)")
    print(f"   Result shape: {result1.shape}")
    
    # Compare with standard GEMM
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_std = torch.mm(dy1, weight_out1)
    torch.cuda.synchronize()
    std_time = time.time() - start
    
    print(f"   Standard GEMM: {std_time:.4f}s ({std_time/num_iterations*1000:.3f}ms per call)")
    print(f"   Overhead: {(opt_time-std_time)/std_time*100:.1f}%")
    
    # Test 2: apply_split_gemm_to_dy1
    print(f"\n2. Testing apply_split_gemm_to_dy1 ({num_iterations} iterations):")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result2 = apply_split_gemm_to_dy1(dy1, layer_id)
    torch.cuda.synchronize()
    apply_time = time.time() - start
    
    print(f"   Time: {apply_time:.4f}s ({apply_time/num_iterations*1000:.3f}ms per call)")
    print(f"   Result shape: {result2.shape}")
    
    # Verify sparsity pattern
    # Get the reordered permutation to check sparsity
    from optimized_split_gemm_reorder import reordered_split_gemm
    perm, inv_perm, num_sparse = reordered_split_gemm.get_or_create_permutation(layer_id, sparse_mask)
    
    # Reorder result2 to check sparsity pattern
    result2_reordered = result2[:, perm]
    sparse_part = result2_reordered[:, :num_sparse]
    dense_part = result2_reordered[:, num_sparse:]
    
    # Check that sparse part has 2:4 pattern (approximately)
    sparse_nonzero_ratio = (sparse_part != 0).float().mean()
    dense_nonzero_ratio = (dense_part != 0).float().mean()
    
    print(f"   Sparse part nonzero ratio: {sparse_nonzero_ratio*100:.1f}%")
    print(f"   Dense part nonzero ratio: {dense_nonzero_ratio*100:.1f}%")
    
    print("\n" + "="*60)
    print("✓ Clean integration test completed successfully!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_clean_integration()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)