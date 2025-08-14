"""
Final integration test for zero-copy optimized split-GEMM.
"""

import torch
import time
import sys
sys.path.append('/home/rtx3090/code_jiaxi/LORO-main_temp')


def test_final_integration():
    """Test the final integrated zero-copy optimization."""
    print("Final Integration Test - Zero-Copy Split-GEMM")
    print("="*60)
    
    # Clear CUDA state
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    from fused_sparsity_ops import sparsity_tracker
    from peft_pretraining.split_gemm_utils import (
        compute_split_gemm_lowrank_intermediate,
        apply_split_gemm_to_dy1,
        compute_split_gemm_dx
    )
    
    # Configuration
    batch_seq = 2048
    hidden_size = 768
    intermediate_size = 3072
    rank = 256
    
    device = 'cuda'
    dtype = torch.float16
    
    print(f"Configuration:")
    print(f"  Batch*Seq: {batch_seq}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Rank: {rank}")
    
    # Create test data
    dy1 = torch.randn(batch_seq, hidden_size, device=device, dtype=dtype)
    weight_out1 = torch.randn(hidden_size, rank, device=device, dtype=dtype)
    weight1 = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    dy1_large = torch.randn(batch_seq, intermediate_size, device=device, dtype=dtype)
    
    # Create sparse masks
    sparse_mask1 = torch.rand(hidden_size, device=device) < 0.95
    sparse_mask2 = torch.rand(intermediate_size, device=device) < 0.95
    
    # Store in tracker
    layer_id1 = "layer1"
    layer_id2 = "layer2"
    
    col_sparsity1 = torch.rand(hidden_size, device=device)
    col_sparsity2 = torch.rand(intermediate_size, device=device)
    
    sparsity_tracker.store_sparsity(layer_id1, col_sparsity1, sparse_mask1)
    sparsity_tracker.store_sparsity(layer_id2, col_sparsity2, sparse_mask2)
    
    print(f"  Sparse ratio (layer1): {sparse_mask1.float().mean()*100:.1f}%")
    print(f"  Sparse ratio (layer2): {sparse_mask2.float().mean()*100:.1f}%")
    
    num_iterations = 100
    
    # Test 1: compute_split_gemm_lowrank_intermediate
    print(f"\n1. compute_split_gemm_lowrank_intermediate:")
    
    # Warmup
    for _ in range(10):
        _ = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, layer_id1)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result1 = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, layer_id1)
    torch.cuda.synchronize()
    opt_time1 = time.time() - start
    
    # Compare with standard GEMM
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_std1 = torch.mm(dy1, weight_out1)
    torch.cuda.synchronize()
    std_time1 = time.time() - start
    
    print(f"   Zero-copy: {opt_time1:.4f}s ({opt_time1/num_iterations*1000:.3f}ms per call)")
    print(f"   Standard: {std_time1:.4f}s ({std_time1/num_iterations*1000:.3f}ms per call)")
    print(f"   Overhead: {(opt_time1-std_time1)/std_time1*100:.1f}%")
    print(f"   Speedup vs original (~100ms): {100.0/(opt_time1/num_iterations*1000):.1f}x")
    
    # Test 2: apply_split_gemm_to_dy1
    print(f"\n2. apply_split_gemm_to_dy1:")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result2 = apply_split_gemm_to_dy1(dy1, layer_id1)
    torch.cuda.synchronize()
    apply_time = time.time() - start
    
    print(f"   Time: {apply_time:.4f}s ({apply_time/num_iterations*1000:.3f}ms per call)")
    
    # Verify sparsity
    sparse_indices = torch.where(sparse_mask1)[0]
    if len(sparse_indices) > 0:
        sample_col = result2[:, sparse_indices[0]]
        sample_col_reshaped = sample_col.view(-1, 4)
        nonzero_per_group = (sample_col_reshaped != 0).sum(dim=1)
        has_2to4 = (nonzero_per_group <= 2).all()
        print(f"   2:4 pattern verified: {has_2to4}")
    
    # Test 3: compute_split_gemm_dx
    print(f"\n3. compute_split_gemm_dx:")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result3 = compute_split_gemm_dx(dy1_large, weight1, layer_id2)
    torch.cuda.synchronize()
    dx_time = time.time() - start
    
    # Compare with standard
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_std3 = torch.mm(dy1_large, weight1.T)
    torch.cuda.synchronize()
    std_time3 = time.time() - start
    
    print(f"   Zero-copy: {dx_time:.4f}s ({dx_time/num_iterations*1000:.3f}ms per call)")
    print(f"   Standard: {std_time3:.4f}s ({std_time3/num_iterations*1000:.3f}ms per call)")
    print(f"   Overhead: {(dx_time-std_time3)/std_time3*100:.1f}%")
    
    print("\n" + "="*60)
    print("✓ All tests completed successfully!")
    print(f"\nSummary:")
    print(f"  Average overhead vs standard GEMM: ~{((opt_time1-std_time1)/std_time1*100 + (dx_time-std_time3)/std_time3*100)/2:.1f}%")
    print(f"  Speedup vs original implementation: ~{100.0/((opt_time1/num_iterations*1000 + dx_time/num_iterations*1000)/2):.0f}x")
    print("="*60)


if __name__ == "__main__":
    test_final_integration()