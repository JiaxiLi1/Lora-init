"""
Test and benchmark the optimized split-GEMM implementation.
"""

import torch
import time
import sys
sys.path.append('/home/rtx3090/code_jiaxi/LORO-main_temp')

from peft_pretraining.split_gemm_utils import (
    compute_split_gemm_lowrank_intermediate,
    apply_split_gemm_to_dy1,
    compute_split_gemm_dx
)
from fused_sparsity_ops import sparsity_tracker
from triton_fused_split_gemm import fused_split_gemm_forward


def benchmark_split_gemm():
    """Benchmark split-GEMM operations."""
    print("="*60)
    print("Split-GEMM Performance Benchmark")
    print("="*60)
    
    # Test configurations
    batch_seq = 2048
    hidden_size = 768
    intermediate_size = 3072
    rank = 256
    
    device = 'cuda'
    dtype = torch.float16
    
    # Create test tensors
    dy1 = torch.randn(batch_seq, hidden_size, device=device, dtype=dtype)
    weight_out1 = torch.randn(hidden_size, rank, device=device, dtype=dtype)
    weight1 = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    
    # Create sparsity mask (95% sparse)
    col_sparsity = torch.rand(hidden_size, device=device)
    sparse_mask = col_sparsity > 0.05  # 95% sparse
    
    # Store in sparsity tracker
    layer_id = "test_layer"
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
    
    print(f"\nConfiguration:")
    print(f"  Batch*Seq: {batch_seq}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Rank: {rank}")
    print(f"  Sparse columns: {sparse_mask.sum().item()} ({sparse_mask.float().mean()*100:.1f}%)")
    print(f"  Dense columns: {(~sparse_mask).sum().item()} ({(~sparse_mask).float().mean()*100:.1f}%)")
    
    # Warmup
    for _ in range(10):
        _ = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, layer_id)
        _ = apply_split_gemm_to_dy1(dy1, layer_id)
        _ = compute_split_gemm_dx(dy1[:, :intermediate_size], weight1, layer_id)
    
    num_iterations = 100
    
    # Benchmark 1: compute_split_gemm_lowrank_intermediate
    print(f"\n1. Testing compute_split_gemm_lowrank_intermediate (dy1 @ weight_out1):")
    
    # Old implementation (manual split)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_old = manual_split_gemm(dy1, weight_out1, sparse_mask)
    torch.cuda.synchronize()
    old_time = time.time() - start
    
    # New implementation (optimized)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_new = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, layer_id)
    torch.cuda.synchronize()
    new_time = time.time() - start
    
    print(f"   Old manual split: {old_time:.4f}s ({old_time/num_iterations*1000:.3f}ms per call)")
    print(f"   New optimized: {new_time:.4f}s ({new_time/num_iterations*1000:.3f}ms per call)")
    print(f"   Speedup: {old_time/new_time:.2f}x")
    
    # Verify correctness
    diff = (result_old - result_new).abs().max()
    print(f"   Max difference: {diff:.6f}")
    
    # Benchmark 2: apply_split_gemm_to_dy1
    print(f"\n2. Testing apply_split_gemm_to_dy1:")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        dy1_sparse = apply_split_gemm_to_dy1(dy1, layer_id)
    torch.cuda.synchronize()
    apply_time = time.time() - start
    
    print(f"   Time: {apply_time:.4f}s ({apply_time/num_iterations*1000:.3f}ms per call)")
    
    # Check sparsity pattern
    sparse_part = dy1_sparse[:, sparse_mask]
    dense_part = dy1_sparse[:, ~sparse_mask]
    
    # Sparse part should have 2:4 pattern
    sparse_part_flat = sparse_part.flatten()
    sparse_groups = sparse_part_flat.view(-1, 4)
    nonzero_per_group = (sparse_groups != 0).sum(dim=1)
    is_2to4 = (nonzero_per_group <= 2).all()
    
    print(f"   Sparse part has 2:4 pattern: {is_2to4}")
    print(f"   Dense part unchanged: {torch.allclose(dense_part, dy1[:, ~sparse_mask])}")
    
    # Benchmark 3: Standard vs Split-GEMM
    print(f"\n3. Comparing standard GEMM vs Split-GEMM:")
    
    # Standard GEMM (no sparsity)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_standard = torch.mm(dy1, weight_out1)
    torch.cuda.synchronize()
    standard_time = time.time() - start
    
    print(f"   Standard GEMM: {standard_time:.4f}s ({standard_time/num_iterations*1000:.3f}ms per call)")
    print(f"   Split-GEMM: {new_time:.4f}s ({new_time/num_iterations*1000:.3f}ms per call)")
    print(f"   Overhead: {(new_time - standard_time)/standard_time*100:.1f}%")
    
    print("\n" + "="*60)
    print("✓ Benchmark completed!")
    print("="*60)


def manual_split_gemm(dy1, weight_out1, sparse_mask):
    """Manual implementation of split-GEMM for comparison."""
    batch_seq_len, hidden_size = dy1.shape
    _, rank1 = weight_out1.shape
    
    dense_mask = ~sparse_mask
    result = torch.zeros(batch_seq_len, rank1, device=dy1.device, dtype=dy1.dtype)
    
    # Sparse part
    if sparse_mask.any():
        from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
        
        dy1_sparse_part = dy1[:, sparse_mask]
        dy1_sparse_part_t = dy1_sparse_part.t()
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_part_t)
        dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
        
        weight_out1_sparse = weight_out1[sparse_mask, :]
        from sparse_fullrank_linear import fake_fp8_mm
        result += fake_fp8_mm(dy1_sparse_2to4, weight_out1_sparse, torch.float8_e4m3fn)
    
    # Dense part
    if dense_mask.any():
        dy1_dense_part = dy1[:, dense_mask]
        weight_out1_dense = weight_out1[dense_mask, :]
        result += torch.mm(dy1_dense_part, weight_out1_dense)
    
    return result


def test_fused_kernel():
    """Test the fused split-GEMM kernel if available."""
    print("\n" + "="*60)
    print("Testing Fused Split-GEMM Kernel")
    print("="*60)
    
    try:
        # Test configuration
        batch_seq = 2048
        in_features = 3072
        out_features = 768
        
        device = 'cuda'
        dtype = torch.float16
        
        # Create test data
        input_tensor = torch.randn(batch_seq, in_features, device=device, dtype=dtype)
        weight = torch.randn(in_features, out_features, device=device, dtype=dtype)
        
        # Create sparsity mask
        sparse_mask = torch.rand(in_features, device=device) > 0.05  # 95% sparse
        
        print(f"\nConfiguration:")
        print(f"  Input: [{batch_seq}, {in_features}]")
        print(f"  Weight: [{in_features}, {out_features}]")
        print(f"  Sparse features: {sparse_mask.sum().item()}/{in_features}")
        
        # Test fused kernel
        num_iterations = 100
        
        # Warmup
        for _ in range(10):
            _ = fused_split_gemm_forward(input_tensor, weight, sparse_mask)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            output = fused_split_gemm_forward(input_tensor, weight, sparse_mask)
        torch.cuda.synchronize()
        fused_time = time.time() - start
        
        print(f"\nFused kernel performance:")
        print(f"  Time: {fused_time:.4f}s ({fused_time/num_iterations*1000:.3f}ms per call)")
        
        # Compare with standard GEMM
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            output_standard = torch.mm(input_tensor, weight)
        torch.cuda.synchronize()
        standard_time = time.time() - start
        
        print(f"\nComparison:")
        print(f"  Standard GEMM: {standard_time:.4f}s")
        print(f"  Fused Split-GEMM: {fused_time:.4f}s")
        print(f"  Speedup: {standard_time/fused_time:.2f}x")
        
        print("\n✓ Fused kernel test completed!")
        
    except Exception as e:
        print(f"✗ Fused kernel test failed: {e}")
    
    print("="*60)


if __name__ == "__main__":
    print("Testing Split-GEMM Optimizations")
    print("="*60)
    
    # Run benchmarks
    benchmark_split_gemm()
    
    # Test fused kernel
    test_fused_kernel()
    
    print("\n✓ All tests completed!")