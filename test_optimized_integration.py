"""
Test the integrated optimized split-GEMM.
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


def test_integration():
    print("Testing integrated optimized split-GEMM...")
    print("="*60)
    
    # Configuration
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
    dy1_large = torch.randn(batch_seq, intermediate_size, device=device, dtype=dtype)
    
    # Create sparsity masks
    sparse_mask1 = torch.rand(hidden_size, device=device) < 0.95
    sparse_mask2 = torch.rand(intermediate_size, device=device) < 0.95
    
    # Store in tracker
    layer_id1 = "test_layer1"
    layer_id2 = "test_layer2"
    
    col_sparsity1 = torch.rand(hidden_size, device=device)
    col_sparsity2 = torch.rand(intermediate_size, device=device)
    
    sparsity_tracker.store_sparsity(layer_id1, col_sparsity1, sparse_mask1)
    sparsity_tracker.store_sparsity(layer_id2, col_sparsity2, sparse_mask2)
    
    print(f"Configuration:")
    print(f"  Batch*Seq: {batch_seq}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Rank: {rank}")
    print(f"  Sparse ratio (layer1): {sparse_mask1.float().mean()*100:.1f}%")
    print(f"  Sparse ratio (layer2): {sparse_mask2.float().mean()*100:.1f}%")
    
    num_iterations = 100
    
    # Test 1: compute_split_gemm_lowrank_intermediate
    print(f"\n1. Testing compute_split_gemm_lowrank_intermediate:")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result1 = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, layer_id1)
    torch.cuda.synchronize()
    time1 = time.time() - start
    
    print(f"   Time: {time1:.4f}s ({time1/num_iterations*1000:.3f}ms per call)")
    
    # Test 2: apply_split_gemm_to_dy1
    print(f"\n2. Testing apply_split_gemm_to_dy1:")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result2 = apply_split_gemm_to_dy1(dy1, layer_id1)
    torch.cuda.synchronize()
    time2 = time.time() - start
    
    print(f"   Time: {time2:.4f}s ({time2/num_iterations*1000:.3f}ms per call)")
    
    # Test 3: compute_split_gemm_dx
    print(f"\n3. Testing compute_split_gemm_dx:")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result3 = compute_split_gemm_dx(dy1_large, weight1, layer_id2)
    torch.cuda.synchronize()
    time3 = time.time() - start
    
    print(f"   Time: {time3:.4f}s ({time3/num_iterations*1000:.3f}ms per call)")
    
    # Compare with standard GEMM
    print(f"\n4. Comparison with standard GEMM:")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        std1 = torch.mm(dy1, weight_out1)
    torch.cuda.synchronize()
    std_time1 = time.time() - start
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        std3 = torch.mm(dy1_large, weight1.T)
    torch.cuda.synchronize()
    std_time3 = time.time() - start
    
    print(f"   Standard GEMM (dy1 @ weight_out1): {std_time1:.4f}s")
    print(f"   Optimized split-GEMM: {time1:.4f}s")
    print(f"   Overhead: {(time1-std_time1)/std_time1*100:.1f}%")
    
    print(f"\n   Standard GEMM (dy1_large @ weight1.T): {std_time3:.4f}s")
    print(f"   Optimized split-GEMM: {time3:.4f}s")
    print(f"   Overhead: {(time3-std_time3)/std_time3*100:.1f}%")
    
    print("\n" + "="*60)
    print("✓ Integration test completed!")
    print("="*60)


if __name__ == "__main__":
    test_integration()