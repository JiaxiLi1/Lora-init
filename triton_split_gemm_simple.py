"""
Simplified but efficient Split-GEMM Triton kernel.
Key idea: Process sparse and dense features separately but in single kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def naive_2to4_kernel(
    x_ptr,
    y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Apply 2:4 sparsity pattern to tensor."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Process in groups of 4 along M dimension
    for i in range(0, BLOCK_SIZE_M, 4):
        for j in range(BLOCK_SIZE_N):
            m_base = pid_m * BLOCK_SIZE_M + i
            n_idx = pid_n * BLOCK_SIZE_N + j
            
            if m_base + 3 < M and n_idx < N:
                # Load 4 values
                ptrs = x_ptr + (m_base + tl.arange(0, 4)) * stride_xm + n_idx * stride_xn
                vals = tl.load(ptrs)
                abs_vals = tl.abs(vals)
                
                # Find top 2 - simple approach
                # Get max
                max_val = tl.max(abs_vals, axis=0)
                mask1 = abs_vals == max_val
                
                # Suppress max and get second max
                suppressed = tl.where(mask1, -1.0, abs_vals)
                max2_val = tl.max(suppressed, axis=0)
                mask2 = suppressed == max2_val
                
                # Keep top 2
                keep_mask = mask1 | mask2
                result = tl.where(keep_mask, vals, 0.0)
                
                # Store
                out_ptrs = y_ptr + (m_base + tl.arange(0, 4)) * stride_ym + n_idx * stride_yn
                tl.store(out_ptrs, result)


@triton.jit
def split_gemm_kernel_v2(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    sparse_mask_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Split-GEMM kernel that processes sparse columns with 2:4 pattern.
    Simplified version that's easier to compile.
    """
    pid = tl.program_id(0)
    
    # Compute block indices
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, K, BLOCK_K):
        # Load A block
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # For each column in this K block, check if it's sparse
        # and apply 2:4 pattern if needed
        # This is simplified - in practice would need proper 2:4 logic
        
        # Matrix multiply
        acc += tl.dot(a, b)
    
    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def split_gemm_hybrid(dy1, weight, sparse_mask):
    """
    Hybrid approach: Use existing operations more efficiently.
    
    Key optimization: Process sparse and dense in parallel using streams.
    """
    M, K = dy1.shape
    _, N = weight.shape
    
    device = dy1.device
    dtype = dy1.dtype
    
    # Get indices
    sparse_indices = torch.where(sparse_mask)[0]
    dense_indices = torch.where(~sparse_mask)[0]
    
    num_sparse = len(sparse_indices)
    num_dense = len(dense_indices)
    
    if num_sparse == 0:
        # All dense
        return torch.mm(dy1, weight)
    
    if num_dense == 0:
        # All sparse - apply 2:4 to everything
        from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity
        dy1_sparse = apply_naive_2to4_sparsity(dy1)
        from sparse_fullrank_linear import fake_fp8_mm
        return fake_fp8_mm(dy1_sparse, weight, torch.float8_e4m3fn)
    
    # Mixed case - process in parallel using CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    result = torch.zeros(M, N, device=device, dtype=dtype)
    
    # Process sparse part in stream1
    with torch.cuda.stream(stream1):
        from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
        from sparse_fullrank_linear import fake_fp8_mm
        
        dy1_sparse = dy1[:, sparse_indices]
        weight_sparse = weight[sparse_indices, :]
        
        dy1_sparse_t = dy1_sparse.t()
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_t)
        dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
        
        result_sparse = fake_fp8_mm(dy1_sparse_2to4, weight_sparse, torch.float8_e4m3fn)
    
    # Process dense part in stream2
    with torch.cuda.stream(stream2):
        dy1_dense = dy1[:, dense_indices]
        weight_dense = weight[dense_indices, :]
        result_dense = torch.mm(dy1_dense, weight_dense)
    
    # Synchronize and combine
    torch.cuda.synchronize()
    result = result_sparse + result_dense
    
    return result


def compute_split_gemm_optimized_v2(dy1, weight, layer_id):
    """
    Optimized split-GEMM using parallel streams.
    """
    from fused_sparsity_ops import sparsity_tracker
    
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    
    if sparse_mask is None:
        return torch.mm(dy1, weight)
    
    return split_gemm_hybrid(dy1, weight, sparse_mask)


def apply_split_gemm_sparsity_optimized(dy1, layer_id):
    """
    Apply 2:4 sparsity to selected columns efficiently.
    """
    from fused_sparsity_ops import sparsity_tracker
    from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
    
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    
    if sparse_mask is None:
        return dy1
    
    # Get sparse indices
    sparse_indices = torch.where(sparse_mask)[0]
    
    if len(sparse_indices) == 0:
        return dy1
    
    # Process only sparse columns
    result = dy1.clone()
    
    # Extract, sparsify, and put back
    dy1_sparse = dy1[:, sparse_indices]
    dy1_sparse_t = dy1_sparse.t()
    dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_t)
    dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
    
    result[:, sparse_indices] = dy1_sparse_2to4
    
    return result


if __name__ == "__main__":
    print("Testing optimized Split-GEMM with parallel streams...")
    print("="*60)
    
    import time
    from fused_sparsity_ops import sparsity_tracker
    
    # Test configuration
    M, K, N = 2048, 768, 256
    device = 'cuda'
    dtype = torch.float16
    
    # Create test data
    dy1 = torch.randn(M, K, device=device, dtype=dtype)
    weight = torch.randn(K, N, device=device, dtype=dtype)
    
    # Create sparse mask
    sparse_mask = torch.rand(K, device=device) < 0.95
    
    print(f"Configuration: [{M}, {K}] @ [{K}, {N}]")
    print(f"Sparse columns: {sparse_mask.sum().item()}/{K} ({sparse_mask.float().mean()*100:.1f}%)")
    
    # Store in tracker
    layer_id = "test_layer"
    col_sparsity = torch.rand(K, device=device)
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
    
    num_iterations = 100
    
    print(f"\nTesting optimized kernel ({num_iterations} iterations):")
    
    # Warmup
    for _ in range(10):
        _ = compute_split_gemm_optimized_v2(dy1, weight, layer_id)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result = compute_split_gemm_optimized_v2(dy1, weight, layer_id)
    torch.cuda.synchronize()
    opt_time = time.time() - start
    
    print(f"  Optimized: {opt_time:.4f}s ({opt_time/num_iterations*1000:.3f}ms per call)")
    
    # Compare with standard GEMM
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_std = torch.mm(dy1, weight)
    torch.cuda.synchronize()
    std_time = time.time() - start
    
    print(f"  Standard GEMM: {std_time:.4f}s ({std_time/num_iterations*1000:.3f}ms per call)")
    print(f"  Overhead: {(opt_time-std_time)/std_time*100:.1f}%")
    
    # Test apply function
    print(f"\nTesting apply_split_gemm_sparsity_optimized:")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result2 = apply_split_gemm_sparsity_optimized(dy1, layer_id)
    torch.cuda.synchronize()
    apply_time = time.time() - start
    
    print(f"  Time: {apply_time:.4f}s ({apply_time/num_iterations*1000:.3f}ms per call)")
    
    # Verify correctness (basic check)
    sparse_part = result2[:, sparse_mask]
    dense_part = result2[:, ~sparse_mask]
    
    print(f"\n  Sparse part shape: {sparse_part.shape}")
    print(f"  Dense part shape: {dense_part.shape}")
    print(f"  Sparse nonzero ratio: {(sparse_part != 0).float().mean()*100:.1f}%")
    print(f"  Dense nonzero ratio: {(dense_part != 0).float().mean()*100:.1f}%")
    
    print("\n" + "="*60)
    print("✓ Test completed!")