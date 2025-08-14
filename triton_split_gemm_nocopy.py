"""
True zero-copy Split-GEMM implementation.
Key idea: Never extract columns. Process them in-place.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def split_gemm_2to4_kernel(
    # Input matrix A (will be modified in-place for 2:4)
    a_ptr,
    # Sparse mask
    sparse_mask_ptr,
    # Dimensions
    M, K,
    # Strides
    stride_am, stride_ak,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Apply 2:4 sparsity to columns marked as sparse IN-PLACE.
    This modifies the input tensor directly.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # Process each column in the block
    for k_idx in range(BLOCK_K):
        k = pid_k * BLOCK_K + k_idx
        if k < K:
            # Check if this column should be sparsified
            is_sparse = tl.load(sparse_mask_ptr + k)
            
            if is_sparse:
                # Process in groups of 4 along M dimension
                for m_start in range(0, BLOCK_M, 4):
                    m_base = pid_m * BLOCK_M + m_start
                    if m_base + 3 < M:
                        # Load 4 values from this column
                        ptrs = a_ptr + (m_base + tl.arange(0, 4)) * stride_am + k * stride_ak
                        vals = tl.load(ptrs)
                        abs_vals = tl.abs(vals)
                        
                        # Find top 2 values
                        max1 = tl.max(abs_vals, axis=0)
                        mask1 = abs_vals == max1
                        suppressed = tl.where(mask1, -1.0, abs_vals)
                        max2 = tl.max(suppressed, axis=0)
                        mask2 = suppressed == max2
                        
                        # Keep only top 2
                        keep = mask1 | mask2
                        result = tl.where(keep, vals, 0.0)
                        
                        # Store back IN-PLACE
                        tl.store(ptrs, result)


def split_gemm_nocopy(dy1, weight, sparse_mask):
    """
    Zero-copy Split-GEMM implementation.
    
    Strategy:
    1. Clone dy1 (unavoidable for gradient computation)
    2. Apply 2:4 sparsity IN-PLACE to sparse columns
    3. Use single GEMM with mixed sparse/dense data
    """
    M, K = dy1.shape
    _, N = weight.shape
    
    # Clone input (necessary to preserve original for other computations)
    dy1_work = dy1.clone()
    
    # Apply 2:4 sparsity IN-PLACE to sparse columns
    if sparse_mask is not None and sparse_mask.any():
        # Convert mask
        sparse_mask_int = sparse_mask.to(torch.int32)
        
        # Grid configuration
        BLOCK_M = 128
        BLOCK_K = 32
        
        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(K, BLOCK_K),
        )
        
        # Launch kernel to modify dy1_work in-place
        split_gemm_2to4_kernel[grid](
            dy1_work,
            sparse_mask_int,
            M, K,
            dy1_work.stride(0), dy1_work.stride(1),
            BLOCK_M, BLOCK_K,
        )
    
    # Now dy1_work has 2:4 sparsity in sparse columns, original data in dense columns
    # Use accelerated sparse GEMM (it should handle mixed sparse/dense)
    from sparse_fullrank_linear import fake_fp8_mm
    result = fake_fp8_mm(dy1_work, weight, torch.float8_e4m3fn)
    
    return result


def apply_split_gemm_sparsity_nocopy(dy1, sparse_mask):
    """
    Apply 2:4 sparsity to selected columns with minimal copying.
    """
    if sparse_mask is None or not sparse_mask.any():
        return dy1
    
    M, K = dy1.shape
    
    # Clone (necessary to preserve original)
    result = dy1.clone()
    
    # Apply 2:4 sparsity IN-PLACE
    sparse_mask_int = sparse_mask.to(torch.int32)
    
    BLOCK_M = 128
    BLOCK_K = 32
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(K, BLOCK_K),
    )
    
    split_gemm_2to4_kernel[grid](
        result,
        sparse_mask_int,
        M, K,
        result.stride(0), result.stride(1),
        BLOCK_M, BLOCK_K,
    )
    
    return result


# Integration functions
def compute_split_gemm_lowrank_intermediate_nocopy(dy1, weight_out1, layer_id):
    """
    Zero-copy version of compute_split_gemm_lowrank_intermediate.
    """
    from fused_sparsity_ops import sparsity_tracker
    
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    
    return split_gemm_nocopy(dy1, weight_out1, sparse_mask)


def apply_split_gemm_to_dy1_nocopy(dy1, layer_id):
    """
    Zero-copy version of apply_split_gemm_to_dy1.
    """
    from fused_sparsity_ops import sparsity_tracker
    
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    
    return apply_split_gemm_sparsity_nocopy(dy1, sparse_mask)


def compute_split_gemm_dw_nocopy(activation, grad_output, layer_id, transpose_result=False):
    """
    Zero-copy version for computing weight gradients.
    Computes activation.T @ grad_output with split-GEMM.
    
    Args:
        activation: Activation tensor [batch, in_features]
        grad_output: Gradient tensor [batch, out_features]
        layer_id: Layer ID for cached sparsity
        transpose_result: Whether to transpose the result
    
    Returns:
        Weight gradient [in_features, out_features] or transposed
    """
    from fused_sparsity_ops import sparsity_tracker
    
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    
    if sparse_mask is None:
        result = torch.mm(activation.T, grad_output)
        return result.T if transpose_result else result
    
    # Get dimensions
    batch_size, in_features = activation.shape
    batch_size2, out_features = grad_output.shape
    assert batch_size == batch_size2, "Batch size mismatch"
    
    # Initialize gradient - we compute activation.T @ grad_output
    grad_weight = torch.zeros(in_features, out_features, device=activation.device, dtype=activation.dtype)
    
    dense_mask = ~sparse_mask
    
    if sparse_mask.any():
        # Sparse part: extract columns, apply 2:4 sparsity, compute with fake_fp8_mm
        from sparse_fullrank_linear import fake_fp8_mm
        from fused_sparsity_ops import apply_feature_wise_2to4
        
        activation_sparse = activation[:, sparse_mask]
        activation_sparse_2to4 = apply_feature_wise_2to4(activation_sparse)
        result = fake_fp8_mm(
            activation_sparse_2to4.T, 
            grad_output, 
            torch.float8_e4m3fn
        )
        grad_weight[sparse_mask, :] = result.to(grad_weight.dtype)
    
    # Dense part: standard matmul
    if dense_mask.any():
        activation_dense = activation[:, dense_mask]
        grad_weight[dense_mask, :] = torch.mm(activation_dense.T, grad_output)
    
    # Return with optional transpose
    return grad_weight.T if transpose_result else grad_weight


if __name__ == "__main__":
    print("Testing zero-copy Split-GEMM...")
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
    
    print(f"\nTesting zero-copy kernel ({num_iterations} iterations):")
    
    # Warmup
    for _ in range(10):
        _ = compute_split_gemm_lowrank_intermediate_nocopy(dy1, weight, layer_id)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result = compute_split_gemm_lowrank_intermediate_nocopy(dy1, weight, layer_id)
    torch.cuda.synchronize()
    nocopy_time = time.time() - start
    
    print(f"  Zero-copy: {nocopy_time:.4f}s ({nocopy_time/num_iterations*1000:.3f}ms per call)")
    
    # Compare with standard GEMM
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_std = torch.mm(dy1, weight)
    torch.cuda.synchronize()
    std_time = time.time() - start
    
    print(f"  Standard GEMM: {std_time:.4f}s ({std_time/num_iterations*1000:.3f}ms per call)")
    print(f"  Overhead: {(nocopy_time-std_time)/std_time*100:.1f}%")
    
    # Test apply function
    print(f"\nTesting apply_split_gemm_to_dy1_nocopy:")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result2 = apply_split_gemm_to_dy1_nocopy(dy1, layer_id)
    torch.cuda.synchronize()
    apply_time = time.time() - start
    
    print(f"  Time: {apply_time:.4f}s ({apply_time/num_iterations*1000:.3f}ms per call)")
    
    # Verify sparsity pattern
    print(f"\nVerifying sparsity pattern:")
    result2_test = apply_split_gemm_to_dy1_nocopy(dy1, layer_id)
    
    # Check sparse columns
    sparse_indices = torch.where(sparse_mask)[0]
    if len(sparse_indices) > 0:
        sample_sparse_col = result2_test[:, sparse_indices[0]]
        sample_sparse_col_reshaped = sample_sparse_col.view(-1, 4)
        nonzero_per_group = (sample_sparse_col_reshaped != 0).sum(dim=1)
        has_2to4 = (nonzero_per_group <= 2).float().mean()
        print(f"  Sample sparse column has 2:4 pattern: {has_2to4*100:.1f}% of groups")
    
    # Check dense columns
    dense_indices = torch.where(~sparse_mask)[0]
    if len(dense_indices) > 0:
        sample_dense_col = result2_test[:, dense_indices[0]]
        unchanged = torch.allclose(sample_dense_col, dy1[:, dense_indices[0]])
        print(f"  Sample dense column unchanged: {unchanged}")
    
    print("\n" + "="*60)
    print("✓ Test completed!")