"""
Real fused Split-GEMM Triton kernel that avoids ALL data copies.
The key insight: process sparse and dense columns directly in the kernel without reorganizing data.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def split_gemm_kernel_fused(
    # Input matrix A
    a_ptr,
    # Weight matrix B  
    b_ptr,
    # Output matrix C
    c_ptr,
    # Sparse mask for columns of A / rows of B
    sparse_mask_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute C = A @ B where columns of A are selectively processed:
    - Sparse columns: apply 2:4 sparsity before multiplication
    - Dense columns: normal multiplication
    
    This is done WITHOUT any data reorganization.
    """
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Compute block indices
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop over K dimension
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # K offsets for this block
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Load sparse mask for this K block
        mask_k = offs_k < K
        sparse_flags = tl.load(sparse_mask_ptr + offs_k, mask=mask_k, other=0)
        
        # Load A block [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_block = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # Apply 2:4 sparsity to sparse columns IN-PLACE
        # Process each column of the block
        for col_idx in range(BLOCK_SIZE_K):
            if col_idx < BLOCK_SIZE_K:
                col_k = k_block * BLOCK_SIZE_K + col_idx
                if col_k < K:
                    # Check if this column should be sparsified
                    is_sparse = tl.load(sparse_mask_ptr + col_k)
                    
                    if is_sparse:
                        # Apply 2:4 sparsity to this column
                        # Process in groups of 4 rows
                        for row_start in range(0, BLOCK_SIZE_M, 4):
                            if row_start + 3 < BLOCK_SIZE_M:
                                # Get 4 values from the column
                                val0 = a_block[row_start, col_idx]
                                val1 = a_block[row_start + 1, col_idx]
                                val2 = a_block[row_start + 2, col_idx]
                                val3 = a_block[row_start + 3, col_idx]
                                
                                # Get absolute values
                                abs0 = tl.abs(val0)
                                abs1 = tl.abs(val1)
                                abs2 = tl.abs(val2)
                                abs3 = tl.abs(val3)
                                
                                # Find top 2 values (keep largest 2, zero out smallest 2)
                                # Simple sorting network for 4 elements
                                # First pass: compare pairs
                                if abs0 < abs1:
                                    abs0, abs1 = abs1, abs0
                                    val0, val1 = val1, val0
                                if abs2 < abs3:
                                    abs2, abs3 = abs3, abs2
                                    val2, val3 = val3, val2
                                
                                # Second pass: compare and swap
                                if abs0 < abs2:
                                    abs0, abs2 = abs2, abs0
                                    val0, val2 = val2, val0
                                if abs1 < abs3:
                                    abs1, abs3 = abs3, abs1
                                    val1, val3 = val3, val1
                                
                                # Third pass: final comparison
                                if abs1 < abs2:
                                    abs1, abs2 = abs2, abs1
                                    val1, val2 = val2, val1
                                
                                # Now val0 and val1 are top 2, zero out val2 and val3
                                a_block[row_start, col_idx] = val0
                                a_block[row_start + 1, col_idx] = val1
                                a_block[row_start + 2, col_idx] = 0.0
                                a_block[row_start + 3, col_idx] = 0.0
        
        # Load B block [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_block = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Matrix multiply and accumulate
        acc += tl.dot(a_block, b_block)
    
    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Convert to output dtype
    acc = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, acc, mask=mask_c)


@triton.jit
def apply_2to4_sparsity_kernel(
    # Input tensor
    input_ptr,
    # Output tensor
    output_ptr,
    # Sparse mask
    sparse_mask_ptr,
    # Dimensions
    M, N,
    # Strides
    stride_im, stride_in,
    stride_om, stride_on,
    # Block size
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Apply 2:4 sparsity to selected columns based on sparse_mask.
    This modifies the tensor in-place conceptually (output can be same as input).
    """
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Compute block indices
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Process each column in the block
    for col_idx in range(BLOCK_SIZE_N):
        n_idx = pid_n * BLOCK_SIZE_N + col_idx
        if n_idx < N:
            # Check if this column should be sparsified
            is_sparse = tl.load(sparse_mask_ptr + n_idx)
            
            # Load column data
            col_ptrs = input_ptr + offs_m * stride_im + n_idx * stride_in
            mask_col = offs_m < M
            col_data = tl.load(col_ptrs, mask=mask_col, other=0.0)
            
            if is_sparse:
                # Apply 2:4 sparsity
                # Process in groups of 4
                for i in range(0, BLOCK_SIZE_M, 4):
                    if i + 3 < BLOCK_SIZE_M and offs_m[i] + 3 < M:
                        # Get 4 values
                        vals = tl.zeros(4, dtype=col_data.dtype)
                        for j in range(4):
                            vals[j] = col_data[i + j]
                        
                        # Find top 2 by magnitude
                        abs_vals = tl.abs(vals)
                        
                        # Simple approach: find max and second max
                        max_idx = 0
                        max_val = abs_vals[0]
                        for j in range(1, 4):
                            if abs_vals[j] > max_val:
                                max_idx = j
                                max_val = abs_vals[j]
                        
                        # Find second max
                        second_idx = (max_idx + 1) % 4
                        second_val = abs_vals[second_idx]
                        for j in range(4):
                            if j != max_idx and abs_vals[j] > second_val:
                                second_idx = j
                                second_val = abs_vals[j]
                        
                        # Zero out bottom 2
                        for j in range(4):
                            if j != max_idx and j != second_idx:
                                col_data[i + j] = 0.0
            
            # Store result
            out_ptrs = output_ptr + offs_m * stride_om + n_idx * stride_on
            tl.store(out_ptrs, col_data, mask=mask_col)


def split_gemm_fused(dy1, weight, sparse_mask):
    """
    Fused Split-GEMM that processes sparse/dense columns without data copies.
    
    Args:
        dy1: Input tensor [M, K]
        weight: Weight tensor [K, N]
        sparse_mask: Boolean mask [K] indicating sparse columns
    
    Returns:
        Result [M, N]
    """
    M, K = dy1.shape
    K2, N = weight.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"
    
    # Allocate output
    output = torch.empty((M, N), device=dy1.device, dtype=dy1.dtype)
    
    # Convert boolean mask to int for Triton
    sparse_mask_int = sparse_mask.to(torch.int32)
    
    # Grid configuration
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    split_gemm_kernel_fused[grid](
        dy1, weight, output,
        sparse_mask_int,
        M, N, K,
        dy1.stride(0), dy1.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output


def apply_split_gemm_sparsity_fused(tensor, sparse_mask):
    """
    Apply 2:4 sparsity to selected columns without data copies.
    
    Args:
        tensor: Input tensor [M, N]
        sparse_mask: Boolean mask [N] indicating which columns to sparsify
    
    Returns:
        Sparsified tensor [M, N]
    """
    M, N = tensor.shape
    
    # Allocate output (could optimize to modify in-place)
    output = torch.empty_like(tensor)
    
    # Convert mask
    sparse_mask_int = sparse_mask.to(torch.int32)
    
    # Grid configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    apply_2to4_sparsity_kernel[grid](
        tensor, output, sparse_mask_int,
        M, N,
        tensor.stride(0), tensor.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output


# Direct replacements for the original functions
def compute_split_gemm_lowrank_intermediate_fused(dy1, weight_out1, layer_id):
    """
    Fused version of compute_split_gemm_lowrank_intermediate.
    """
    from fused_sparsity_ops import sparsity_tracker
    
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    
    if sparse_mask is None:
        return torch.mm(dy1, weight_out1)
    
    # Use fused kernel
    return split_gemm_fused(dy1, weight_out1, sparse_mask)


def apply_split_gemm_to_dy1_fused(dy1, layer_id):
    """
    Fused version of apply_split_gemm_to_dy1.
    """
    from fused_sparsity_ops import sparsity_tracker
    
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    
    if sparse_mask is None:
        return dy1
    
    # Use fused kernel
    return apply_split_gemm_sparsity_fused(dy1, sparse_mask)


if __name__ == "__main__":
    print("Testing real fused Split-GEMM kernel...")
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
    
    print(f"\nTesting fused kernel ({num_iterations} iterations):")
    
    # Warmup
    for _ in range(10):
        _ = compute_split_gemm_lowrank_intermediate_fused(dy1, weight, layer_id)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result = compute_split_gemm_lowrank_intermediate_fused(dy1, weight, layer_id)
    torch.cuda.synchronize()
    fused_time = time.time() - start
    
    print(f"  Fused kernel: {fused_time:.4f}s ({fused_time/num_iterations*1000:.3f}ms per call)")
    
    # Compare with standard GEMM
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_std = torch.mm(dy1, weight)
    torch.cuda.synchronize()
    std_time = time.time() - start
    
    print(f"  Standard GEMM: {std_time:.4f}s ({std_time/num_iterations*1000:.3f}ms per call)")
    print(f"  Overhead: {(fused_time-std_time)/std_time*100:.1f}%")
    
    # Test apply function
    print(f"\nTesting apply_split_gemm_to_dy1_fused:")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result2 = apply_split_gemm_to_dy1_fused(dy1, layer_id)
    torch.cuda.synchronize()
    apply_time = time.time() - start
    
    print(f"  Time: {apply_time:.4f}s ({apply_time/num_iterations*1000:.3f}ms per call)")
    
    print("\n" + "="*60)
    print("✓ Test completed!")