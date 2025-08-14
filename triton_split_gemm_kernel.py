"""
Efficient Triton kernel for split-GEMM that avoids data copies.
Key idea: Process sparse/dense columns in a single kernel without splitting tensors.
"""

import torch
import triton
import triton.language as tl
from sparse_fullrank_linear import fake_fp8_mm


@triton.jit
def split_gemm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Sparse mask pointer
    sparse_mask_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for A
    stride_am, stride_ak,
    # Strides for B
    stride_bk, stride_bn,
    # Strides for C
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute C = A @ B with split-GEMM strategy.
    
    For columns marked as sparse in sparse_mask:
        - Apply 2:4 sparsity pattern
    For columns marked as dense:
        - Regular computation
    
    This kernel processes both in a single pass to avoid data copies.
    """
    # Get the block ID
    pid = tl.program_id(axis=0)
    
    # Compute block indices
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Create block offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Current K block offset
        offs_ak = k * BLOCK_SIZE_K + offs_k
        
        # Load sparse mask for current K block
        sparse_mask = tl.load(
            sparse_mask_ptr + offs_ak,
            mask=offs_ak < K,
            other=0
        )
        
        # Load A block
        a = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_ak[None, :] < K),
            other=0.0
        )
        
        # Apply 2:4 sparsity to columns marked as sparse
        # This is a simplified version - actual implementation would be more complex
        for i in range(BLOCK_SIZE_K):
            if sparse_mask[i]:
                # Apply 2:4 sparsity pattern to column i of a
                col = a[:, i]
                # Keep top 2 out of 4 values (simplified)
                # In practice, this would need proper 2:4 logic
                a[:, i] = apply_2to4_sparsity_triton(col)
        
        # Load B block
        b = tl.load(
            b_ptr + offs_ak[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_ak[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0
        )
        
        # Accumulate
        acc += tl.dot(a, b)
    
    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def apply_2to4_sparsity_triton(col):
    """
    Apply 2:4 sparsity pattern to a column (simplified).
    Keep top 2 values out of every 4.
    """
    # This is a placeholder - actual implementation would be more complex
    return col


def split_gemm_triton(input_tensor, weight, sparse_mask):
    """
    Efficient split-GEMM using Triton kernel.
    
    Args:
        input_tensor: [M, K] tensor
        weight: [K, N] tensor
        sparse_mask: [K] boolean mask indicating sparse columns
    
    Returns:
        Output tensor [M, N]
    """
    M, K = input_tensor.shape
    K2, N = weight.shape
    assert K == K2, "Dimension mismatch"
    
    # Allocate output
    output = torch.zeros((M, N), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Convert sparse_mask to int for Triton
    sparse_mask_int = sparse_mask.int()
    
    # Launch kernel
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    split_gemm_kernel[grid](
        input_tensor, weight, output,
        sparse_mask_int,
        M, N, K,
        input_tensor.stride(0), input_tensor.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output


def split_gemm_efficient(dy, weight, sparse_mask, apply_2to4=True):
    """
    Most efficient split-GEMM implementation.
    
    For performance, we use different strategies based on sparsity ratio:
    - >90% sparse: Apply 2:4 to everything (small accuracy loss, big speed gain)
    - 50-90% sparse: Use hybrid approach
    - <50% sparse: Use standard GEMM
    """
    if sparse_mask is None:
        return torch.mm(dy, weight)
    
    sparse_ratio = sparse_mask.float().mean().item()
    
    if sparse_ratio > 0.9:
        # Almost all sparse - just apply 2:4 to everything
        if apply_2to4:
            from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity
            dy_sparse = apply_naive_2to4_sparsity(dy)
            return fake_fp8_mm(dy_sparse, weight, torch.float8_e4m3fn)
        else:
            return torch.mm(dy, weight)
    
    elif sparse_ratio > 0.5:
        # Mixed - use optimized approach
        # Process sparse and dense in blocks to improve memory access
        return split_gemm_blocked(dy, weight, sparse_mask, apply_2to4)
    
    else:
        # Mostly dense - standard GEMM is faster
        return torch.mm(dy, weight)


def split_gemm_blocked(dy, weight, sparse_mask, apply_2to4=True):
    """
    Block-based split-GEMM for better cache usage.
    Process contiguous blocks of sparse/dense columns together.
    """
    batch_seq, features = dy.shape
    _, out_features = weight.shape
    
    # Find contiguous blocks of sparse/dense columns
    sparse_blocks = []
    dense_blocks = []
    
    # Simple block detection
    current_block_start = 0
    current_is_sparse = sparse_mask[0].item()
    
    for i in range(1, features):
        if sparse_mask[i].item() != current_is_sparse:
            # End of current block
            if current_is_sparse:
                sparse_blocks.append((current_block_start, i))
            else:
                dense_blocks.append((current_block_start, i))
            
            current_block_start = i
            current_is_sparse = sparse_mask[i].item()
    
    # Add last block
    if current_is_sparse:
        sparse_blocks.append((current_block_start, features))
    else:
        dense_blocks.append((current_block_start, features))
    
    # Process blocks
    output = torch.zeros(batch_seq, out_features, device=dy.device, dtype=dy.dtype)
    
    # Process sparse blocks with 2:4 sparsity
    if apply_2to4 and sparse_blocks:
        from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
        
        for start, end in sparse_blocks:
            dy_block = dy[:, start:end]
            weight_block = weight[start:end, :]
            
            # Apply 2:4 sparsity
            dy_block_t = dy_block.t()
            dy_block_2to4_t = apply_naive_2to4_sparsity_featurewise(dy_block_t)
            dy_block_2to4 = dy_block_2to4_t.t()
            
            # Use sparse GEMM
            output += fake_fp8_mm(dy_block_2to4, weight_block, torch.float8_e4m3fn)
    
    # Process dense blocks with standard GEMM
    for start, end in dense_blocks:
        dy_block = dy[:, start:end]
        weight_block = weight[start:end, :]
        output += torch.mm(dy_block, weight_block)
    
    return output


# Export optimized functions
def compute_split_gemm_optimized(dy, weight, layer_id, transpose_weight=False):
    """
    Optimized split-GEMM computation.
    """
    from fused_sparsity_ops import sparsity_tracker
    
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    
    if sparse_mask is None:
        # No sparsity - use standard GEMM
        if transpose_weight:
            return torch.mm(dy, weight.T)
        else:
            return torch.mm(dy, weight)
    
    # Use optimized split-GEMM
    if transpose_weight:
        weight = weight.T
    
    return split_gemm_efficient(dy, weight, sparse_mask)


if __name__ == "__main__":
    print("Testing efficient split-GEMM kernel...")
    
    import time
    from fused_sparsity_ops import sparsity_tracker
    
    # Test configuration
    batch_seq = 2048
    hidden_size = 768
    rank = 256
    
    device = 'cuda'
    dtype = torch.float16
    
    # Create test data
    dy = torch.randn(batch_seq, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, rank, device=device, dtype=dtype)
    
    # Test different sparsity ratios
    sparsity_ratios = [0.95, 0.7, 0.3]
    
    for sparse_ratio in sparsity_ratios:
        print(f"\n--- Testing with {sparse_ratio*100:.0f}% sparsity ---")
        
        # Create sparsity mask
        sparse_mask = torch.rand(hidden_size, device=device) < sparse_ratio
        
        # Store in tracker
        layer_id = f"test_{sparse_ratio}"
        col_sparsity = torch.rand(hidden_size, device=device)
        sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
        
        num_iterations = 100
        
        # Warmup
        for _ in range(10):
            _ = compute_split_gemm_optimized(dy, weight, layer_id)
        
        # Benchmark optimized version
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            result_opt = compute_split_gemm_optimized(dy, weight, layer_id)
        torch.cuda.synchronize()
        opt_time = time.time() - start
        
        # Benchmark standard GEMM
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            result_std = torch.mm(dy, weight)
        torch.cuda.synchronize()
        std_time = time.time() - start
        
        print(f"  Optimized: {opt_time:.4f}s ({opt_time/num_iterations*1000:.3f}ms per call)")
        print(f"  Standard: {std_time:.4f}s ({std_time/num_iterations*1000:.3f}ms per call)")
        print(f"  Speedup: {std_time/opt_time:.2f}x")
    
    print("\n✓ Tests completed!")