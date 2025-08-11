"""
Triton kernel for fused GEMM with column-level sparsity computation in epilogue.
This kernel computes Y = X @ W and column sparsity stats in a single pass.
"""

import torch
import triton
import triton.language as tl
import numpy as np


@triton.jit
def gemm_with_sparsity_kernel(
    # Input matrices
    x_ptr, w_ptr, y_ptr,
    # Sparsity output buffers
    col_nonzero_ptr, col_total_ptr,
    # Dimensions
    M, N, K,
    # Strides for X
    stride_xm, stride_xk,
    # Strides for W
    stride_wk, stride_wn,
    # Strides for Y
    stride_ym, stride_yn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # ReLU² activation flag
    ACTIVATION_RELU2: tl.constexpr,
):
    """
    Compute Y = X @ W with optional ReLU² activation and column sparsity stats.
    
    This kernel fuses the GEMM operation with sparsity computation in the epilogue,
    avoiding the need for a separate pass to compute sparsity.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Compute block indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Offsets for the current block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to the first block of X and W
    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Perform matrix multiplication
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load current block
        x_block = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w_block = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Compute
        accumulator += tl.dot(x_block, w_block)
        
        # Advance pointers
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
    
    # Apply activation if needed
    if ACTIVATION_RELU2:
        # ReLU²: max(0, x)²
        accumulator = tl.where(accumulator > 0, accumulator * accumulator, 0.0)
    
    # Keep as float32 for now, convert outside kernel
    y_block = accumulator
    
    # Store output
    offs_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
    mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, y_block, mask=mask)
    
    # Compute column sparsity stats in epilogue
    # Count non-zero elements per column
    nonzero_mask = (y_block != 0.0)
    col_nonzero = tl.sum(nonzero_mask.to(tl.int32), axis=0)
    col_total = tl.sum(mask.to(tl.int32), axis=0)
    
    # Atomic add to global column stats
    col_nonzero_ptrs = col_nonzero_ptr + offs_yn
    col_total_ptrs = col_total_ptr + offs_yn
    
    tl.atomic_add(col_nonzero_ptrs, col_nonzero, mask=offs_yn < N)
    tl.atomic_add(col_total_ptrs, col_total, mask=offs_yn < N)


@triton.jit
def gemm_with_sparsity_and_2to4_kernel(
    # Input matrices
    x_ptr, w_ptr, y_ptr, y_sparse_ptr,
    # Sparsity output buffers
    col_sparsity_ptr,
    # Dimensions
    M, N, K,
    # Strides for X
    stride_xm, stride_xk,
    # Strides for W
    stride_wk, stride_wn,
    # Strides for Y
    stride_ym, stride_yn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # ReLU² activation flag
    ACTIVATION_RELU2: tl.constexpr,
    # Apply 2:4 sparsity
    APPLY_2TO4: tl.constexpr,
):
    """
    Enhanced kernel that computes Y = X @ W with:
    1. Optional ReLU² activation
    2. Column sparsity computation
    3. Optional 2:4 sparsification in epilogue
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Compute block indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Offsets for the current block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to the first block of X and W
    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Perform matrix multiplication
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load current block
        x_block = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w_block = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Compute
        accumulator += tl.dot(x_block, w_block)
        
        # Advance pointers
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
    
    # Apply activation if needed
    if ACTIVATION_RELU2:
        # ReLU²: max(0, x)²
        accumulator = tl.where(accumulator > 0, accumulator * accumulator, 0.0)
    
    # Keep as float32 for now, convert outside kernel
    y_block = accumulator
    
    # Store dense output
    offs_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
    mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, y_block, mask=mask)
    
    # Apply 2:4 sparsity if requested
    y_sparse_block = y_block
    if APPLY_2TO4:
        # For each group of 4 elements in each row, keep top 2
        # This is simplified - actual 2:4 needs more complex logic
        # For demonstration, we'll use a simpler approach
        abs_vals = tl.abs(y_block)
        
        # This is a simplified version - actual implementation would need
        # proper 2:4 pattern enforcement
        threshold = tl.max(abs_vals) * 0.5  # Simple threshold for demo
        y_sparse_block = tl.where(abs_vals >= threshold, y_block, 0.0)
    
    # Store sparse output if different from dense
    if APPLY_2TO4:
        y_sparse_ptrs = y_sparse_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
        tl.store(y_sparse_ptrs, y_sparse_block, mask=mask)
    
    # Compute column sparsity in epilogue
    nonzero_count = tl.sum((y_sparse_block != 0.0).to(tl.float32), axis=0)
    total_count = tl.sum(mask.to(tl.float32), axis=0)
    col_sparsity = 1.0 - (nonzero_count / (total_count + 1e-8))
    
    # Store column sparsity
    col_sparsity_ptrs = col_sparsity_ptr + offs_yn
    tl.store(col_sparsity_ptrs, col_sparsity, mask=offs_yn < N)


class FusedGEMMWithSparsity(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for the fused GEMM with sparsity kernel.
    """
    
    @staticmethod
    def forward(ctx, x, w, activation_relu2=False, compute_2to4=False):
        """
        Forward pass: Y = X @ W with sparsity computation in epilogue.
        
        Args:
            x: Input tensor [M, K]
            w: Weight tensor [K, N]
            activation_relu2: Apply ReLU² activation
            compute_2to4: Also compute 2:4 sparse version
        
        Returns:
            y: Output tensor [M, N]
            col_sparsity: Column sparsity stats [N]
            y_sparse: Optional 2:4 sparse output
        """
        M, K = x.shape
        K_w, N = w.shape
        assert K == K_w, f"Dimension mismatch: {K} != {K_w}"
        
        # Allocate output tensors
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        col_sparsity = torch.zeros(N, device=x.device, dtype=torch.float32)
        
        # Configure block sizes
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 64
        
        # Grid configuration
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        if compute_2to4:
            y_sparse = torch.empty_like(y)
            
            # Use enhanced kernel with 2:4 support
            gemm_with_sparsity_and_2to4_kernel[grid](
                x, w, y, y_sparse,
                col_sparsity,
                M, N, K,
                x.stride(0), x.stride(1),
                w.stride(0), w.stride(1),
                y.stride(0), y.stride(1),
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                ACTIVATION_RELU2=activation_relu2,
                APPLY_2TO4=compute_2to4,
            )
            
            # Save for backward
            ctx.save_for_backward(x, w, y, y_sparse)
            ctx.activation_relu2 = activation_relu2
            return y, col_sparsity, y_sparse
        else:
            # Use basic kernel
            col_nonzero = torch.zeros(N, device=x.device, dtype=torch.int32)
            col_total = torch.zeros(N, device=x.device, dtype=torch.int32)
            
            gemm_with_sparsity_kernel[grid](
                x, w, y,
                col_nonzero, col_total,
                M, N, K,
                x.stride(0), x.stride(1),
                w.stride(0), w.stride(1),
                y.stride(0), y.stride(1),
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                ACTIVATION_RELU2=activation_relu2,
            )
            
            # Compute sparsity ratio
            col_sparsity = 1.0 - (col_nonzero.float() / (col_total.float() + 1e-8))
            
            # Save for backward
            ctx.save_for_backward(x, w, y)
            ctx.activation_relu2 = activation_relu2
            return y, col_sparsity, None
    
    @staticmethod
    def backward(ctx, grad_output, grad_col_sparsity, grad_y_sparse):
        """
        Backward pass implementation.
        """
        if len(ctx.saved_tensors) == 4:
            x, w, y, y_sparse = ctx.saved_tensors
        else:
            x, w, y = ctx.saved_tensors
            y_sparse = None
        
        grad_x = grad_w = None
        
        # Gradient through ReLU² if needed
        if ctx.activation_relu2:
            # d/dx[ReLU²(x)] = 2*ReLU(x)
            grad_output = grad_output * 2 * torch.where(y > 0, torch.sqrt(y), torch.zeros_like(y))
        
        # Standard GEMM gradients
        if ctx.needs_input_grad[0]:
            grad_x = grad_output @ w.T
        if ctx.needs_input_grad[1]:
            grad_w = x.T @ grad_output
        
        return grad_x, grad_w, None, None


def gemm_with_sparsity(x, w, activation_relu2=False, compute_2to4=False):
    """
    User-friendly wrapper for the fused GEMM with sparsity computation.
    
    This function performs Y = X @ W and computes column sparsity statistics
    in a single fused kernel pass, avoiding the overhead of separate sparsity
    computation.
    
    Args:
        x: Input tensor [batch_size, in_features]
        w: Weight tensor [in_features, out_features]
        activation_relu2: Whether to apply ReLU² activation
        compute_2to4: Whether to also compute 2:4 sparse version
    
    Returns:
        output: Dense output tensor
        col_sparsity: Column-wise sparsity ratios
        sparse_output: Optional 2:4 sparse output (if compute_2to4=True)
    """
    return FusedGEMMWithSparsity.apply(x, w, activation_relu2, compute_2to4)


# Integration function for existing codebase
def compute_split_gemm_with_fused_sparsity(x, w, forward_mask=None, threshold=0.95):
    """
    Replacement for existing split_gemm that uses fused sparsity computation.
    
    This function computes the GEMM and determines sparse/dense split in one pass,
    eliminating the separate sparsity computation overhead.
    """
    # Compute GEMM with sparsity stats in single pass
    y, col_sparsity, _ = gemm_with_sparsity(x, w, activation_relu2=False)
    
    # Determine sparse columns based on sparsity threshold
    num_features = col_sparsity.shape[0]
    num_sparse = int(threshold * num_features)
    
    # Get indices of most sparse columns
    sparse_indices = torch.topk(col_sparsity, num_sparse).indices
    sparse_mask = torch.zeros(num_features, dtype=torch.bool, device=x.device)
    sparse_mask[sparse_indices] = True
    
    return y, sparse_mask, col_sparsity


# Example usage and testing
if __name__ == "__main__":
    # Test the fused kernel
    batch_size = 512
    in_features = 4096
    out_features = 11008
    
    # Create test tensors
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float16)
    w = torch.randn(in_features, out_features, device='cuda', dtype=torch.float16)
    
    # Test basic GEMM with sparsity
    print("Testing basic GEMM with sparsity computation...")
    y, col_sparsity, _ = gemm_with_sparsity(x, w, activation_relu2=False)
    print(f"Output shape: {y.shape}")
    print(f"Column sparsity shape: {col_sparsity.shape}")
    print(f"Average sparsity: {col_sparsity.mean().item():.4f}")
    
    # Test with ReLU² activation
    print("\nTesting with ReLU² activation...")
    y_relu2, col_sparsity_relu2, _ = gemm_with_sparsity(x, w, activation_relu2=True)
    print(f"Average sparsity with ReLU²: {col_sparsity_relu2.mean().item():.4f}")
    
    # Test with 2:4 sparsification
    print("\nTesting with 2:4 sparsification...")
    y_dense, col_sparsity, y_sparse = gemm_with_sparsity(x, w, activation_relu2=True, compute_2to4=True)
    actual_sparsity = (y_sparse == 0).float().mean()
    print(f"Actual 2:4 sparsity: {actual_sparsity.item():.4f}")
    
    # Verify correctness against standard PyTorch
    print("\nVerifying correctness...")
    y_ref = x @ w
    error = (y - y_ref).abs().max()
    print(f"Max error vs PyTorch: {error.item():.6f}")
    
    print("\nFused GEMM with sparsity computation test completed!")