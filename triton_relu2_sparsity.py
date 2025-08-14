"""
Triton kernel for ReLU² with column sparsity computation.
Computes y = x * x (where x > 0) and tracks sparsity in epilogue.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def relu2_with_sparsity_kernel(
    # Pointers
    x_ptr, y_ptr,
    # Shape
    M, N,
    # Strides
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    # Sparsity tracking
    col_nnz_ptr,  # Count of non-zeros per column
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel for computing ReLU²(x) with column sparsity tracking."""
    
    # Block indices
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Create block pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask for valid elements
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Load input
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Compute ReLU²
    relu_mask = x > 0
    y = tl.where(relu_mask, x * x, 0.0)
    
    # Store output
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask)
    
    # Count non-zeros per column and update atomically
    if col_nnz_ptr:
        nnz = (y != 0).to(tl.int32)
        col_nnz = tl.sum(nnz, axis=0)  # Sum over rows for each column
        col_nnz_ptrs = col_nnz_ptr + offs_n
        tl.atomic_add(col_nnz_ptrs, col_nnz, mask=offs_n < N)


def relu2_with_sparsity(x):
    """
    Compute ReLU²(x) with column sparsity tracking.
    
    Args:
        x: Input tensor [M, N]
    
    Returns:
        y: Output tensor [M, N] where y = x² if x > 0, else 0
        col_sparsity: Sparsity ratio per column [N]
    """
    M, N = x.shape
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Allocate sparsity counter
    col_nnz = torch.zeros(N, device=x.device, dtype=torch.int32)
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    
    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    relu2_with_sparsity_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        col_nnz,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Compute sparsity ratios
    col_sparsity = 1.0 - (col_nnz.float() / M)
    
    return y, col_sparsity


# Test the implementation
if __name__ == "__main__":
    print("Testing ReLU² with Sparsity Tracking")
    print("=" * 60)
    
    # Test
    M, N = 4096, 3072
    x = torch.randn(M, N, device='cuda', dtype=torch.float16)
    
    # Reference implementation
    relu_x = torch.relu(x)
    y_ref = relu_x * relu_x
    
    # Triton implementation
    y_triton, col_sparsity = relu2_with_sparsity(x)
    
    # Check correctness
    error = (y_triton - y_ref).abs().max().item()
    print(f"Max error: {error:.6f}")
    assert error < 0.01, f"Error too large: {error}"
    
    # Check sparsity computation
    ref_sparsity = (y_ref == 0).float().mean(dim=0)
    sparsity_error = (col_sparsity - ref_sparsity).abs().max().item()
    print(f"Sparsity error: {sparsity_error:.6f}")
    assert sparsity_error < 0.01, f"Sparsity error too large: {sparsity_error}"
    
    print("✓ ReLU² with sparsity tracking passed")
    print(f"Average sparsity: {col_sparsity.mean().item():.2%}")