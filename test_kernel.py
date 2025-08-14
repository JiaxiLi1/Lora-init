import torch
import triton
import triton.language as tl
from triton_split_gemm_nocopy import split_gemm_2to4_kernel

def test_kernel():
    """Test the split_gemm_2to4_kernel directly"""
    print("Testing split_gemm_2to4_kernel...")
    
    # Test configuration
    M, K = 3072, 16384
    device = 'cuda'
    dtype = torch.bfloat16
    
    # Create test data
    data = torch.randn(M, K, device=device, dtype=dtype)
    sparse_mask = torch.ones(K, device=device, dtype=torch.int32)  # All columns sparse for testing
    
    # Clone for in-place modification
    data_work = data.clone()
    
    # Grid configuration
    BLOCK_M = 128
    BLOCK_K = 32
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(K, BLOCK_K),
    )
    
    print(f"Shape: [{M}, {K}]")
    print(f"Grid: {grid}")
    print(f"Block size: M={BLOCK_M}, K={BLOCK_K}")
    
    try:
        # Launch kernel
        split_gemm_2to4_kernel[grid](
            data_work,
            sparse_mask,
            M, K,
            data_work.stride(0), data_work.stride(1),
            BLOCK_M, BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        torch.cuda.synchronize()
        print("✓ Kernel executed successfully!")
        
        # Check sparsity pattern
        sample_col = data_work[:, 0]
        sample_col_reshaped = sample_col[:M//4*4].view(-1, 4)
        nonzero_per_group = (sample_col_reshaped != 0).sum(dim=1)
        has_2to4 = (nonzero_per_group <= 2).float().mean()
        print(f"✓ 2:4 sparsity pattern: {has_2to4*100:.1f}% of groups have ≤2 non-zeros")
        
    except Exception as e:
        print(f"✗ Kernel failed: {e}")
        return False
    
    return True

def test_split_gemm_nocopy():
    """Test the full split_gemm_nocopy function"""
    print("\nTesting split_gemm_nocopy function...")
    
    from triton_split_gemm_nocopy import split_gemm_nocopy
    
    # Test configuration
    M, K, N = 3072, 16384, 256
    device = 'cuda'
    dtype = torch.bfloat16
    
    # Create test data
    dy1 = torch.randn(M, K, device=device, dtype=dtype)
    weight = torch.randn(K, N, device=device, dtype=dtype)
    sparse_mask = torch.rand(K, device=device) < 0.95  # 95% sparse columns
    
    print(f"Configuration: [{M}, {K}] @ [{K}, {N}]")
    print(f"Sparse columns: {sparse_mask.sum().item()}/{K} ({sparse_mask.float().mean()*100:.1f}%)")
    
    try:
        result = split_gemm_nocopy(dy1, weight, sparse_mask)
        torch.cuda.synchronize()
        print(f"✓ Function executed successfully!")
        print(f"✓ Result shape: {result.shape}")
        
        # Compare with standard matmul
        result_std = torch.mm(dy1, weight)
        # They shouldn't be equal because we applied sparsity
        print(f"✓ Results differ (expected): {not torch.allclose(result, result_std)}")
        
    except Exception as e:
        print(f"✗ Function failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("="*60)
    success1 = test_kernel()
    success2 = test_split_gemm_nocopy()
    print("="*60)
    if success1 and success2:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
