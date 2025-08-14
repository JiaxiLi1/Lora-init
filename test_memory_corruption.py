import torch
import triton
from triton_split_gemm_nocopy import split_gemm_2to4_kernel

def test_memory_corruption():
    """Test if kernel corrupts memory"""
    M, K = 512, 768
    device = 'cuda'
    dtype = torch.bfloat16
    
    # Create test data
    dy1 = torch.randn(M, K, device=device, dtype=dtype)
    
    # Create sparse mask
    sparse_mask = torch.rand(K, device=device) < 0.95
    sparse_mask_int = sparse_mask.to(torch.int32)
    
    print(f"Testing with shape [{M}, {K}]")
    print(f"Sparse columns: {sparse_mask.sum().item()}/{K}")
    
    # Test 1: Check if we can access sparse_mask before kernel
    try:
        print(f"Before kernel - sparse_mask.any(): {sparse_mask.any()}")
        print(f"Before kernel - sparse_mask.numel(): {sparse_mask.numel()}")
    except Exception as e:
        print(f"ERROR before kernel: {e}")
        return
    
    # Grid configuration
    BLOCK_M = 128
    BLOCK_K = 32
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(K, BLOCK_K),
    )
    
    print(f"Launching kernel with grid {grid}")
    
    # Launch kernel
    try:
        split_gemm_2to4_kernel[grid](
            dy1,
            sparse_mask_int,
            M, K,
            dy1.stride(0), dy1.stride(1),
            BLOCK_M, BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        torch.cuda.synchronize()
        print("Kernel executed successfully")
    except Exception as e:
        print(f"ERROR during kernel: {e}")
        return
    
    # Test 2: Check if we can still access sparse_mask after kernel
    try:
        print(f"After kernel - sparse_mask.any(): {sparse_mask.any()}")
        print(f"After kernel - sparse_mask.numel(): {sparse_mask.numel()}")
        print("✓ No memory corruption detected")
    except Exception as e:
        print(f"ERROR after kernel: {e}")
        print("✗ Memory corruption detected!")
        
    # Test 3: Check dy1
    try:
        print(f"dy1 shape: {dy1.shape}")
        print(f"dy1 min/max: {dy1.min()}, {dy1.max()}")
        print("✓ dy1 accessible")
    except Exception as e:
        print(f"ERROR accessing dy1: {e}")

if __name__ == "__main__":
    test_memory_corruption()