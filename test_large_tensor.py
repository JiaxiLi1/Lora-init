import torch
import triton
from triton_split_gemm_nocopy import split_gemm_2to4_kernel

def test_large_tensor():
    """Test kernel with large tensors to find memory access issues"""
    
    # Test different sizes
    test_cases = [
        (512, 768),
        (1024, 1536),
        (2048, 3072),
        (4096, 3072),
        (8192, 3072),
        (16384, 3072),  # This is the actual size in training
    ]
    
    device = 'cuda'
    dtype = torch.bfloat16
    
    for M, K in test_cases:
        print(f"\nTesting size [{M}, {K}]...")
        
        # Create test data
        dy1 = torch.randn(M, K, device=device, dtype=dtype)
        sparse_mask = torch.ones(K, device=device, dtype=torch.bool)  # All sparse for testing
        sparse_mask_int = sparse_mask.to(torch.int32)
        
        # Grid configuration
        BLOCK_M = 128
        BLOCK_K = 32
        
        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(K, BLOCK_K),
        )
        
        print(f"  Grid: {grid}, Blocks: M={BLOCK_M}, K={BLOCK_K}")
        
        try:
            # Launch kernel
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
            print(f"  ✓ Success!")
            
            # Check a sample column
            col_sample = dy1[:min(16, M), 0].view(-1, 4)
            nonzero_per_group = (col_sample != 0).sum(dim=1)
            print(f"  Sample column sparsity: {nonzero_per_group.float().mean():.2f} non-zeros per group")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            # Try to identify the issue
            print(f"  M % BLOCK_M = {M % BLOCK_M}")
            print(f"  K % BLOCK_K = {K % BLOCK_K}")
            print(f"  M % 4 = {M % 4}")
            
            # Check memory usage
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    test_large_tensor()