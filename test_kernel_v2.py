import torch
import triton
from triton_split_gemm_nocopy import split_gemm_2to4_kernel

def test_kernel():
    # Test configuration
    M, K = 512, 768
    device = 'cuda'
    dtype = torch.float32  # Use float32 first for easier debugging
    
    # Create test data with some large values
    dy1 = torch.randn(M, K, device=device, dtype=dtype) * 10.0
    dy1_original = dy1.clone()
    
    # Create sparse mask - only first few columns are sparse for testing
    sparse_mask = torch.zeros(K, device=device, dtype=torch.bool)
    sparse_mask[:10] = True  # Only first 10 columns are sparse
    sparse_mask_int = sparse_mask.to(torch.int32)
    
    print(f"Testing kernel with shape [{M}, {K}]")
    print(f"Sparse columns: {sparse_mask.sum().item()}/{K}")
    
    # Grid configuration
    BLOCK_M = 128
    BLOCK_K = 32
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(K, BLOCK_K),
    )
    
    print(f"Grid: {grid}")
    print(f"Blocks: BLOCK_M={BLOCK_M}, BLOCK_K={BLOCK_K}")
    
    # Check some values before kernel
    print("\nBefore kernel:")
    for k in range(min(5, K)):
        if sparse_mask[k]:
            sample_vals = dy1_original[:4, k]
            print(f"Column {k} first 4 values: {sample_vals.cpu().numpy()}")
    
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
        print("\n✓ Kernel executed successfully!")
        
        # Check result
        print("\nAfter kernel:")
        for k in range(min(5, K)):
            if sparse_mask[k]:
                # Check first group of 4
                original_vals = dy1_original[:4, k]
                result_vals = dy1[:4, k]
                print(f"\nColumn {k}:")
                print(f"  Original: {original_vals.cpu().numpy()}")
                print(f"  Result:   {result_vals.cpu().numpy()}")
                print(f"  Non-zeros: {(result_vals != 0).sum().item()}/4")
                
                # Check overall column sparsity
                col = dy1[:, k].view(-1, 4)
                nonzero_per_group = (col != 0).sum(dim=1)
                print(f"  Avg nonzeros per group: {nonzero_per_group.float().mean():.2f}")
        
        # Check dense columns are unchanged
        print("\nDense column check:")
        for k in range(K):
            if not sparse_mask[k]:
                if not torch.allclose(dy1[:, k], dy1_original[:, k]):
                    print(f"✗ Dense column {k} was modified!")
                    break
        else:
            print("✓ All dense columns unchanged")
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kernel()