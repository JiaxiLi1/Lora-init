import torch
from triton_split_gemm_nocopy import compute_split_gemm_lowrank_intermediate_nocopy
from fused_sparsity_ops import sparsity_tracker

def test_full_function():
    """Test the full compute_split_gemm_lowrank_intermediate_nocopy function"""
    
    # Test configuration
    M, K, N = 512, 768, 256
    device = 'cuda'
    dtype = torch.bfloat16
    
    # Create test data
    dy1 = torch.randn(M, K, device=device, dtype=dtype)
    weight_out1 = torch.randn(K, N, device=device, dtype=dtype)
    
    # Test case 1: No cached sparsity (should use default)
    print("Test 1: No cached sparsity")
    layer_id = "test_layer_nocache"
    
    try:
        result = compute_split_gemm_lowrank_intermediate_nocopy(dy1, weight_out1, layer_id)
        print(f"✓ Success with no cached sparsity. Result shape: {result.shape}")
    except Exception as e:
        print(f"✗ Error with no cached sparsity: {e}")
        import traceback
        traceback.print_exc()
    
    # Test case 2: With cached sparsity
    print("\nTest 2: With cached sparsity")
    layer_id = "test_layer_cached"
    
    # Create and store sparsity
    col_sparsity = torch.rand(K, device=device)
    sparse_mask = torch.rand(K, device=device) < 0.95
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
    
    try:
        result = compute_split_gemm_lowrank_intermediate_nocopy(dy1, weight_out1, layer_id)
        print(f"✓ Success with cached sparsity. Result shape: {result.shape}")
    except Exception as e:
        print(f"✗ Error with cached sparsity: {e}")
        import traceback
        traceback.print_exc()
    
    # Test case 3: Empty sparse mask
    print("\nTest 3: Empty sparse mask")
    layer_id = "test_layer_empty"
    sparse_mask_empty = torch.zeros(K, device=device, dtype=torch.bool)
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask_empty)
    
    try:
        result = compute_split_gemm_lowrank_intermediate_nocopy(dy1, weight_out1, layer_id)
        print(f"✓ Success with empty sparse mask. Result shape: {result.shape}")
    except Exception as e:
        print(f"✗ Error with empty sparse mask: {e}")
        import traceback
        traceback.print_exc()
    
    # Test case 4: All sparse mask
    print("\nTest 4: All sparse mask")
    layer_id = "test_layer_all"
    sparse_mask_all = torch.ones(K, device=device, dtype=torch.bool)
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask_all)
    
    try:
        result = compute_split_gemm_lowrank_intermediate_nocopy(dy1, weight_out1, layer_id)
        print(f"✓ Success with all sparse mask. Result shape: {result.shape}")
    except Exception as e:
        print(f"✗ Error with all sparse mask: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_function()