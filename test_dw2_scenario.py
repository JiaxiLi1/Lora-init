import torch
from triton_split_gemm_nocopy import split_gemm_nocopy
from fused_sparsity_ops import sparsity_tracker

def test_dw2_scenario():
    """Test the specific scenario from compute_split_gemm_dw2_lowrank"""
    
    # Dimensions from the error case
    batch_seq = 64 * 256  # batch_size * seq_len
    intermediate_size = 3072
    rank2 = 256
    
    device = 'cuda'
    dtype = torch.bfloat16
    
    # Create test tensors
    y2 = torch.randn(batch_seq, intermediate_size, device=device, dtype=dtype)
    d_intermediate_2 = torch.randn(batch_seq, rank2, device=device, dtype=dtype)
    
    # Create sparse mask for intermediate_size dimension
    sparse_mask = torch.rand(intermediate_size, device=device) < 0.95
    
    print(f"y2 shape: {y2.shape}")
    print(f"d_intermediate_2 shape: {d_intermediate_2.shape}")
    print(f"sparse_mask shape: {sparse_mask.shape}")
    print(f"Sparse columns: {sparse_mask.sum().item()}/{intermediate_size}")
    
    # Store sparsity in tracker
    layer_id = "test_layer_y2"
    col_sparsity = torch.rand(intermediate_size, device=device)
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
    
    # Compute y2.T @ d_intermediate_2 with split-GEMM
    print("\nComputing y2.T @ d_intermediate_2...")
    
    # Transpose y2
    y2_t = y2.t()  # [intermediate_size, batch*seq]
    print(f"y2_t shape: {y2_t.shape}")
    
    try:
        # This is the exact call that fails
        result = split_gemm_nocopy(y2_t, d_intermediate_2, sparse_mask)
        print(f"✓ Success! Result shape: {result.shape}")
        
        # Verify result shape
        expected_shape = (intermediate_size, rank2)
        assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dw2_scenario()