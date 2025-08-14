"""
Minimal test for optimized split-GEMM.
"""

import torch
import sys
sys.path.append('/home/rtx3090/code_jiaxi/LORO-main_temp')

# Clear CUDA state
torch.cuda.empty_cache()
torch.cuda.synchronize()


def minimal_test():
    """Minimal end-to-end test."""
    print("Running minimal test...")
    print("="*60)
    
    from fused_sparsity_ops import sparsity_tracker
    from peft_pretraining.split_gemm_utils import compute_split_gemm_lowrank_intermediate
    
    # Small configuration
    batch_seq = 256
    hidden_size = 128
    rank = 64
    
    device = 'cuda'
    dtype = torch.float16
    
    # Create test data
    dy1 = torch.randn(batch_seq, hidden_size, device=device, dtype=dtype)
    weight_out1 = torch.randn(hidden_size, rank, device=device, dtype=dtype)
    
    print(f"Configuration:")
    print(f"  dy1: {dy1.shape}")
    print(f"  weight_out1: {weight_out1.shape}")
    
    # Create sparse mask
    sparse_mask = torch.rand(hidden_size, device=device) < 0.9
    layer_id = "minimal_test_layer"
    col_sparsity = torch.rand(hidden_size, device=device)
    
    print(f"  Sparse ratio: {sparse_mask.float().mean()*100:.1f}%")
    
    # Store in tracker
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
    
    # Run the function
    print("\nRunning compute_split_gemm_lowrank_intermediate...")
    try:
        result = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, layer_id)
        print(f"✓ Success! Result shape: {result.shape}")
        
        # Verify result
        expected_shape = (batch_seq, rank)
        assert result.shape == expected_shape, f"Shape mismatch: {result.shape} != {expected_shape}"
        print(f"✓ Shape verified: {result.shape}")
        
        # Check for NaN/Inf
        assert not torch.isnan(result).any(), "Result contains NaN"
        assert not torch.isinf(result).any(), "Result contains Inf"
        print("✓ No NaN/Inf in result")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = minimal_test()
    
    if success:
        print("\n" + "="*60)
        print("✓ Minimal test PASSED!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Minimal test FAILED!")
        print("="*60)
        sys.exit(1)