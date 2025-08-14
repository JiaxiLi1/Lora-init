"""
Test dtype compatibility for soft_threshold functions.
"""

import torch
from peft_pretraining.modeling_llama import (
    apply_soft_threshold_weights_2to4_sparsity,
    apply_soft_threshold_dynamic_activation_2to4_sparsity
)

def test_bfloat16_compatibility():
    """Test that soft_threshold functions work with bfloat16."""
    print("Testing bfloat16 compatibility...")
    
    # Test with different dtypes
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    
    for dtype in dtypes:
        print(f"\nTesting with dtype: {dtype}")
        
        # Create test data
        batch_size = 2
        seq_len = 64
        hidden_size = 768
        intermediate_size = 3072
        
        y2 = torch.randn(batch_size * seq_len, intermediate_size, device='cuda', dtype=dtype)
        weight_out = torch.randn(hidden_size, intermediate_size, device='cuda', dtype=dtype)
        
        try:
            # Test soft_threshold_weights
            layer_id = f"test_{dtype}"
            y2_sparse_weights = apply_soft_threshold_weights_2to4_sparsity(y2, weight_out, layer_id)
            assert y2_sparse_weights.dtype == dtype, f"Output dtype mismatch for soft_threshold_weights"
            print(f"  ✓ soft_threshold_weights works with {dtype}")
            
            # Test soft_dynamic
            y2_sparse_dynamic = apply_soft_threshold_dynamic_activation_2to4_sparsity(
                y2, layer_id=0, current_step=0, dynamic_steps=10, calibration_samples=50
            )
            assert y2_sparse_dynamic.dtype == dtype, f"Output dtype mismatch for soft_dynamic"
            print(f"  ✓ soft_dynamic works with {dtype}")
            
            # Verify sparsity pattern
            y2_sparse_reshaped = y2_sparse_weights.view(-1, 4)
            nonzero_per_group = (y2_sparse_reshaped != 0).sum(dim=1)
            assert (nonzero_per_group <= 2).all(), "Should have at most 2 non-zeros per group of 4"
            
        except Exception as e:
            print(f"  ✗ Failed with {dtype}: {e}")
            return False
    
    return True


def test_scale_computation():
    """Test that scale computation works correctly with different dtypes."""
    print("\nTesting scale computation with different dtypes...")
    
    from peft_pretraining.modeling_llama import WeightBasedScaleManager, ActivationSoftThresholdManager
    
    # Clear any cached scales
    WeightBasedScaleManager._scales.clear()
    ActivationSoftThresholdManager._dynamic_scales.clear()
    
    dtype = torch.bfloat16
    
    # Create test data
    activations = torch.randn(128, 3072, device='cuda', dtype=dtype)
    weight = torch.randn(768, 3072, device='cuda', dtype=dtype)
    
    # Test weight-based scale computation
    scale = WeightBasedScaleManager._compute_weight_based_scale(activations, weight)
    assert isinstance(scale, float), "Scale should be a float"
    assert 0.5 <= scale <= 2.0, f"Scale {scale} out of expected range"
    print(f"  Weight-based scale: {scale:.4f}")
    
    # Test activation-based scale computation
    scale = ActivationSoftThresholdManager._compute_optimal_scale(activations)
    assert isinstance(scale, float), "Scale should be a float"
    assert 0.5 <= scale <= 2.0, f"Scale {scale} out of expected range"
    print(f"  Activation-based scale: {scale:.4f}")
    
    print("  ✓ Scale computation works correctly")
    return True


if __name__ == "__main__":
    print("="*60)
    print("Testing dtype compatibility fixes")
    print("="*60)
    
    success = True
    
    if not test_bfloat16_compatibility():
        success = False
    
    if not test_scale_computation():
        success = False
    
    if success:
        print("\n" + "="*60)
        print("All dtype tests passed!")
        print("="*60)
    else:
        print("\n✗ Some tests failed")