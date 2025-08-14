"""
Test script to verify soft_threshold_weights and soft_dynamic implementations.
"""

import torch
from peft_pretraining.modeling_llama import (
    apply_soft_threshold_weights_2to4_sparsity,
    apply_soft_threshold_dynamic_activation_2to4_sparsity,
    WeightBasedScaleManager,
    ActivationSoftThresholdManager
)

def test_soft_threshold_weights():
    """Test soft_threshold_weights with weight-based scaling."""
    print("Testing soft_threshold_weights...")
    
    # Create test data
    batch_size = 8
    seq_len = 256
    hidden_size = 768
    intermediate_size = 3072
    
    # Test activation and weight
    y2 = torch.randn(batch_size * seq_len, intermediate_size, device='cuda', dtype=torch.float16)
    weight_out = torch.randn(hidden_size, intermediate_size, device='cuda', dtype=torch.float16)
    
    # First call - should compute and cache scale
    layer_id = "test_layer_1"
    y2_sparse_1 = apply_soft_threshold_weights_2to4_sparsity(y2, weight_out, layer_id)
    
    # Second call - should use cached scale
    y2_sparse_2 = apply_soft_threshold_weights_2to4_sparsity(y2, weight_out, layer_id)
    
    # Check that scale is cached (no recomputation)
    assert layer_id in WeightBasedScaleManager._scales
    scale = WeightBasedScaleManager._scales[layer_id]
    print(f"  Cached scale for {layer_id}: {scale:.4f}")
    
    # Verify sparsity pattern (2 out of 4)
    y2_sparse_reshaped = y2_sparse_1.view(-1, 4)
    nonzero_per_group = (y2_sparse_reshaped != 0).sum(dim=1)
    assert (nonzero_per_group <= 2).all(), "Should have at most 2 non-zeros per group of 4"
    
    print("  ✓ soft_threshold_weights test passed")


def test_soft_dynamic():
    """Test soft_dynamic with dynamic activation-based scaling."""
    print("\nTesting soft_dynamic...")
    
    # Create test data
    batch_size = 8
    seq_len = 256
    intermediate_size = 3072
    
    y2 = torch.randn(batch_size * seq_len, intermediate_size, device='cuda', dtype=torch.float16)
    
    # Test dynamic scaling with different steps
    layer_id = 0
    dynamic_steps = 10
    calibration_samples = 50
    
    # First call at step 0 - should compute scale
    current_step = 0
    y2_sparse_1 = apply_soft_threshold_dynamic_activation_2to4_sparsity(
        y2, layer_id, current_step, dynamic_steps, calibration_samples
    )
    
    # Second call at step 5 - should use cached scale
    current_step = 5
    y2_sparse_2 = apply_soft_threshold_dynamic_activation_2to4_sparsity(
        y2, layer_id, current_step, dynamic_steps, calibration_samples
    )
    
    # Third call at step 10 - should recompute scale
    current_step = 10
    y2_sparse_3 = apply_soft_threshold_dynamic_activation_2to4_sparsity(
        y2, layer_id, current_step, dynamic_steps, calibration_samples
    )
    
    # Verify sparsity pattern
    y2_sparse_reshaped = y2_sparse_1.view(-1, 4)
    nonzero_per_group = (y2_sparse_reshaped != 0).sum(dim=1)
    assert (nonzero_per_group <= 2).all(), "Should have at most 2 non-zeros per group of 4"
    
    print(f"  Dynamic scale at step 0: {ActivationSoftThresholdManager._dynamic_scales.get(layer_id, 1.0):.4f}")
    print("  ✓ soft_dynamic test passed")


def test_no_split_gemm_confusion():
    """Verify that soft_threshold methods don't involve split-GEMM."""
    print("\nVerifying no split-GEMM confusion...")
    
    # The soft_threshold methods should NOT:
    # 1. Compute column sparsity for split-GEMM
    # 2. Partition features into 95%/5%
    # 3. Use any split-GEMM related functions
    
    import inspect
    
    # Check apply_soft_threshold_weights_2to4_sparsity
    source = inspect.getsource(apply_soft_threshold_weights_2to4_sparsity)
    assert "split_gemm" not in source.lower(), "soft_threshold_weights should not mention split_gemm"
    assert "95%" not in source and "0.95" not in source, "soft_threshold_weights should not use 95% partitioning"
    print("  ✓ soft_threshold_weights does not use split-GEMM")
    
    # Check apply_soft_threshold_dynamic_activation_2to4_sparsity
    source = inspect.getsource(apply_soft_threshold_dynamic_activation_2to4_sparsity)
    assert "split_gemm" not in source.lower(), "soft_dynamic should not mention split_gemm"
    assert "95%" not in source and "0.95" not in source, "soft_dynamic should not use 95% partitioning"
    print("  ✓ soft_dynamic does not use split-GEMM")
    
    print("  ✓ No split-GEMM confusion verified")


if __name__ == "__main__":
    print("="*60)
    print("Testing Soft Threshold Implementations")
    print("="*60)
    
    try:
        test_soft_threshold_weights()
        test_soft_dynamic()
        test_no_split_gemm_confusion()
        
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()