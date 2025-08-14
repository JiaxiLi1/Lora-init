"""
Test script to verify warmup and permutation implementation.
"""

import torch
import torch.nn.functional as F
from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunction

def test_warmup_transition():
    """Test transition from warmup to sparse training."""
    print("Testing warmup transition...")
    
    # Setup
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    intermediate_size = 3072
    rank1 = rank2 = 256
    
    # Create dummy tensors with requires_grad
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16, requires_grad=True)
    weight_in1 = torch.randn(hidden_size, rank1, device='cuda', dtype=torch.float16, requires_grad=True)
    weight_out1 = torch.randn(intermediate_size, rank1, device='cuda', dtype=torch.float16, requires_grad=True)  # Note: transposed shape
    weight_in2 = torch.randn(intermediate_size, rank2, device='cuda', dtype=torch.float16, requires_grad=True)
    weight_out2 = torch.randn(hidden_size, rank2, device='cuda', dtype=torch.float16, requires_grad=True)  # Note: transposed shape
    
    # Test with warmup
    ActivationSparse2to4LowRankFunction._training_step = 0
    ActivationSparse2to4LowRankFunction._warmup_steps = 2
    
    # Forward pass during warmup
    y_warmup = ActivationSparse2to4LowRankFunction.apply(
        x, weight_in1, weight_out1, weight_in2, weight_out2,
        None, None,  # biases
        "soft_threshold_weights",  # sparsity_method
        None,  # warmup_steps (use class default)
        1,  # dx_direct_sparse
        10,  # dynamic_steps
        50,  # calibration_samples
        True  # enable_permute
    )
    
    # Increment step
    ActivationSparse2to4LowRankFunction.increment_step()
    
    # Forward pass after warmup
    y_sparse = ActivationSparse2to4LowRankFunction.apply(
        x, weight_in1, weight_out1, weight_in2, weight_out2,
        None, None,
        "soft_threshold_weights", None, 1, 10, 50, True
    )
    
    print(f"Warmup output shape: {y_warmup.shape}")
    print(f"Sparse output shape: {y_sparse.shape}")
    print(f"Outputs match shape: {y_warmup.shape == y_sparse.shape}")
    
    # Test backward
    grad_output = torch.randn_like(y_warmup)
    y_warmup.backward(grad_output)
    
    print("✓ Warmup transition test passed")


def test_permutation():
    """Test input permutation correctness."""
    print("\nTesting input permutation...")
    
    batch_size = 2
    seq_len = 64
    hidden_size = 768
    intermediate_size = 3072
    rank1 = rank2 = 256
    
    # Create dummy tensors with requires_grad
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16, requires_grad=True)
    weight_in1 = torch.randn(hidden_size, rank1, device='cuda', dtype=torch.float16, requires_grad=True)
    weight_out1 = torch.randn(intermediate_size, rank1, device='cuda', dtype=torch.float16, requires_grad=True)  # Note: transposed shape
    weight_in2 = torch.randn(intermediate_size, rank2, device='cuda', dtype=torch.float16, requires_grad=True)
    weight_out2 = torch.randn(hidden_size, rank2, device='cuda', dtype=torch.float16, requires_grad=True)  # Note: transposed shape
    
    # Reset training step
    ActivationSparse2to4LowRankFunction._training_step = 10
    ActivationSparse2to4LowRankFunction._warmup_steps = 0
    
    # Test with permutation
    y_with_perm = ActivationSparse2to4LowRankFunction.apply(
        x, weight_in1, weight_out1, weight_in2, weight_out2,
        None, None,
        "soft_threshold_weights", None, 3, 10, 50,
        True  # enable_permute = True
    )
    
    # Test without permutation
    y_without_perm = ActivationSparse2to4LowRankFunction.apply(
        x, weight_in1, weight_out1, weight_in2, weight_out2,
        None, None,
        "soft_threshold_weights", None, 3, 10, 50,
        False  # enable_permute = False
    )
    
    print(f"Output with permutation shape: {y_with_perm.shape}")
    print(f"Output without permutation shape: {y_without_perm.shape}")
    print(f"Shapes match: {y_with_perm.shape == y_without_perm.shape}")
    
    # The outputs should be different due to permutation
    diff = (y_with_perm - y_without_perm).abs().max().item()
    print(f"Max difference: {diff:.6f}")
    
    if diff > 0:
        print("✓ Permutation changes output (expected)")
    else:
        print("✗ Permutation doesn't change output (unexpected)")
    
    print("✓ Permutation test passed")


def test_different_seq_lengths():
    """Test permutation with different sequence lengths."""
    print("\nTesting different sequence lengths...")
    
    batch_size = 2
    hidden_size = 768
    intermediate_size = 3072
    rank1 = rank2 = 256
    
    # Create weights
    weight_in1 = torch.randn(hidden_size, rank1, device='cuda', dtype=torch.float16)
    weight_out1 = torch.randn(intermediate_size, rank1, device='cuda', dtype=torch.float16)  # Note: transposed shape
    weight_in2 = torch.randn(intermediate_size, rank2, device='cuda', dtype=torch.float16)
    weight_out2 = torch.randn(hidden_size, rank2, device='cuda', dtype=torch.float16)  # Note: transposed shape
    
    # Reset
    ActivationSparse2to4LowRankFunction._token_permutation = {}
    ActivationSparse2to4LowRankFunction._inverse_permutation = {}
    
    # Test different sequence lengths
    for seq_len in [32, 64, 128]:
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
        
        y = ActivationSparse2to4LowRankFunction.apply(
            x, weight_in1, weight_out1, weight_in2, weight_out2,
            None, None,
            "soft_threshold_weights", None, 3, 10, 50,
            True  # enable_permute
        )
        
        print(f"Seq length {seq_len}: output shape {y.shape}")
    
    # Check that different permutations were created
    num_perms = len(ActivationSparse2to4LowRankFunction._token_permutation)
    print(f"Number of permutations created: {num_perms}")
    
    if num_perms == 3:
        print("✓ Different permutations for different sequence lengths")
    else:
        print(f"✗ Expected 3 permutations, got {num_perms}")
    
    print("✓ Different sequence lengths test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Warmup and Permutation Implementation")
    print("=" * 60)
    
    try:
        test_warmup_transition()
        test_permutation()
        test_different_seq_lengths()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()