#!/usr/bin/env python3
"""
Test script for activation 2:4 sparsification implementation
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from peft_pretraining.modeling_llama import (
    ActivationSparse2to4Function,
    LlamaMLP,
    apply_naive_2to4_sparsity,
    apply_mvue_2to4_sparsity,
    apply_soft_threshold_2to4_sparsity,
    apply_feature_wise_2to4_sparsity,
    apply_naive_2to4_sparsity_featurewise
)

def test_2to4_sparsity_pattern(tensor, name):
    """Test if tensor follows 2:4 sparsity pattern"""
    print(f"\n🔍 Testing {name}:")
    print(f"  Input shape: {tensor.shape}")
    
    # Flatten to 2D for easier checking
    if tensor.dim() > 2:
        tensor_2d = tensor.view(-1, tensor.shape[-1])
    else:
        tensor_2d = tensor
    
    batch_size, hidden_size = tensor_2d.shape
    
    # Pad to make hidden_size divisible by 4
    if hidden_size % 4 != 0:
        pad_size = 4 - (hidden_size % 4)
        tensor_padded = torch.nn.functional.pad(tensor_2d, (0, pad_size), value=0)
    else:
        tensor_padded = tensor_2d
    
    # Reshape to check 2:4 pattern
    tensor_reshaped = tensor_padded.view(batch_size, -1, 4)
    
    # Count non-zero elements in each group of 4
    non_zero_counts = (tensor_reshaped != 0).sum(dim=-1)
    
    # Check if all groups have at most 2 non-zero elements
    valid_groups = (non_zero_counts <= 2).all()
    
    print(f"  ✅ Valid 2:4 pattern: {valid_groups}")
    print(f"  📊 Non-zero counts distribution: {torch.bincount(non_zero_counts.flatten())}")
    
    # Calculate actual sparsity
    total_elements = tensor_2d.numel()
    non_zero_elements = (tensor_2d != 0).sum().item()
    sparsity = 1.0 - (non_zero_elements / total_elements)
    print(f"  📊 Actual sparsity: {sparsity:.2%}")
    
    return valid_groups

def test_token_permutation():
    """Test token permutation functionality"""
    print("\n🔧 Testing Token Permutation:")
    
    # Reset step counter
    ActivationSparse2to4Function._training_step = 1500  # Beyond warmup
    
    batch_size, seq_len, hidden_size = 2, 8, 16
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    
    # Apply activation sparse function
    output1 = ActivationSparse2to4Function.apply(input_tensor, "naive")
    output2 = ActivationSparse2to4Function.apply(input_tensor, "naive")
    
    # Check if permutation is consistent
    permutation_consistent = torch.allclose(output1, output2, atol=1e-6)
    print(f"  ✅ Permutation consistency: {permutation_consistent}")
    
    # Check if permutation keys are stored
    perm_key = f"{seq_len}_cuda:0"
    has_permutation = perm_key in ActivationSparse2to4Function._token_permutation
    print(f"  ✅ Permutation stored: {has_permutation}")
    
    if has_permutation:
        perm = ActivationSparse2to4Function._token_permutation[perm_key]
        inv_perm = ActivationSparse2to4Function._inverse_permutation[perm_key]
        print(f"  📊 Permutation: {perm}")
        print(f"  📊 Inverse permutation: {inv_perm}")
        
        # Verify inverse permutation
        identity = torch.arange(seq_len, device='cuda')
        is_inverse = torch.equal(identity[perm][inv_perm], identity)
        print(f"  ✅ Inverse permutation correct: {is_inverse}")

def test_dense_warmup():
    """Test dense warmup functionality"""
    print("\n🔧 Testing Dense Warmup:")
    
    batch_size, seq_len, hidden_size = 2, 8, 16
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    
    # Test custom warmup steps
    custom_warmup_steps = 500
    ActivationSparse2to4Function.set_warmup_steps(custom_warmup_steps)
    
    # Test warmup phase (step < custom_warmup_steps)
    ActivationSparse2to4Function._training_step = 300
    
    output_warmup = ActivationSparse2to4Function.apply(input_tensor, "naive", custom_warmup_steps)
    
    # During warmup, output should be identical to input
    is_identity_warmup = torch.allclose(input_tensor, output_warmup, atol=1e-6)
    print(f"  ✅ Warmup identity (step 300, warmup={custom_warmup_steps}): {is_identity_warmup}")
    
    # Test post-warmup phase (step >= custom_warmup_steps)
    ActivationSparse2to4Function._training_step = 600
    
    output_sparse = ActivationSparse2to4Function.apply(input_tensor, "naive", custom_warmup_steps)
    
    # After warmup, output should be different (sparsified)
    is_different_post_warmup = not torch.allclose(input_tensor, output_sparse, atol=1e-6)
    print(f"  ✅ Post-warmup sparsification (step 600, warmup={custom_warmup_steps}): {is_different_post_warmup}")
    
    # Test default warmup steps (1000)
    ActivationSparse2to4Function.set_warmup_steps(1000)
    ActivationSparse2to4Function._training_step = 800
    
    output_default_warmup = ActivationSparse2to4Function.apply(input_tensor, "naive")
    is_identity_default = torch.allclose(input_tensor, output_default_warmup, atol=1e-6)
    print(f"  ✅ Default warmup identity (step 800, warmup=1000): {is_identity_default}")
    
    # Test sparsity pattern
    test_2to4_sparsity_pattern(output_sparse, "Post-warmup output")

def test_feature_wise_backward():
    """Test feature-wise backward pass"""
    print("\n🔧 Testing Feature-wise Backward Pass:")
    
    batch_seq_len, hidden_size = 16, 64
    grad_tensor = torch.randn(batch_seq_len, hidden_size, device='cuda')
    forward_mask = torch.rand(batch_seq_len, hidden_size, device='cuda') > 0.5
    
    # Apply feature-wise sparsity
    grad_sparse = apply_feature_wise_2to4_sparsity(grad_tensor, forward_mask.float())
    
    # Test that forward mask is preserved
    forward_mask_preserved = torch.all((grad_sparse != 0) <= forward_mask)
    print(f"  ✅ Forward mask preserved: {forward_mask_preserved}")
    
    # Test feature-wise sparsity pattern
    # We need to check if 95% of features are sparsified
    feature_sparsity = torch.mean((grad_sparse != 0).float(), dim=0)
    sparse_features = (feature_sparsity > 0.75).sum().item()
    total_features = hidden_size
    sparse_ratio = sparse_features / total_features
    print(f"  📊 Sparse features ratio: {sparse_ratio:.2%}")
    
    # Test that sparse features follow 2:4 pattern along batch dimension
    if sparse_features > 0:
        sparse_mask = feature_sparsity > 0.75
        sparse_grad = grad_sparse[:, sparse_mask]
        
        # Check 2:4 pattern along batch dimension (transpose for checking)
        if sparse_grad.numel() > 0:
            sparse_grad_t = sparse_grad.t()
            is_valid_2to4 = test_2to4_sparsity_pattern(sparse_grad_t, "Feature-wise sparse gradients")
            print(f"  ✅ Feature-wise 2:4 pattern: {is_valid_2to4}")

def test_backward_pass_integration():
    """Test backward pass integration with autograd"""
    print("\n🔧 Testing Backward Pass Integration:")
    
    # Set beyond warmup
    ActivationSparse2to4Function._training_step = 1500
    
    batch_size, seq_len, hidden_size = 2, 8, 16
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda', requires_grad=True)
    
    # Forward pass
    output = ActivationSparse2to4Function.apply(input_tensor, "naive")
    
    # Create a dummy loss
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check if gradients are computed
    has_gradients = input_tensor.grad is not None
    print(f"  ✅ Gradients computed: {has_gradients}")
    
    if has_gradients:
        grad_shape_correct = input_tensor.grad.shape == input_tensor.shape
        print(f"  ✅ Gradient shape correct: {grad_shape_correct}")
        
        # Test that gradients follow the expected pattern
        grad_non_zero = (input_tensor.grad != 0).sum().item()
        grad_total = input_tensor.grad.numel()
        grad_sparsity = 1.0 - (grad_non_zero / grad_total)
        print(f"  📊 Gradient sparsity: {grad_sparsity:.2%}")

def test_mlp_integration():
    """Test MLP integration with squared ReLU and activation sparsity"""
    print("\n🔧 Testing MLP Integration:")
    
    # Reset step counter beyond warmup
    ActivationSparse2to4Function._training_step = 1500
    
    # Create config
    class MockConfig:
        def __init__(self):
            self.squ_relu = True
            self.activation_sparse_method = "naive"
    
    config = MockConfig()
    
    # Create MLP
    hidden_size = 32
    intermediate_size = 64
    mlp = LlamaMLP(hidden_size, intermediate_size, "silu", config=config).cuda()
    
    # Test forward pass
    batch_size, seq_len = 2, 8
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda', requires_grad=True)
    
    output = mlp(input_tensor)
    
    # Check output shape
    output_shape_correct = output.shape == (batch_size, seq_len, hidden_size)
    print(f"  ✅ Output shape correct: {output_shape_correct}")
    
    # Check if gate_proj is None (squared ReLU architecture)
    has_no_gate = mlp.gate_proj is None
    print(f"  ✅ No gate projection (squared ReLU): {has_no_gate}")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    # Check if gradients are computed
    has_gradients = input_tensor.grad is not None
    print(f"  ✅ Gradients computed: {has_gradients}")
    
    print(f"  📊 MLP architecture: up_proj({hidden_size} -> {mlp.new_intermediate_size}) -> squared_relu -> sparse_2to4 -> down_proj({mlp.new_intermediate_size} -> {hidden_size})")

def main():
    print("🚀 Starting Activation 2:4 Sparsity Tests")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. These tests require GPU.")
        return
    
    try:
        # Test individual components
        test_token_permutation()
        test_dense_warmup()
        test_feature_wise_backward()
        test_backward_pass_integration()
        test_mlp_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        
        # Print current step counter
        current_step = ActivationSparse2to4Function.get_training_step()
        print(f"📊 Final training step: {current_step}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 