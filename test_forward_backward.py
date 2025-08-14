import torch
from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunction
from fused_sparsity_ops import sparsity_tracker

def test_forward_backward():
    """Test forward and backward passes with sparsity tracking"""
    
    # Reset tracker
    sparsity_tracker.reset()
    
    # Test configuration
    batch_size, seq_len, hidden_size = 2, 128, 768
    intermediate_size = 3072
    rank1, rank2 = 256, 256
    
    # Create test data
    device = 'cuda'
    dtype = torch.bfloat16
    
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
    # Weight shapes follow LowRankLinear convention: weight_out is [out_dim, rank]
    weight_in1 = torch.randn(hidden_size, rank1, device=device, dtype=dtype, requires_grad=True)
    weight_out1 = torch.randn(intermediate_size, rank1, device=device, dtype=dtype, requires_grad=True)
    weight_in2 = torch.randn(intermediate_size, rank2, device=device, dtype=dtype, requires_grad=True)
    weight_out2 = torch.randn(hidden_size, rank2, device=device, dtype=dtype, requires_grad=True)
    
    bias1 = None
    bias2 = None
    perm = None
    
    # Set configuration
    ActivationSparse2to4LowRankFunction._dx_direct_sparse = 1  # Enable split-GEMM
    ActivationSparse2to4LowRankFunction._warmup_steps = 0
    ActivationSparse2to4LowRankFunction._training_step = 1
    ActivationSparse2to4LowRankFunction._sparsity_method = 'soft_dynamic'
    
    print("Running forward pass...")
    
    # Forward pass
    # Note: the 8th parameter is sparsity_method, not perm
    output = ActivationSparse2to4LowRankFunction.apply(
        input_tensor, weight_in1, weight_out1, weight_in2, weight_out2,
        bias1, bias2, 'soft_dynamic'  # sparsity_method
    )
    
    print(f"Forward pass completed. Output shape: {output.shape}")
    
    # Check sparsity tracker
    print(f"Sparsity tracker has {len(sparsity_tracker.forward_masks)} entries")
    for layer_id in list(sparsity_tracker.forward_masks.keys())[:5]:
        print(f"  - {layer_id}")
    
    print("\nRunning backward pass...")
    
    # Backward pass
    grad_output = torch.randn_like(output)
    
    try:
        output.backward(grad_output)
        print("✓ Backward pass completed successfully!")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_forward_backward()