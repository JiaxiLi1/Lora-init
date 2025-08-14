"""Test script to diagnose split-GEMM issue"""

import torch
import sys
sys.path.append('.')

from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunction

# Set up test configuration
batch_size = 64
seq_len = 256
hidden_size = 768
intermediate_size = 3072
rank1 = 256
rank2 = 256

device = 'cuda'
dtype = torch.float16

# Create test tensors
input = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
weight_in1 = torch.randn(hidden_size, rank1, device=device, dtype=dtype, requires_grad=True)
weight_out1 = torch.randn(intermediate_size, rank1, device=device, dtype=dtype, requires_grad=True)
weight_in2 = torch.randn(intermediate_size, rank2, device=device, dtype=dtype, requires_grad=True)
weight_out2 = torch.randn(hidden_size, rank2, device=device, dtype=dtype, requires_grad=True)
bias1 = torch.randn(intermediate_size, device=device, dtype=dtype, requires_grad=True)
bias2 = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)

# Set up function configuration
ActivationSparse2to4LowRankFunction._training_step = 0
ActivationSparse2to4LowRankFunction._warmup_steps = 1

print("Testing ActivationSparse2to4LowRankFunction...")
print(f"Warmup steps: {ActivationSparse2to4LowRankFunction._warmup_steps}")

# Test warmup phase (step 0)
print("\n=== Step 0 (Warmup) ===")
ActivationSparse2to4LowRankFunction._training_step = 0
output1 = ActivationSparse2to4LowRankFunction.apply(
    input, weight_in1, weight_out1, weight_in2, weight_out2, bias1, bias2,
    "soft_threshold_weights",  # sparsity_method
    1,  # warmup_steps
    1,  # dx_direct_sparse (1 = use split-GEMM)
    10,  # dynamic_steps
    50,  # calibration_samples
    True  # enable_permute
)
print(f"Forward pass completed. Output shape: {output1.shape}")

# Backward pass
loss1 = output1.sum()
loss1.backward()
print("Backward pass completed successfully")

# Clear gradients
input.grad = None
weight_in1.grad = None
weight_out1.grad = None
weight_in2.grad = None
weight_out2.grad = None
bias1.grad = None
bias2.grad = None

# Test non-warmup phase (step 1)
print("\n=== Step 1 (After warmup) ===")
ActivationSparse2to4LowRankFunction._training_step = 1
try:
    output2 = ActivationSparse2to4LowRankFunction.apply(
        input, weight_in1, weight_out1, weight_in2, weight_out2, bias1, bias2,
        "soft_threshold_weights",  # sparsity_method
        1,  # warmup_steps
        1,  # dx_direct_sparse (1 = use split-GEMM)
        10,  # dynamic_steps
        50,  # calibration_samples
        True  # enable_permute
    )
    print(f"Forward pass completed. Output shape: {output2.shape}")
    
    # Backward pass
    loss2 = output2.sum()
    loss2.backward()
    print("Backward pass completed successfully")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Test completed!")