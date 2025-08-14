import torch

# Test configuration
batch_size, seq_len, hidden_size = 2, 128, 768
intermediate_size = 3072
rank1 = 256

# Create test tensors
input_2d = torch.randn(batch_size * seq_len, hidden_size)
weight_in1 = torch.randn(hidden_size, rank1)
weight_out1 = torch.randn(rank1, intermediate_size)

# Compute intermediate_1
intermediate_1 = torch.mm(input_2d, weight_in1)

print(f"input_2d shape: {input_2d.shape}")
print(f"weight_in1 shape: {weight_in1.shape}")
print(f"intermediate_1 shape: {intermediate_1.shape}")
print(f"weight_out1 shape: {weight_out1.shape}")
print(f"weight_out1.T shape: {weight_out1.T.shape}")

# Expected computation: intermediate_1 @ weight_out1.T
print(f"\nExpected: intermediate_1 @ weight_out1.T")
print(f"  {intermediate_1.shape} @ {weight_out1.T.shape}")
print(f"  [{batch_size * seq_len}, {rank1}] @ [{rank1}, {intermediate_size}]")

result = torch.mm(intermediate_1, weight_out1.T)
print(f"Result shape: {result.shape}")