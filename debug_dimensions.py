#!/usr/bin/env python3
"""Debug matrix dimensions for LORO"""

import torch

print('Testing matrix dimensions:')
print('x:', (8, 1024, 768))
print('weight_in (rank, in_features):', (64, 768))
print('weight_out (out_features, rank):', (256, 64))
print()

print('For FP8SparseLinear.apply(x, weight, bias):')
print('  Input: x has shape (B, S, in_features)')
print('  Weight: weight has shape (out_features, in_features)')  
print('  Inside FP8SparseLinear: uses weight.t() -> (in_features, out_features)')
print('  Computation: x @ weight.t() -> (B, S, out_features)')
print()

print('Step 1: x @ weight_in -> x_proj')
print('  x shape:', (8, 1024, 768))
print('  Need weight shape (rank, in_features) = (64, 768) for FP8SparseLinear')
print('  weight_in shape:', (64, 768), '✓ Correct!')
print('  Result x_proj shape:', (8, 1024, 64))
print()

print('Step 2: x_proj @ weight_out -> output')  
print('  x_proj shape:', (8, 1024, 64))
print('  Need weight shape (out_features, rank) = (256, 64) for FP8SparseLinear')
print('  weight_out shape:', (256, 64), '✓ Correct!')
print('  Result output shape:', (8, 1024, 256))

# Test actual computation
print('\n' + '='*50)
print('Testing actual computation:')

x = torch.randn(8, 1024, 768)
weight_in = torch.randn(64, 768)  # (rank, in_features)
weight_out = torch.randn(256, 64)  # (out_features, rank)

# Step 1: Manual matrix multiplication
x_flat = x.view(-1, x.shape[-1])  # (8*1024, 768)
print('x_flat shape:', x_flat.shape)
print('weight_in.t() shape:', weight_in.t().shape)

try:
    result1 = torch.matmul(x_flat, weight_in.t())  # (8*1024, 768) @ (768, 64) = (8*1024, 64)
    print('Step 1 result shape:', result1.shape)
    x_proj = result1.view(*x.shape[:-1], -1)  # (8, 1024, 64)
    print('x_proj shape:', x_proj.shape)
    
    # Step 2
    x_proj_flat = x_proj.view(-1, x_proj.shape[-1])  # (8*1024, 64)
    print('x_proj_flat shape:', x_proj_flat.shape)
    print('weight_out.t() shape:', weight_out.t().shape)
    
    result2 = torch.matmul(x_proj_flat, weight_out.t())  # (8*1024, 64) @ (64, 256) = (8*1024, 256)
    print('Step 2 result shape:', result2.shape)
    output = result2.view(*x.shape[:-1], -1)  # (8, 1024, 256)
    print('Final output shape:', output.shape)
    
    print('\n✓ Matrix dimensions are correct!')
    
except Exception as e:
    print(f'\n❌ Error: {e}') 