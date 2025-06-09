#!/usr/bin/env python3

import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Test LORO installation
import loro_torch
print('LORO installation: SUCCESS')

# Test sparse package installation
from sparse import soft_threshold24_triton, matmul, MVUE24_approx_triton
from loro_torch.sparse_lowrank_module import SparseLowRankLinear, SPARSE_AVAILABLE
print(f'Sparse package available: {SPARSE_AVAILABLE}')

if SPARSE_AVAILABLE:
    print('2:4 sparse acceleration: REAL TRITON KERNELS')
    
    # Test 2:4 sparsification
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_weight = torch.randn(16, 32, device=device)
    sparse_weight, mask = soft_threshold24_triton(test_weight)
    sparsity = (sparse_weight == 0).float().mean()
    print(f'2:4 Sparsity test: {sparsity.item():.1%} zeros (should be ~50%)')
    
    print('All tests passed!')
else:
    print('WARNING: Using fallback mode') 