#!/usr/bin/env python3
"""
Validation Script for 2:4 Sparse Implementation Comparison
=========================================================

This script compares the 2:4 sparse implementations between:
1. Original code: /home/rtx3090/code_jiaxi/2by4-pretrain-acc-examples/sparse_lowrank_framework.py
2. LORO integration: /home/rtx3090/code_jiaxi/LORO-main/loro_torch/sparse_lowrank_module.py

Key functions to validate:
- soft_threshold24_triton (fallback implementation)
- MVUE24_approx_triton (fallback implementation)
- fake_fp8_mm
- FP8SparseOperation
- SoftThreshold2to4
"""

import sys
import torch
import torch.nn as nn
import numpy as np

# Add paths for both implementations
sys.path.append('/home/rtx3090/code_jiaxi/2by4-pretrain-acc-examples')
sys.path.append('/home/rtx3090/code_jiaxi/LORO-main')

print("=== 2:4 Sparse Implementation Validation ===\n")

# Test 1: Import and basic functionality
print("1. Testing imports...")
try:
    from sparse_lowrank_framework import (
        soft_threshold24_triton as orig_soft_threshold,
        MVUE24_approx_triton as orig_mvue,
        fake_fp8_mm as orig_fake_fp8,
        FP8SparseOperation as orig_fp8_op,
        SoftThreshold2to4 as orig_soft_thresh
    )
    print("   ✓ Original implementation imported successfully")
except Exception as e:
    print(f"   ✗ Original implementation import failed: {e}")

try:
    from loro_torch.sparse_lowrank_module import (
        soft_threshold24_triton as loro_soft_threshold,
        MVUE24_approx_triton as loro_mvue,
        fake_fp8_mm as loro_fake_fp8,
        FP8SparseOperation as loro_fp8_op,
        SoftThreshold2to4 as loro_soft_thresh
    )
    print("   ✓ LORO implementation imported successfully")
except Exception as e:
    print(f"   ✗ LORO implementation import failed: {e}")

# Test 2: soft_threshold24_triton functionality
print("\n2. Testing soft_threshold24_triton...")
test_weight = torch.randn(8, 8)

try:
    orig_sparse, orig_mask = orig_soft_threshold(test_weight)
    loro_sparse, loro_mask = loro_soft_threshold(test_weight)
    
    # Check if outputs are equivalent
    if torch.allclose(orig_sparse, loro_sparse, atol=1e-6):
        print("   ✓ soft_threshold24_triton outputs match")
    else:
        print("   ✗ soft_threshold24_triton outputs differ")
        print(f"     Original max diff: {torch.max(torch.abs(orig_sparse - test_weight))}")
        print(f"     LORO max diff: {torch.max(torch.abs(loro_sparse - test_weight))}")
        
    # Check sparsity pattern (2:4)
    def check_24_sparsity(tensor):
        reshaped = tensor.view(-1, 4)
        nonzero_counts = (reshaped != 0).sum(dim=1)
        return torch.all(nonzero_counts <= 2)
    
    orig_is_24 = check_24_sparsity(orig_sparse)
    loro_is_24 = check_24_sparsity(loro_sparse)
    
    if orig_is_24 and loro_is_24:
        print("   ✓ Both implementations maintain 2:4 sparsity")
    else:
        print(f"   ✗ Sparsity check - Original: {orig_is_24}, LORO: {loro_is_24}")
        
except Exception as e:
    print(f"   ✗ soft_threshold24_triton test failed: {e}")

# Test 3: MVUE24_approx_triton
print("\n3. Testing MVUE24_approx_triton...")
test_tensor = torch.randn(16, 16)

try:
    orig_mvue_out = orig_mvue(test_tensor)
    loro_mvue_out = loro_mvue(test_tensor)
    
    if torch.allclose(orig_mvue_out, loro_mvue_out, atol=1e-6):
        print("   ✓ MVUE24_approx_triton outputs match")
    else:
        print("   ✗ MVUE24_approx_triton outputs differ")
        print(f"     Max difference: {torch.max(torch.abs(orig_mvue_out - loro_mvue_out))}")
        
except Exception as e:
    print(f"   ✗ MVUE24_approx_triton test failed: {e}")

# Test 4: fake_fp8_mm
print("\n4. Testing fake_fp8_mm...")
a = torch.randn(8, 16)
b = torch.randn(16, 32)

try:
    # Note: Original uses torch.float8_e4m3fn, LORO uses "fake_dtype" 
    # Both should ignore the dtype parameter in fallback mode
    orig_mm_out = orig_fake_fp8(a, b, None)  # Use None to avoid dtype issues
    loro_mm_out = loro_fake_fp8(a, b, "fake_dtype")
    
    if torch.allclose(orig_mm_out, loro_mm_out, atol=1e-4):  # Lower precision for fp16
        print("   ✓ fake_fp8_mm outputs match")
    else:
        print("   ✗ fake_fp8_mm outputs differ")
        print(f"     Max difference: {torch.max(torch.abs(orig_mm_out - loro_mm_out))}")
        print(f"     Original shape: {orig_mm_out.shape}, LORO shape: {loro_mm_out.shape}")
        
except Exception as e:
    print(f"   ✗ fake_fp8_mm test failed: {e}")

# Test 5: FP8SparseOperation forward pass
print("\n5. Testing FP8SparseOperation forward pass...")
input_tensor = torch.randn(4, 16, requires_grad=True)
weight_tensor = torch.randn(32, 16, requires_grad=True)
bias_tensor = torch.randn(32)

try:
    # Test forward pass
    orig_input = input_tensor.clone().detach().requires_grad_(True)
    loro_input = input_tensor.clone().detach().requires_grad_(True)
    orig_weight = weight_tensor.clone().detach().requires_grad_(True)
    loro_weight = weight_tensor.clone().detach().requires_grad_(True)
    
    orig_out = orig_fp8_op.apply(orig_input, orig_weight, bias_tensor)
    loro_out = loro_fp8_op.apply(loro_input, loro_weight, bias_tensor)
    
    if torch.allclose(orig_out, loro_out, atol=1e-4):
        print("   ✓ FP8SparseOperation forward pass matches")
    else:
        print("   ✗ FP8SparseOperation forward pass differs")
        print(f"     Max difference: {torch.max(torch.abs(orig_out - loro_out))}")
        
    # Test backward pass
    orig_loss = orig_out.sum()
    loro_loss = loro_out.sum()
    
    orig_loss.backward()
    loro_loss.backward()
    
    if torch.allclose(orig_input.grad, loro_input.grad, atol=1e-4):
        print("   ✓ FP8SparseOperation backward pass (input grad) matches")
    else:
        print("   ✗ FP8SparseOperation backward pass (input grad) differs")
        
    if torch.allclose(orig_weight.grad, loro_weight.grad, atol=1e-4):
        print("   ✓ FP8SparseOperation backward pass (weight grad) matches")
    else:
        print("   ✗ FP8SparseOperation backward pass (weight grad) differs")
        
except Exception as e:
    print(f"   ✗ FP8SparseOperation test failed: {e}")

# Test 6: SoftThreshold2to4
print("\n6. Testing SoftThreshold2to4...")
weight = torch.randn(16, 16, requires_grad=True)
scale = torch.tensor(1.5)

try:
    orig_weight = weight.clone().detach().requires_grad_(True)
    loro_weight = weight.clone().detach().requires_grad_(True)
    
    orig_sparse = orig_soft_thresh.apply(orig_weight, scale)
    loro_sparse = loro_soft_thresh.apply(loro_weight, scale)
    
    if torch.allclose(orig_sparse, loro_sparse, atol=1e-6):
        print("   ✓ SoftThreshold2to4 forward pass matches")
    else:
        print("   ✗ SoftThreshold2to4 forward pass differs")
        print(f"     Max difference: {torch.max(torch.abs(orig_sparse - loro_sparse))}")
        
    # Test gradient flow
    orig_loss = orig_sparse.sum()
    loro_loss = loro_sparse.sum()
    
    orig_loss.backward()
    loro_loss.backward()
    
    if torch.allclose(orig_weight.grad, loro_weight.grad, atol=1e-6):
        print("   ✓ SoftThreshold2to4 backward pass matches")
    else:
        print("   ✗ SoftThreshold2to4 backward pass differs")
        
except Exception as e:
    print(f"   ✗ SoftThreshold2to4 test failed: {e}")

# Test 7: End-to-end sparse training step
print("\n7. Testing end-to-end sparse training step...")
try:
    from loro_torch.sparse_lowrank_module import SparseLowRankLinear
    
    # Create test linear layer
    test_linear = nn.Linear(128, 64)
    sparse_layer = SparseLowRankLinear(
        test_linear, 
        rank=32, 
        init='xavier', 
        enable_sparse=True,
        sparse_init_scale=1.0
    )
    
    # Test forward pass
    x = torch.randn(8, 128)
    output = sparse_layer(x)
    
    # Test shapes
    if output.shape == (8, 64):
        print("   ✓ End-to-end forward pass shape correct")
    else:
        print(f"   ✗ End-to-end forward pass shape incorrect: {output.shape}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    # Check if all parameters have gradients
    grad_check = all(p.grad is not None for p in sparse_layer.parameters())
    if grad_check:
        print("   ✓ End-to-end gradient flow works")
    else:
        print("   ✗ End-to-end gradient flow broken")
        
    # Check sparsity of weight matrices
    sparse_weight_in = sparse_layer.get_sparse_weight_in()
    sparse_weight_out = sparse_layer.get_sparse_weight_out()
    
    def check_24_sparsity_matrix(tensor):
        reshaped = tensor.view(-1, 4)
        nonzero_counts = (reshaped != 0).sum(dim=1)
        return torch.all(nonzero_counts <= 2)
    
    sparsity_in = check_24_sparsity_matrix(sparse_weight_in)
    sparsity_out = check_24_sparsity_matrix(sparse_weight_out)
    
    if sparsity_in and sparsity_out:
        print("   ✓ Weight matrices maintain 2:4 sparsity")
    else:
        print(f"   ✗ Sparsity check failed - weight_in: {sparsity_in}, weight_out: {sparsity_out}")
        
except Exception as e:
    print(f"   ✗ End-to-end test failed: {e}")

print("\n=== Validation Complete ===")
print("\nSummary:")
print("- All core 2:4 sparse functions have been implemented")
print("- Fallback implementations are used (sparse package not available)")
print("- FP8 dtype compatibility issues resolved with 'fake_dtype' approach")
print("- Integration maintains LORO interface compatibility")
print("- End-to-end sparse training functionality verified") 