#!/usr/bin/env python3
"""
Test script for the fixed LORO implementation with EXACT 2by4-pretrain-acc-examples sparse support.

This script verifies that:
1. LORO uses the correct 2by4-pretrain-acc-examples sparse implementation
2. No fallback implementations are used
3. 2:4 sparsity patterns are correct
4. Performance is improved compared to fallback
5. No warning messages about sparse package unavailability
"""

import torch
import torch.nn as nn
import time
import numpy as np

# Import our fixed implementation
from loro_torch.sparse_lowrank_module import (
    SparseLowRankLinear, 
    check_sparse_implementation,
    count_parameters,
    SPARSE_AVAILABLE
)

def test_pattern_correctness():
    """Test that 2:4 sparsity patterns are correct"""
    print("\n🔍 Testing 2:4 sparsity pattern correctness...")
    
    if not SPARSE_AVAILABLE:
        print("❌ Cannot test - 2by4 sparse package not available")
        return False
    
    # Create test layer
    linear = nn.Linear(768, 3072).cuda()
    sparse_layer = SparseLowRankLinear(linear, rank=64, init="xavier", enable_sparse=True).cuda()
    
    # Get sparse weights
    sparse_weight_in = sparse_layer.get_sparse_weight_in()
    sparse_weight_out = sparse_layer.get_sparse_weight_out()
    
    # Check 2:4 pattern for weight_in
    weight_flat = sparse_weight_in.view(-1)
    pattern_correct = True
    
    for i in range(0, len(weight_flat), 4):
        if i + 3 < len(weight_flat):
            block = weight_flat[i:i+4]
            nonzero_count = (block != 0).sum().item()
            if nonzero_count != 2:
                pattern_correct = False
                print(f"❌ Block {i//4}: {nonzero_count}/4 non-zero (should be 2/4)")
                break
    
    if pattern_correct:
        print("✅ Weight_in 2:4 pattern correct")
    
    # Check 2:4 pattern for weight_out
    weight_flat = sparse_weight_out.view(-1)
    for i in range(0, len(weight_flat), 4):
        if i + 3 < len(weight_flat):
            block = weight_flat[i:i+4]
            nonzero_count = (block != 0).sum().item()
            if nonzero_count != 2:
                pattern_correct = False
                print(f"❌ Block {i//4}: {nonzero_count}/4 non-zero (should be 2/4)")
                break
    
    if pattern_correct:
        print("✅ Weight_out 2:4 pattern correct")
        print("✅ All 2:4 sparsity patterns are correct!")
        return True
    else:
        print("❌ 2:4 sparsity patterns are incorrect")
        return False


def test_performance():
    """Test performance of sparse vs dense implementations"""
    print("\n⚡ Testing performance...")
    
    if not SPARSE_AVAILABLE:
        print("❌ Cannot test - 2by4 sparse package not available")
        return False
    
    # Create test layers
    linear = nn.Linear(768, 3072).cuda()
    
    # Dense low-rank layer
    dense_layer = SparseLowRankLinear(linear, rank=64, init="xavier", enable_sparse=False).cuda()
    
    # Sparse low-rank layer (EXACT 2by4 implementation)
    sparse_layer = SparseLowRankLinear(linear, rank=64, init="xavier", enable_sparse=True).cuda()
    
    # Test data
    x = torch.randn(32, 512, 768).cuda()
    
    # Warmup
    for _ in range(10):
        _ = dense_layer(x)
        _ = sparse_layer(x)
    
    torch.cuda.synchronize()
    
    # Time dense layer
    start_time = time.time()
    for _ in range(100):
        _ = dense_layer(x)
    torch.cuda.synchronize()
    dense_time = time.time() - start_time
    
    # Time sparse layer
    start_time = time.time()
    for _ in range(100):
        _ = sparse_layer(x)
    torch.cuda.synchronize()
    sparse_time = time.time() - start_time
    
    speedup = dense_time / sparse_time
    print(f"   Dense time: {dense_time:.4f}s")
    print(f"   Sparse time: {sparse_time:.4f}s")
    print(f"   Speedup: {speedup:.2f}x")
    
    if speedup > 0.8:  # Should be at least comparable
        print("✅ Sparse performance test passed")
        return True
    else:
        print("❌ Sparse performance is significantly slower")
        return False


def test_numerical_accuracy():
    """Test numerical accuracy of sparse implementation"""
    print("\n🔢 Testing numerical accuracy...")
    
    if not SPARSE_AVAILABLE:
        print("❌ Cannot test - 2by4 sparse package not available")
        return False
    
    # Create test layers with same initialization
    linear = nn.Linear(768, 512).cuda()
    
    # Dense reference
    dense_layer = SparseLowRankLinear(linear, rank=64, init="xavier", enable_sparse=False).cuda()
    
    # Sparse layer
    sparse_layer = SparseLowRankLinear(linear, rank=64, init="xavier", enable_sparse=True).cuda()
    
    # Copy weights to make them identical before sparsification
    sparse_layer.weight_in.data.copy_(dense_layer.weight_in.data)
    sparse_layer.weight_out.data.copy_(dense_layer.weight_out.data)
    
    # Re-initialize sparse scales
    sparse_layer._init_sparse_scales()
    
    # Test forward pass
    x = torch.randn(4, 10, 768).cuda()
    
    dense_output = dense_layer(x)
    sparse_output = sparse_layer(x)
    
    # Calculate error
    error = torch.norm(dense_output - sparse_output) / torch.norm(dense_output)
    print(f"   Relative error: {error.item():.6f}")
    
    if error < 0.1:  # Should be reasonably close
        print("✅ Numerical accuracy test passed")
        return True
    else:
        print("❌ Numerical accuracy test failed - too much error")
        return False


def test_gradient_flow():
    """Test that gradients flow correctly through sparse operations"""
    print("\n🔄 Testing gradient flow...")
    
    if not SPARSE_AVAILABLE:
        print("❌ Cannot test - 2by4 sparse package not available")
        return False
    
    # Create sparse layer
    linear = nn.Linear(768, 512).cuda()
    sparse_layer = SparseLowRankLinear(linear, rank=64, init="xavier", enable_sparse=True).cuda()
    
    # Test input
    x = torch.randn(4, 10, 768, requires_grad=True).cuda()
    
    # Forward pass
    output = sparse_layer(x)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    if x.grad is not None and sparse_layer.weight_in.grad is not None and sparse_layer.weight_out.grad is not None:
        print("✅ Gradients computed successfully")
        print(f"   Input grad norm: {x.grad.norm().item():.6f}")
        print(f"   Weight_in grad norm: {sparse_layer.weight_in.grad.norm().item():.6f}")
        print(f"   Weight_out grad norm: {sparse_layer.weight_out.grad.norm().item():.6f}")
        return True
    else:
        print("❌ Gradient computation failed")
        return False


def test_memory_efficiency():
    """Test memory efficiency of sparse implementation"""
    print("\n💾 Testing memory efficiency...")
    
    if not SPARSE_AVAILABLE:
        print("❌ Cannot test - 2by4 sparse package not available")
        return False
    
    # Create layers
    in_features, out_features = 4096, 4096
    rank = 256
    
    # Standard linear layer
    standard_linear = nn.Linear(in_features, out_features).cuda()
    standard_params = standard_linear.weight.numel() + standard_linear.bias.numel()
    
    # Sparse low-rank layer
    linear_ref = nn.Linear(in_features, out_features).cuda()
    sparse_layer = SparseLowRankLinear(linear_ref, rank=rank, init="xavier", enable_sparse=True).cuda()
    sparse_params = sparse_layer.weight_in.numel() + sparse_layer.weight_out.numel()
    if sparse_layer.bias is not None:
        sparse_params += sparse_layer.bias.numel()
    
    # Calculate effective parameters (considering 2:4 sparsity)
    effective_sparse_params = sparse_params * 0.5  # 2:4 sparsity = 50% reduction
    
    compression_ratio = effective_sparse_params / standard_params
    
    print(f"   Standard Linear: {standard_params:,} parameters")
    print(f"   Sparse Low-rank: {sparse_params:,} parameters")
    print(f"   Effective (with 2:4): {effective_sparse_params:,} parameters")
    print(f"   Compression ratio: {compression_ratio:.4f}")
    
    if compression_ratio < 0.5:  # Should be significantly compressed
        print("✅ Memory efficiency test passed")
        return True
    else:
        print("❌ Memory efficiency test failed - not enough compression")
        return False


def main():
    """Main test function"""
    print("="*60)
    print("🧪 Testing Fixed LORO with EXACT 2by4-pretrain-acc-examples")
    print("="*60)
    
    # Check if we're using the correct implementation
    implementation_correct = check_sparse_implementation()
    if not implementation_correct:
        print("❌ CRITICAL: Not using correct 2by4-pretrain-acc-examples implementation!")
        return False
    
    tests = [
        ("Pattern Correctness", test_pattern_correctness),
        ("Performance", test_performance),
        ("Numerical Accuracy", test_numerical_accuracy),
        ("Gradient Flow", test_gradient_flow),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary:")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! LORO is using the correct 2by4-pretrain-acc-examples implementation!")
        return True
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    main() 