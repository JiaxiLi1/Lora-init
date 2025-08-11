#!/usr/bin/env python3
"""
Test that the optimized ReLU² implementation works correctly in the model.
"""

import torch
import torch.nn.functional as F
from peft_pretraining.modeling_llama import (
    ActivationSparse2to4Function,
    ActivationSparse2to4LowRankFunction,
)
import time


def test_correctness():
    """Test that optimized ReLU² produces correct results."""
    print("Testing Optimized ReLU² Implementation")
    print("="*60)
    
    # Test data
    batch_size = 2
    seq_len = 128
    hidden_size = 256
    intermediate_size = 768
    
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32, requires_grad=True)
    weight1 = torch.randn(intermediate_size, hidden_size, device='cuda', dtype=torch.float32, requires_grad=True)
    weight2 = torch.randn(hidden_size, intermediate_size, device='cuda', dtype=torch.float32, requires_grad=True)
    
    # Forward pass
    y = ActivationSparse2to4Function.apply(
        x, weight1, weight2, None, None,
        "naive", 100, 3, 10, 100, False  # dx_direct_sparse=3 to avoid fused kernel
    )
    
    # Backward pass
    loss = y.sum()
    loss.backward()
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {y.shape}")
    print(f"  Output mean: {y.mean().item():.6f}")
    
    print(f"✓ Backward pass successful")
    print(f"  Input gradient norm: {x.grad.norm().item():.6f}")
    print(f"  Weight1 gradient norm: {weight1.grad.norm().item():.6f}")
    print(f"  Weight2 gradient norm: {weight2.grad.norm().item():.6f}")
    
    # Verify gradients are non-zero
    assert x.grad.norm() > 0, "Input gradient is zero"
    assert weight1.grad.norm() > 0, "Weight1 gradient is zero"
    assert weight2.grad.norm() > 0, "Weight2 gradient is zero"
    
    print("\n✓ All correctness tests passed")


def benchmark_performance():
    """Benchmark the performance improvement."""
    print("\nBenchmarking ReLU² Performance")
    print("="*60)
    
    # Larger size for meaningful benchmark
    batch_size = 8
    seq_len = 512
    hidden_size = 1024
    intermediate_size = 4096
    
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)
    
    # Test standalone ReLU² performance
    print("\nStandalone ReLU² benchmark:")
    
    # Old implementation (torch.where)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y_old = torch.where(x > 0, x * x, torch.zeros_like(x))
    torch.cuda.synchronize()
    time_old = time.time() - start
    
    # New implementation (F.relu)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        relu_x = F.relu(x)
        y_new = relu_x * relu_x
    torch.cuda.synchronize()
    time_new = time.time() - start
    
    speedup = time_old / time_new
    print(f"  Old (torch.where): {time_old:.3f}s")
    print(f"  New (F.relu):      {time_new:.3f}s")
    print(f"  Speedup:           {speedup:.2f}x")
    
    # Verify results match
    y_old = torch.where(x > 0, x * x, torch.zeros_like(x))
    relu_x = F.relu(x)
    y_new = relu_x * relu_x
    assert torch.allclose(y_old, y_new, atol=1e-6), "Results don't match!"
    
    print("\n✓ Results are identical")
    
    return speedup


def test_with_gradient():
    """Test gradient computation with optimized ReLU²."""
    print("\nTesting Gradient Computation")
    print("="*60)
    
    x = torch.randn(100, 100, device='cuda', dtype=torch.float32, requires_grad=True)
    
    # Forward with optimized ReLU²
    relu_x = F.relu(x)
    y = relu_x * relu_x
    
    # Backward
    grad_output = torch.randn_like(y)
    y.backward(grad_output)
    
    # Expected gradient: 2*ReLU(x) * grad_output
    expected_grad = 2 * F.relu(x.detach()) * grad_output
    
    assert torch.allclose(x.grad, expected_grad, atol=1e-6), "Gradient incorrect!"
    
    print("✓ Gradient computation is correct")
    print(f"  Gradient norm: {x.grad.norm().item():.6f}")


def main():
    """Run all tests."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("Testing Optimized ReLU² Integration")
    print("="*60)
    print()
    
    # Run tests
    test_correctness()
    speedup = benchmark_performance()
    test_with_gradient()
    
    print("\n" + "="*60)
    print(f"🎉 All tests passed!")
    print(f"🚀 ReLU² is {speedup:.2f}x faster with F.relu implementation")
    print("="*60)


if __name__ == "__main__":
    main()