"""
Optimized ReLU² implementation for PyTorch with custom backward.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class OptimizedReLU2Function(Function):
    """
    Optimized ReLU² with custom backward.
    Uses F.relu for better performance than torch.where.
    """
    
    @staticmethod
    def forward(ctx, input):
        # Use F.relu for better performance
        relu_output = torch.nn.functional.relu(input)
        output = relu_output * relu_output
        
        # Save for backward - we only need to know where input > 0
        ctx.save_for_backward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        
        # Gradient of ReLU²(x) is 2*x if x > 0, else 0
        # This is equivalent to 2 * ReLU(x)
        grad_input = 2 * torch.nn.functional.relu(input) * grad_output
        return grad_input


class OptimizedReLU2(nn.Module):
    """Module wrapper for optimized ReLU²."""
    
    def forward(self, x):
        return OptimizedReLU2Function.apply(x)


def benchmark_relu2_implementations():
    """Compare different ReLU² implementations including our optimized version."""
    import time
    
    x = torch.randn(2048, 11008, device='cuda', dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn_like(x)
    
    print("ReLU² Implementation Benchmark (Forward + Backward)")
    print("="*60)
    
    # Method 1: torch.where (current implementation)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        x1 = x.clone().requires_grad_(True)
        y1 = torch.where(x1 > 0, x1 * x1, torch.zeros_like(x1))
        y1.backward(grad_output)
    torch.cuda.synchronize()
    time1 = time.time() - start
    
    # Method 2: Optimized with F.relu
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        x2 = x.clone().requires_grad_(True)
        y2 = OptimizedReLU2Function.apply(x2)
        y2.backward(grad_output)
    torch.cuda.synchronize()
    time2 = time.time() - start
    
    # Method 3: Direct F.relu + square (no custom backward)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        x3 = x.clone().requires_grad_(True)
        y3 = torch.nn.functional.relu(x3) ** 2
        y3.backward(grad_output)
    torch.cuda.synchronize()
    time3 = time.time() - start
    
    print(f"torch.where:           {time1:.3f}s (baseline)")
    print(f"Optimized (F.relu):    {time2:.3f}s ({time1/time2:.2f}x speedup)")
    print(f"F.relu + square:       {time3:.3f}s ({time1/time3:.2f}x speedup)")
    
    # Verify correctness
    x_test = torch.randn(100, 100, device='cuda', dtype=torch.float32, requires_grad=True)
    
    # Forward check
    y_where = torch.where(x_test > 0, x_test * x_test, torch.zeros_like(x_test))
    y_opt = OptimizedReLU2Function.apply(x_test.clone())
    
    assert torch.allclose(y_where, y_opt, atol=1e-6), "Forward results don't match!"
    
    # Backward check
    x1 = torch.randn(100, 100, device='cuda', dtype=torch.float32, requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)
    
    y1 = torch.where(x1 > 0, x1 * x1, torch.zeros_like(x1))
    y2 = OptimizedReLU2Function.apply(x2)
    
    grad_out = torch.randn_like(y1)
    y1.backward(grad_out)
    y2.backward(grad_out)
    
    # Both should produce same gradient
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6), "Gradients don't match!"
    
    print("\n✓ All implementations produce correct results")
    print("✓ Gradients match expected values")
    
    return time1/time2  # Return speedup factor


if __name__ == "__main__":
    if torch.cuda.is_available():
        speedup = benchmark_relu2_implementations()
        print(f"\n🚀 Optimized ReLU² is {speedup:.2f}x faster than torch.where implementation")
    else:
        print("CUDA not available")