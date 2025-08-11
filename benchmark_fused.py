#!/usr/bin/env python3
"""
Benchmark comparing fused kernel vs standard implementation.
Tests both bf16 and fp16 precision.
"""

import torch
import time
import numpy as np
from triton_fused_gemm import triton_matmul_with_sparsity


def benchmark_standard_method(x, w, num_iters=100):
    """Standard method: GEMM then compute sparsity."""
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iters):
        # Forward GEMM
        y = torch.matmul(x, w)
        
        # Apply ReLU²
        y_relu2 = torch.where(y > 0, y * y, torch.zeros_like(y))
        
        # Compute column sparsity (this is the expensive part)
        col_nonzero = (y_relu2 != 0).float().sum(dim=0)
        col_sparsity = 1.0 - col_nonzero / y_relu2.shape[0]
        
        # Determine sparse columns for split-GEMM (95% threshold)
        num_sparse = int(0.95 * col_sparsity.shape[0])
        _, sparse_indices = torch.topk(col_sparsity, num_sparse)
        sparse_mask = torch.zeros_like(col_sparsity, dtype=torch.bool)
        sparse_mask[sparse_indices] = True
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return elapsed


def benchmark_fused_method(x, w, num_iters=100):
    """Fused method: GEMM with sparsity in epilogue."""
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iters):
        # Fused GEMM + ReLU² + sparsity computation
        y, col_sparsity = triton_matmul_with_sparsity(
            x, w, activation='relu2', track_sparsity=True
        )
        
        # Determine sparse columns (same as standard)
        num_sparse = int(0.95 * col_sparsity.shape[0])
        _, sparse_indices = torch.topk(col_sparsity, num_sparse)
        sparse_mask = torch.zeros_like(col_sparsity, dtype=torch.bool)
        sparse_mask[sparse_indices] = True
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return elapsed


def benchmark_pure_gemm(x, w, num_iters=100):
    """Pure GEMM without any sparsity computation."""
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iters):
        y = torch.matmul(x, w)
        y_relu2 = torch.where(y > 0, y * y, torch.zeros_like(y))
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return elapsed


def test_bf16_support():
    """Test if BF16 is supported in fused kernel."""
    print("\n" + "="*60)
    print("Testing BF16 Support")
    print("="*60)
    
    try:
        x = torch.randn(256, 1024, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(1024, 2048, device='cuda', dtype=torch.bfloat16)
        
        # Test fused kernel with bf16
        y, sparsity = triton_matmul_with_sparsity(x, w, activation='relu2')
        
        print(f"✓ BF16 is supported!")
        print(f"  Output dtype: {y.dtype}")
        print(f"  Output shape: {y.shape}")
        print(f"  Average sparsity: {sparsity.mean().item():.4f}")
        return True
    except Exception as e:
        print(f"✗ BF16 not supported: {e}")
        return False


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("="*60)
    print("Performance Benchmark: Fused vs Standard Implementation")
    print("="*60)
    
    configs = [
        (512, 4096, 11008, torch.float16, "FP16 Small"),
        (1024, 4096, 11008, torch.float16, "FP16 Medium"),
        (2048, 4096, 11008, torch.float16, "FP16 Large"),
    ]
    
    # Add BF16 configs if supported
    if test_bf16_support():
        configs.extend([
            (512, 4096, 11008, torch.bfloat16, "BF16 Small"),
            (1024, 4096, 11008, torch.bfloat16, "BF16 Medium"),
            (2048, 4096, 11008, torch.bfloat16, "BF16 Large"),
        ])
    
    results = []
    
    for batch, in_feat, out_feat, dtype, name in configs:
        print(f"\n{name}: [{batch}, {in_feat}] x [{in_feat}, {out_feat}], dtype={dtype}")
        
        # Create test data
        x = torch.randn(batch, in_feat, device='cuda', dtype=dtype)
        w = torch.randn(in_feat, out_feat, device='cuda', dtype=dtype)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(x, w)
            _ = triton_matmul_with_sparsity(x, w, activation='relu2')
        
        # Benchmark
        pure_time = benchmark_pure_gemm(x, w, num_iters=50)
        standard_time = benchmark_standard_method(x, w, num_iters=50)
        fused_time = benchmark_fused_method(x, w, num_iters=50)
        
        # Calculate speedups
        speedup = standard_time / fused_time
        overhead_vs_pure = (fused_time - pure_time) / pure_time * 100
        
        print(f"  Pure GEMM+ReLU²:     {pure_time:.3f}s")
        print(f"  Standard (separate): {standard_time:.3f}s")
        print(f"  Fused (epilogue):    {fused_time:.3f}s")
        print(f"  Speedup:             {speedup:.2f}x")
        print(f"  Overhead vs pure:    {overhead_vs_pure:+.1f}%")
        
        results.append({
            'config': name,
            'speedup': speedup,
            'overhead': overhead_vs_pure
        })
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    avg_speedup = np.mean([r['speedup'] for r in results])
    avg_overhead = np.mean([r['overhead'] for r in results])
    
    print(f"Average speedup over standard method: {avg_speedup:.2f}x")
    print(f"Average overhead vs pure GEMM: {avg_overhead:+.1f}%")
    
    if avg_overhead < 5:
        print("✓ Fused kernel adds minimal overhead (<5%) vs pure GEMM")
    else:
        print(f"⚠ Fused kernel adds {avg_overhead:.1f}% overhead vs pure GEMM")


def test_relu2_implementations():
    """Compare different ReLU² implementations."""
    print("\n" + "="*60)
    print("ReLU² Implementation Comparison")
    print("="*60)
    
    x = torch.randn(2048, 11008, device='cuda', dtype=torch.float32)
    
    # Method 1: torch.where
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        y1 = torch.where(x > 0, x * x, torch.zeros_like(x))
    torch.cuda.synchronize()
    time1 = time.time() - start
    
    # Method 2: mask multiplication
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        mask = (x > 0).float()
        y2 = x * x * mask
    torch.cuda.synchronize()
    time2 = time.time() - start
    
    # Method 3: F.relu then square
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        y3 = torch.nn.functional.relu(x) ** 2
    torch.cuda.synchronize()
    time3 = time.time() - start
    
    print(f"torch.where:        {time1:.3f}s (baseline)")
    print(f"mask multiply:      {time2:.3f}s ({time2/time1:.2f}x)")
    print(f"F.relu + square:    {time3:.3f}s ({time3/time1:.2f}x)")
    
    # Verify correctness
    assert torch.allclose(y1, y2, atol=1e-5)
    assert torch.allclose(y1, y3, atol=1e-5)
    print("✓ All methods produce identical results")


def main():
    """Run all tests and benchmarks."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("Comprehensive Benchmark Suite for Fused GEMM with Sparsity")
    print("="*60)
    
    # Run benchmarks
    run_benchmarks()
    
    # Test ReLU² implementations
    test_relu2_implementations()
    
    print("\n" + "="*60)
    print("Benchmark Complete")
    print("="*60)


if __name__ == "__main__":
    main()