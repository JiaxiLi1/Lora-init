#!/usr/bin/env python3
"""
Test 2:4 sparse matmul vs dense matmul speed
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from sparse import matmul

def create_2to4_mask(n_cols, device, dtype=torch.bfloat16):
    """Create a 2:4 sparsity mask"""
    mask = torch.ones(n_cols, dtype=dtype, device=device)
    mask[2::4] = 0  # Zero out every 3rd column in groups of 4
    mask[3::4] = 0  # Zero out every 4th column in groups of 4
    return mask

def benchmark_matmul(M, N, K, device='cuda', dtype=torch.bfloat16, num_iterations=100, warmup=10):
    """
    Benchmark matrix multiplication
    A: [M, N]
    B: [N, K]
    Output: [M, K]
    """
    print(f"\n{'='*60}")
    print(f"Testing matrix multiplication: [{M}, {N}] x [{N}, {K}] = [{M}, {K}]")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"{'='*60}")

    # Create random matrices
    A_dense = torch.randn(M, N, device=device, dtype=dtype)
    B = torch.randn(N, K, device=device, dtype=dtype)

    # Create 2:4 sparse version of A
    mask = create_2to4_mask(N, device, dtype)
    A_sparse = A_dense * mask

    # Count actual sparsity
    sparsity = (A_sparse == 0).float().mean().item()
    print(f"Actual sparsity in A_sparse: {sparsity*100:.1f}%")

    # Test 1: Dense matmul
    print("\n1. Dense matmul (torch.mm):")
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = torch.mm(A_dense, B)

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            result_dense = torch.mm(A_dense, B)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    dense_time = (end_time - start_time) / num_iterations * 1000
    print(f"   Time: {dense_time:.3f} ms")

    # Test 2: 2:4 Sparse matmul using sparse package
    print("\n2. 2:4 Sparse matmul (sparse.matmul):")
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = matmul(A_sparse, B)

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            result_sparse = matmul(A_sparse, B)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    sparse_time = (end_time - start_time) / num_iterations * 1000
    print(f"   Time: {sparse_time:.3f} ms")

    # Test 3: Dense matmul with the sparse matrix (for comparison)
    print("\n3. Dense matmul with sparse matrix (torch.mm on A_sparse):")
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = torch.mm(A_sparse, B)

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            result_sparse_dense = torch.mm(A_sparse, B)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    sparse_dense_time = (end_time - start_time) / num_iterations * 1000
    print(f"   Time: {sparse_dense_time:.3f} ms")

    # Calculate speedup
    print(f"\n📊 Speedup Analysis:")
    print(f"   2:4 Sparse vs Dense: {dense_time/sparse_time:.2f}x")
    print(f"   Dense on sparse vs Dense: {dense_time/sparse_dense_time:.2f}x")

    # Verify correctness (results should be close but not exact due to sparsity)
    diff = torch.norm(result_sparse - result_sparse_dense) / torch.norm(result_sparse_dense)
    print(f"\n✓ Relative difference between sparse.matmul and torch.mm on sparse: {diff:.6f}")

    return dense_time, sparse_time, sparse_dense_time


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    print("🚀 2:4 Sparse Matmul Benchmark")

    # Test different matrix sizes (corresponding to our model sizes)
    test_configs = [
        # (batch*seq, intermediate_size_new, hidden_size) for different models
        ("60M MLP", 4096, 2064, 512),    # 60M: batch*seq=8*512=4096, new_intermediate=2064, hidden=512
        ("130M MLP", 4096, 3072, 768),   # 130M
        ("350M MLP", 4096, 4104, 1024),  # 350M
        ("1B MLP", 4096, 8191, 2048),    # 1B
        ("7B MLP", 4096, 16512, 4096),   # 7B
    ]

    results = {}

    for name, M, N, K in test_configs:
        print(f"\n{'='*70}")
        print(f"🔧 Testing {name}")
        dense_time, sparse_time, sparse_dense_time = benchmark_matmul(
            M, N, K, device, dtype, num_iterations=100
        )
        results[name] = {
            'dense': dense_time,
            'sparse_matmul': sparse_time,
            'sparse_dense': sparse_dense_time,
            'speedup': dense_time / sparse_time
        }

    # Summary table
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Dense (ms)':<12} {'Sparse (ms)':<12} {'Speedup':<10}")
    print(f"{'-'*46}")

    for name, res in results.items():
        print(f"{name:<12} {res['dense']:<12.3f} {res['sparse_matmul']:<12.3f} {res['speedup']:<10.2f}x")

    # Also test with square matrices of different sizes
    print(f"\n{'='*70}")
    print(f"🔧 Testing Square Matrices")
    print(f"{'='*70}")

    square_sizes = [512, 1024, 2048, 4096, 8192]
    square_results = {}

    for size in square_sizes:
        print(f"\n📏 Square matrix [{size}x{size}]")
        dense_time, sparse_time, sparse_dense_time = benchmark_matmul(
            size, size, size, device, dtype, num_iterations=100
        )
        square_results[size] = {
            'dense': dense_time,
            'sparse_matmul': sparse_time,
            'speedup': dense_time / sparse_time
        }

    print(f"\n{'='*70}")
    print(f"📊 SQUARE MATRIX SUMMARY")
    print(f"{'='*70}")
    print(f"{'Size':<10} {'Dense (ms)':<12} {'Sparse (ms)':<12} {'Speedup':<10}")
    print(f"{'-'*44}")

    for size, res in square_results.items():
        print(f"{size:<10} {res['dense']:<12.3f} {res['sparse_matmul']:<12.3f} {res['speedup']:<10.2f}x")


if __name__ == "__main__":
    main()