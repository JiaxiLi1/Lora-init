"""
Benchmark cheap argsort vs torch.topk performance.
"""

import torch
import time
from triton_cheap_argsort import fast_threshold_partition


def benchmark_topk_vs_cheap(N=3072, num_iterations=1000):
    """Benchmark torch.topk vs cheap argsort."""
    print("="*60)
    print(f"Benchmarking with N={N} features, {num_iterations} iterations")
    print("="*60)
    
    # Create test data
    col_sparsity = torch.rand(N, device='cuda')
    sparsity_ratio = 0.95
    num_sparse = int(sparsity_ratio * N)
    
    # Warmup
    for _ in range(10):
        _ = torch.topk(col_sparsity, num_sparse)
        _ = fast_threshold_partition(col_sparsity, sparsity_ratio)
    
    # Benchmark torch.topk
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _, indices = torch.topk(col_sparsity, num_sparse)
        sparse_mask_topk = torch.zeros(N, dtype=torch.bool, device='cuda')
        sparse_mask_topk[indices] = True
    torch.cuda.synchronize()
    topk_time = time.time() - start
    
    # Benchmark cheap argsort
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        sparse_mask_cheap = fast_threshold_partition(col_sparsity, sparsity_ratio)
    torch.cuda.synchronize()
    cheap_time = time.time() - start
    
    print(f"\nResults:")
    print(f"  torch.topk time: {topk_time:.4f}s ({topk_time/num_iterations*1000:.3f}ms per call)")
    print(f"  cheap argsort time: {cheap_time:.4f}s ({cheap_time/num_iterations*1000:.3f}ms per call)")
    print(f"  Speedup: {topk_time/cheap_time:.2f}x")
    
    # Verify correctness (they should select approximately the same number)
    sparse_count_topk = sparse_mask_topk.sum().item()
    sparse_count_cheap = sparse_mask_cheap.sum().item()
    print(f"\nSparse columns selected:")
    print(f"  torch.topk: {sparse_count_topk}")
    print(f"  cheap argsort: {sparse_count_cheap}")
    
    return topk_time, cheap_time


def benchmark_different_sizes():
    """Test with different feature sizes."""
    print("\n" + "="*60)
    print("Testing different feature sizes")
    print("="*60)
    
    sizes = [768, 1536, 3072, 6144, 12288]
    speedups = []
    
    for N in sizes:
        print(f"\n--- N = {N} ---")
        topk_time, cheap_time = benchmark_topk_vs_cheap(N, num_iterations=100)
        speedup = topk_time / cheap_time
        speedups.append(speedup)
    
    print("\n" + "="*60)
    print("Summary of speedups:")
    for N, speedup in zip(sizes, speedups):
        print(f"  N={N:5d}: {speedup:.2f}x faster")
    print(f"  Average speedup: {sum(speedups)/len(speedups):.2f}x")
    print("="*60)


def test_in_training_context():
    """Test in a context similar to actual training."""
    print("\n" + "="*60)
    print("Testing in training-like context")
    print("="*60)
    
    batch_size = 8
    seq_len = 256
    hidden_size = 3072
    
    # Simulate forward pass computation
    x = torch.randn(batch_size * seq_len, hidden_size, device='cuda', dtype=torch.float16)
    
    # Simulate getting sparsity (would come from ReLU² computation)
    col_sparsity = (x == 0).float().mean(dim=0)
    
    num_iterations = 100
    
    # Measure overhead in training context
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        # Old way with torch.topk
        num_sparse = int(0.95 * hidden_size)
        _, indices = torch.topk(col_sparsity, num_sparse)
        sparse_mask = torch.zeros(hidden_size, dtype=torch.bool, device='cuda')
        sparse_mask[indices] = True
        # Simulate using the mask
        _ = x[:, sparse_mask]
    torch.cuda.synchronize()
    old_time = time.time() - start
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        # New way with cheap argsort
        sparse_mask = fast_threshold_partition(col_sparsity, 0.95)
        # Simulate using the mask
        _ = x[:, sparse_mask]
    torch.cuda.synchronize()
    new_time = time.time() - start
    
    print(f"Training context results ({num_iterations} iterations):")
    print(f"  Old method (torch.topk): {old_time:.4f}s")
    print(f"  New method (cheap argsort): {new_time:.4f}s")
    print(f"  Speedup: {old_time/new_time:.2f}x")
    print(f"  Time saved per iteration: {(old_time-new_time)/num_iterations*1000:.3f}ms")


if __name__ == "__main__":
    print("Cheap Argsort Performance Benchmark")
    print("="*60)
    
    # Basic benchmark
    benchmark_topk_vs_cheap()
    
    # Test different sizes
    benchmark_different_sizes()
    
    # Test in training context
    test_in_training_context()
    
    print("\n✓ Benchmark completed!")