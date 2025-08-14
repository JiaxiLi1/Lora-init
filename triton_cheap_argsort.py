"""
Cheap argsort kernel for partitioning features into sparse and dense.
Instead of full sorting, we just need to partition based on a threshold.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def partition_by_sparsity_kernel(
    col_sparsity_ptr,
    sparse_mask_ptr,
    sparse_indices_ptr,
    dense_indices_ptr,
    n_sparse_ptr,
    n_dense_ptr,
    N,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for partitioning columns into sparse and dense based on sparsity threshold.
    This is much cheaper than full sorting as we only need a binary partition.
    """
    pid = tl.program_id(axis=0)
    
    # Each block processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load sparsity values
    sparsity = tl.load(col_sparsity_ptr + offsets, mask=mask, other=0.0)
    
    # Check if sparse (sparsity > threshold)
    is_sparse = sparsity > threshold
    
    # Store mask
    tl.store(sparse_mask_ptr + offsets, is_sparse, mask=mask)
    
    # Count sparse/dense in this block (for index computation)
    # This is done atomically across blocks
    if mask.any():
        # Use atomic operations to build indices
        for idx in range(BLOCK_SIZE):
            if idx + block_start < N:
                offset = block_start + idx
                is_sparse_val = tl.load(sparse_mask_ptr + offset)
                
                if is_sparse_val:
                    # Atomic increment and get index for sparse
                    sparse_idx = tl.atomic_add(n_sparse_ptr, 1)
                    tl.store(sparse_indices_ptr + sparse_idx, offset)
                else:
                    # Atomic increment and get index for dense
                    dense_idx = tl.atomic_add(n_dense_ptr, 1)
                    tl.store(dense_indices_ptr + dense_idx, offset)


def partition_features_by_sparsity(col_sparsity, threshold=0.95):
    """
    Fast partitioning of features into sparse and dense based on sparsity threshold.
    
    This is much faster than torch.topk as we only need to partition, not sort.
    
    Args:
        col_sparsity: Column sparsity ratios [N]
        threshold: Sparsity threshold (default 0.95 means 95% sparse)
    
    Returns:
        sparse_mask: Boolean mask [N] where True indicates sparse column
        sparse_indices: Indices of sparse columns
        dense_indices: Indices of dense columns
    """
    N = col_sparsity.shape[0]
    device = col_sparsity.device
    
    # Allocate outputs
    sparse_mask = torch.zeros(N, dtype=torch.bool, device=device)
    sparse_indices = torch.zeros(N, dtype=torch.int32, device=device)
    dense_indices = torch.zeros(N, dtype=torch.int32, device=device)
    
    # Counters for number of sparse/dense
    n_sparse = torch.zeros(1, dtype=torch.int32, device=device)
    n_dense = torch.zeros(1, dtype=torch.int32, device=device)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    partition_by_sparsity_kernel[grid](
        col_sparsity,
        sparse_mask,
        sparse_indices,
        dense_indices,
        n_sparse,
        n_dense,
        N,
        threshold,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Trim arrays to actual sizes
    num_sparse = n_sparse.item()
    num_dense = n_dense.item()
    
    sparse_indices = sparse_indices[:num_sparse]
    dense_indices = dense_indices[:num_dense]
    
    return sparse_mask, sparse_indices, dense_indices


def fast_threshold_partition(col_sparsity, sparsity_ratio=0.95):
    """
    Even simpler threshold-based partitioning without any sorting.
    This is the cheapest possible approach.
    
    Args:
        col_sparsity: Column sparsity values [N]
        sparsity_ratio: Ratio of columns to mark as sparse (e.g., 0.95 for 95%)
    
    Returns:
        sparse_mask: Boolean mask indicating sparse columns
    """
    N = col_sparsity.shape[0]
    num_sparse = int(sparsity_ratio * N)
    
    if num_sparse == 0:
        return torch.zeros(N, dtype=torch.bool, device=col_sparsity.device)
    
    # Find threshold using partial sort (much faster than full sort)
    # We only need the k-th value, not the full sorted array
    if num_sparse < N:
        # Use kthvalue which is O(n) instead of O(n log n)
        kth_val = torch.kthvalue(col_sparsity, N - num_sparse + 1)[0]
        sparse_mask = col_sparsity >= kth_val
        
        # Ensure exactly num_sparse columns are marked
        # (handle ties at the threshold)
        if sparse_mask.sum() > num_sparse:
            # Too many marked as sparse due to ties
            # Keep only the first num_sparse
            indices = torch.where(sparse_mask)[0]
            sparse_mask.fill_(False)
            sparse_mask[indices[:num_sparse]] = True
    else:
        # All columns are sparse
        sparse_mask = torch.ones(N, dtype=torch.bool, device=col_sparsity.device)
    
    return sparse_mask


def adaptive_threshold_partition(col_sparsity, target_sparsity_ratio=0.95):
    """
    Adaptive threshold partitioning that finds natural clustering in sparsity values.
    This avoids arbitrary cutoffs and respects the actual sparsity distribution.
    
    Args:
        col_sparsity: Column sparsity values [N] 
        target_sparsity_ratio: Target ratio of sparse columns
        
    Returns:
        sparse_mask: Boolean mask indicating sparse columns
    """
    N = col_sparsity.shape[0]
    
    # Compute simple statistics
    mean_sparsity = col_sparsity.mean()
    std_sparsity = col_sparsity.std()
    
    # Use mean + std as adaptive threshold
    # This naturally separates sparse from dense columns
    adaptive_threshold = mean_sparsity + 0.5 * std_sparsity
    
    # Apply threshold
    sparse_mask = col_sparsity > adaptive_threshold
    
    # If we're too far from target, adjust
    actual_ratio = sparse_mask.float().mean()
    if abs(actual_ratio - target_sparsity_ratio) > 0.1:
        # Fall back to fast_threshold_partition
        return fast_threshold_partition(col_sparsity, target_sparsity_ratio)
    
    return sparse_mask


# Test implementation
if __name__ == "__main__":
    print("Testing cheap argsort implementations...")
    print("=" * 60)
    
    # Test data
    N = 3072
    col_sparsity = torch.rand(N, device='cuda')
    
    # Test 1: Triton partition kernel
    print("\n1. Testing Triton partition kernel...")
    sparse_mask, sparse_idx, dense_idx = partition_features_by_sparsity(col_sparsity, 0.5)
    print(f"   Sparse columns: {len(sparse_idx)}, Dense columns: {len(dense_idx)}")
    
    # Test 2: Fast threshold partition
    print("\n2. Testing fast threshold partition...")
    import time
    
    # Benchmark vs torch.topk
    start = time.time()
    for _ in range(100):
        sparse_mask_fast = fast_threshold_partition(col_sparsity, 0.95)
    fast_time = time.time() - start
    
    start = time.time()
    for _ in range(100):
        num_sparse = int(0.95 * N)
        _, indices = torch.topk(col_sparsity, num_sparse)
        sparse_mask_topk = torch.zeros(N, dtype=torch.bool, device='cuda')
        sparse_mask_topk[indices] = True
    topk_time = time.time() - start
    
    print(f"   Fast partition time: {fast_time:.4f}s")
    print(f"   Torch.topk time: {topk_time:.4f}s")
    print(f"   Speedup: {topk_time/fast_time:.2f}x")
    
    # Test 3: Adaptive threshold
    print("\n3. Testing adaptive threshold partition...")
    sparse_mask_adaptive = adaptive_threshold_partition(col_sparsity, 0.95)
    print(f"   Sparse ratio: {sparse_mask_adaptive.float().mean():.2%}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed!")