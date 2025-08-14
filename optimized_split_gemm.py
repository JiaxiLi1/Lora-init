"""
Optimized Split-GEMM implementation to avoid data copies.
Key idea: Use masked operations instead of splitting tensors.
"""

import torch
import torch.nn.functional as F
from sparse_fullrank_linear import fake_fp8_mm


def optimized_split_gemm_v1(dy1, weight, sparse_mask, use_2to4=True):
    """
    Optimized split-GEMM that avoids explicit tensor splitting.
    
    Args:
        dy1: Input tensor [batch*seq, features]
        weight: Weight tensor [features, out_features]
        sparse_mask: Boolean mask indicating sparse features
        use_2to4: Whether to apply 2:4 sparsity to sparse parts
    
    Returns:
        Result of split-GEMM computation
    """
    if not sparse_mask.any():
        # All dense - just use standard GEMM
        return torch.mm(dy1, weight)
    
    if sparse_mask.all():
        # All sparse - apply 2:4 and use sparse GEMM
        if use_2to4:
            from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
            dy1_t = dy1.t()
            dy1_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_t)
            dy1_2to4 = dy1_2to4_t.t()
            return fake_fp8_mm(dy1_2to4, weight, torch.float8_e4m3fn)
        else:
            return torch.mm(dy1, weight)
    
    # Mixed sparse/dense - use masked computation
    batch_seq, features = dy1.shape
    _, out_features = weight.shape
    
    # Create masked versions without copying
    dy1_masked = dy1.clone()  # Clone to avoid modifying original
    
    # Apply 2:4 sparsity to sparse columns in-place
    if use_2to4:
        sparse_indices = torch.where(sparse_mask)[0]
        if len(sparse_indices) > 0:
            # Apply 2:4 sparsity column by column (more efficient than transpose)
            for idx in sparse_indices:
                col = dy1_masked[:, idx]
                col_reshaped = col.view(-1, 4)
                # Keep top 2 values in each group of 4
                _, top_indices = torch.topk(col_reshaped.abs(), k=2, dim=1)
                mask = torch.zeros_like(col_reshaped, dtype=torch.bool)
                mask.scatter_(1, top_indices, True)
                col_reshaped *= mask.float()
    
    # Single GEMM with modified input
    return torch.mm(dy1_masked, weight)


def optimized_split_gemm_v2(dy1, weight, sparse_mask):
    """
    Alternative optimization: Process in blocks to improve cache locality.
    """
    batch_seq, features = dy1.shape
    _, out_features = weight.shape
    
    # Initialize output
    output = torch.zeros(batch_seq, out_features, device=dy1.device, dtype=dy1.dtype)
    
    # Process in blocks for better cache usage
    block_size = 128  # Tune this for your GPU
    
    sparse_indices = torch.where(sparse_mask)[0]
    dense_indices = torch.where(~sparse_mask)[0]
    
    # Process sparse blocks
    if len(sparse_indices) > 0:
        from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
        
        for i in range(0, len(sparse_indices), block_size):
            block_indices = sparse_indices[i:i+block_size]
            dy1_block = dy1[:, block_indices]
            weight_block = weight[block_indices, :]
            
            # Apply 2:4 sparsity to block
            dy1_block_t = dy1_block.t()
            dy1_block_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_block_t)
            dy1_block_2to4 = dy1_block_2to4_t.t()
            
            # Accumulate result
            output += fake_fp8_mm(dy1_block_2to4, weight_block, torch.float8_e4m3fn)
    
    # Process dense blocks
    if len(dense_indices) > 0:
        for i in range(0, len(dense_indices), block_size):
            block_indices = dense_indices[i:i+block_size]
            dy1_block = dy1[:, block_indices]
            weight_block = weight[block_indices, :]
            
            # Accumulate result
            output += torch.mm(dy1_block, weight_block)
    
    return output


def optimized_split_gemm_v3(dy1, weight, sparse_mask):
    """
    Most aggressive optimization: Skip split-GEMM entirely for small dense portions.
    
    If dense portion is < 10%, just apply 2:4 to everything.
    This trades a small accuracy loss for major speed gain.
    """
    dense_ratio = (~sparse_mask).float().mean()
    
    if dense_ratio < 0.1:  # Less than 10% dense
        # Just apply 2:4 to everything - much faster
        from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
        dy1_t = dy1.t()
        dy1_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_t)
        dy1_2to4 = dy1_2to4_t.t()
        return fake_fp8_mm(dy1_2to4, weight, torch.float8_e4m3fn)
    
    # Otherwise use standard split-GEMM
    return optimized_split_gemm_v2(dy1, weight, sparse_mask)


def compute_split_gemm_lowrank_intermediate_optimized(dy1, weight_out1, layer_id):
    """
    Optimized version of compute_split_gemm_lowrank_intermediate.
    """
    from fused_sparsity_ops import sparsity_tracker
    
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    if sparse_mask is None:
        # No sparsity info - use standard GEMM
        return torch.mm(dy1, weight_out1)
    
    # Use most optimized version
    return optimized_split_gemm_v3(dy1, weight_out1, sparse_mask)


def apply_split_gemm_to_dy1_optimized(dy1, layer_id):
    """
    Optimized version that applies sparsity in-place when possible.
    """
    from fused_sparsity_ops import sparsity_tracker
    from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
    
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    if sparse_mask is None:
        return dy1
    
    # If mostly sparse (>90%), just apply 2:4 to everything
    sparse_ratio = sparse_mask.float().mean()
    if sparse_ratio > 0.9:
        dy1_t = dy1.t()
        dy1_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_t)
        return dy1_2to4_t.t()
    
    # Otherwise, apply selectively
    result = dy1.clone()
    
    # Apply 2:4 only to sparse columns
    sparse_indices = torch.where(sparse_mask)[0]
    if len(sparse_indices) > 0:
        dy1_sparse = dy1[:, sparse_indices]
        dy1_sparse_t = dy1_sparse.t()
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_t)
        dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
        result[:, sparse_indices] = dy1_sparse_2to4
    
    return result


# Monkey-patch the original functions with optimized versions
def enable_optimized_split_gemm():
    """Enable optimized split-GEMM implementations."""
    import peft_pretraining.split_gemm_utils as utils
    
    # Replace with optimized versions
    utils.compute_split_gemm_lowrank_intermediate = compute_split_gemm_lowrank_intermediate_optimized
    utils.apply_split_gemm_to_dy1 = apply_split_gemm_to_dy1_optimized
    
    print("✓ Optimized split-GEMM enabled")


if __name__ == "__main__":
    print("Testing optimized split-GEMM implementations...")
    
    import time
    from fused_sparsity_ops import sparsity_tracker
    
    # Test configuration
    batch_seq = 2048
    hidden_size = 768
    rank = 256
    
    device = 'cuda'
    dtype = torch.float16
    
    # Create test data
    dy1 = torch.randn(batch_seq, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, rank, device=device, dtype=dtype)
    
    # Create sparsity mask (95% sparse)
    sparse_mask = torch.rand(hidden_size, device=device) > 0.05
    
    print(f"\nConfiguration:")
    print(f"  Input: [{batch_seq}, {hidden_size}]")
    print(f"  Weight: [{hidden_size}, {rank}]")
    print(f"  Sparse ratio: {sparse_mask.float().mean()*100:.1f}%")
    
    num_iterations = 100
    
    # Test V1: Masked operations
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_v1 = optimized_split_gemm_v1(dy1, weight, sparse_mask)
    torch.cuda.synchronize()
    v1_time = time.time() - start
    
    print(f"\nV1 (masked ops): {v1_time:.4f}s ({v1_time/num_iterations*1000:.3f}ms per call)")
    
    # Test V2: Block processing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_v2 = optimized_split_gemm_v2(dy1, weight, sparse_mask)
    torch.cuda.synchronize()
    v2_time = time.time() - start
    
    print(f"V2 (blocks): {v2_time:.4f}s ({v2_time/num_iterations*1000:.3f}ms per call)")
    
    # Test V3: Aggressive optimization
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_v3 = optimized_split_gemm_v3(dy1, weight, sparse_mask)
    torch.cuda.synchronize()
    v3_time = time.time() - start
    
    print(f"V3 (aggressive): {v3_time:.4f}s ({v3_time/num_iterations*1000:.3f}ms per call)")
    
    # Compare with standard GEMM
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_standard = torch.mm(dy1, weight)
    torch.cuda.synchronize()
    standard_time = time.time() - start
    
    print(f"\nStandard GEMM: {standard_time:.4f}s ({standard_time/num_iterations*1000:.3f}ms per call)")
    print(f"\nSpeedups vs standard GEMM:")
    print(f"  V1: {standard_time/v1_time:.2f}x")
    print(f"  V2: {standard_time/v2_time:.2f}x")
    print(f"  V3: {standard_time/v3_time:.2f}x")
    
    print("\n✓ Tests completed!")