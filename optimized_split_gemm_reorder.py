"""
Optimized split-GEMM by reordering columns for contiguous memory access.
Key idea: Reorder columns so sparse columns are contiguous, then use views instead of copies.
"""

import torch
from sparse_fullrank_linear import fake_fp8_mm
from fused_sparsity_ops import sparsity_tracker


class ReorderedSplitGEMM:
    """
    Manager for reordered split-GEMM operations.
    Maintains column permutations to ensure sparse columns are contiguous.
    """
    
    def __init__(self):
        self.permutations = {}
        self.inverse_permutations = {}
        self.sparse_counts = {}
    
    def get_or_create_permutation(self, layer_id, sparse_mask):
        """
        Get or create a permutation that puts sparse columns first.
        """
        if layer_id in self.permutations:
            return (self.permutations[layer_id], 
                    self.inverse_permutations[layer_id],
                    self.sparse_counts[layer_id])
        
        # Create permutation: sparse columns first, then dense
        sparse_indices = torch.where(sparse_mask)[0]
        dense_indices = torch.where(~sparse_mask)[0]
        
        # Concatenate indices
        perm = torch.cat([sparse_indices, dense_indices])
        
        # Create inverse permutation
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm), device=perm.device)
        
        # Store
        self.permutations[layer_id] = perm
        self.inverse_permutations[layer_id] = inv_perm
        self.sparse_counts[layer_id] = len(sparse_indices)
        
        return perm, inv_perm, len(sparse_indices)
    
    def compute_split_gemm_reordered(self, dy1, weight, layer_id):
        """
        Compute split-GEMM with reordered columns for efficiency.
        """
        # Get cached sparsity
        col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
        
        if sparse_mask is None or not sparse_mask.any():
            return torch.mm(dy1, weight)
        
        # Check dimension compatibility
        if dy1.shape[1] != len(sparse_mask):
            # Dimension mismatch - fallback to standard GEMM
            return torch.mm(dy1, weight)
        
        # Get permutation
        perm, inv_perm, num_sparse = self.get_or_create_permutation(layer_id, sparse_mask)
        
        # Reorder columns of dy1 and rows of weight
        # This makes sparse columns contiguous
        dy1_reordered = dy1[:, perm]
        weight_reordered = weight[perm, :]
        
        # Now we can use views instead of copies!
        dy1_sparse = dy1_reordered[:, :num_sparse]
        dy1_dense = dy1_reordered[:, num_sparse:]
        
        weight_sparse = weight_reordered[:num_sparse, :]
        weight_dense = weight_reordered[num_sparse:, :]
        
        # Initialize result
        result = torch.zeros(dy1.shape[0], weight.shape[1], 
                            device=dy1.device, dtype=dy1.dtype)
        
        # Process sparse part with 2:4 sparsity
        if num_sparse > 0:
            from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
            
            # Apply 2:4 sparsity
            dy1_sparse_t = dy1_sparse.t()
            dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_t)
            dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
            
            # Compute with fake_fp8_mm
            result += fake_fp8_mm(dy1_sparse_2to4, weight_sparse, torch.float8_e4m3fn)
        
        # Process dense part
        if dy1_dense.shape[1] > 0:
            result += torch.mm(dy1_dense, weight_dense)
        
        return result


# Global instance
reordered_split_gemm = ReorderedSplitGEMM()


def compute_split_gemm_lowrank_intermediate_reordered(dy1, weight_out1, layer_id):
    """
    Reordered version of split-GEMM for low-rank intermediate computation.
    """
    return reordered_split_gemm.compute_split_gemm_reordered(dy1, weight_out1, layer_id)


def apply_split_gemm_to_dy1_reordered(dy1, layer_id):
    """
    Apply split-GEMM sparsification with reordering.
    """
    # Get cached sparsity
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    
    if sparse_mask is None:
        return dy1
    
    # Get permutation
    perm, inv_perm, num_sparse = reordered_split_gemm.get_or_create_permutation(
        layer_id, sparse_mask)
    
    # Reorder columns
    dy1_reordered = dy1[:, perm]
    
    # Apply 2:4 to sparse part (first num_sparse columns)
    if num_sparse > 0:
        from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity_featurewise
        
        dy1_sparse = dy1_reordered[:, :num_sparse]
        dy1_sparse_t = dy1_sparse.t()
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_t)
        dy1_reordered[:, :num_sparse] = dy1_sparse_2to4_t.t()
    
    # Reorder back - create result tensor and scatter values back
    result = torch.zeros_like(dy1)
    result[:, perm] = dy1_reordered
    
    return result


if __name__ == "__main__":
    print("Testing reordered split-GEMM...")
    
    import time
    
    # Test configuration
    M, K, N = 2048, 768, 256
    device = 'cuda'
    dtype = torch.float16
    
    # Create test data
    dy1 = torch.randn(M, K, device=device, dtype=dtype)
    weight = torch.randn(K, N, device=device, dtype=dtype)
    
    # Create sparse mask (95% sparse)
    sparse_mask = torch.rand(K, device=device) < 0.95
    
    print(f"Configuration: [{M}, {K}] @ [{K}, {N}]")
    print(f"Sparse columns: {sparse_mask.sum().item()}/{K} ({sparse_mask.float().mean()*100:.1f}%)")
    
    # Store in tracker
    layer_id = "test_layer"
    col_sparsity = torch.rand(K, device=device)
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
    
    num_iterations = 100
    
    # Test 1: Original split-GEMM
    from peft_pretraining.split_gemm_utils import compute_split_gemm_lowrank_intermediate
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_original = compute_split_gemm_lowrank_intermediate(dy1, weight, layer_id)
    torch.cuda.synchronize()
    original_time = time.time() - start
    
    # Test 2: Reordered split-GEMM
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_reordered = compute_split_gemm_lowrank_intermediate_reordered(dy1, weight, layer_id)
    torch.cuda.synchronize()
    reordered_time = time.time() - start
    
    # Test 3: Standard GEMM
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_standard = torch.mm(dy1, weight)
    torch.cuda.synchronize()
    standard_time = time.time() - start
    
    print(f"\nResults ({num_iterations} iterations):")
    print(f"  Original split-GEMM: {original_time:.4f}s ({original_time/num_iterations*1000:.3f}ms per call)")
    print(f"  Reordered split-GEMM: {reordered_time:.4f}s ({reordered_time/num_iterations*1000:.3f}ms per call)")
    print(f"  Standard GEMM: {standard_time:.4f}s ({standard_time/num_iterations*1000:.3f}ms per call)")
    print(f"\nSpeedup:")
    print(f"  Reordered vs Original: {original_time/reordered_time:.2f}x")
    print(f"  Overhead vs Standard: {(reordered_time-standard_time)/standard_time*100:.1f}%")
    
    # Verify correctness
    if result_original is not None and result_reordered is not None:
        diff = (result_original - result_reordered).abs().max()
        print(f"\nMax difference: {diff:.6f}")
    
    print("\n✓ Test completed!")