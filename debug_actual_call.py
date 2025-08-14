"""
Debug the actual function calls to find the error.
"""

import torch
import sys
sys.path.append('/home/rtx3090/code_jiaxi/LORO-main_temp')

from peft_pretraining.split_gemm_utils import compute_split_gemm_lowrank_intermediate
from fused_sparsity_ops import sparsity_tracker


def debug_actual_call():
    print("Debugging actual function call...")
    print("="*60)
    
    # Configuration
    batch_seq = 2048
    hidden_size = 768
    rank = 256
    
    device = 'cuda'
    dtype = torch.float16
    
    # Create test data
    dy1 = torch.randn(batch_seq, hidden_size, device=device, dtype=dtype)
    weight_out1 = torch.randn(hidden_size, rank, device=device, dtype=dtype)
    
    print(f"dy1 shape: {dy1.shape}")
    print(f"weight_out1 shape: {weight_out1.shape}")
    
    # Create and store sparse mask
    sparse_mask = torch.rand(hidden_size, device=device) < 0.95
    layer_id = "test_layer"
    col_sparsity = torch.rand(hidden_size, device=device)
    
    print(f"\nSparse mask shape: {sparse_mask.shape}")
    print(f"Sparse count: {sparse_mask.sum().item()}")
    
    # Store in tracker
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
    
    # Verify storage
    retrieved_sparsity, retrieved_mask = sparsity_tracker.get_sparsity(layer_id)
    print(f"\nRetrieved mask shape: {retrieved_mask.shape if retrieved_mask is not None else 'None'}")
    print(f"Masks match: {torch.equal(sparse_mask, retrieved_mask) if retrieved_mask is not None else 'N/A'}")
    
    # Call the function
    print("\nCalling compute_split_gemm_lowrank_intermediate...")
    try:
        result = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, layer_id)
        print(f"✓ Success! Result shape: {result.shape}")
    except Exception as e:
        print(f"✗ Failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Debug completed")


if __name__ == "__main__":
    debug_actual_call()