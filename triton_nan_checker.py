"""
Triton kernel NaN checker - Add NaN checks to Triton kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def check_nan_kernel(
    tensor_ptr,
    result_ptr,  # Output: 0 if no NaN, 1 if NaN found
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Check if a tensor contains NaN values."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    values = tl.load(tensor_ptr + offsets, mask=mask)
    
    # Check for NaN (NaN != NaN)
    has_nan = values != values
    
    # If any NaN found in this block, set result to 1
    if tl.sum(has_nan.to(tl.int32)) > 0:
        tl.atomic_max(result_ptr, 1)


def check_tensor_for_nan_triton(tensor):
    """Check if a tensor contains NaN using Triton kernel."""
    if not tensor.is_cuda:
        return torch.isnan(tensor).any().item()
    
    result = torch.zeros(1, dtype=torch.int32, device=tensor.device)
    n_elements = tensor.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    flat_tensor = tensor.flatten()
    check_nan_kernel[grid](
        flat_tensor,
        result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result.item() > 0


def wrap_triton_kernel_with_nan_check(kernel_func, kernel_name="triton_kernel"):
    """Wrap a Triton kernel to check for NaN in inputs and outputs."""
    def wrapped(*args, **kwargs):
        # Check inputs for NaN
        input_has_nan = False
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and torch.is_floating_point(arg):
                if check_tensor_for_nan_triton(arg):
                    print(f"\n⚠️ NaN detected in {kernel_name} input {i}")
                    print(f"   Shape: {list(arg.shape)}, dtype: {arg.dtype}")
                    input_has_nan = True
        
        # Run kernel
        result = kernel_func(*args, **kwargs)
        
        # Check output for NaN
        if not input_has_nan:  # Only check if inputs were clean
            if isinstance(result, torch.Tensor) and torch.is_floating_point(result):
                if check_tensor_for_nan_triton(result):
                    print(f"\n🔴 NaN GENERATED in {kernel_name}!")
                    print(f"   Output shape: {list(result.shape)}, dtype: {result.dtype}")
                    
                    # Try to identify which part of the kernel caused NaN
                    analyze_kernel_nan_cause(kernel_name, args, result)
        
        return result
    
    return wrapped


def analyze_kernel_nan_cause(kernel_name, args, result):
    """Analyze why a Triton kernel produced NaN."""
    print(f"\n[Analyzing {kernel_name} for NaN cause...]")
    
    if "split_gemm" in kernel_name.lower():
        print("  Possible causes in split_gemm:")
        print("  1. Invalid 2:4 sparsity pattern selection")
        print("  2. Out-of-bounds memory access")
        print("  3. Numerical overflow in accumulation")
        
        # Check input ranges
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and torch.is_floating_point(arg):
                if not torch.isnan(arg).any():
                    max_val = arg.abs().max().item()
                    if max_val > 1e10:
                        print(f"  ⚠️ Large values in input {i}: max abs = {max_val:.4e}")
    
    elif "2to4" in kernel_name.lower() or "sparsity" in kernel_name.lower():
        print("  Possible causes in 2:4 sparsity kernel:")
        print("  1. Incorrect comparison logic for selecting top-2 values")
        print("  2. Division by zero in scaling")
        print("  3. Boundary condition errors")
        
        # Check if input has special patterns
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            tensor = args[0]
            if not torch.isnan(tensor).any():
                # Check for all-zero rows/columns that might cause issues
                if len(tensor.shape) == 2:
                    zero_rows = (tensor == 0).all(dim=1).sum().item()
                    zero_cols = (tensor == 0).all(dim=0).sum().item()
                    if zero_rows > 0:
                        print(f"  ⚠️ {zero_rows} all-zero rows detected")
                    if zero_cols > 0:
                        print(f"  ⚠️ {zero_cols} all-zero columns detected")


def add_debug_to_split_gemm_kernel():
    """Add debugging to the split_gemm kernel."""
    import triton_split_gemm_nocopy
    
    original_kernel = triton_split_gemm_nocopy.split_gemm_nocopy
    
    def debug_split_gemm(dy1, weight, sparse_mask):
        """Debug version of split_gemm with NaN checking."""
        print("\n[DEBUG] split_gemm_nocopy called")
        print(f"  dy1 shape: {dy1.shape}, dtype: {dy1.dtype}")
        print(f"  weight shape: {weight.shape}, dtype: {weight.dtype}")
        
        # Check inputs
        dy1_has_nan = torch.isnan(dy1).any().item()
        weight_has_nan = torch.isnan(weight).any().item()
        
        if dy1_has_nan:
            print("  ⚠️ dy1 contains NaN!")
        if weight_has_nan:
            print("  ⚠️ weight contains NaN!")
        
        if not dy1_has_nan and not weight_has_nan:
            print(f"  dy1 range: [{dy1.min().item():.4e}, {dy1.max().item():.4e}]")
            print(f"  weight range: [{weight.min().item():.4e}, {weight.max().item():.4e}]")
        
        # Check sparse mask
        if sparse_mask is not None:
            print(f"  sparse_mask shape: {sparse_mask.shape}")
            sparse_count = sparse_mask.sum().item() if sparse_mask.numel() > 0 else 0
            print(f"  Number of sparse columns: {sparse_count}")
        else:
            print("  sparse_mask is None")
        
        # Call original kernel
        result = original_kernel(dy1, weight, sparse_mask)
        
        # Check output
        if torch.isnan(result).any().item():
            print("  🔴 OUTPUT CONTAINS NaN!")
            nan_count = torch.isnan(result).sum().item()
            print(f"     NaN count: {nan_count} / {result.numel()}")
            
            # Find first NaN location
            nan_indices = torch.where(torch.isnan(result))
            if len(nan_indices[0]) > 0:
                first_row = nan_indices[0][0].item()
                first_col = nan_indices[1][0].item() if len(nan_indices) > 1 else 0
                print(f"     First NaN at [{first_row}, {first_col}]")
                
                # Check corresponding input values
                if sparse_mask is not None and first_col < sparse_mask.shape[0]:
                    is_sparse_col = sparse_mask[first_col].item() if sparse_mask.numel() > first_col else False
                    print(f"     Column {first_col} is {'sparse' if is_sparse_col else 'dense'}")
        else:
            print(f"  ✅ Output clean, range: [{result.min().item():.4e}, {result.max().item():.4e}]")
        
        return result
    
    # Replace the function
    triton_split_gemm_nocopy.split_gemm_nocopy = debug_split_gemm
    return original_kernel  # Return original for restoration


def check_2to4_sparsity_kernel():
    """Test the 2:4 sparsity kernel for NaN generation."""
    print("\n[Testing 2:4 sparsity kernel...]")
    
    # Test cases that might cause NaN
    test_cases = [
        # All zeros
        torch.zeros(4, 4, dtype=torch.bfloat16, device='cuda'),
        # Very small values
        torch.full((4, 4), 1e-10, dtype=torch.bfloat16, device='cuda'),
        # Mixed signs
        torch.tensor([[-1, 2, -3, 4]], dtype=torch.bfloat16, device='cuda').repeat(4, 1),
        # Large values
        torch.full((4, 4), 1e10, dtype=torch.bfloat16, device='cuda'),
    ]
    
    for i, test_tensor in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"  Input: {test_tensor[0].tolist()}")
        
        # Apply 2:4 sparsity
        from peft_pretraining.modeling_llama import apply_naive_2to4_sparsity
        result = apply_naive_2to4_sparsity(test_tensor)
        
        if torch.isnan(result).any():
            print(f"  🔴 NaN detected in output!")
        else:
            print(f"  ✅ Output: {result[0].tolist()}")