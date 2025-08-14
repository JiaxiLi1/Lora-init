import torch
import triton
import triton.language as tl

@triton.jit
def simple_2to4_kernel(
    a_ptr,
    M, K,
    stride_am, stride_ak,
):
    """Simplified kernel to test 2:4 sparsity on first column only"""
    pid = tl.program_id(0)
    
    # Process groups of 4
    m_base = pid * 4
    if m_base + 4 <= M:
        # First column only (k=0)
        k = 0
        
        # Load 4 values
        ptr0 = a_ptr + (m_base + 0) * stride_am + k * stride_ak
        ptr1 = a_ptr + (m_base + 1) * stride_am + k * stride_ak
        ptr2 = a_ptr + (m_base + 2) * stride_am + k * stride_ak
        ptr3 = a_ptr + (m_base + 3) * stride_am + k * stride_ak
        
        val0 = tl.load(ptr0)
        val1 = tl.load(ptr1)
        val2 = tl.load(ptr2)
        val3 = tl.load(ptr3)
        
        # Get absolute values
        abs0 = tl.abs(val0)
        abs1 = tl.abs(val1)
        abs2 = tl.abs(val2)
        abs3 = tl.abs(val3)
        
        # Count comparisons - convert bools to integers explicitly
        count0 = tl.where(abs0 >= abs1, 1, 0) + tl.where(abs0 >= abs2, 1, 0) + tl.where(abs0 >= abs3, 1, 0)
        count1 = tl.where(abs1 > abs0, 1, 0) + tl.where(abs1 >= abs2, 1, 0) + tl.where(abs1 >= abs3, 1, 0)
        count2 = tl.where(abs2 > abs0, 1, 0) + tl.where(abs2 > abs1, 1, 0) + tl.where(abs2 >= abs3, 1, 0)
        count3 = tl.where(abs3 > abs0, 1, 0) + tl.where(abs3 > abs1, 1, 0) + tl.where(abs3 > abs2, 1, 0)
        
        # Keep top 2
        keep0 = count0 >= 2
        keep1 = count1 >= 2
        keep2 = count2 >= 2
        keep3 = count3 >= 2
        
        # Apply mask
        result0 = tl.where(keep0, val0, 0.0)
        result1 = tl.where(keep1, val1, 0.0)
        result2 = tl.where(keep2, val2, 0.0)
        result3 = tl.where(keep3, val3, 0.0)
        
        # Store results
        tl.store(ptr0, result0)
        tl.store(ptr1, result1)
        tl.store(ptr2, result2)
        tl.store(ptr3, result3)

def test_simple():
    M, K = 16, 4
    device = 'cuda'
    dtype = torch.float32
    
    # Create simple test data
    data = torch.tensor([
        [0.5, 1.0, 1.5, 2.0],    # Group 0: should keep 1.5, 2.0
        [-0.3, -1.2, 0.8, -0.1], # 
        [0.8, 0.2, 0.4, 0.6],    # 
        [-0.1, -0.9, -0.5, -0.7],# 
        
        [10.0, 5.0, 3.0, 1.0],   # Group 1: should keep 10.0, 5.0
        [-2.0, -8.0, -4.0, -6.0],#
        [7.0, 9.0, 11.0, 13.0],  #
        [-12.0, -14.0, -10.0, -8.0], #
        
        [1.0, 1.0, 1.0, 1.0],    # Group 2: all equal - behavior undefined
        [2.0, 2.0, 2.0, 2.0],    #
        [3.0, 3.0, 3.0, 3.0],    #
        [4.0, 4.0, 4.0, 4.0],    #
        
        [0.0, 0.0, 0.0, 0.0],    # Group 3: all zeros
        [0.0, 0.0, 0.0, 0.0],    #
        [0.0, 0.0, 0.0, 0.0],    #
        [0.0, 0.0, 0.0, 0.0],    #
    ], device=device, dtype=dtype)
    
    original = data.clone()
    
    print("Original data (first column):")
    for i in range(0, M, 4):
        print(f"Group {i//4}: {original[i:i+4, 0].cpu().numpy()}")
    
    # Launch kernel (only processes first column)
    grid = (M // 4,)
    simple_2to4_kernel[grid](
        data,
        M, K,
        data.stride(0), data.stride(1),
    )
    torch.cuda.synchronize()
    
    print("\nAfter 2:4 sparsity (first column):")
    for i in range(0, M, 4):
        group_orig = original[i:i+4, 0].cpu().numpy()
        group_result = data[i:i+4, 0].cpu().numpy()
        nonzeros = (group_result != 0).sum()
        print(f"Group {i//4}: {group_result} (nonzeros: {nonzeros}/4)")
        
        # Verify correctness
        abs_vals = [abs(v) for v in group_orig]
        sorted_abs = sorted(enumerate(abs_vals), key=lambda x: x[1], reverse=True)
        expected_indices = [sorted_abs[0][0], sorted_abs[1][0]]
        for j in range(4):
            if j in expected_indices:
                if group_result[j] == 0:
                    print(f"  ERROR: Index {j} should be kept but is zero!")
            else:
                if group_result[j] != 0:
                    print(f"  ERROR: Index {j} should be zero but is {group_result[j]}!")

if __name__ == "__main__":
    test_simple()