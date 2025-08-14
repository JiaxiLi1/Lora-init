import torch

def test_2to4_logic():
    # Test the 2:4 sparsity logic
    vals = torch.tensor([0.5, -0.3, 0.8, -0.1], dtype=torch.float32)
    print(f"Original values: {vals}")
    
    abs_vals = vals.abs()
    print(f"Absolute values: {abs_vals}")
    
    # Count comparisons for each element
    abs0, abs1, abs2, abs3 = abs_vals[0], abs_vals[1], abs_vals[2], abs_vals[3]
    
    count0 = int(abs0 >= abs1) + int(abs0 >= abs2) + int(abs0 >= abs3)
    count1 = int(abs1 > abs0) + int(abs1 >= abs2) + int(abs1 >= abs3)
    count2 = int(abs2 > abs0) + int(abs2 > abs1) + int(abs2 >= abs3)
    count3 = int(abs3 > abs0) + int(abs3 > abs1) + int(abs3 > abs2)
    
    print(f"Counts: [{count0}, {count1}, {count2}, {count3}]")
    
    # Keep top 2 (those with count >= 2)
    keep = [count0 >= 2, count1 >= 2, count2 >= 2, count3 >= 2]
    print(f"Keep mask: {keep}")
    
    # Apply mask
    result = vals * torch.tensor(keep, dtype=torch.float32)
    print(f"Result: {result}")
    print(f"Non-zeros: {(result != 0).sum().item()}")
    
    # Expected: keep the two largest absolute values (0.8 and 0.5)
    print(f"\nExpected to keep: 0.5 and 0.8 (indices 0 and 2)")

if __name__ == "__main__":
    test_2to4_logic()