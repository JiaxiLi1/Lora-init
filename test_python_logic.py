def test_logic():
    """Test the 2:4 logic in pure Python"""
    
    test_cases = [
        [0.5, -0.3, 0.8, -0.1],
        [10.0, -2.0, 7.0, -12.0],
        [1.0, 2.0, 3.0, 4.0],
    ]
    
    for vals in test_cases:
        print(f"\nInput: {vals}")
        
        abs_vals = [abs(v) for v in vals]
        print(f"Absolute: {abs_vals}")
        
        # Count for each element
        counts = []
        for i in range(4):
            count = 0
            for j in range(4):
                if i != j:
                    if i == 0 or i == 2:  # Use >= for first comparison
                        if abs_vals[i] >= abs_vals[j]:
                            count += 1
                    else:  # Use > for others to break ties
                        if abs_vals[i] > abs_vals[j]:
                            count += 1
            counts.append(count)
        
        print(f"Counts: {counts}")
        
        # My original logic
        abs0, abs1, abs2, abs3 = abs_vals
        count0 = int(abs0 >= abs1) + int(abs0 >= abs2) + int(abs0 >= abs3)
        count1 = int(abs1 > abs0) + int(abs1 >= abs2) + int(abs1 >= abs3)
        count2 = int(abs2 > abs0) + int(abs2 > abs1) + int(abs2 >= abs3)
        count3 = int(abs3 > abs0) + int(abs3 > abs1) + int(abs3 > abs2)
        
        print(f"Original logic counts: [{count0}, {count1}, {count2}, {count3}]")
        
        keep = [c >= 2 for c in [count0, count1, count2, count3]]
        print(f"Keep mask: {keep}")
        print(f"Kept values: {[v if k else 0 for v, k in zip(vals, keep)]}")
        
        # Expected
        sorted_idx = sorted(range(4), key=lambda i: abs_vals[i], reverse=True)
        print(f"Expected to keep indices: {sorted_idx[:2]}")

if __name__ == "__main__":
    test_logic()