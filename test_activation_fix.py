#!/usr/bin/env python3
"""
测试relu2激活函数是否正确注册
"""

import torch
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after adding path
from transformers.activations import ACT2FN

def test_relu2_activation():
    """测试relu2激活函数"""
    print("🧪 Testing relu2 activation function registration")
    
    # Test that relu2 is registered
    assert "relu2" in ACT2FN, "relu2 should be registered in ACT2FN"
    print("✅ relu2 is registered in ACT2FN")
    
    # Create instance
    relu2_fn = ACT2FN["relu2"]()
    print("✅ relu2 instance created successfully")
    
    # Test input
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Test forward pass
    output = relu2_fn(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 4.0])
    
    assert torch.allclose(output, expected), f"Expected {expected}, got {output}"
    print(f"✅ relu2 forward pass works correctly: {output}")
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    test_relu2_activation() 