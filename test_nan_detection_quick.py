#!/usr/bin/env python3
"""
Quick test for NaN detection enhancement.
This script tests if the enhanced NaN tracking works correctly.
"""

import torch
import torch.nn as nn
from nan_detection_enhanced import NaNTracker, TensorInfo


class SimpleModel(nn.Module):
    """Simple model for testing NaN propagation."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 10)
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        # Introduce NaN intentionally
        x = x / 0  # This will create NaN
        x = self.layer3(x)
        return x


def test_nan_tracker():
    """Test the NaN tracker functionality."""
    print("Testing Enhanced NaN Tracker")
    print("="*60)
    
    # Create model and tracker
    model = SimpleModel()
    tracker = NaNTracker(model, verbose=True)
    
    # Create input
    batch_size = 2
    input_dim = 10
    x = torch.randn(batch_size, input_dim)
    
    # Test tensor analysis
    print("\n1. Testing tensor analysis:")
    info = tracker.analyze_tensor(x, "test_input")
    print(f"   Input tensor info: {info}")
    
    # Test forward pass with NaN
    print("\n2. Testing forward pass with intentional NaN:")
    tracker.register_hooks()
    
    try:
        output = model(x)
        print(f"   Output shape: {output.shape}")
        print(f"   Output has NaN: {torch.isnan(output).any().item()}")
    except Exception as e:
        print(f"   Exception during forward: {e}")
    
    tracker.remove_hooks()
    
    # Test parameter checking
    print("\n3. Testing parameter checking:")
    param_issues = tracker.check_model_parameters()
    if param_issues:
        print(f"   Found {len(param_issues)} parameters with issues")
    else:
        print("   No parameter issues found")
    
    print("\n" + "="*60)
    print("Test completed!")


if __name__ == "__main__":
    test_nan_tracker()