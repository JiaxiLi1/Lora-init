#!/usr/bin/env python3
"""
Test script for flip rate functionality in both pure Sparse2to4Linear and LORO+Sparse combinations.
"""

import torch
import torch.nn as nn
from sparse_fullrank_linear import (
    Sparse2to4Linear, 
    apply_sparse2to4_to_model,
    enable_flip_rate_tracking_for_model,
    calculate_model_flip_rate
)

def test_sparse2to4_flip_rate():
    """Test flip rate functionality for pure Sparse2to4Linear modules"""
    print("=" * 60)
    print("Testing Sparse2to4Linear Flip Rate Functionality")
    print("=" * 60)
    
    # Create a simple model with Linear layers
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=False)
            self.dense = nn.Linear(64, 64, bias=False)  # This should not be replaced
            
        def forward(self, x):
            x = self.q_proj(x)
            x = self.v_proj(x)
            x = self.dense(x)
            return x
    
    model = TestModel()
    print(f"Original model structure:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: {type(module).__name__}")
    
    # Apply Sparse2to4Linear replacement
    target_modules = ["q_proj", "v_proj"]  # Don't replace 'dense'
    model = apply_sparse2to4_to_model(model, target_modules=target_modules)
    
    print(f"\nAfter applying Sparse2to4Linear:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: {type(module).__name__}")
    
    # Enable flip rate tracking
    enable_flip_rate_tracking_for_model(model, enabled=True)
    
    # Create test input
    x = torch.randn(4, 64)
    
    # First forward pass (should return 0 flip rate since no previous mask)
    print(f"\n--- First Forward Pass ---")
    _ = model(x)
    flip_rates = calculate_model_flip_rate(model)
    print(f"Flip rates: {flip_rates}")
    
    # Second forward pass (should still be 0 since no parameter updates)
    print(f"\n--- Second Forward Pass (no parameter update) ---")
    _ = model(x)
    flip_rates = calculate_model_flip_rate(model)
    print(f"Flip rates: {flip_rates}")
    
    # Simulate parameter update by modifying weights slightly
    print(f"\n--- After simulated parameter update ---")
    with torch.no_grad():
        # Modify some weights to trigger mask changes
        for name, module in model.named_modules():
            if isinstance(module, Sparse2to4Linear):
                # Add small random noise
                module.weight.data += torch.randn_like(module.weight.data) * 0.01
    
    _ = model(x)
    flip_rates = calculate_model_flip_rate(model)
    print(f"Flip rates after weight modification: {flip_rates}")
    
    # Test with flip rate disabled
    print(f"\n--- Testing with flip rate disabled ---")
    enable_flip_rate_tracking_for_model(model, enabled=False)
    _ = model(x)
    flip_rates = calculate_model_flip_rate(model)
    print(f"Flip rates (disabled): {flip_rates}")


def test_loro_sparse_flip_rate():
    """Test flip rate functionality for LORO+Sparse combination"""
    print("\n" + "=" * 60)
    print("Testing LORO+Sparse Flip Rate Functionality")
    print("=" * 60)
    
    try:
        from loro_torch.sparse_overlay import (
            enable_flip_rate_tracking_for_sparse_overlay,
            calculate_sparse_overlay_flip_rate
        )
        
        # Create a mock LORO module for testing
        class MockLoroModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(32, 64))
                self.lora_B = nn.Parameter(torch.randn(64, 32))
                
                # Add sparse scale buffers (simulating what apply_sparse_overlay_on_loro would do)
                self.register_buffer('sparse_scale_A', torch.tensor(1.0))
                self.register_buffer('sparse_scale_B', torch.tensor(1.0))
                
                # Add flip rate tracking attributes
                self.register_buffer('previous_mask_A', None)
                self.register_buffer('previous_mask_B', None)
                self._flip_rate_enabled_A = False
                self._flip_rate_enabled_B = False
                self._first_mask_recorded_A = False
                self._first_mask_recorded_B = False
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = MockLoroModule()
                self.mlp = MockLoroModule()
        
        model = TestModel()
        print(f"Created mock LORO model with sparse overlay simulation")
        
        # Enable flip rate tracking
        enable_flip_rate_tracking_for_sparse_overlay(model, enabled=True)
        
        # Test flip rate calculation
        print(f"\n--- First flip rate calculation ---")
        flip_rates = calculate_sparse_overlay_flip_rate(model)
        print(f"LORO Sparse flip rates: {flip_rates}")
        
        # Simulate parameter updates
        print(f"\n--- After simulated parameter update ---")
        with torch.no_grad():
            model.attention.lora_A.data += torch.randn_like(model.attention.lora_A.data) * 0.01
            model.mlp.lora_B.data += torch.randn_like(model.mlp.lora_B.data) * 0.01
        
        flip_rates = calculate_sparse_overlay_flip_rate(model)
        print(f"LORO Sparse flip rates after update: {flip_rates}")
        
        print("✅ LORO+Sparse flip rate test completed successfully")
        
    except ImportError as e:
        print(f"⚠️ Cannot test LORO+Sparse flip rate: {e}")
        print("   This is expected if SparseOverlayFunction is not properly implemented")


if __name__ == "__main__":
    # Test pure Sparse2to4Linear flip rate
    test_sparse2to4_flip_rate()
    
    # Test LORO+Sparse flip rate
    test_loro_sparse_flip_rate()
    
    print("\n" + "=" * 60)
    print("All flip rate tests completed!")
    print("=" * 60) 