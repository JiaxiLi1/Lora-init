#!/usr/bin/env python
"""Test script to diagnose and fix CoLA ReLU2 training issue"""
import torch
import torch.nn as nn
import numpy as np

def test_cola_initialization():
    """Test different initialization scales for CoLA with ReLU2"""
    in_dim = 512
    out_dim = 512
    rank = 256
    
    # Test different scale factors
    scales = [0.5, 1.0, 2.0, 4.0]
    
    for scale_mult in scales:
        target_sdv = (in_dim + out_dim) ** (-0.5)
        # Original problematic initialization
        scale_factor = (rank ** (-0.25)) * (target_sdv ** 0.5) * 0.5 * scale_mult
        
        # Initialize weights
        weight_in = torch.randn(in_dim, rank) * scale_factor
        weight_out = torch.randn(out_dim, rank) * scale_factor
        
        # Simulate forward pass with random input
        x = torch.randn(32, 128, in_dim) * 0.1  # Small input
        x_2d = x.view(-1, in_dim)
        
        # Forward through CoLA
        intermediate = torch.mm(x_2d, weight_in)  # y1
        relu_y1 = torch.relu(intermediate)
        y2 = relu_y1 * relu_y1  # ReLU²
        
        # Calculate statistics
        sparsity = (y2 == 0).float().mean().item()
        dead_neurons = (intermediate <= 0).all(dim=0).float().mean().item()
        mean_activation = y2.mean().item()
        std_activation = y2.std().item()
        
        print(f"\nScale multiplier: {scale_mult}")
        print(f"  Sparsity: {sparsity:.3f}")
        print(f"  Dead neurons ratio: {dead_neurons:.3f}")
        print(f"  Mean activation: {mean_activation:.6f}")
        print(f"  Std activation: {std_activation:.6f}")
        print(f"  Actual scale factor: {scale_factor:.6f}")

def proposed_fix_initialization(in_dim, out_dim, rank, weight_in, weight_out, use_relu2=True):
    """Proposed fix for CoLA initialization with ReLU2"""
    with torch.no_grad():
        target_sdv = (in_dim + out_dim) ** (-0.5)
        
        if use_relu2:
            # Use He initialization for ReLU-family activations
            # He et al. suggests std = sqrt(2/fan_in) for ReLU
            # For ReLU², we need slightly larger initialization
            fan_in = in_dim
            he_scale = np.sqrt(2.0 / fan_in)
            
            # Combine with rank-aware scaling
            scale_factor = he_scale * (rank ** (-0.25))
            
            # Add small positive bias to prevent dead neurons
            bias_init = 0.01
        else:
            # Original SiLU initialization
            scale_factor = (rank ** (-0.25)) * (target_sdv ** 0.5)
            bias_init = 0.0
        
        # Initialize weights
        weight_in.data.normal_(0, scale_factor)
        weight_out.data.normal_(0, scale_factor)
        
        # Optional: Add small positive bias to first layer to prevent dead ReLU
        if use_relu2:
            weight_in.data += bias_init
    
    return scale_factor, bias_init

if __name__ == "__main__":
    print("=" * 60)
    print("Testing CoLA initialization with ReLU2")
    print("=" * 60)
    
    test_cola_initialization()
    
    print("\n" + "=" * 60)
    print("Testing proposed fix")
    print("=" * 60)
    
    in_dim = 512
    out_dim = 512 
    rank = 256
    
    weight_in = torch.zeros(in_dim, rank)
    weight_out = torch.zeros(out_dim, rank)
    
    scale_factor, bias_init = proposed_fix_initialization(
        in_dim, out_dim, rank, weight_in, weight_out, use_relu2=True
    )
    
    print(f"\nProposed initialization:")
    print(f"  Scale factor: {scale_factor:.6f}")
    print(f"  Bias init: {bias_init:.4f}")
    
    # Test with random input
    x = torch.randn(32, 128, in_dim) * 0.1
    x_2d = x.view(-1, in_dim)
    
    intermediate = torch.mm(x_2d, weight_in)
    relu_y1 = torch.relu(intermediate)
    y2 = relu_y1 * relu_y1
    
    sparsity = (y2 == 0).float().mean().item()
    dead_neurons = (intermediate <= 0).all(dim=0).float().mean().item()
    
    print(f"  Sparsity: {sparsity:.3f}")
    print(f"  Dead neurons ratio: {dead_neurons:.3f}")
    print(f"  Mean activation: {y2.mean().item():.6f}")
    print(f"  Std activation: {y2.std().item():.6f}")