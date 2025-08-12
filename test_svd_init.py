#!/usr/bin/env python
"""Test SVD initialization vs cola_init for ReLU2"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def test_svd_vs_cola_init():
    """Compare SVD and cola_init initialization for CoLA with ReLU2"""
    in_dim = 512
    out_dim = 512
    rank = 256
    batch_size = 32
    seq_len = 128
    
    # Create a pretrained weight matrix (simulate a real model)
    # Use Kaiming initialization as baseline
    original_weight = torch.randn(out_dim, in_dim) * (2.0 / in_dim) ** 0.5
    
    print("=" * 60)
    print("Testing different initialization methods with ReLU²")
    print("=" * 60)
    
    # Test input
    x = torch.randn(batch_size, seq_len, in_dim) * 0.1
    x_2d = x.view(-1, in_dim)
    
    # Method 1: SVD initialization
    print("\n1. SVD Initialization:")
    U, S, Vh = torch.linalg.svd(original_weight.to(torch.float32), full_matrices=False)
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]
    S_sqrt = torch.sqrt(S_k)
    
    weight_in_svd = (Vh_k.T * S_sqrt)
    weight_out_svd = (U_k * S_sqrt)
    
    # Forward pass with SVD weights
    intermediate_svd = torch.mm(x_2d, weight_in_svd)
    relu_y1_svd = F.relu(intermediate_svd)
    y2_svd = relu_y1_svd * relu_y1_svd
    
    print(f"  Weight_in norm: {weight_in_svd.norm():.3f}")
    print(f"  Weight_out norm: {weight_out_svd.norm():.3f}")
    print(f"  Singular values (first 5): {S_k[:5].tolist()}")
    print(f"  Intermediate mean: {intermediate_svd.mean():.6f}")
    print(f"  Intermediate std: {intermediate_svd.std():.6f}")
    print(f"  ReLU² sparsity: {(y2_svd == 0).float().mean():.3f}")
    print(f"  ReLU² mean: {y2_svd.mean():.6f}")
    print(f"  ReLU² std: {y2_svd.std():.6f}")
    
    # Method 2: cola_init with more_activation_relu2=True
    print("\n2. CoLA Init (with ReLU² scaling):")
    target_sdv = (in_dim + out_dim) ** (-0.5)
    scale_factor = (rank ** (-0.25)) * (target_sdv ** 0.5) * 0.5  # Note the 0.5!
    
    weight_in_cola = torch.randn(in_dim, rank) * scale_factor
    weight_out_cola = torch.randn(out_dim, rank) * scale_factor
    
    # Forward pass with cola weights
    intermediate_cola = torch.mm(x_2d, weight_in_cola)
    relu_y1_cola = F.relu(intermediate_cola)
    y2_cola = relu_y1_cola * relu_y1_cola
    
    print(f"  Scale factor: {scale_factor:.6f}")
    print(f"  Weight_in norm: {weight_in_cola.norm():.3f}")
    print(f"  Weight_out norm: {weight_out_cola.norm():.3f}")
    print(f"  Intermediate mean: {intermediate_cola.mean():.6f}")
    print(f"  Intermediate std: {intermediate_cola.std():.6f}")
    print(f"  ReLU² sparsity: {(y2_cola == 0).float().mean():.3f}")
    print(f"  ReLU² mean: {y2_cola.mean():.6f}")
    print(f"  ReLU² std: {y2_cola.std():.6f}")
    
    # Method 3: cola_init without the 0.5 scaling
    print("\n3. CoLA Init (without extra 0.5 scaling):")
    scale_factor_fixed = (rank ** (-0.25)) * (target_sdv ** 0.5)  # No 0.5
    
    weight_in_fixed = torch.randn(in_dim, rank) * scale_factor_fixed
    weight_out_fixed = torch.randn(out_dim, rank) * scale_factor_fixed
    
    # Forward pass with fixed weights
    intermediate_fixed = torch.mm(x_2d, weight_in_fixed)
    relu_y1_fixed = F.relu(intermediate_fixed)
    y2_fixed = relu_y1_fixed * relu_y1_fixed
    
    print(f"  Scale factor: {scale_factor_fixed:.6f}")
    print(f"  Weight_in norm: {weight_in_fixed.norm():.3f}")
    print(f"  Weight_out norm: {weight_out_fixed.norm():.3f}")
    print(f"  Intermediate mean: {intermediate_fixed.mean():.6f}")
    print(f"  Intermediate std: {intermediate_fixed.std():.6f}")
    print(f"  ReLU² sparsity: {(y2_fixed == 0).float().mean():.3f}")
    print(f"  ReLU² mean: {y2_fixed.mean():.6f}")
    print(f"  ReLU² std: {y2_fixed.std():.6f}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print("SVD initialization preserves the scale of the original weights")
    print("and should work well with ReLU² without adjustment.")
    print("\nThe cola_init with 0.5 scaling causes values to be too small,")
    print("leading to vanishing gradients and eventual NaN loss.")

if __name__ == "__main__":
    test_svd_vs_cola_init()