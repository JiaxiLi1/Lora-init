#!/usr/bin/env python3

import torch
import torch.nn as nn
from loro_torch.sparse_lowrank_module import SparseLowRankLinear, apply_sparse_lowrank_param, get_sparse_lowrank_param
from loro_torch.loro_optim import LOROAdamW

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        
    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

def test_basic_integration():
    """Test basic sparse lowrank integration"""
    print("=== Testing Basic Integration ===")
    
    model = SimpleModel()
    original_params = sum(p.numel() for p in model.parameters())
    print(f'Original model parameters: {original_params}')

    # Apply sparse lowrank parameterization
    apply_sparse_lowrank_param(
        model, 
        None,  # model_config
        'simple',  # model_type
        'all',  # scope
        32,  # attn_rank
        32,  # mlp_rank
        'xavier',  # init
        enable_sparse=True
    )

    new_params = sum(p.numel() for p in model.parameters())
    print(f'After sparse lowrank parameters: {new_params}')
    print(f'Parameter reduction: {(original_params - new_params) / original_params * 100:.1f}%')

    # Test forward pass
    x = torch.randn(4, 128)
    out = model(x)
    print(f'Model forward pass successful: {out.shape}')
    
    return model

def test_optimizer_integration():
    """Test optimizer integration"""
    print("\n=== Testing Optimizer Integration ===")
    
    model = SimpleModel()
    
    # Apply sparse lowrank parameterization
    apply_sparse_lowrank_param(
        model, 
        None,  # model_config
        'simple',  # model_type
        'all',  # scope
        32,  # attn_rank
        32,  # mlp_rank
        'xavier',  # init
        enable_sparse=True
    )
    
    # Get parameter groups
    param_groups = get_sparse_lowrank_param(model, None)
    print(f'Number of parameter groups: {len(param_groups)}')
    for i, group in enumerate(param_groups):
        print(f'  Group {i}: {len(group["params"])} parameters')
    
    # Create optimizer
    optimizer = LOROAdamW(
        param_groups,
        lr=0.001,
        weight_decay=0.01,
        loro_type="loro",
        model=model,
    )
    
    print("Optimizer created successfully")
    
    # Test training step
    x = torch.randn(4, 128)
    target = torch.randn(4, 32)
    
    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step with use_exact_loro parameter
    optimizer.step(use_exact_loro=False)
    optimizer.zero_grad()
    
    print(f"Training step successful, loss: {loss.item():.4f}")

def test_non_sparse_mode():
    """Test without sparsity enabled"""
    print("\n=== Testing Non-Sparse Mode ===")
    
    model = SimpleModel()
    
    # Apply lowrank parameterization without sparsity
    apply_sparse_lowrank_param(
        model, 
        None,  # model_config
        'simple',  # model_type
        'all',  # scope
        32,  # attn_rank
        32,  # mlp_rank
        'xavier',  # init
        enable_sparse=False  # Disable sparsity
    )
    
    # Test forward pass
    x = torch.randn(4, 128)
    out = model(x)
    print(f'Non-sparse mode forward pass successful: {out.shape}')

if __name__ == "__main__":
    print("Testing LORO + 2:4 Sparse Integration\n")
    
    try:
        model = test_basic_integration()
        test_optimizer_integration()
        test_non_sparse_mode()
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 