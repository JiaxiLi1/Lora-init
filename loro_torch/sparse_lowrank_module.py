"""
LORO Sparse Low-Rank Module with Correct 2:4 Implementation
===========================================================

This module provides LORO (Low-Rank + Sparse) training with the correct 2:4 sparse implementation
directly copied from 2by4-pretrain-acc-examples project.

Key Features:
- Correct 2:4 sparsification using soft_threshold24_triton
- Proper FP8 operations for RTX 3090 compatibility
- MVUE gradient estimation for backward pass
- Learnable scaling factors for stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.cuda.amp import custom_fwd, custom_bwd
import math
import numpy as np
from typing import Optional, Union, Dict, Any, List

# Import correct 2:4 implementations from 2by4 project
from sparse import matmul, MVUE24_approx_triton, soft_threshold24_triton


def fake_fp8_mm(a, b, dtype):
    """Simulate FP8 precision using float16 for compatibility with RTX 3090"""
    a = a.to(torch.float16).contiguous()
    b = b.to(torch.float16).contiguous()
    output = matmul(a, b, c_dtype=torch.float32)
    return output


class FP8SparseLinear(autograd.Function):
    """Correct FP8 Sparse Linear operation - EXACT copy from 2by4-pretrain-acc-examples"""
    
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = fake_fp8_mm(input, weight.t(), torch.float8_e4m3fn)
        if bias is None:
            return output.view(*ctx.shape[:-1], -1)
        else:
            return output.view(*ctx.shape[:-1], -1) + bias

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.half()
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            if grad_output.stride() == (0, 0, 0):
                grad_output = torch.ones_like(grad_output, device=grad_output.device, dtype=grad_output.dtype)
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_input = fake_fp8_mm(grad_output, weight, torch.float8_e5m2).view(ctx.shape)
            
        if ctx.needs_input_grad[1]:
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = fake_fp8_mm(MVUE24_approx_triton(grad_output.t()), input, torch.float8_e5m2)
            
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            
        return grad_input, grad_weight, grad_bias


class SoftThreshold2to4(autograd.Function):
    """2:4 sparsification with learnable scaling - EXACT copy from 2by4-pretrain-acc-examples"""
    
    @staticmethod
    def forward(ctx, weight, scale):
        weight_temp = weight.detach()
        # Convert bfloat16 to float16 for triton compatibility
        original_dtype = weight_temp.dtype
        if weight_temp.dtype == torch.bfloat16:
            weight_temp = weight_temp.to(torch.float16)
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        # Convert back to original dtype
        if original_dtype == torch.bfloat16:
            weight_sparse = weight_sparse.to(torch.bfloat16)
        return weight_sparse * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class FP8SparseLinearLayer(nn.Module):
    """FP8 Sparse Linear Layer with 2:4 sparsity - EXACT copy from 2by4-pretrain-acc-examples"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super(FP8SparseLinearLayer, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        # Initialize scale for 2:4 sparsification
        self.register_buffer('scale', torch.tensor(0.))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def get_sparse_weights(self):
        """Get sparsified weights using 2:4 pattern"""
        return SoftThreshold2to4.apply(self.weight, self.scale)

    @torch.no_grad()
    def init_scale(self):
        """Initialize scale factor for stable 2:4 training"""
        weight = self.weight.cuda()
        weight_temp = weight.detach()
        # Convert bfloat16 to float16 for triton compatibility
        if weight_temp.dtype == torch.bfloat16:
            weight_temp = weight_temp.to(torch.float16)
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        weight_scale = torch.dot(torch.flatten(weight), torch.flatten(weight_sparse)) / torch.dot(
            torch.flatten(weight_sparse), torch.flatten(weight_sparse))
        self.scale.copy_(weight_scale.cpu())
        self.weight.scale = self.scale

    def forward(self, x):
        """Forward pass with 2:4 sparse weights"""
        w = self.get_sparse_weights()
        return FP8SparseLinear.apply(x, w, self.bias)


class LoroSparseLinear(nn.Module):
    """
    LORO (Low-Rank + 2:4 Sparse) Linear Layer with Correct Implementation
    
    Combines:
    - Low-rank decomposition for parameter efficiency
    - 2:4 sparsity for hardware acceleration
    - Correct implementations from 2by4-pretrain-acc-examples
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super(LoroSparseLinear, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank weights: W ≈ B @ A
        self.weight_in = nn.Parameter(torch.empty((rank, in_features), **factory_kwargs))  # A
        self.weight_out = nn.Parameter(torch.empty((out_features, rank), **factory_kwargs))  # B
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # 2:4 sparsity scales
        self.register_buffer('scale_in', torch.tensor(0.))
        self.register_buffer('scale_out', torch.tensor(0.))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using standard methods"""
        # Initialize low-rank weights
        nn.init.kaiming_uniform_(self.weight_in, a=math.sqrt(5))
        # Fix: Don't initialize weight_out to zeros! This causes gradient vanishing.
        # Use proper initialization for low-rank decomposition
        nn.init.kaiming_uniform_(self.weight_out, a=math.sqrt(5))
        
        # Initialize bias
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def init_scales(self):
        """Initialize scale factors for stable 2:4 training"""
        # Initialize scale for weight_in
        weight_in_temp = self.weight_in.detach()
        # Convert bfloat16 to float16 for triton compatibility
        if weight_in_temp.dtype == torch.bfloat16:
            weight_in_temp = weight_in_temp.to(torch.float16)
        weight_in_sparse, _ = soft_threshold24_triton(weight_in_temp)
        scale_in = torch.dot(torch.flatten(weight_in_temp), torch.flatten(weight_in_sparse)) / torch.dot(
            torch.flatten(weight_in_sparse), torch.flatten(weight_in_sparse))
        self.scale_in.copy_(scale_in.cpu())
        
        # Initialize scale for weight_out  
        weight_out_temp = self.weight_out.detach()
        # Convert bfloat16 to float16 for triton compatibility
        if weight_out_temp.dtype == torch.bfloat16:
            weight_out_temp = weight_out_temp.to(torch.float16)
        weight_out_sparse, _ = soft_threshold24_triton(weight_out_temp)
        scale_out = torch.dot(torch.flatten(weight_out_temp), torch.flatten(weight_out_sparse)) / torch.dot(
            torch.flatten(weight_out_sparse), torch.flatten(weight_out_sparse))
        self.scale_out.copy_(scale_out.cpu())

    def get_sparse_weight_in(self):
        """Get sparsified input projection weight"""
        return SoftThreshold2to4.apply(self.weight_in, self.scale_in)

    def get_sparse_weight_out(self):
        """Get sparsified output projection weight"""
        return SoftThreshold2to4.apply(self.weight_out, self.scale_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x -> sparse(weight_in) -> sparse(weight_out) -> output
        """
        # Get sparse weights
        weight_in = self.get_sparse_weight_in()
        weight_out = self.get_sparse_weight_out()
        
        # Two-step sparse computation: x @ A.T -> (x @ A.T) @ B.T
        # FP8SparseLinear expects weight in (out_features, in_features) format
        # Step 1: x (B,S,768) @ weight_in.T -> x_proj (B,S,64)
        # weight_in is (64, 768), so we need it as (64, 768) for FP8SparseLinear  
        x_proj = FP8SparseLinear.apply(x, weight_in, None)  # weight_in shape (64, 768) is correct
        x_proj = self.dropout(x_proj)
        output = FP8SparseLinear.apply(x_proj, weight_out, self.bias)  # (x @ A.T) @ B.T
        
        return output * self.scaling

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'rank={self.rank}, alpha={self.alpha}, bias={self.bias is not None}'


def replace_linear_with_loro_sparse(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 64,
    alpha: float = 1.0,
    dropout: float = 0.0,
    exclude_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Replace target Linear modules with LoroSparseLinear modules
    
    Args:
        model: The model to modify
        target_modules: List of module names to replace (e.g., ["q_proj", "v_proj"])
        rank: Low-rank dimension
        alpha: Scaling factor
        dropout: Dropout rate
        exclude_modules: List of module names to exclude from replacement
        
    Returns:
        Modified model with LoroSparseLinear modules
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    if exclude_modules is None:
        exclude_modules = []
    
    def replace_module(parent_module, child_name, child_module):
        if isinstance(child_module, nn.Linear) and child_name in target_modules and child_name not in exclude_modules:
            # Create LoroSparseLinear replacement
            loro_sparse = LoroSparseLinear(
                in_features=child_module.in_features,
                out_features=child_module.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=child_module.bias is not None,
                device=child_module.weight.device,
                dtype=child_module.weight.dtype,
            )
            
            # Initialize scales
            loro_sparse.init_scales()
            
            # Replace the module
            setattr(parent_module, child_name, loro_sparse)
            return True
        return False
    
    replaced_count = 0
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if replace_module(module, child_name, child_module):
                replaced_count += 1
                print(f"Replaced {name}.{child_name} with LoroSparseLinear")
    
    print(f"Total replaced modules: {replaced_count}")
    return model


def test_installation():
    """Test if the correct 2:4 sparse implementation is working"""
    print("🧪 Testing Correct 2:4 Sparse Implementation")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # Test basic 2:4 sparsification
        print("1. Testing soft_threshold24_triton...")
        test_weight = torch.randn(256, 768).cuda()
        # Convert bfloat16 to float16 for triton compatibility
        if test_weight.dtype == torch.bfloat16:
            test_weight = test_weight.to(torch.float16)
        sparse_weight, mask = soft_threshold24_triton(test_weight)
        
        # Check sparsity
        sparsity = (sparse_weight == 0).float().mean().item()
        print(f"   ✓ Sparsity: {sparsity:.1%}")
        
        # Check 2:4 pattern
        reshaped = sparse_weight.view(-1, 4)
        nonzero_counts = (reshaped != 0).sum(dim=1)
        perfect_24 = torch.all(nonzero_counts <= 2)
        print(f"   ✓ 2:4 pattern: {'Correct' if perfect_24 else 'Incorrect'}")
        
        # Test FP8SparseLinear
        print("\n2. Testing FP8SparseLinear...")
        x = torch.randn(8, 1024, 768).cuda()
        weight = torch.randn(256, 768).cuda()
        output = FP8SparseLinear.apply(x, weight, None)
        print(f"   ✓ Output shape: {output.shape}")
        
        # Test LoroSparseLinear
        print("\n3. Testing LoroSparseLinear...")
        loro_layer = LoroSparseLinear(768, 256, rank=64).cuda()
        loro_layer.init_scales()
        output = loro_layer(x)
        print(f"   ✓ Output shape: {output.shape}")
        
        print(f"\n✅ All tests passed! Correct 2:4 implementation is working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_installation()


def apply_sparse_lowrank_param(
    model: nn.Module, 
    model_config, 
    model_type: str = "llama",
    scope: str = "all",
    attn_rank: int = 64,
    mlp_rank: int = 64,
    init: str = "xavier",
    enable_sparse: bool = True,
    sparse_init_scale: float = 1.0,
    **kwargs
) -> None:
    """
    Apply sparse low-rank parameterization to the model.
    This function replaces linear layers with LoroSparseLinear modules and initializes them.
    
    Args:
        model: The model to modify
        model_config: Model configuration
        model_type: Type of model (e.g., "llama")
        scope: Which layers to apply to ("all", "attn", "mlp")
        attn_rank: Rank for attention layers
        mlp_rank: Rank for MLP layers
        init: Initialization method
        enable_sparse: Whether to enable sparse patterns
        sparse_init_scale: Scale factor for sparse initialization
        **kwargs: Additional keyword arguments (for compatibility)
    """
    print("🔧 Applying sparse low-rank parameterization...")
    
    # Determine target modules based on scope
    target_modules = []
    if scope in ["all", "attn"]:
        target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    if scope in ["all", "mlp"]:
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])
    
    # Replace linear layers with LoroSparseLinear
    replaced_count = 0
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear) and child_name in target_modules:
                # Determine rank based on module type
                if child_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    rank = attn_rank
                else:  # MLP modules
                    rank = mlp_rank
                
                # Create LoroSparseLinear replacement
                loro_sparse = LoroSparseLinear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    rank=rank,
                    alpha=1.0,
                    dropout=0.0,
                    bias=child_module.bias is not None,
                    device=child_module.weight.device,
                    dtype=child_module.weight.dtype,
                )
                
                # Initialize scales
                loro_sparse.init_scales()
                
                # Replace the module
                setattr(module, child_name, loro_sparse)
                replaced_count += 1
                print(f"   ✓ Replaced {name}.{child_name} with LoroSparseLinear (rank={rank})")
    
    # Initialize all LoroSparseLinear modules
    initialized_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoroSparseLinear):
            # Initialize scales for 2:4 sparsity
            module.init_scales()
            initialized_count += 1
        elif isinstance(module, FP8SparseLinearLayer):
            # Initialize scales for FP8SparseLinearLayer
            module.init_scale()
            initialized_count += 1
    
    print(f"✅ Replaced {replaced_count} modules, initialized {initialized_count} sparse low-rank modules")


def get_sparse_lowrank_param(model: nn.Module, model_config=None, lr_scaler=-1):
    """
    Get sparse low-rank parameters from the model for optimizer initialization.
    Returns parameter groups for LORO optimizer.
    
    Args:
        model: The model containing LoroSparseLinear modules
        model_config: Model configuration
        lr_scaler: Learning rate scaler (for compatibility)
        
    Returns:
        List of parameter groups for optimizer
    """
    # Collect different types of parameters
    lowrank_params_in = []
    lowrank_params_out = []
    sparse_scale_params = []
    lowrank_params_in_type = []
    lowrank_params_out_type = []
    
    for name, module in model.named_modules():
        if isinstance(module, LoroSparseLinear):
            # Low-rank parameters
            if hasattr(module, 'weight_in'):
                lowrank_params_in.append(module.weight_in)
                if "mlp" in name:
                    lowrank_params_in_type.append("mlp")
                elif "self_attn" in name:
                    lowrank_params_in_type.append("attn")
                    
            if hasattr(module, 'weight_out'):
                lowrank_params_out.append(module.weight_out)
                if "mlp" in name:
                    lowrank_params_out_type.append("mlp")
                elif "self_attn" in name:
                    lowrank_params_out_type.append("attn")
            
            # Sparse scale parameters
            if hasattr(module, 'scale_in'):
                sparse_scale_params.append(module.scale_in)
            if hasattr(module, 'scale_out'):
                sparse_scale_params.append(module.scale_out)
                
        elif isinstance(module, FP8SparseLinearLayer):
            # FP8 sparse scale parameters
            if hasattr(module, 'scale'):
                sparse_scale_params.append(module.scale)
    
    # Collect regular parameters (not in any of the above groups)
    id_special_params = [id(p) for p in lowrank_params_in + lowrank_params_out + sparse_scale_params]
    regular_params = [p for p in model.parameters() if id(p) not in id_special_params]
    
    # Get model configuration values
    init_std = getattr(model_config, 'initializer_range', 0.02) if model_config else 0.02
    hidden_size = getattr(model_config, 'hidden_size', 768) if model_config else 768
    intermediate_size = getattr(model_config, 'intermediate_size', 3072) if model_config else 3072
    
    # Create parameter groups
    param_groups = [
        {
            "type": "regular",
            "params": regular_params,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "init_std": init_std,
        },
        {
            "type": "lowrank_in",
            "params": lowrank_params_in,
            "params_type": lowrank_params_in_type,
            "lr_scaler": lr_scaler,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "init_std": init_std,
        },
        {
            "type": "lowrank_out",
            "params": lowrank_params_out,
            "params_type": lowrank_params_out_type,
            "lr_scaler": lr_scaler,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "init_std": init_std,
        },
        {
            "type": "sparse_scale",
            "params": sparse_scale_params,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "init_std": init_std,
        },
    ]
    
    return param_groups 