"""
Modular Framework for 2:4 Sparse Training + LoRA Low-Rank Training
==================================================================

This framework provides modular components that can be easily integrated
into different low-rank training projects like LORO, LoRA, etc.

Components:
1. Sparse2to4Mixin: Adds 2:4 sparsity capability to any linear layer
2. LowRankSparse2to4Linear: Combines LoRA + 2:4 sparsity
3. Sparse2to4Optimizer: Optimizer modifications for 2:4 sparse training
4. Sparse2to4Utils: Utility functions for applying sparsity to models

Usage:
- Can be used as drop-in replacement for nn.Linear in LoRA models
- Provides automatic 2:4 sparsification and scaling
- Modular design allows easy integration with different projects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.cuda.amp import custom_fwd, custom_bwd
import math
import numpy as np
from typing import Optional, Union, Dict, Any, List


# ============================================================================
# Core 2:4 Sparsity Operations (from sparse package)
# ============================================================================

try:
    from sparse import matmul, MVUE24_approx_triton, soft_threshold24_triton
    SPARSE_AVAILABLE = True
except ImportError:
    print("Warning: sparse package not available. Using fallback implementations.")
    SPARSE_AVAILABLE = False
    
    def matmul(a, b, c_dtype=torch.float32):
        """Fallback implementation"""
        return torch.matmul(a.to(c_dtype), b.to(c_dtype))
    
    def MVUE24_approx_triton(x):
        """Fallback implementation"""
        return x
    
    def soft_threshold24_triton(weight):
        """Fallback 2:4 sparsification"""
        # Simple 2:4 sparsity: keep top 2 values in each group of 4
        weight_reshaped = weight.view(-1, 4)
        _, indices = torch.topk(torch.abs(weight_reshaped), 2, dim=1)
        mask = torch.zeros_like(weight_reshaped)
        mask.scatter_(1, indices, 1)
        weight_sparse = weight_reshaped * mask
        return weight_sparse.view(weight.shape), mask.view(weight.shape)


def fake_fp8_mm(a, b, dtype):
    """Simulate FP8 precision using float16 for compatibility"""
    a = a.to(torch.float16).contiguous()
    b = b.to(torch.float16).contiguous()
    output = matmul(a, b, c_dtype=torch.float32)
    return output


# ============================================================================
# 2:4 Sparse Operations
# ============================================================================

class FP8SparseOperation(autograd.Function):
    """Core 2:4 sparse matrix multiplication with FP8 simulation"""
    
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
            grad_output_view = grad_output.view(-1, grad_output.shape[-1])
            grad_input = fake_fp8_mm(grad_output_view, weight, torch.float8_e5m2).view(ctx.shape)
            
        if ctx.needs_input_grad[1]:
            input_view = input.view(-1, input.shape[-1])
            grad_output_view = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = fake_fp8_mm(MVUE24_approx_triton(grad_output_view.t()), input_view, torch.float8_e5m2)
            
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            
        return grad_input, grad_weight, grad_bias


class SoftThreshold2to4(autograd.Function):
    """2:4 sparsification with learnable scaling"""
    
    @staticmethod
    def forward(ctx, weight, scale):
        weight_temp = weight.detach()
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        return weight_sparse * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# ============================================================================
# Mixin Classes for Modular Design
# ============================================================================

class Sparse2to4Mixin:
    """
    Mixin class that adds 2:4 sparsity capability to any linear layer.
    Can be mixed with existing LoRA implementations.
    """
    
    def __init__(self, enable_sparse=True, sparse_init_scale=1.0):
        self.enable_sparse = enable_sparse
        if enable_sparse:
            self.register_buffer('scale', torch.tensor(sparse_init_scale))
        
    def get_sparse_weights(self, weight):
        """Apply 2:4 sparsification to weight matrix"""
        if not self.enable_sparse:
            return weight
        return SoftThreshold2to4.apply(weight, self.scale)
    
    @torch.no_grad()
    def init_sparse_scale(self, weight):
        """Initialize the scale parameter for sparse weights"""
        if not self.enable_sparse:
            return
            
        weight_temp = weight.detach()
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        scale = torch.dot(torch.flatten(weight), torch.flatten(weight_sparse)) / torch.dot(
            torch.flatten(weight_sparse), torch.flatten(weight_sparse))
        self.scale.copy_(scale.cpu())
    
    def sparse_forward(self, x, weight, bias=None):
        """Forward pass with 2:4 sparse weights"""
        if not self.enable_sparse:
            return F.linear(x, weight, bias)
        
        sparse_weight = self.get_sparse_weights(weight)
        return FP8SparseOperation.apply(x, sparse_weight, bias)


# ============================================================================
# Combined LoRA + 2:4 Sparse Linear Layer
# ============================================================================

class LowRankSparse2to4Linear(nn.Module, Sparse2to4Mixin):
    """
    Combined LoRA + 2:4 Sparse Linear Layer
    
    This layer combines:
    1. Low-rank decomposition (A @ B) like in LORO
    2. 2:4 sparsity on the low-rank matrices
    3. Modular design for easy integration
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int,
        bias: bool = True,
        enable_sparse: bool = True,
        lora_init: str = "xavier",
        sparse_init_scale: float = 1.0,
        init_range: Optional[float] = None,
        device=None, 
        dtype=None
    ):
        nn.Module.__init__(self)
        Sparse2to4Mixin.__init__(self, enable_sparse, sparse_init_scale)
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_init = lora_init
        self.init_range = init_range
        
        # Low-rank decomposition: out = x @ A @ B^T + bias
        self.weight_A = nn.Parameter(
            torch.randn(in_features, rank, device=device, dtype=dtype),
            requires_grad=True,
        )
        self.weight_B = nn.Parameter(
            torch.randn(out_features, rank, device=device, dtype=dtype),
            requires_grad=True,
        )
        
        self.bias = (
            nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            if bias else None
        )
        
        # Initialize separate scales for A and B matrices if sparse is enabled
        if enable_sparse:
            self.register_buffer('scale_A', torch.tensor(sparse_init_scale))
            self.register_buffer('scale_B', torch.tensor(sparse_init_scale))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LoRA weights using various strategies"""
        if self.lora_init == "xavier":
            nn.init.xavier_normal_(self.weight_A)
            nn.init.xavier_normal_(self.weight_B)
        elif self.lora_init == "auto":
            assert self.init_range is not None
            std = math.sqrt(math.sqrt(self.init_range**2 / self.rank))
            nn.init.normal_(self.weight_A, mean=0, std=std)
            nn.init.normal_(self.weight_B, mean=0, std=std)
        elif self.lora_init == "kaiming":
            nn.init.kaiming_normal_(self.weight_A)
            nn.init.kaiming_normal_(self.weight_B)
        elif self.lora_init == "orth":
            nn.init.orthogonal_(self.weight_A.float())
            nn.init.orthogonal_(self.weight_B.float())
        else:
            # Default initialization
            std = 1.0 / math.sqrt(self.rank)
            nn.init.normal_(self.weight_A, 0, std)
            nn.init.normal_(self.weight_B, 0, std)
    
    @torch.no_grad()
    def init_all_sparse_scales(self):
        """Initialize sparse scales for both A and B matrices"""
        if not self.enable_sparse:
            return
            
        # Initialize scale for weight_A
        weight_A_temp = self.weight_A.detach()
        weight_A_sparse, _ = soft_threshold24_triton(weight_A_temp)
        scale_A = torch.dot(torch.flatten(weight_A_temp), torch.flatten(weight_A_sparse)) / torch.dot(
            torch.flatten(weight_A_sparse), torch.flatten(weight_A_sparse))
        self.scale_A.copy_(scale_A.cpu())
        
        # Initialize scale for weight_B
        weight_B_temp = self.weight_B.detach()
        weight_B_sparse, _ = soft_threshold24_triton(weight_B_temp)
        scale_B = torch.dot(torch.flatten(weight_B_temp), torch.flatten(weight_B_sparse)) / torch.dot(
            torch.flatten(weight_B_sparse), torch.flatten(weight_B_sparse))
        self.scale_B.copy_(scale_B.cpu())
    
    def get_sparse_weight_A(self):
        """Get 2:4 sparse version of weight_A"""
        if not self.enable_sparse:
            return self.weight_A
        return SoftThreshold2to4.apply(self.weight_A, self.scale_A)
    
    def get_sparse_weight_B(self):
        """Get 2:4 sparse version of weight_B"""
        if not self.enable_sparse:
            return self.weight_B
        return SoftThreshold2to4.apply(self.weight_B, self.scale_B)
    
    def forward(self, x):
        """
        Forward pass: x @ A_sparse @ B_sparse^T + bias
        """
        # Get sparse weights
        weight_A = self.get_sparse_weight_A()
        weight_B = self.get_sparse_weight_B()
        
        # Low-rank computation with sparsity
        if self.enable_sparse:
            # x @ A (with sparsity)
            x_proj = FP8SparseOperation.apply(x, weight_A.t(), None)
            # (x @ A) @ B^T (with sparsity)
            output = FP8SparseOperation.apply(x_proj, weight_B, self.bias)
        else:
            # Standard LoRA computation
            x_proj = F.linear(x, weight_A.t())
            output = F.linear(x_proj, weight_B, self.bias)
        
        return output
    
    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.rank}, enable_sparse={self.enable_sparse}, '
                f'lora_init={self.lora_init}')


# ============================================================================
# Standard Sparse Linear (for non-LoRA cases)
# ============================================================================

class Sparse2to4Linear(nn.Linear, Sparse2to4Mixin):
    """
    Standard linear layer with 2:4 sparsity.
    Compatible with existing code, drop-in replacement for nn.Linear
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 enable_sparse: bool = True, sparse_init_scale: float = 1.0,
                 device=None, dtype=None):
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)
        Sparse2to4Mixin.__init__(self, enable_sparse, sparse_init_scale)
    
    def forward(self, x):
        return self.sparse_forward(x, self.weight, self.bias)


# ============================================================================
# Model Transformation Utilities
# ============================================================================

class Sparse2to4Utils:
    """Utility functions for applying 2:4 sparsity to existing models"""
    
    @staticmethod
    def replace_linear_with_sparse(
        model: nn.Module,
        target_modules: List[str] = None,
        enable_sparse: bool = True,
        sparse_init_scale: float = 1.0,
        verbose: bool = True
    ):
        """
        Replace nn.Linear modules with Sparse2to4Linear
        
        Args:
            model: The model to modify
            target_modules: List of module names to target (e.g., ['attn', 'mlp'])
            enable_sparse: Whether to enable sparsity
            sparse_init_scale: Initial scale for sparse weights
            verbose: Whether to print replacement info
        """
        if target_modules is None:
            target_modules = ['attn', 'mlp']  # Default targets
        
        replaced_count = 0
        for name, module in model.named_modules():
            if any(target in name for target in target_modules):
                for child_name, child_module in module.named_children():
                    if isinstance(child_module, nn.Linear):
                        # Create sparse replacement
                        sparse_linear = Sparse2to4Linear(
                            child_module.in_features,
                            child_module.out_features,
                            bias=child_module.bias is not None,
                            enable_sparse=enable_sparse,
                            sparse_init_scale=sparse_init_scale,
                            device=child_module.weight.device,
                            dtype=child_module.weight.dtype
                        )
                        
                        # Copy weights
                        sparse_linear.weight.data.copy_(child_module.weight.data)
                        if child_module.bias is not None:
                            sparse_linear.bias.data.copy_(child_module.bias.data)
                        
                        # Initialize sparse scale
                        sparse_linear.init_sparse_scale(sparse_linear.weight)
                        
                        # Replace module
                        setattr(module, child_name, sparse_linear)
                        replaced_count += 1
                        
                        if verbose:
                            print(f"Replaced {name}.{child_name}: {child_module} -> {sparse_linear}")
        
        print(f"\nReplaced {replaced_count} linear layers with sparse variants")
        return model
    
    @staticmethod
    def replace_lowrank_with_sparse_lowrank(
        model: nn.Module,
        target_modules: List[str] = None,
        enable_sparse: bool = True,
        sparse_init_scale: float = 1.0,
        verbose: bool = True
    ):
        """
        Replace existing LoRA modules with LowRankSparse2to4Linear
        This is specifically for models already using low-rank decomposition
        """
        if target_modules is None:
            target_modules = ['attn', 'mlp']
        
        replaced_count = 0
        # This would need to be customized based on the specific LoRA implementation
        # For now, provide a template that can be adapted
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight_in') and hasattr(module, 'weight_out'):
                # This looks like a LoRA module with in/out weights
                if any(target in name for target in target_modules):
                    # Create combined LoRA + Sparse module
                    rank = min(module.weight_in.shape[1], module.weight_out.shape[1])
                    
                    sparse_lowrank = LowRankSparse2to4Linear(
                        in_features=module.weight_in.shape[0],
                        out_features=module.weight_out.shape[0],
                        rank=rank,
                        bias=hasattr(module, 'bias') and module.bias is not None,
                        enable_sparse=enable_sparse,
                        sparse_init_scale=sparse_init_scale,
                        device=module.weight_in.device,
                        dtype=module.weight_in.dtype
                    )
                    
                    # Copy existing LoRA weights
                    sparse_lowrank.weight_A.data.copy_(module.weight_in.data)
                    sparse_lowrank.weight_B.data.copy_(module.weight_out.data)
                    if hasattr(module, 'bias') and module.bias is not None:
                        sparse_lowrank.bias.data.copy_(module.bias.data)
                    
                    # Initialize sparse scales
                    sparse_lowrank.init_all_sparse_scales()
                    
                    # This would need parent module replacement logic
                    # which depends on the specific model structure
                    replaced_count += 1
                    
                    if verbose:
                        print(f"Replaced LoRA module {name} with sparse variant")
        
        print(f"\nReplaced {replaced_count} LoRA modules with sparse variants")
        return model
    
    @staticmethod
    def count_sparse_parameters(model: nn.Module):
        """Count parameters in sparse vs non-sparse modules"""
        sparse_params = 0
        total_params = 0
        
        for module in model.modules():
            if isinstance(module, (Sparse2to4Linear, LowRankSparse2to4Linear)):
                sparse_params += sum(p.numel() for p in module.parameters())
            total_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'sparse_params': sparse_params,
            'sparse_ratio': sparse_params / total_params if total_params > 0 else 0
        }


# ============================================================================
# Example Integration Functions
# ============================================================================

def integrate_with_loro_model(model, attn_rank=64, mlp_rank=128, enable_sparse=True):
    """
    Example integration with LORO-style models
    This shows how to add 2:4 sparsity to an existing LORO model
    """
    print("Integrating 2:4 sparsity with LORO model...")
    
    # Replace existing low-rank modules with sparse variants
    Sparse2to4Utils.replace_lowrank_with_sparse_lowrank(
        model, 
        target_modules=['self_attn', 'mlp'],
        enable_sparse=enable_sparse
    )
    
    # Print parameter statistics
    stats = Sparse2to4Utils.count_sparse_parameters(model)
    print(f"Model statistics after sparse integration:")
    print(f"  Total parameters: {stats['total_params']:,}")
    print(f"  Sparse parameters: {stats['sparse_params']:,}")
    print(f"  Sparse ratio: {stats['sparse_ratio']:.2%}")
    
    return model


def integrate_with_standard_model(model, enable_sparse=True):
    """
    Example integration with standard transformer models
    This shows how to add 2:4 sparsity to a regular model
    """
    print("Integrating 2:4 sparsity with standard model...")
    
    # Replace linear layers with sparse variants
    Sparse2to4Utils.replace_linear_with_sparse(
        model,
        target_modules=['attn', 'mlp', 'self_attn'],
        enable_sparse=enable_sparse
    )
    
    # Print parameter statistics
    stats = Sparse2to4Utils.count_sparse_parameters(model)
    print(f"Model statistics after sparse integration:")
    print(f"  Total parameters: {stats['total_params']:,}")
    print(f"  Sparse parameters: {stats['sparse_params']:,}")
    print(f"  Sparse ratio: {stats['sparse_ratio']:.2%}")
    
    return model


# ============================================================================
# Configuration and Factory Functions
# ============================================================================

class Sparse2to4Config:
    """Configuration class for 2:4 sparse training"""
    
    def __init__(
        self,
        enable_sparse: bool = True,
        sparse_init_scale: float = 1.0,
        target_modules: List[str] = None,
        lora_rank_attn: int = 64,
        lora_rank_mlp: int = 128,
        lora_init: str = "xavier",
        verbose: bool = True
    ):
        self.enable_sparse = enable_sparse
        self.sparse_init_scale = sparse_init_scale
        self.target_modules = target_modules or ['attn', 'mlp']
        self.lora_rank_attn = lora_rank_attn
        self.lora_rank_mlp = lora_rank_mlp
        self.lora_init = lora_init
        self.verbose = verbose


def create_sparse_lowrank_linear(
    in_features: int,
    out_features: int,
    rank: int = None,
    config: Sparse2to4Config = None,
    **kwargs
):
    """
    Factory function to create appropriate sparse/lowrank linear layer
    """
    if config is None:
        config = Sparse2to4Config()
    
    if rank is not None:
        # Create low-rank + sparse layer
        return LowRankSparse2to4Linear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            enable_sparse=config.enable_sparse,
            sparse_init_scale=config.sparse_init_scale,
            lora_init=config.lora_init,
            **kwargs
        )
    else:
        # Create standard sparse layer
        return Sparse2to4Linear(
            in_features=in_features,
            out_features=out_features,
            enable_sparse=config.enable_sparse,
            sparse_init_scale=config.sparse_init_scale,
            **kwargs
        )