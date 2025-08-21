"""
Lightweight backward-only NaN detector for debugging split_gemm issues.
Focuses on memory-efficient detection without storing large tensors.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class NaNInfo:
    """Lightweight info about NaN occurrence"""
    layer_name: str
    param_name: str
    grad_shape: tuple
    nan_ratio: float
    inf_ratio: float
    
    
class BackwardNaNDetector:
    """
    Lightweight NaN detector for backward pass only.
    Focuses on split_gemm and activation 2:4 operations.
    """
    
    def __init__(self, model: nn.Module, args):
        self.model = model
        self.args = args
        self.nan_locations: List[NaNInfo] = []
        self.hooks = []
        
    def register_hooks(self):
        """Register backward hooks on relevant layers"""
        
        def make_backward_hook(layer_name: str, param_name: str):
            def hook(grad):
                if grad is not None:
                    # Check for NaN/Inf without storing tensors
                    has_nan = torch.isnan(grad).any().item()
                    has_inf = torch.isinf(grad).any().item()
                    
                    if has_nan or has_inf:
                        nan_ratio = torch.isnan(grad).float().mean().item() if has_nan else 0
                        inf_ratio = torch.isinf(grad).float().mean().item() if has_inf else 0
                        
                        info = NaNInfo(
                            layer_name=layer_name,
                            param_name=param_name,
                            grad_shape=tuple(grad.shape),
                            nan_ratio=nan_ratio,
                            inf_ratio=inf_ratio
                        )
                        self.nan_locations.append(info)
                        
                        # Print immediately for debugging
                        print(f"\n❌ NaN/Inf in backward: {layer_name}.{param_name}")
                        print(f"   Shape: {grad.shape}")
                        print(f"   NaN ratio: {nan_ratio:.2%}, Inf ratio: {inf_ratio:.2%}")
                        
                        # For MLP layers with activation 2:4, check specific components
                        if "mlp" in layer_name.lower() and self.args.activation_2by4:
                            self._analyze_mlp_gradient(layer_name, grad)
                            
                return grad
            return hook
        
        # Register hooks on all parameters in relevant layers
        for name, module in self.model.named_modules():
            # Focus on MLP and attention layers
            if any(key in name.lower() for key in ["mlp", "attn", "ffn"]):
                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        h = param.register_hook(make_backward_hook(name, param_name))
                        self.hooks.append(h)
                        
    def _analyze_mlp_gradient(self, layer_name: str, grad: torch.Tensor):
        """Analyze gradient patterns in MLP layers using activation 2:4"""
        print(f"   [Analyzing MLP gradient patterns]")
        
        # Check gradient statistics without storing full tensor
        grad_abs = grad.abs()
        grad_mean = grad_abs.mean().item()
        grad_std = grad_abs.std().item()
        grad_max = grad_abs.max().item()
        
        print(f"   Gradient stats: mean={grad_mean:.3e}, std={grad_std:.3e}, max={grad_max:.3e}")
        
        # Check if gradient explosion might be happening
        if grad_max > 1e6:
            print(f"   ⚠️ Gradient explosion detected! Max value: {grad_max:.3e}")
            
        # For 2D gradients, check column-wise patterns (relevant for split_gemm)
        if grad.dim() == 2:
            # Check columns with high NaN ratio
            nan_per_col = torch.isnan(grad).float().mean(dim=0)
            high_nan_cols = (nan_per_col > 0.5).sum().item()
            if high_nan_cols > 0:
                print(f"   ⚠️ {high_nan_cols}/{grad.shape[1]} columns have >50% NaN")
                
    def track_backward(self, batch: Dict, labels: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Track backward pass for NaN detection.
        Returns the loss value if computed.
        """
        self.nan_locations.clear()
        
        # Register hooks
        self.register_hooks()
        
        try:
            # First check if forward pass is clean
            print("\n[Checking forward pass...]")
            with torch.no_grad():
                output = self.model(**batch, labels=labels)
                loss_check = output.loss if hasattr(output, 'loss') else output
                
                if torch.isnan(loss_check).any():
                    print("❌ NaN already present in forward pass loss")
                    return loss_check
                    
                print(f"✓ Forward pass clean, loss = {loss_check.item():.4f}")
            
            # Now run with gradients and track backward
            print("\n[Running backward pass with NaN detection...]")
            self.model.zero_grad()
            
            # Run forward again with gradients enabled
            output = self.model(**batch, labels=labels)
            loss = output.loss if hasattr(output, 'loss') else output
            
            # Check intermediate activation sparsity if using activation 2:4
            if self.args.activation_2by4:
                self._check_activation_sparsity()
            
            # Run backward
            loss.backward()
            
            # Report findings
            if self.nan_locations:
                print(f"\n🔴 Found {len(self.nan_locations)} locations with NaN/Inf in gradients")
                
                # Group by layer type
                mlp_issues = [loc for loc in self.nan_locations if "mlp" in loc.layer_name.lower()]
                attn_issues = [loc for loc in self.nan_locations if "attn" in loc.layer_name.lower()]
                
                if mlp_issues:
                    print(f"\n  MLP layers with issues: {len(mlp_issues)}")
                    if self.args.activation_2by4 and self.args.dx_direct_sparse == 1:
                        print("  ⚠️ Using split_gemm - check kernel stability")
                        
                if attn_issues:
                    print(f"  Attention layers with issues: {len(attn_issues)}")
                    
            else:
                print("\n✓ No NaN/Inf detected in gradients")
                
            return loss
            
        finally:
            # Clean up hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
            
    def _check_activation_sparsity(self):
        """Check activation sparsity levels in MLP layers"""
        print("\n[Checking activation sparsity levels...]")
        
        for name, module in self.model.named_modules():
            if "mlp" in name.lower() and hasattr(module, 'last_sparsity'):
                sparsity = module.last_sparsity
                if sparsity is not None:
                    print(f"  {name}: sparsity = {sparsity:.2%}")
                    if sparsity < 0.4:  # Less than 40% sparse (i.e., >60% dense)
                        print(f"    ⚠️ Low sparsity might cause split_gemm issues")
                        
    def get_summary(self) -> Dict:
        """Get summary of NaN detection results"""
        summary = {
            'total_nan_locations': len(self.nan_locations),
            'mlp_issues': sum(1 for loc in self.nan_locations if "mlp" in loc.layer_name.lower()),
            'attn_issues': sum(1 for loc in self.nan_locations if "attn" in loc.layer_name.lower()),
            'max_nan_ratio': max((loc.nan_ratio for loc in self.nan_locations), default=0),
            'locations': self.nan_locations
        }
        return summary