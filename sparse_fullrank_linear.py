"""
Full-rank Linear Layer with 2:4 Sparsity Training
================================================

This module provides Sparse2to4Linear - a full-rank linear layer that applies 
2:4 sparsity training using the EXACT same implementation as LORO+2:4, but 
without low-rank decomposition.

Use case: Control experiments to isolate whether issues come from LORO or 2:4 sparsity.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List

# Import the exact same 2:4 sparse implementations used in LORO
from loro_torch.sparse_overlay import SparseOverlayFunction
from loro_torch.triton_kernels.sparse_kernels import soft_threshold24_triton


class Sparse2to4Linear(nn.Module):
    """
    Full-rank Linear layer with 2:4 sparsity training.
    
    This uses the EXACT same 2:4 sparse implementation as LORO+2:4:
    - Same SparseOverlayFunction for forward/backward
    - Same MVUE bias correction in backward pass  
    - Same scaling mechanism
    - Same Triton kernels
    
    But operates on full-rank weights instead of low-rank decomposition.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparse_init_scale: float = 1.0,
        device=None,
        dtype=None,
    ):
        super(Sparse2to4Linear, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        
        # Full-rank weight (not low-rank like LORO)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # 2:4 sparsity scale factor (same as LORO implementation)
        self.register_buffer('sparse_scale', torch.tensor(sparse_init_scale, **factory_kwargs))
        
        # Lazy initialization flag
        self._scales_initialized = False
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using standard methods"""
        # Standard initialization for full-rank weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def init_sparse_scales(self):
        """Initialize sparse scale factors for stable training (same as LORO)"""
        if self._scales_initialized:
            return
            
        try:
            # Use same scale initialization as LORO
            weight_temp = self.weight.detach()
            
            # Convert bfloat16 to float16 for triton compatibility
            if weight_temp.dtype == torch.bfloat16:
                weight_temp = weight_temp.to(torch.float16)
                
            weight_sparse, _ = soft_threshold24_triton(weight_temp)
            
            # Convert back to original dtype before computing scale
            if self.weight.dtype == torch.bfloat16:
                weight_sparse = weight_sparse.to(torch.bfloat16)
            
            # Same scale computation as LORO
            dense_flat = torch.flatten(self.weight)
            sparse_flat = torch.flatten(weight_sparse)
            
            dense_norm_sq = torch.sum(dense_flat * dense_flat)
            sparse_norm_sq = torch.sum(sparse_flat * sparse_flat)
            
            if sparse_norm_sq > 1e-12:  # Avoid division by near-zero
                scale = torch.sqrt(dense_norm_sq / sparse_norm_sq)
                # Clamp scale to prevent extreme values
                scale = torch.clamp(scale, min=0.1, max=10.0)
            else:
                scale = torch.tensor(1.0, device=self.weight.device)
                print(f"⚠️  Weight sparse norm too small, using scale=1.0")
            
            self.sparse_scale.copy_(scale.detach())
            self._scales_initialized = True
            
        except Exception as e:
            # If Triton functions fail, use conservative initialization
            print(f"⚠️  Triton initialization failed, using conservative scale initialization: {e}")
            self.sparse_scale.copy_(torch.tensor(1.0, device=self.weight.device))
            self._scales_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with 2:4 sparsity (same as LORO implementation)
        """
        # Ensure scales are initialized
        if not self._scales_initialized:
            self.init_sparse_scales()
        
        # Use the EXACT same SparseOverlayFunction as LORO
        # This includes the same forward pass and MVUE backward pass
        output = SparseOverlayFunction.apply(x, self.weight, self.bias, self.sparse_scale)
        
        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, sparse_scale={self.sparse_scale.item():.4f}'


def replace_linear_with_sparse2to4(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    sparse_init_scale: float = 1.0,
) -> nn.Module:
    """
    Replace target Linear modules with Sparse2to4Linear modules
    
    Args:
        model: The model to modify
        target_modules: List of module names to replace (e.g., ["q_proj", "v_proj"])
        exclude_modules: List of module names to exclude from replacement
        sparse_init_scale: Initial scale for sparse weights
        
    Returns:
        Modified model with Sparse2to4Linear modules
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    if exclude_modules is None:
        exclude_modules = []
    
    def replace_module(parent_module, child_name, child_module):
        if isinstance(child_module, nn.Linear) and child_name in target_modules and child_name not in exclude_modules:
            # Create Sparse2to4Linear replacement
            sparse_linear = Sparse2to4Linear(
                in_features=child_module.in_features,
                out_features=child_module.out_features,
                bias=child_module.bias is not None,
                sparse_init_scale=sparse_init_scale,
                device=child_module.weight.device,
                dtype=child_module.weight.dtype,
            )
            
            # Initialize scales
            sparse_linear.init_sparse_scales()
            
            # Replace the module
            setattr(parent_module, child_name, sparse_linear)
            return True
        return False
    
    replaced_count = 0
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if replace_module(module, child_name, child_module):
                replaced_count += 1
                print(f"Replaced {name}.{child_name} with Sparse2to4Linear")
    
    print(f"✅ Total replaced modules: {replaced_count}")
    return model


def test_sparse2to4_linear():
    """Test if the Sparse2to4Linear implementation is working"""
    print("🧪 Testing Sparse2to4Linear Implementation")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # Test basic functionality
        print("1. Testing Sparse2to4Linear...")
        layer = Sparse2to4Linear(768, 256).cuda()
        layer.init_sparse_scales()
        
        x = torch.randn(8, 1024, 768).cuda()
        output = layer(x)
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Sparse scale: {layer.sparse_scale.item():.4f}")
        
        # Test backward pass
        print("\n2. Testing backward pass...")
        loss = output.sum()
        loss.backward()
        print(f"   ✓ Weight gradient shape: {layer.weight.grad.shape}")
        print(f"   ✓ Weight gradient norm: {layer.weight.grad.norm().item():.6f}")
        
        # Test replacement function
        print("\n3. Testing module replacement...")
        import torch.nn as nn
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(768, 256)
                self.k_proj = nn.Linear(768, 256)
                self.other_linear = nn.Linear(256, 128)
        
        model = TestModel()
        model = replace_linear_with_sparse2to4(
            model, 
            target_modules=["q_proj", "k_proj"],
            exclude_modules=["other_linear"]
        )
        
        assert isinstance(model.q_proj, Sparse2to4Linear)
        assert isinstance(model.k_proj, Sparse2to4Linear) 
        assert isinstance(model.other_linear, nn.Linear)  # Should not be replaced
        
        print(f"\n✅ All tests passed! Sparse2to4Linear is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_sparse2to4_linear() 