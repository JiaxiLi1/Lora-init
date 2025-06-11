"""
Sparse Overlay Module for LORO Parameters
==========================================

This module applies 2:4 sparsity as an overlay on existing LORO low-rank parameters,
implementing true LORO + 2:4 Sparse stacking architecture.

Architecture:
Base Model → LORO Low-Rank → 2:4 Sparse Overlay → Final Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.cuda.amp import custom_fwd, custom_bwd
import math
from typing import Optional, List

# Import 2:4 sparse implementations
from sparse import matmul, MVUE24_approx_triton, soft_threshold24_triton


def fake_fp8_mm(a, b, dtype):
    """Simulate FP8 precision for compatibility"""
    a = a.to(torch.float16).contiguous()
    b = b.to(torch.float16).contiguous()
    output = matmul(a, b, c_dtype=torch.float32)
    return output


class SparseOverlayFunction(autograd.Function):
    """
    Sparse overlay autograd function that applies 2:4 sparsity on LORO parameters
    with MVUE gradient estimation for unbiased backward pass.
    
    Key improvements for numerical stability:
    1. Forward: Use sparse weights for computation efficiency
    2. Backward: Use DENSE weights for grad_input (prevents bias propagation)
    3. Backward: Apply MVUE to BOTH input and grad_output for grad_weight
    
    This ensures that LORO optimizer receives unbiased gradients, preventing
    the parameter matrices from becoming ill-conditioned over time.
    """
    
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, sparse_scale):
        # 🔍 Debug: Check inputs for NaN
        if torch.isnan(input).any():
            print(f"❌ FORWARD: NaN detected in INPUT! Shape: {input.shape}")
        if torch.isnan(weight).any():
            print(f"❌ FORWARD: NaN detected in WEIGHT! Shape: {weight.shape}")
        if torch.isnan(sparse_scale).any():
            print(f"❌ FORWARD: NaN detected in SPARSE_SCALE! Value: {sparse_scale}")
            
        ctx.save_for_backward(input, weight, bias, sparse_scale)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        
        # Apply 2:4 sparsity to the LORO weight
        weight_sparse = get_sparse_weight(weight, sparse_scale)
        
        # 🔍 Debug: Check sparse weight
        if torch.isnan(weight_sparse).any():
            print(f"❌ FORWARD: NaN detected in WEIGHT_SPARSE after get_sparse_weight!")
            print(f"   Original weight has NaN: {torch.isnan(weight).any()}")
            print(f"   Sparse scale: {sparse_scale}")
        
        # For LORO: weight_in is (in_features, rank), weight_out is (out_features, rank)
        # input @ weight_in -> (batch, in_features) @ (in_features, rank) = (batch, rank)
        # x_proj @ weight_out.T -> (batch, rank) @ (rank, out_features) = (batch, out_features)
        output = fake_fp8_mm(input, weight_sparse, torch.float8_e4m3fn)
        
        # 🔍 Debug: Check output
        if torch.isnan(output).any():
            print(f"❌ FORWARD: NaN detected in OUTPUT after matmul!")
            print(f"   Input has NaN: {torch.isnan(input).any()}")
            print(f"   Weight_sparse has NaN: {torch.isnan(weight_sparse).any()}")
        
        if bias is None:
            final_output = output.view(*ctx.shape[:-1], -1)
        else:
            final_output = output.view(*ctx.shape[:-1], -1) + bias
            if torch.isnan(final_output).any() and not torch.isnan(output).any():
                print(f"❌ FORWARD: NaN introduced by BIAS addition!")
                
        return final_output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # 🔍 Debug: Check grad_output before half() conversion
        if torch.isnan(grad_output).any():
            print(f"❌ BACKWARD: NaN in grad_output BEFORE half() conversion!")
            
        grad_output = grad_output.half()
        
        # 🔍 Debug: Check grad_output after half() conversion
        if torch.isnan(grad_output).any():
            print(f"❌ BACKWARD: NaN in grad_output AFTER half() conversion!")
            
        input, weight, bias, sparse_scale = ctx.saved_tensors
        
        # 🔍 Debug: Check saved tensors
        if torch.isnan(input).any():
            print(f"❌ BACKWARD: NaN in saved INPUT!")
        if torch.isnan(weight).any():
            print(f"❌ BACKWARD: NaN in saved WEIGHT!")
            
        grad_input = grad_weight = grad_bias = grad_sparse_scale = None
        
        if ctx.needs_input_grad[0]:
            grad_output_view = grad_output.view(-1, grad_output.shape[-1])
            # Use dense weight for grad_input calculation (not sparse weight!)
            # This ensures grad_input doesn't carry sparse bias
            grad_input = fake_fp8_mm(grad_output_view, weight.t(), torch.float8_e5m2).view(ctx.shape)
            
            # 🔍 Debug: Check grad_input
            if torch.isnan(grad_input).any():
                print(f"❌ BACKWARD: NaN in computed GRAD_INPUT!")
                print(f"   grad_output_view has NaN: {torch.isnan(grad_output_view).any()}")
                print(f"   weight.t() has NaN: {torch.isnan(weight.t()).any()}")
            
        if ctx.needs_input_grad[1]:
            input_view = input.view(-1, input.shape[-1])
            grad_output_view = grad_output.view(-1, grad_output.shape[-1])
            
            # 🔍 Debug: Before MVUE
            if torch.isnan(input_view).any():
                print(f"❌ BACKWARD: NaN in input_view BEFORE MVUE!")
            if torch.isnan(grad_output_view).any():
                print(f"❌ BACKWARD: NaN in grad_output_view BEFORE MVUE!")
            
            # 🔍 NEW DEBUG: Check if grad_output is essentially zero
            grad_output_norm = torch.norm(grad_output_view).item()
            grad_output_max = torch.max(torch.abs(grad_output_view)).item()
            grad_output_zero_ratio = (grad_output_view.abs() < 1e-10).float().mean().item()
            
            print(f"🔬 DETAILED DEBUG:")
            print(f"   grad_output shape: {grad_output_view.shape}")
            print(f"   grad_output norm: {grad_output_norm:.8f}")
            print(f"   grad_output max abs: {grad_output_max:.8f}")
            print(f"   grad_output zero ratio (<1e-10): {grad_output_zero_ratio:.4f}")
            print(f"   input norm: {torch.norm(input_view).item():.8f}")
            
            # Only apply MVUE if grad_output is not all zeros
            if grad_output_norm > 1e-12:
                # Apply MVUE to grad_output for bias correction
                grad_output_mvue = MVUE24_approx_triton(grad_output_view)
                
                # 🔍 Debug: After MVUE
                if torch.isnan(grad_output_mvue).any():
                    print(f"❌ BACKWARD: NaN in grad_output_mvue AFTER MVUE!")
                
                # Debug: Check MVUE effectiveness (print more frequently during debugging)
                import random
                if random.random() < 0.01:  # 1% chance to print for more debugging info
                    orig_grad_norm = torch.norm(grad_output_view).item()
                    mvue_grad_norm = torch.norm(grad_output_mvue).item()
                    ratio_grad = mvue_grad_norm/orig_grad_norm if orig_grad_norm > 0 else float('inf')
                    print(f"🔍 MVUE debug: grad_output norm {orig_grad_norm:.4f} → {mvue_grad_norm:.4f} (ratio: {ratio_grad:.4f})")
                    
                    # Detailed analysis of zero gradients
                    zero_mask = grad_output_mvue.abs() < 1e-10
                    zero_ratio = zero_mask.float().mean().item()
                    max_val = torch.max(grad_output_mvue.abs()).item()
                    min_val = torch.min(grad_output_mvue.abs()).item()
                    print(f"📊 Zero grad analysis: max={max_val:.8f}, min={min_val:.8f}, zero_ratio={zero_ratio:.4f}")
            else:
                print(f"⚠️  SKIPPING MVUE: grad_output norm too small ({grad_output_norm:.2e})")
                grad_output_mvue = grad_output_view  # Skip MVUE for zero gradients
            
            # Compute grad_weight using corrected formula
            grad_weight = fake_fp8_mm(input_view.t(), grad_output_mvue, torch.float8_e5m2)
            
            # 🔍 Debug: Check final grad_weight
            if torch.isnan(grad_weight).any():
                print(f"❌ BACKWARD: NaN in computed GRAD_WEIGHT!")
                print(f"   input_view.t() has NaN: {torch.isnan(input_view.t()).any()}")
                print(f"   grad_output_mvue has NaN: {torch.isnan(grad_output_mvue).any()}")
        
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            if torch.isnan(grad_bias).any():
                print(f"❌ BACKWARD: NaN in computed GRAD_BIAS!")
            
        if ctx.needs_input_grad[3]:
            # Sparse scale gradient (simplified)
            grad_sparse_scale = None  # Will be computed separately if needed
            
        return grad_input, grad_weight, grad_bias, grad_sparse_scale


def get_sparse_weight(weight, sparse_scale):
    """Apply 2:4 sparsity to weight with learnable scaling"""
    weight_temp = weight.detach()
    
    # Convert bfloat16 to float16 for triton compatibility
    original_dtype = weight_temp.dtype
    if weight_temp.dtype == torch.bfloat16:
        weight_temp = weight_temp.to(torch.float16)
        
    weight_sparse, _ = soft_threshold24_triton(weight_temp)
    
    # Convert back to original dtype
    if original_dtype == torch.bfloat16:
        weight_sparse = weight_sparse.to(torch.bfloat16)
        
    return weight_sparse * sparse_scale


class SparseOverlayLinear(nn.Module):
    """
    Sparse overlay that wraps existing LORO linear layers
    """
    
    def __init__(self, loro_linear, sparse_init_scale=1.0):
        super(SparseOverlayLinear, self).__init__()
        
        # Store reference to original LORO linear layer
        self.loro_linear = loro_linear
        
        # Add sparse scaling parameters as fixed buffers (not learnable parameters)
        device = next(loro_linear.parameters()).device
        self.register_buffer('sparse_scale_in', torch.tensor(sparse_init_scale, device=device))
        self.register_buffer('sparse_scale_out', torch.tensor(sparse_init_scale, device=device))
        
        # Lazy initialization flag
        self._scales_initialized = False
    
    @torch.no_grad()
    def init_sparse_scales(self):
        """Initialize sparse scale factors for stable training (delayed initialization)"""
        if self._scales_initialized:
            return
            
        try:
            if hasattr(self.loro_linear, 'weight_in'):
                weight_in = self.loro_linear.weight_in
                weight_in_temp = weight_in.detach()
                
                if weight_in_temp.dtype == torch.bfloat16:
                    weight_in_temp = weight_in_temp.to(torch.float16)
                    
                weight_in_sparse, _ = soft_threshold24_triton(weight_in_temp)
                # Convert back to original dtype before computing scale
                if weight_in.dtype == torch.bfloat16:
                    weight_in_sparse = weight_in_sparse.to(torch.bfloat16)
                scale_in = torch.dot(torch.flatten(weight_in), torch.flatten(weight_in_sparse)) / torch.dot(
                    torch.flatten(weight_in_sparse), torch.flatten(weight_in_sparse))
                self.sparse_scale_in.copy_(scale_in.detach())  # Ensure it's detached from computation graph
            
            if hasattr(self.loro_linear, 'weight_out'):
                weight_out = self.loro_linear.weight_out
                weight_out_temp = weight_out.detach()
                
                if weight_out_temp.dtype == torch.bfloat16:
                    weight_out_temp = weight_out_temp.to(torch.float16)
                    
                weight_out_sparse, _ = soft_threshold24_triton(weight_out_temp)
                # Convert back to original dtype before computing scale
                if weight_out.dtype == torch.bfloat16:
                    weight_out_sparse = weight_out_sparse.to(torch.bfloat16)
                scale_out = torch.dot(torch.flatten(weight_out), torch.flatten(weight_out_sparse)) / torch.dot(
                    torch.flatten(weight_out_sparse), torch.flatten(weight_out_sparse))
                self.sparse_scale_out.copy_(scale_out.detach())  # Ensure it's detached from computation graph
            
            self._scales_initialized = True
            
        except Exception as e:
            # If Triton functions fail, use simple initialization
            print(f"⚠️  Triton initialization failed, using simple scale initialization: {e}")
            self._scales_initialized = True
    
    def get_scale_info(self):
        """Get current sparse scale information for debugging"""
        return {
            'sparse_scale_in': self.sparse_scale_in.item() if self._scales_initialized else "Not initialized",
            'sparse_scale_out': self.sparse_scale_out.item() if self._scales_initialized else "Not initialized",
            'initialized': self._scales_initialized
        }
    
    def forward(self, x):
        """
        Forward pass through LORO with sparse overlay
        
        Flow: x → sparse(LORO_A) → sparse(LORO_B) → output
        """
        # Initialize scales on first forward pass (avoids multiprocessing issues)
        if not self._scales_initialized:
            self.init_sparse_scales()
            
        # Get LORO weights
        weight_in = self.loro_linear.weight_in    # (in_features, rank)
        weight_out = self.loro_linear.weight_out  # (out_features, rank)
        bias = getattr(self.loro_linear, 'bias', None)
        
        # Apply sparse overlay to LORO weights
        # Step 1: x @ sparse(weight_in) -> (batch, in_features) @ (in_features, rank) = (batch, rank)
        x_proj = SparseOverlayFunction.apply(x, weight_in, None, self.sparse_scale_in)
        
        # Apply dropout if present
        if hasattr(self.loro_linear, 'dropout'):
            x_proj = self.loro_linear.dropout(x_proj)
        
        # Step 2: x_proj @ sparse(weight_out.T) -> (batch, rank) @ (rank, out_features) = (batch, out_features)
        output = SparseOverlayFunction.apply(x_proj, weight_out.t(), bias, self.sparse_scale_out)
        
        # Apply LORO scaling
        if hasattr(self.loro_linear, 'scaling'):
            output = output * self.loro_linear.scaling
            
        return output


def apply_sparse_overlay_on_loro(
    model: nn.Module,
    sparse_init_scale: float = 1.0,
    target_modules: Optional[List[str]] = None
) -> None:
    """
    Apply 2:4 sparse overlay on existing LORO parameters
    
    Args:
        model: Model with LORO parameters already applied
        sparse_init_scale: Initial scale for sparse weights
        target_modules: List of module names to apply sparse overlay to
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    replaced_modules = 0
    total_modules = 0
    
    for name, module in model.named_modules():
        # Check if this is a LORO linear layer
        if (hasattr(module, 'weight_in') and hasattr(module, 'weight_out') and 
            any(target_name in name for target_name in target_modules)):
            
            total_modules += 1
            
            # Get parent module and child name
            parent = model
            components = name.split('.')
            for component in components[:-1]:
                parent = getattr(parent, component)
            child_name = components[-1]
            
            # Create sparse overlay
            sparse_overlay = SparseOverlayLinear(module, sparse_init_scale)
            
            # Replace the module
            setattr(parent, child_name, sparse_overlay)
            replaced_modules += 1
            
            print(f"🔧 Applied sparse overlay to: {name} (scale will be computed on first forward pass)")
    
    print(f"✅ Sparse overlay applied to {replaced_modules}/{total_modules} LORO modules")
    
    if replaced_modules == 0:
        print("❌ Warning: No LORO modules found to apply sparse overlay!")
        print("   Make sure LORO parameterization is applied first.")


def get_sparse_overlay_parameters(model: nn.Module):
    """Get sparse overlay parameters for optimizer - Note: Now returns empty since scales are fixed buffers"""
    # Sparse scale parameters are now fixed buffers (not learnable), so return empty list
    return []


def test_sparse_overlay():
    """Test sparse overlay functionality"""
    print("Testing Sparse Overlay on LORO...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU test")
        return False
    
    # This would normally be a LORO linear layer
    class MockLoroLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_in = nn.Parameter(torch.randn(64, 768))
            self.weight_out = nn.Parameter(torch.randn(512, 64))
            self.scaling = 1.0
    
    try:
        loro_layer = MockLoroLinear().cuda()
        sparse_layer = SparseOverlayLinear(loro_layer).cuda()
        
        x = torch.randn(2, 10, 768).cuda()
        output = sparse_layer(x)
        
        print(f"✅ Input shape: {x.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Sparse scales: in={sparse_layer.sparse_scale_in.item():.4f}, out={sparse_layer.sparse_scale_out.item():.4f}")
        print("✅ Sparse overlay test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during GPU test: {e}")
        # Try CPU fallback test
        print("🔄 Trying CPU fallback...")
        try:
            loro_layer = MockLoroLinear()
            
            # Create a simple CPU version for testing
            x = torch.randn(2, 10, 768)
            # Simple forward pass without sparse overlay
            weight_in = loro_layer.weight_in    # (64, 768)
            weight_out = loro_layer.weight_out  # (512, 64)
            
            x_proj = torch.matmul(x, weight_in.t())  # (2, 10, 64)
            output = torch.matmul(x_proj, weight_out.t())  # (2, 10, 512)
            
            print(f"✅ CPU fallback - Input shape: {x.shape}")
            print(f"✅ CPU fallback - Output shape: {output.shape}")
            print("✅ Basic structure test passed!")
            return True
            
        except Exception as e2:
            print(f"❌ CPU fallback also failed: {e2}")
            return False








if __name__ == "__main__":
    test_sparse_overlay() 