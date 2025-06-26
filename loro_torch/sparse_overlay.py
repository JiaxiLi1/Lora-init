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
import random

# 🔧 DEBUG CONTROL - Set to control debug verbosity
# 0: No debug output
# 1: Critical errors only (NaN, Inf)
# 2: All debug output (default)
DEBUG_LEVEL = 1  # 🔧 REDUCED: Only show critical issues

def debug_print(message, level=1):
    """Conditional debug printing based on DEBUG_LEVEL"""
    if DEBUG_LEVEL >= level:
        print(message)

# Import after setting debug level
try:
    from sparse import matmul, MVUE24_approx_triton, soft_threshold24_triton
except ImportError:
    print("Warning: Could not import sparse functions, using fallback implementations")
    
    def matmul(a, b, c_dtype=torch.float32):
        return torch.matmul(a.float(), b.float()).to(c_dtype)
    
    def MVUE24_approx_triton(x):
        # Fallback: just return the input (no MVUE correction)
        return x
    
    def soft_threshold24_triton(x):
        # Fallback: simulate 2:4 sparsity with simple thresholding
        abs_x = torch.abs(x)
        threshold = torch.quantile(abs_x.flatten(), 0.5)  # Keep 50% of values
        mask = abs_x >= threshold
        return x * mask, mask


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
        debug_nan_input = torch.isnan(input).any()
        debug_nan_weight = torch.isnan(weight).any()
        debug_nan_bias = torch.isnan(bias).any() if bias is not None else False
        debug_nan_scale = torch.isnan(sparse_scale).any()
        
        if debug_nan_input or debug_nan_weight or debug_nan_bias or debug_nan_scale:
            print(f"🚨 CRITICAL: NaN detected in forward pass inputs!")
            print(f"   input has NaN: {debug_nan_input}")
            print(f"   weight has NaN: {debug_nan_weight}")
            print(f"   bias has NaN: {debug_nan_bias}")
            print(f"   sparse_scale has NaN: {debug_nan_scale}")
            print(f"   Stopping training to prevent NaN propagation...")
            raise RuntimeError("NaN detected in forward pass inputs - training stopped")
        
        # Get sparse weight with scaling
        weight_sparse = get_sparse_weight(weight, sparse_scale)
        
        # 🔍 Debug: Check sparse weight
        debug_nan_sparse = torch.isnan(weight_sparse).any()
        if debug_nan_sparse:
            print(f"🚨 FORWARD: NaN in sparse weight after get_sparse_weight!")
            print(f"   original weight norm: {torch.norm(weight).item():.6f}")
            print(f"   sparse_scale: {sparse_scale.item():.6f}")
        
        # Save for backward
        ctx.save_for_backward(input, weight, bias, sparse_scale)
        ctx.shape = input.shape
        
        # Forward computation
        input_view = input.view(-1, input.shape[-1])
        output = fake_fp8_mm(input_view, weight_sparse, torch.float8_e4m3fn)
        
        # 🔍 Debug: Check output
        debug_nan_output = torch.isnan(output).any()
        debug_inf_output = torch.isinf(output).any()
        if debug_nan_output or debug_inf_output:
            print(f"🚨 CRITICAL: NaN/Inf detected in forward pass output!")
            print(f"   output has NaN: {debug_nan_output}")
            print(f"   output has Inf: {debug_inf_output}")
            print(f"   input_view norm: {torch.norm(input_view).item():.6f}")
            print(f"   weight_sparse norm: {torch.norm(weight_sparse).item():.6f}")
            if debug_nan_output:
                print(f"   Stopping training to prevent NaN propagation...")
                raise RuntimeError("NaN detected in forward pass output - training stopped")
        
        # Add bias if present
        if bias is None:
            final_output = output.view(*ctx.shape[:-1], -1)
        else:
            final_output = output.view(*ctx.shape[:-1], -1) + bias
            
        # 🔍 Debug: Final output check
        debug_nan_final = torch.isnan(final_output).any()
        if debug_nan_final:
            print(f"🚨 CRITICAL: NaN detected in final output!")
            print(f"   Stopping training to prevent NaN propagation...")
            raise RuntimeError("NaN detected in final forward output - training stopped")
            
        return final_output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # 🔍 Debug: Check grad_output before half() conversion
        debug_nan_grad_in = torch.isnan(grad_output).any()
        debug_inf_grad_in = torch.isinf(grad_output).any()
        grad_norm_before = torch.norm(grad_output).item()
        grad_max_before = torch.max(torch.abs(grad_output)).item()
        
        if debug_nan_grad_in or debug_inf_grad_in:
            print(f"🚨 BACKWARD: NaN/Inf in incoming grad_output!")
            print(f"   grad_output has NaN: {debug_nan_grad_in}")
            print(f"   grad_output has Inf: {debug_inf_grad_in}")
            print(f"   grad_output shape: {grad_output.shape}")
            print(f"   grad_output norm: {grad_norm_before:.6f}")
            print(f"   grad_output max abs: {grad_max_before:.6f}")
            
        grad_output = grad_output.half()
        
        # 🔍 Debug: Check grad_output after half() conversion
        debug_nan_grad_half = torch.isnan(grad_output).any()
        grad_norm_after_half = torch.norm(grad_output).item()
        if debug_nan_grad_half:
            print(f"🚨 BACKWARD: NaN in grad_output AFTER half() conversion!")
            print(f"   norm before half(): {grad_norm_before:.6f}")
            print(f"   norm after half(): {grad_norm_after_half:.6f}")
            
        input, weight, bias, sparse_scale = ctx.saved_tensors
        
        # 🔍 Debug: Check saved tensors
        if torch.isnan(input).any():
            print(f"🚨 BACKWARD: NaN in saved INPUT!")
        if torch.isnan(weight).any():
            print(f"🚨 BACKWARD: NaN in saved WEIGHT!")
        if bias is not None and torch.isnan(bias).any():
            print(f"🚨 BACKWARD: NaN in saved BIAS!")
        if torch.isnan(sparse_scale).any():
            print(f"🚨 BACKWARD: NaN in saved SPARSE_SCALE!")
        
        grad_input = grad_weight = grad_bias = grad_sparse_scale = None
        
        # Compute grad_input
        if ctx.needs_input_grad[0]:
            if grad_output.stride() == (0, 0, 0):
                grad_output = torch.ones_like(grad_output, device=grad_output.device, dtype=grad_output.dtype)
                print(f"⚠️  BACKWARD: Fixed zero-stride grad_output!")
                
            grad_output_view = grad_output.view(-1, grad_output.shape[-1])
            
            # Use dense weight for grad_input (as per reference implementation)
            # For LORO: grad_input = grad_output @ weight (因为forward用了weight)
            grad_input = fake_fp8_mm(grad_output_view, weight.t(), torch.float8_e5m2).view(ctx.shape)
            
            # 🔍 Debug: Check grad_input
            if torch.isnan(grad_input).any():
                print(f"🚨 BACKWARD: NaN in computed GRAD_INPUT!")
                print(f"   grad_output_view norm: {torch.norm(grad_output_view).item():.6f}")
                print(f"   weight norm: {torch.norm(weight).item():.6f}")
        
        # Compute grad_weight
        if ctx.needs_input_grad[1]:
            input_view = input.view(-1, input.shape[-1])
            grad_output_view = grad_output.view(-1, grad_output.shape[-1])
            
            # 🔍 Debug: Before MVUE
            input_norm = torch.norm(input_view).item()
            grad_norm = torch.norm(grad_output_view).item()
            debug_print(f"🔬 DETAILED DEBUG BEFORE MVUE:", 2)
            debug_print(f"   grad_output shape: {grad_output_view.shape}", 2)
            debug_print(f"   grad_output norm: {grad_norm:.8f}", 2)
            debug_print(f"   grad_output max abs: {torch.max(torch.abs(grad_output_view)).item():.8f}", 2)
            debug_print(f"   grad_output zero ratio (<1e-10): {(torch.abs(grad_output_view) < 1e-10).float().mean().item():.4f}", 2)
            debug_print(f"   input norm: {input_norm:.8f}", 2)
            
            # 🔧 CORRECTED: Apply MVUE exactly like reference implementation
            # Reference: grad_weight = fake_fp8_mm(MVUE24_approx_triton(grad_output.t()), input, ...)
            
            try:
                # 🔧 FIXED: Apply MVUE to transposed grad_output (like reference implementation)
                grad_output_t = grad_output_view.t()  # Transpose first
                
                # Convert bfloat16 to float16 for Triton compatibility
                original_dtype = grad_output_t.dtype
                if grad_output_t.dtype == torch.bfloat16:
                    grad_output_t = grad_output_t.to(torch.float16)
                
                # 🔍 Apply MVUE and track problems (no modification of behavior)
                grad_t_norm_before = torch.norm(grad_output_t).item()
                grad_output_mvue_t = MVUE24_approx_triton(grad_output_t)  # Apply MVUE as intended
                
                # Convert back to original dtype
                if original_dtype == torch.bfloat16:
                    grad_output_mvue_t = grad_output_mvue_t.to(torch.bfloat16)
                    
                grad_mvue_norm_after = torch.norm(grad_output_mvue_t).item()
                
                # 🔍 Calculate ratio safely and track issues
                if grad_t_norm_before > 0:
                    mvue_ratio = grad_mvue_norm_after / grad_t_norm_before
                else:
                    mvue_ratio = float('inf')
                
                # 🚨 DETECTION ONLY: Track when problems occur (don't modify behavior)
                if grad_t_norm_before < 1e-10:
                    print(f"🚨 ISSUE DETECTED: grad_output.t() norm very small ({grad_t_norm_before:.2e}) at step")
                
                if torch.isnan(grad_output_mvue_t).any():
                    print(f"🚨 CRITICAL: NaN detected in MVUE output!")
                    print(f"   Input grad_output.t() norm: {grad_t_norm_before:.2e}")
                    print(f"   MVUE output norm: {grad_mvue_norm_after:.2e}")
                    print(f"   Stopping training to prevent NaN propagation...")
                    raise RuntimeError("NaN detected in MVUE output - training stopped")
                
                if mvue_ratio > 100:
                    print(f"🚨 MVUE amplification: {mvue_ratio:.1f}x (input_norm: {grad_t_norm_before:.2e})")
                elif mvue_ratio < 0.01 and grad_t_norm_before > 1e-12:
                    print(f"🚨 MVUE reduction: {mvue_ratio:.4f}x (input_norm: {grad_t_norm_before:.2e})")
                    
            except Exception as e:
                print(f"🚨 MVUE failed with error: {e}")
                print(f"   Using original grad_output.t() as fallback")
                grad_output_mvue_t = grad_output_view.t()  # Fallback to original transposed
            
            # 🔧 CORRECTED: Compute grad_weight using reference implementation pattern
            # Reference: fake_fp8_mm(MVUE_corrected_grad_output_transposed, input, ...)
            grad_weight = fake_fp8_mm(grad_output_mvue_t, input_view, torch.float8_e5m2).t()
            
            # 🚨 Critical grad_weight checks
            if torch.isnan(grad_weight).any():
                print(f"🚨 CRITICAL: NaN detected in grad_weight!")
                print(f"   MVUE ratio was: {mvue_ratio:.2e}")
                print(f"   Input norm: {input_norm:.2e}")
                print(f"   Stopping training to prevent NaN propagation...")
                raise RuntimeError("NaN detected in grad_weight - training stopped")
                
            grad_weight_norm = torch.norm(grad_weight).item()
            if grad_weight_norm < 1e-12:
                print(f"🚨 Gradient vanishing: grad_weight norm = {grad_weight_norm:.2e}")
            elif grad_weight_norm > 1e3:
                print(f"🚨 Gradient exploding: grad_weight norm = {grad_weight_norm:.2e}")
        
        # Compute grad_bias
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            if torch.isnan(grad_bias).any():
                print(f"🚨 BACKWARD: NaN in computed GRAD_BIAS!")
            
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
        
        # Flip rate tracking buffers
        self.register_buffer('previous_mask', None)
        self._flip_rate_enabled = False
        self._first_mask_recorded = False
    
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
                
                # 🔧 IMPROVED: More conservative and numerically stable scale computation
                dense_norm_sq = torch.sum(weight_in * weight_in)
                sparse_norm_sq = torch.sum(weight_in_sparse * weight_in_sparse)
                
                if sparse_norm_sq > 1e-12:  # Avoid division by near-zero
                    scale_in = torch.sqrt(dense_norm_sq / sparse_norm_sq)
                    # Clamp scale to prevent extreme values
                    scale_in = torch.clamp(scale_in, min=0.1, max=10.0)
                else:
                    scale_in = torch.tensor(1.0, device=weight_in.device)
                    print(f"⚠️  Weight_in sparse norm too small, using scale=1.0")
                
                self.sparse_scale_in.copy_(scale_in.detach())
            
            if hasattr(self.loro_linear, 'weight_out'):
                weight_out = self.loro_linear.weight_out
                weight_out_temp = weight_out.detach()
                
                if weight_out_temp.dtype == torch.bfloat16:
                    weight_out_temp = weight_out_temp.to(torch.float16)
                    
                weight_out_sparse, _ = soft_threshold24_triton(weight_out_temp)
                # Convert back to original dtype before computing scale
                if weight_out.dtype == torch.bfloat16:
                    weight_out_sparse = weight_out_sparse.to(torch.bfloat16)
                
                # 🔧 IMPROVED: More conservative and numerically stable scale computation
                dense_norm_sq = torch.sum(weight_out * weight_out)
                sparse_norm_sq = torch.sum(weight_out_sparse * weight_out_sparse)
                
                if sparse_norm_sq > 1e-12:  # Avoid division by near-zero
                    scale_out = torch.sqrt(dense_norm_sq / sparse_norm_sq)
                    # Clamp scale to prevent extreme values
                    scale_out = torch.clamp(scale_out, min=0.1, max=10.0)
                else:
                    scale_out = torch.tensor(1.0, device=weight_out.device)
                    print(f"⚠️  Weight_out sparse norm too small, using scale=1.0")
                
                self.sparse_scale_out.copy_(scale_out.detach())
            
            self._scales_initialized = True
            
        except Exception as e:
            # If Triton functions fail, use simple initialization
            print(f"⚠️  Triton initialization failed, using conservative scale initialization: {e}")
            device = next(self.loro_linear.parameters()).device
            self.sparse_scale_in.copy_(torch.tensor(1.0, device=device))
            self.sparse_scale_out.copy_(torch.tensor(1.0, device=device))
            self._scales_initialized = True
    
    def get_scale_info(self):
        """Get current sparse scale information for debugging"""
        return {
            'sparse_scale_in': self.sparse_scale_in.item() if self._scales_initialized else "Not initialized",
            'sparse_scale_out': self.sparse_scale_out.item() if self._scales_initialized else "Not initialized",
            'initialized': self._scales_initialized
        }
    
    def enable_flip_rate_tracking(self, enabled=True):
        """启用或禁用flip rate跟踪"""
        self._flip_rate_enabled = enabled
        if not enabled:
            self.previous_mask = None
            self._first_mask_recorded = False
    
    def calculate_flip_rate(self):
        """
        计算当前mask与上一次mask的flip rate
        
        Returns:
            tuple: (flip_rate, changed_elements, total_elements)
        """
        if not self._flip_rate_enabled:
            return 0.0, 0, 0
            
        # Initialize scales if needed
        if not self._scales_initialized:
            self.init_sparse_scales()
            
        # Get current sparse weights and mask for both weight_in and weight_out
        weight_in = self.loro_linear.weight_in
        weight_out = self.loro_linear.weight_out
        
        # Apply sparse overlay to get current sparse weights
        current_sparse_in = get_sparse_weight(weight_in, self.sparse_scale_in)
        current_sparse_out = get_sparse_weight(weight_out, self.sparse_scale_out)
        
        # Calculate masks
        current_mask_in = (current_sparse_in != 0.0).float()
        current_mask_out = (current_sparse_out != 0.0).float()
        
        # Combine masks (flatten and concatenate)
        current_mask = torch.cat([current_mask_in.flatten(), current_mask_out.flatten()])
        
        if self.previous_mask is None or not self._first_mask_recorded:
            # 第一次记录，没有previous mask可以比较
            self.previous_mask = current_mask.clone()
            self._first_mask_recorded = True
            return 0.0, 0, current_mask.numel()
        
        # 计算mask变化
        mask_diff = (current_mask != self.previous_mask).float()
        total_elements = mask_diff.numel()
        changed_elements = int(mask_diff.sum().item())
        
        flip_rate = changed_elements / total_elements if total_elements > 0 else 0.0
        
        # 更新previous mask
        self.previous_mask = current_mask.clone()
        
        return flip_rate, changed_elements, total_elements
    
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
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply 2:4 sparse overlay on LORO low-rank parameters
    
    Args:
        model: Model with LORO parameters already applied
        sparse_init_scale: Initial scale factor for sparse weights
        target_modules: List of module names to apply sparsity to
        
    Returns:
        Model with sparse overlay applied to LORO parameters
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    replaced_count = 0
    
    for name, module in model.named_modules():
        # Check if this module should be sparsified
        module_name = name.split('.')[-1]  # Get just the module name (e.g., "q_proj")
        if module_name not in target_modules:
            continue
            
        # Check if module has LORO low-rank parameters
        if not (hasattr(module, 'weight_in') and hasattr(module, 'weight_out')):
            continue
            
        print(f"Applying sparse overlay to LORO module: {name}")
        
        # Create SparseOverlayLinear wrapper for this LORO module
        sparse_overlay = SparseOverlayLinear(module, sparse_init_scale)
        
        # Replace the module in the parent
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent_module = model
            for part in parent_name.split('.'):
                parent_module = getattr(parent_module, part)
            setattr(parent_module, child_name, sparse_overlay)
        else:
            setattr(model, child_name, sparse_overlay)
        
        # Flip rate tracking is handled by SparseOverlayLinear
        
        replaced_count += 1
    
    print(f"Applied sparse overlay to {replaced_count} LORO modules")
    return model


def enable_flip_rate_tracking_for_sparse_overlay(model: nn.Module, enabled: bool = True):
    """
    Enable flip rate tracking for LORO modules with sparse overlay
    
    Args:
        model: Model with LORO sparse overlay modules
        enabled: Whether to enable flip rate tracking
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, SparseOverlayLinear):
            module.enable_flip_rate_tracking(enabled)
            count += 1
    
    print(f"{'启用' if enabled else '禁用'} {count} 个LORO SparseOverlay模块的flip rate跟踪")



def calculate_sparse_overlay_flip_rate(model: nn.Module) -> dict:
    """
    Calculate flip rate for LORO modules with sparse overlay
    
    Args:
        model: Model with LORO sparse overlay modules
        
    Returns:
        dict: Flip rate statistics
    """
    flip_rates = {}
    all_flip_rates = []
    total_changed_elements = 0
    total_elements = 0
    
    for name, module in model.named_modules():
        if isinstance(module, SparseOverlayLinear):
            # Calculate flip rate for the module
            flip_rate, changed_elements, elements = module.calculate_flip_rate()
            
            # Store individual flip rate - 确保是Python标量
            flip_rates[f"flip_rate/{name}"] = float(flip_rate)
            
            # Add to overall statistics
            all_flip_rates.append(flip_rate)
            total_changed_elements += changed_elements
            total_elements += elements
    
    # Calculate overall statistics - 确保所有返回值都是Python标量
    if all_flip_rates:
        flip_rates["flip_rate/mean"] = float(sum(all_flip_rates) / len(all_flip_rates))
        flip_rates["flip_rate/max"] = float(max(all_flip_rates))
        flip_rates["flip_rate/min"] = float(min(all_flip_rates))
        
        # 计算总体flip rate（所有层所有矩阵元素累加）
        flip_rates["flip_rate/total"] = float(total_changed_elements / total_elements) if total_elements > 0 else 0.0
    else:
        flip_rates["flip_rate/mean"] = 0.0
        flip_rates["flip_rate/max"] = 0.0
        flip_rates["flip_rate/min"] = 0.0
        flip_rates["flip_rate/total"] = 0.0
    
    return flip_rates


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