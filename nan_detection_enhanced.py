"""
Enhanced NaN detection module for debugging training issues.
This module provides detailed tracking of NaN propagation through the model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import traceback
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TensorInfo:
    """Store detailed information about a tensor."""
    shape: List[int]
    dtype: str
    device: str
    has_nan: bool
    has_inf: bool
    nan_count: int = 0
    inf_count: int = 0
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None
    zero_ratio: float = 0.0
    grad_norm: Optional[float] = None


class NaNTracker:
    """Advanced NaN tracking system for debugging neural network training."""
    
    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.nan_history = []
        self.first_nan_location = None
        self.hooks = []
        self.intermediate_values = {}
        self.gradient_info = {}
        
    def analyze_tensor(self, tensor: torch.Tensor, name: str = "") -> TensorInfo:
        """Analyze a tensor and return detailed information."""
        if tensor is None:
            return TensorInfo([], "None", "None", False, False)
        
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        
        # Skip non-floating point tensors
        if not torch.is_floating_point(tensor) and not torch.is_complex(tensor):
            return TensorInfo(
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                has_nan=False,
                has_inf=False
            )
        
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        info = TensorInfo(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            has_nan=has_nan,
            has_inf=has_inf
        )
        
        if has_nan:
            info.nan_count = torch.isnan(tensor).sum().item()
        if has_inf:
            info.inf_count = torch.isinf(tensor).sum().item()
        
        # Calculate statistics only for clean tensors
        if not has_nan and not has_inf:
            info.min_val = tensor.min().item()
            info.max_val = tensor.max().item()
            info.mean_val = tensor.mean().item()
            info.std_val = tensor.std().item() if tensor.numel() > 1 else 0.0
            info.zero_ratio = (tensor == 0).float().mean().item()
        
        # Check gradient if available
        if hasattr(tensor, 'grad') and tensor.grad is not None:
            if not torch.isnan(tensor.grad).any() and not torch.isinf(tensor.grad).any():
                info.grad_norm = tensor.grad.norm().item()
        
        return info
    
    def check_model_parameters(self) -> Dict[str, TensorInfo]:
        """Check all model parameters for NaN/Inf."""
        param_info = {}
        for name, param in self.model.named_parameters():
            if param is not None:
                info = self.analyze_tensor(param, name)
                if info.has_nan or info.has_inf:
                    param_info[name] = info
                    if self.verbose:
                        print(f"⚠️ NaN/Inf in parameter {name}: {info}")
        return param_info
    
    def check_model_gradients(self) -> Dict[str, TensorInfo]:
        """Check all model gradients for NaN/Inf."""
        grad_info = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                info = self.analyze_tensor(param.grad, f"{name}.grad")
                if info.has_nan or info.has_inf:
                    grad_info[name] = info
                    if self.verbose:
                        print(f"⚠️ NaN/Inf in gradient {name}: {info}")
        return grad_info
    
    def register_hooks(self):
        """Register forward and backward hooks for all modules."""
        self.remove_hooks()  # Clean up any existing hooks
        
        def make_forward_hook(module_name):
            def forward_hook(module, input, output):
                # Check input
                input_info = None
                if isinstance(input, torch.Tensor):
                    input_info = self.analyze_tensor(input, f"{module_name}_input")
                elif isinstance(input, tuple) and len(input) > 0:
                    input_info = self.analyze_tensor(input[0], f"{module_name}_input")
                
                # Check output
                output_info = None
                if isinstance(output, torch.Tensor):
                    output_info = self.analyze_tensor(output, f"{module_name}_output")
                elif isinstance(output, tuple) and len(output) > 0:
                    output_info = self.analyze_tensor(output[0], f"{module_name}_output")
                
                # Store intermediate values
                self.intermediate_values[f"{module_name}_forward"] = {
                    'input': input_info,
                    'output': output_info
                }
                
                # Track first NaN occurrence
                if (input_info and (input_info.has_nan or input_info.has_inf)) or \
                   (output_info and (output_info.has_nan or output_info.has_inf)):
                    if self.first_nan_location is None:
                        self.first_nan_location = {
                            'module': module_name,
                            'stage': 'forward',
                            'input_info': input_info,
                            'output_info': output_info
                        }
                        if self.verbose:
                            print(f"\n🔴 First NaN/Inf detected in {module_name} (forward)")
                            if input_info and (input_info.has_nan or input_info.has_inf):
                                print(f"   Input: {input_info}")
                            if output_info and (output_info.has_nan or output_info.has_inf):
                                print(f"   Output: {output_info}")
                            self._check_module_internals(module, module_name)
            return forward_hook
        
        def make_backward_hook(module_name):
            def backward_hook(module, grad_input, grad_output):
                # Check grad_output (gradient flowing into this module)
                grad_out_info = None
                if isinstance(grad_output, torch.Tensor):
                    grad_out_info = self.analyze_tensor(grad_output, f"{module_name}_grad_output")
                elif isinstance(grad_output, tuple) and len(grad_output) > 0 and grad_output[0] is not None:
                    grad_out_info = self.analyze_tensor(grad_output[0], f"{module_name}_grad_output")
                
                # Check grad_input (gradient flowing out of this module)
                grad_in_info = None
                if isinstance(grad_input, torch.Tensor):
                    grad_in_info = self.analyze_tensor(grad_input, f"{module_name}_grad_input")
                elif isinstance(grad_input, tuple) and len(grad_input) > 0 and grad_input[0] is not None:
                    grad_in_info = self.analyze_tensor(grad_input[0], f"{module_name}_grad_input")
                
                # Store gradient info
                self.gradient_info[f"{module_name}_backward"] = {
                    'grad_input': grad_in_info,
                    'grad_output': grad_out_info
                }
                
                # Track first NaN in backward
                if (grad_in_info and (grad_in_info.has_nan or grad_in_info.has_inf)) or \
                   (grad_out_info and (grad_out_info.has_nan or grad_out_info.has_inf)):
                    if self.first_nan_location is None or self.first_nan_location.get('stage') != 'backward':
                        self.first_nan_location = {
                            'module': module_name,
                            'stage': 'backward',
                            'grad_input_info': grad_in_info,
                            'grad_output_info': grad_out_info
                        }
                        if self.verbose:
                            print(f"\n🔴 First NaN/Inf detected in {module_name} (backward)")
                            if grad_out_info and (grad_out_info.has_nan or grad_out_info.has_inf):
                                print(f"   Grad Output: {grad_out_info}")
                            if grad_in_info and (grad_in_info.has_nan or grad_in_info.has_inf):
                                print(f"   Grad Input: {grad_in_info}")
            return backward_hook
        
        # Register hooks for all modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                forward_hook = module.register_forward_hook(make_forward_hook(name))
                backward_hook = module.register_backward_hook(make_backward_hook(name))
                self.hooks.extend([forward_hook, backward_hook])
    
    def _check_module_internals(self, module: nn.Module, module_name: str):
        """Check internal state of specific module types."""
        # Check for low-rank modules
        if hasattr(module, 'weight_in') and hasattr(module, 'weight_out'):
            win_info = self.analyze_tensor(module.weight_in, "weight_in")
            wout_info = self.analyze_tensor(module.weight_out, "weight_out")
            if win_info.has_nan or win_info.has_inf:
                print(f"   ⚠️ weight_in has NaN/Inf: {win_info}")
            if wout_info.has_nan or wout_info.has_inf:
                print(f"   ⚠️ weight_out has NaN/Inf: {wout_info}")
        
        # Check for standard linear layers
        if hasattr(module, 'weight'):
            weight_info = self.analyze_tensor(module.weight, "weight")
            if weight_info.has_nan or weight_info.has_inf:
                print(f"   ⚠️ weight has NaN/Inf: {weight_info}")
        
        # Check for activation sparse modules
        if hasattr(module, 'scale_factor'):
            if torch.is_tensor(module.scale_factor):
                scale_info = self.analyze_tensor(module.scale_factor, "scale_factor")
                if scale_info.has_nan or scale_info.has_inf:
                    print(f"   ⚠️ scale_factor has NaN/Inf: {scale_info}")
            else:
                print(f"   scale_factor: {module.scale_factor}")
        
        # Check sparsity tracker if available
        if hasattr(module, 'sparsity_tracker'):
            print(f"   sparsity_tracker: {module.sparsity_tracker}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def track_forward_backward(self, batch, labels, loss_fn=None):
        """Track a complete forward-backward pass."""
        self.register_hooks()
        self.intermediate_values.clear()
        self.gradient_info.clear()
        self.first_nan_location = None
        
        try:
            # Forward pass
            if self.verbose:
                print("\n" + "="*80)
                print("Starting NaN tracking for forward pass...")
            
            output = self.model(**batch, labels=labels)
            loss = output.loss if hasattr(output, 'loss') else loss_fn(output, labels)
            
            loss_info = self.analyze_tensor(loss, "loss")
            if self.verbose:
                print(f"\nLoss info: {loss_info}")
            
            if loss_info.has_nan or loss_info.has_inf:
                print("\n" + "="*80)
                print("⚠️ NaN/Inf detected in loss!")
                self.summarize_nan_propagation()
            
            # Backward pass
            if self.verbose:
                print("\n" + "="*80)
                print("Starting NaN tracking for backward pass...")
            
            loss.backward()
            
            # Check gradients
            grad_issues = self.check_model_gradients()
            if grad_issues and self.verbose:
                print(f"\nFound {len(grad_issues)} gradients with NaN/Inf")
            
        finally:
            self.remove_hooks()
        
        return loss, self.first_nan_location
    
    def summarize_nan_propagation(self):
        """Summarize NaN propagation through the model."""
        print("\n" + "="*80)
        print("NaN PROPAGATION SUMMARY")
        print("="*80)
        
        if self.first_nan_location:
            print(f"\n🔴 First NaN/Inf detected:")
            print(f"   Module: {self.first_nan_location['module']}")
            print(f"   Stage: {self.first_nan_location['stage']}")
            
            if self.first_nan_location['stage'] == 'forward':
                if 'input_info' in self.first_nan_location:
                    print(f"   Input: {self.first_nan_location['input_info']}")
                if 'output_info' in self.first_nan_location:
                    print(f"   Output: {self.first_nan_location['output_info']}")
            else:
                if 'grad_output_info' in self.first_nan_location:
                    print(f"   Grad Output: {self.first_nan_location['grad_output_info']}")
                if 'grad_input_info' in self.first_nan_location:
                    print(f"   Grad Input: {self.first_nan_location['grad_input_info']}")
        
        # Count total NaN occurrences
        nan_count = 0
        for key, value in self.intermediate_values.items():
            if value.get('input') and value['input'].has_nan:
                nan_count += 1
            if value.get('output') and value['output'].has_nan:
                nan_count += 1
        
        for key, value in self.gradient_info.items():
            if value.get('grad_input') and value['grad_input'].has_nan:
                nan_count += 1
            if value.get('grad_output') and value['grad_output'].has_nan:
                nan_count += 1
        
        print(f"\n📊 Total NaN/Inf occurrences: {nan_count}")
        
        # Show propagation path (first 10 occurrences)
        print("\n📍 NaN Propagation Path:")
        count = 0
        for key, value in self.intermediate_values.items():
            if count >= 10:
                break
            if (value.get('input') and value['input'].has_nan) or \
               (value.get('output') and value['output'].has_nan):
                print(f"   {count+1}. {key}")
                count += 1


def debug_split_gemm(dy1, weight1, layer_id=None):
    """Debug split_gemm computation to find NaN sources."""
    print("\n" + "="*80)
    print("DEBUG: Split-GEMM Analysis")
    print("="*80)
    
    # Check inputs
    dy1_info = TensorInfo(
        shape=list(dy1.shape),
        dtype=str(dy1.dtype),
        device=str(dy1.device),
        has_nan=torch.isnan(dy1).any().item(),
        has_inf=torch.isinf(dy1).any().item()
    )
    
    weight1_info = TensorInfo(
        shape=list(weight1.shape),
        dtype=str(weight1.dtype),
        device=str(weight1.device),
        has_nan=torch.isnan(weight1).any().item(),
        has_inf=torch.isinf(weight1).any().item()
    )
    
    print(f"dy1: {dy1_info}")
    print(f"weight1: {weight1_info}")
    
    if dy1_info.has_nan or dy1_info.has_inf:
        print("⚠️ dy1 already contains NaN/Inf!")
        return
    
    if weight1_info.has_nan or weight1_info.has_inf:
        print("⚠️ weight1 already contains NaN/Inf!")
        return
    
    # Check sparsity calculation
    feature_sparsity = torch.mean((dy1 != 0).float(), dim=0)
    print(f"\nFeature sparsity shape: {feature_sparsity.shape}")
    print(f"Feature sparsity range: [{feature_sparsity.min():.4f}, {feature_sparsity.max():.4f}]")
    
    # Check for NaN in sparsity
    if torch.isnan(feature_sparsity).any():
        print("⚠️ NaN in feature sparsity calculation!")
        nan_indices = torch.where(torch.isnan(feature_sparsity))[0]
        print(f"   NaN at indices: {nan_indices[:10].tolist()}...")
    
    # Try the split
    num_features = feature_sparsity.shape[0]
    num_sparse = int(0.95 * num_features)
    
    # Check if we have cached sparsity
    if layer_id:
        try:
            from fused_sparsity_ops import sparsity_tracker
            col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
            print(f"\n✅ Found cached sparsity for layer {layer_id}")
            print(f"   Sparse mask shape: {sparse_mask.shape}")
            print(f"   Number of sparse features: {sparse_mask.sum().item()}")
        except:
            print(f"\n⚠️ No cached sparsity found for layer {layer_id}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Test the NaN tracker
    print("NaN Tracker Test Module")
    print("This module provides enhanced NaN detection for debugging training issues.")