"""
Full-rank Linear Layer with 2:4 Sparsity Training
================================================

This module provides Sparse2to4Linear - a full-rank linear layer that applies 
2:4 sparsity training using the EXACT same implementation as the original 
2by4-pretrain-acc-examples nanoGPT implementation.

Use case: Control experiments to isolate whether issues come from LORO or 2:4 sparsity.
"""

import torch
from torch import nn, autograd
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Optional, List

# Import the exact same functions as original 2by4-pretrain-acc-examples
from sparse import matmul, MVUE24_approx_triton, soft_threshold24_triton


def calculate_mask_from_sparse_weights(sparse_weights):
    """从稀疏权重计算mask"""
    return (sparse_weights != 0.0).float()


def fake_fp8_mm(a, b, dtype):
    """EXACT copy of fake_fp8_mm from 2by4-pretrain-acc-examples/v2/nanoGPT/sparse_ops.py"""
    # Store original dtypes for potential conversion back
    original_dtype_a = a.dtype
    original_dtype_b = b.dtype
    
    # Convert to float16 for Triton compatibility (handles bfloat16 → float16)
    a = a.to(torch.float16).contiguous()
    b = b.to(torch.float16).contiguous()
    output = matmul(a, b, c_dtype=torch.float32)
    
    # Convert output back to appropriate precision based on input types
    if original_dtype_a == torch.bfloat16 or original_dtype_b == torch.bfloat16:
        output = output.to(torch.bfloat16)
    
    return output


class fp8_linear(autograd.Function):
    """EXACT copy of fp8_linear from 2by4-pretrain-acc-examples/v2/nanoGPT/sparse_ops.py"""
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


class SoftThreshold(autograd.Function):
    """EXACT copy of SoftThreshold from 2by4-pretrain-acc-examples/v2/nanoGPT/sparse_ops.py"""
    @staticmethod
    def forward(ctx, weight, scale):
        weight_temp = weight.detach()
        
        # Convert bfloat16 to float16 for Triton compatibility
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


class Sparse2to4Linear(nn.Linear):
    """
    EXACT copy of FP8SparseLinear from 2by4-pretrain-acc-examples/v2/nanoGPT/sparse_ops.py
    
    This inherits from nn.Linear which is automatically multiprocessing-safe.
    Using the exact same implementation as the working reference code.
    
    CRITICAL FIX: Delays Triton kernel calls to avoid multiprocessing serialization issues.
    
    Added: Flip Rate calculation to track mask changes during training.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super(Sparse2to4Linear, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.register_buffer('scale', torch.tensor(0.))
        self._scale_initialized = False
        
        # Flip rate tracking buffers
        self.register_buffer('previous_mask', None)
        self._flip_rate_enabled = False
        self._first_mask_recorded = False

    def enable_flip_rate_tracking(self, enabled=True):
        """启用或禁用flip rate跟踪"""
        self._flip_rate_enabled = enabled
        if not enabled:
            self.previous_mask = None
            self._first_mask_recorded = False

    def get_sparse_weights(self):
        # CRITICAL: Only call Triton kernels when actually needed (in forward pass)
        # This avoids serialization issues in DataLoader workers
        if not self._scale_initialized:
            self._lazy_init_scale()
        return SoftThreshold.apply(self.weight, self.scale)

    def calculate_flip_rate(self):
        """
        计算当前mask与上一次mask的flip rate
        
        Returns:
            tuple: (flip_rate, changed_elements, total_elements)
        """
        if not self._flip_rate_enabled:
            return 0.0, 0, 0
            
        # 获取当前sparse权重和mask
        current_sparse_weights = self.get_sparse_weights()
        current_mask = calculate_mask_from_sparse_weights(current_sparse_weights)
        
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

    @torch.no_grad()
    def _lazy_init_scale(self):
        """延迟初始化scale，避免在DataLoader序列化时调用Triton kernels"""
        if self._scale_initialized:
            return
            
        weight = self.weight.cuda()
        weight_temp = weight.detach()
        
        # Convert bfloat16 to float16 for Triton compatibility
        if weight_temp.dtype == torch.bfloat16:
            weight_temp = weight_temp.to(torch.float16)
            
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        
        # CRITICAL FIX: Convert weight_sparse back to original dtype for scale calculation
        if weight.dtype == torch.bfloat16:
            weight_sparse = weight_sparse.to(torch.bfloat16)
        
        # Now use original weight exactly as in reference implementation
        scale_value = torch.dot(torch.flatten(weight), torch.flatten(weight_sparse)) / torch.dot(
            torch.flatten(weight_sparse), torch.flatten(weight_sparse))
        self.scale.copy_(scale_value.cpu())
        self._scale_initialized = True

    @torch.no_grad()
    def init_scale(self):
        """公共接口，确保scale已初始化"""
        self._lazy_init_scale()

    def forward(self, x):
        w = self.get_sparse_weights()
        x = fp8_linear.apply(x, w, self.bias)
        return x


def apply_sparse2to4_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply 2:4 sparsity to target Linear modules in the model
    
    CRITICAL FIX: Uses lazy initialization to avoid Triton kernel calls 
    during model setup, preventing DataLoader multiprocessing issues.
    
    Args:
        model: The model to modify
        target_modules: List of module names to replace (e.g., ["q_proj", "v_proj"])
        exclude_modules: List of module names to exclude from replacement
        
    Returns:
        Modified model with Sparse2to4Linear modules
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    if exclude_modules is None:
        exclude_modules = []
    
    replaced_count = 0
    
    # Replace modules
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if (isinstance(child_module, nn.Linear) and 
                child_name in target_modules and 
                child_name not in exclude_modules):
                
                # Create Sparse2to4Linear replacement
                sparse_linear = Sparse2to4Linear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    bias=child_module.bias is not None,
                    device=child_module.weight.device,
                    dtype=child_module.weight.dtype,
                )
                
                # Copy the original weights and bias
                with torch.no_grad():
                    sparse_linear.weight.copy_(child_module.weight)
                    if sparse_linear.bias is not None and child_module.bias is not None:
                        sparse_linear.bias.copy_(child_module.bias)
                
                # Replace the module
                setattr(module, child_name, sparse_linear)
                replaced_count += 1
                print(f"Replaced {name}.{child_name} with Sparse2to4Linear")
    
    if replaced_count == 0:
        print("⚠️  No Linear modules were replaced. Check your target_modules list.")
    else:
        print(f"✅ Successfully replaced {replaced_count} Linear modules with Sparse2to4Linear")
    
    return model


def enable_flip_rate_tracking_for_model(model: nn.Module, enabled: bool = True):
    """
    为模型中所有的Sparse2to4Linear模块启用或禁用flip rate跟踪
    
    Args:
        model: 包含Sparse2to4Linear模块的模型
        enabled: 是否启用flip rate跟踪
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Sparse2to4Linear):
            module.enable_flip_rate_tracking(enabled)
            count += 1
    
    print(f"{'启用' if enabled else '禁用'} {count} 个Sparse2to4Linear模块的flip rate跟踪")


def calculate_model_flip_rate(model: nn.Module) -> dict:
    """
    计算模型中所有Sparse2to4Linear模块的flip rate
    
    Args:
        model: 包含Sparse2to4Linear模块的模型
        
    Returns:
        dict: 包含各层flip rate和总体统计的字典
    """
    flip_rates = {}
    all_flip_rates = []
    total_changed_elements = 0
    total_elements = 0
    
    for name, module in model.named_modules():
        if isinstance(module, Sparse2to4Linear):
            flip_rate, changed_elements, elements = module.calculate_flip_rate()
            flip_rates[f"flip_rate/{name}"] = flip_rate
            all_flip_rates.append(flip_rate)
            
            # 累加所有层的元素数用于计算总体flip rate
            total_changed_elements += changed_elements
            total_elements += elements
    
    # 计算总体统计
    if all_flip_rates:
        flip_rates["flip_rate/mean"] = sum(all_flip_rates) / len(all_flip_rates)
        flip_rates["flip_rate/max"] = max(all_flip_rates)
        flip_rates["flip_rate/min"] = min(all_flip_rates)
        
        # 计算总体flip rate（所有层元素累加）
        flip_rates["flip_rate/total"] = total_changed_elements / total_elements if total_elements > 0 else 0.0
    else:
        # 没有Sparse2to4Linear模块时返回0
        flip_rates["flip_rate/mean"] = 0.0
        flip_rates["flip_rate/max"] = 0.0
        flip_rates["flip_rate/min"] = 0.0
        flip_rates["flip_rate/total"] = 0.0
    
    return flip_rates


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
        layer.init_scale()
        
        x = torch.randn(8, 1024, 768).cuda()
        output = layer(x)
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Sparse scale: {layer.scale.item():.4f}")
        
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
        model = apply_sparse2to4_to_model(
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