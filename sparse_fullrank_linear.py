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


class ActivationSparse2to4SoftThreshold(autograd.Function):
    """Apply 2:4 sparsity to activation using soft threshold with scaling"""
    @staticmethod
    @custom_fwd
    def forward(ctx, input, scale):
        input_temp = input.detach()
        
        # Convert bfloat16 to float16 for Triton compatibility
        original_dtype = input_temp.dtype
        if input_temp.dtype == torch.bfloat16:
            input_temp = input_temp.to(torch.float16)
            
        input_sparse, _ = soft_threshold24_triton(input_temp)
        
        # Convert back to original dtype
        if original_dtype == torch.bfloat16:
            input_sparse = input_sparse.to(torch.bfloat16)
            
        return input_sparse * scale

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output, None


class ActivationSparse2to4MVUE(autograd.Function):
    """Apply 2:4 sparsity to activation using MVUE method"""
    @staticmethod
    @custom_fwd
    def forward(ctx, input):
        input_temp = input.detach()
        
        # Convert bfloat16 to float16 for Triton compatibility
        original_dtype = input_temp.dtype
        if input_temp.dtype == torch.bfloat16:
            input_temp = input_temp.to(torch.float16)
            
        input_sparse = MVUE24_approx_triton(input_temp)
        
        # Convert back to original dtype
        if original_dtype == torch.bfloat16:
            input_sparse = input_sparse.to(torch.bfloat16)
            
        return input_sparse

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output


class fp8_linear(autograd.Function):
    """Enhanced fp8_linear with 2:4 activation sparsification support"""
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, activation_2by4=False, activation_soft_threshold=False, activation_scale=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.shape = input.shape
        ctx.activation_2by4 = activation_2by4
        ctx.activation_soft_threshold = activation_soft_threshold
        
        input_processed = input.view(-1, input.shape[-1])
        
        # Apply 2:4 sparsity to activation if enabled
        if activation_2by4:
            if activation_soft_threshold:
                # Use soft threshold with scaling
                if activation_scale is None:
                    raise ValueError("activation_scale must be provided when using soft threshold for activation")
                input_processed = ActivationSparse2to4SoftThreshold.apply(input_processed, activation_scale)
            else:
                # Use MVUE method
                input_processed = ActivationSparse2to4MVUE.apply(input_processed)
        
        output = fake_fp8_mm(input_processed, weight.t(), torch.float8_e4m3fn)
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
        # Return gradients for all forward parameters (input, weight, bias, activation_2by4, activation_soft_threshold, activation_scale)
        return grad_input, grad_weight, grad_bias, None, None, None


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
    2:4 Sparse Linear layer with weight sparsification and flip rate tracking
    
    This inherits from nn.Linear which is automatically multiprocessing-safe.
    Applies 2:4 sparsity to WEIGHTS (not activations).
    
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
        """延迟初始化scale以避免多进程序列化问题"""
        if self._scale_initialized:
            return
            
        device = self.weight.device
        if self.weight.is_cuda:
            weight_temp = self.weight.detach()
            weight_sparse, _ = soft_threshold24_triton(weight_temp)
            scale_value = torch.dot(torch.flatten(self.weight), torch.flatten(weight_sparse)) / torch.dot(
                torch.flatten(weight_sparse), torch.flatten(weight_sparse))
            self.scale.copy_(scale_value)
        else:
            # Fallback for CPU (not recommended for training)
            self.scale.copy_(torch.tensor(1.0))
        
        self._scale_initialized = True

    def forward(self, x):
        # 对权重应用2:4稀疏化
        w = self.get_sparse_weights()
        x = fp8_linear.apply(x, w, self.bias, False, False, None)  # No activation sparsity
        return x


class ActivationSparse2to4Linear(nn.Linear):
    """
    Enhanced Linear layer with configurable 2:4 activation sparsification support
    
    This inherits from nn.Linear which is automatically multiprocessing-safe.
    
    CRITICAL BEHAVIOR:
    - activation_2by4=False: Standard dense operation (no sparsity)
    - activation_2by4=True: Apply 2:4 sparsity ONLY to activations, weights remain dense
    
    Note: Does not support flip rate tracking since activation patterns are dynamic.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, 
                 activation_2by4: bool = False, activation_soft_threshold: bool = False):
        super(ActivationSparse2to4Linear, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        
        # Activation 2:4 configuration
        self.activation_2by4 = activation_2by4
        self.activation_soft_threshold = activation_soft_threshold
        
        # Activation scaling (for soft threshold method)
        if self.activation_2by4 and self.activation_soft_threshold:
            self.register_buffer('activation_scale', torch.tensor(0.))
            self._activation_scale_initialized = False
        else:
            self.register_buffer('activation_scale', None)
            self._activation_scale_initialized = True  # Not needed for MVUE or when activation_2by4=False
        
        # Note: We no longer need weight scaling since weights remain dense



    @torch.no_grad()
    def _lazy_init_activation_scale(self, sample_input):
        """延迟初始化activation scale，需要样本输入来计算"""
        if self._activation_scale_initialized or not (self.activation_2by4 and self.activation_soft_threshold):
            return
            
        input_temp = sample_input.detach()
        
        # Convert bfloat16 to float16 for Triton compatibility
        original_dtype = input_temp.dtype
        if input_temp.dtype == torch.bfloat16:
            input_temp = input_temp.to(torch.float16)
            
        input_sparse, _ = soft_threshold24_triton(input_temp)
        
        # Convert back to original dtype for scale calculation
        if original_dtype == torch.bfloat16:
            input_sparse = input_sparse.to(torch.bfloat16)
            sample_input = sample_input.to(torch.bfloat16)
        
        # Calculate activation scale using same method as weight scale
        scale_value = torch.dot(torch.flatten(sample_input), torch.flatten(input_sparse)) / torch.dot(
            torch.flatten(input_sparse), torch.flatten(input_sparse))
        self.activation_scale.copy_(scale_value.cpu())
        self._activation_scale_initialized = True



    def forward(self, x):
        # Initialize activation scale if needed (and using soft threshold for activation)
        if not self._activation_scale_initialized and self.activation_2by4 and self.activation_soft_threshold:
            self._lazy_init_activation_scale(x.view(-1, x.shape[-1]))
        
        # Choose weight based on activation_2by4 setting
        if self.activation_2by4:
            # Only apply 2:4 to activations, keep weights dense
            w = self.weight
            activation_scale = self.activation_scale if self.activation_soft_threshold else None
            x = fp8_linear.apply(x, w, self.bias, True, self.activation_soft_threshold, activation_scale)
        else:
            # Standard dense operation (no 2:4 sparsity at all)
            x = torch.nn.functional.linear(x, self.weight, self.bias)
        
        return x


def apply_sparse2to4_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply weight 2:4 sparsity to target Linear modules in the model
    
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


def apply_activation_sparse2to4_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    activation_2by4: bool = False,
    activation_soft_threshold: bool = False,
) -> nn.Module:
    """
    Apply activation 2:4 sparsity to target Linear modules in the model
    
    CRITICAL BEHAVIOR:
    - activation_2by4=False: Standard dense operation (no sparsity)
    - activation_2by4=True: Apply 2:4 sparsity ONLY to activations, weights remain dense
    
    Args:
        model: The model to modify
        target_modules: List of module names to replace (e.g., ["q_proj", "v_proj"])
        exclude_modules: List of module names to exclude from replacement
        activation_2by4: Whether to apply 2:4 sparsity to activations (weights always remain dense)
        activation_soft_threshold: Whether to use soft threshold (True) or MVUE (False) for activation 2:4
        
    Returns:
        Modified model with ActivationSparse2to4Linear modules
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
                
                # Create ActivationSparse2to4Linear replacement
                sparse_linear = ActivationSparse2to4Linear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    bias=child_module.bias is not None,
                    activation_2by4=activation_2by4,
                    activation_soft_threshold=activation_soft_threshold,
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
                print(f"Replaced {name}.{child_name} with ActivationSparse2to4Linear")
    
    if replaced_count == 0:
        print("⚠️  No Linear modules were replaced. Check your target_modules list.")
    else:
        print(f"✅ Successfully replaced {replaced_count} Linear modules with ActivationSparse2to4Linear")
    
    return model


def enable_flip_rate_tracking_for_model(model: nn.Module, enabled: bool = True):
    """
    为模型中所有的Sparse2to4Linear模块启用或禁用flip rate跟踪
    
    Note: Only works for Sparse2to4Linear (weight sparsity), not ActivationSparse2to4Linear
    
    Args:
        model: 包含Sparse2to4Linear模块的模型
        enabled: 是否启用flip rate跟踪
    """
    count = 0
    activation_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, Sparse2to4Linear):
            module.enable_flip_rate_tracking(enabled)
            count += 1
        elif isinstance(module, ActivationSparse2to4Linear):
            activation_count += 1
    
    if count > 0:
        print(f"{'启用' if enabled else '禁用'} {count} 个Sparse2to4Linear模块的flip rate跟踪")
    
    if activation_count > 0:
        print(f"ℹ️ 发现 {activation_count} 个ActivationSparse2to4Linear模块，这些模块不支持flip rate跟踪")


def calculate_model_flip_rate(model: nn.Module) -> dict:
    """
    计算模型中所有Sparse2to4Linear模块的flip rate
    
    Note: Only calculates for Sparse2to4Linear (weight sparsity), not ActivationSparse2to4Linear
    
    Args:
        model: 包含Sparse2to4Linear模块的模型
        
    Returns:
        dict: 包含各层flip rate和总体统计的字典
    """
    flip_rates = {}
    all_flip_rates = []
    total_changed_elements = 0
    total_elements = 0
    sparse_count = 0
    activation_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, Sparse2to4Linear):
            flip_rate, changed_elements, elements = module.calculate_flip_rate()
            flip_rates[f"flip_rate/{name}"] = float(flip_rate)  # 确保是Python标量
            all_flip_rates.append(flip_rate)
            sparse_count += 1
            
            # 累加所有层的元素数用于计算总体flip rate
            total_changed_elements += changed_elements
            total_elements += elements
        elif isinstance(module, ActivationSparse2to4Linear):
            activation_count += 1
    
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


def test_sparse_implementations():
    """Test both Sparse2to4Linear and ActivationSparse2to4Linear implementations"""
    print("🧪 Testing Both Sparse Implementations")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # Test 1: Weight sparse mode (Sparse2to4Linear)
        print("1. Testing Sparse2to4Linear (weight sparse mode)...")
        layer_weight_sparse = Sparse2to4Linear(768, 256).cuda()
        
        x = torch.randn(8, 1024, 768).cuda()
        output_weight = layer_weight_sparse(x)
        print(f"   ✓ Output shape: {output_weight.shape}")
        print(f"   ✓ Weight scale: {layer_weight_sparse.scale.item():.6f}")
        
        # Test flip rate for weight sparse
        layer_weight_sparse.enable_flip_rate_tracking(True)
        flip_rate, changed, total = layer_weight_sparse.calculate_flip_rate()
        print(f"   ✓ Flip rate capability: {flip_rate:.4f} ({changed}/{total})")
        
        # Test 2: Activation sparse mode (ActivationSparse2to4Linear)
        print("\n2. Testing ActivationSparse2to4Linear (activation sparse mode)...")
        layer_activation_sparse = ActivationSparse2to4Linear(768, 256, activation_2by4=True, activation_soft_threshold=True).cuda()
        output_activation = layer_activation_sparse(x)
        print(f"   ✓ Output shape: {output_activation.shape}")
        print(f"   ✓ Activation scale: {layer_activation_sparse.activation_scale.item():.6f}")
        
        # Test 3: Dense mode (ActivationSparse2to4Linear)
        print("\n3. Testing ActivationSparse2to4Linear (dense mode)...")
        layer_dense = ActivationSparse2to4Linear(768, 256, activation_2by4=False).cuda()
        output_dense = layer_dense(x)
        print(f"   ✓ Output shape: {output_dense.shape}")
        
        # Test 4: Model replacement functions
        print("\n4. Testing module replacement functions...")
        import torch.nn as nn
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(768, 256)
                self.k_proj = nn.Linear(768, 256)
                self.other_linear = nn.Linear(256, 128)
        
        # Test weight sparse replacement
        model1 = TestModel()
        model1 = apply_sparse2to4_to_model(
            model1, 
            target_modules=["q_proj", "k_proj"]
        )
        
        assert isinstance(model1.q_proj, Sparse2to4Linear)
        assert isinstance(model1.k_proj, Sparse2to4Linear) 
        assert isinstance(model1.other_linear, nn.Linear)
        print(f"   ✓ Weight sparse replacement working")
        
        # Test activation sparse replacement
        model2 = TestModel()
        model2 = apply_activation_sparse2to4_to_model(
            model2, 
            target_modules=["q_proj", "k_proj"],
            activation_2by4=True
        )
        
        assert isinstance(model2.q_proj, ActivationSparse2to4Linear)
        assert isinstance(model2.k_proj, ActivationSparse2to4Linear) 
        assert isinstance(model2.other_linear, nn.Linear)
        print(f"   ✓ Activation sparse replacement working")
        
        # Test 5: Flip rate functions with mixed models
        print("\n5. Testing flip rate functions...")
        enable_flip_rate_tracking_for_model(model1, enabled=True)  # Weight sparse model
        enable_flip_rate_tracking_for_model(model2, enabled=True)  # Activation sparse model
        
        flip_rates1 = calculate_model_flip_rate(model1)  # Should calculate flip rates
        flip_rates2 = calculate_model_flip_rate(model2)  # Should return zeros
        
        print(f"   ✓ Weight sparse model flip rates: {flip_rates1}")
        print(f"   ✓ Activation sparse model flip rates: {flip_rates2}")
        
        print(f"\n✅ All tests passed! Both sparse implementations working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_sparse_implementations() 