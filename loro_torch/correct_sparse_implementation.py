"""
Correct 2:4 Sparse Implementation matching 2by4-pretrain-acc-examples exactly.
This file contains the EXACT same implementation as 2by4-pretrain-acc-examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.cuda.amp import custom_fwd, custom_bwd
import math

# Import the correct 2by4 sparse implementation
try:
    from sparse import matmul, MVUE24_approx_triton, soft_threshold24_triton
    SPARSE_AVAILABLE = True
    print("✓ Using correct 2by4-pretrain-acc-examples sparse implementation")
except ImportError:
    SPARSE_AVAILABLE = False
    print("✗ 2by4-pretrain-acc-examples sparse package not available")
    
    def matmul(a, b, c_dtype=torch.float32):
        """Fallback implementation - NOT CORRECT"""
        return torch.matmul(a.to(c_dtype), b.to(c_dtype))
    
    def MVUE24_approx_triton(x):
        """Fallback implementation - NOT CORRECT"""
        return x
    
    def soft_threshold24_triton(weight):
        """Fallback implementation - NOT CORRECT"""
        raise RuntimeError("Fallback 2:4 sparse implementation detected! Must use correct 2by4-pretrain-acc-examples implementation!")


def fake_fp8_mm(a, b, dtype):
    """Simulate FP8 precision - EXACT copy from 2by4-pretrain-acc-examples"""
    a = a.to(torch.float16).contiguous()
    b = b.to(torch.float16).contiguous()
    output = matmul(a, b, c_dtype=torch.float32)
    return output


class FP8SparseOperation(autograd.Function):
    """Core 2:4 sparse matrix multiplication - EXACT copy from 2by4-pretrain-acc-examples"""
    
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
            grad_output_view = grad_output.view(-1, grad_output.shape[-1])
            grad_input = fake_fp8_mm(grad_output_view, weight, torch.float8_e5m2).view(ctx.shape)
            
        if ctx.needs_input_grad[1]:
            input_view = input.view(-1, input.shape[-1])
            grad_output_view = grad_output.view(-1, grad_output.shape[-1])
            # 关键：使用MVUE梯度估算 - EXACT copy from 2by4-pretrain-acc-examples
            grad_weight = fake_fp8_mm(MVUE24_approx_triton(grad_output_view.t()), input_view, torch.float8_e5m2)
            
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            
        return grad_input, grad_weight, grad_bias


class SoftThreshold2to4(autograd.Function):
    """2:4 sparsification with learnable scaling - EXACT copy from 2by4-pretrain-acc-examples"""
    
    @staticmethod
    def forward(ctx, weight, scale):
        weight_temp = weight.detach()
        # 关键：使用正确的soft_threshold24_triton实现
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        return weight_sparse * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class CorrectFP8SparseLinear(nn.Linear):
    """
    Correct 2:4 sparse linear layer - EXACT copy from 2by4-pretrain-acc-examples/v2/nanoGPT/sparse_ops.py
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.register_buffer('scale', torch.tensor(0.))

    def get_sparse_weights(self):
        return SoftThreshold2to4.apply(self.weight, self.scale)

    @torch.no_grad()
    def init_scale(self):
        """Initialize scale parameter - EXACT copy from 2by4-pretrain-acc-examples"""
        weight = self.weight.cuda()
        weight_temp = weight.detach()
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        
        # 关键：正确的缩放计算
        weight.scale = torch.dot(torch.flatten(weight), torch.flatten(weight_sparse)) / torch.dot(
            torch.flatten(weight_sparse), torch.flatten(weight_sparse))
        self.scale.copy_(weight.scale.cpu())
        self.weight.scale = self.scale

    def forward(self, x):
        w = self.get_sparse_weights()
        x = FP8SparseOperation.apply(x, w, self.bias)
        return x


class CorrectLowRankSparse2to4Linear(nn.Module):
    """
    Correct LoRA + 2:4 Sparse combination using exact 2by4-pretrain-acc-examples implementation
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
        init_range: float = None,
        device=None, 
        dtype=None
    ):
        super().__init__()
        
        if not SPARSE_AVAILABLE:
            raise RuntimeError("Cannot create CorrectLowRankSparse2to4Linear: 2by4-pretrain-acc-examples sparse package not available!")
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.enable_sparse = enable_sparse
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
        """Initialize sparse scales for both A and B matrices - EXACT copy from 2by4-pretrain-acc-examples"""
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
        
        # Low-rank computation with sparsity using EXACT 2by4 implementation
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


def check_sparse_implementation():
    """验证我们使用的是正确的2by4实现"""
    if not SPARSE_AVAILABLE:
        print("❌ 错误：未使用2by4-pretrain-acc-examples的sparse实现")
        return False
    
    # 测试软阈值函数
    test_weight = torch.randn(4, 4).cuda()
    try:
        sparse_weight, mask = soft_threshold24_triton(test_weight)
        # 验证2:4模式
        for i in range(0, test_weight.numel(), 4):
            block = mask.view(-1)[i:i+4]
            if block.sum().item() != 2:
                print(f"❌ 第{i//4}个block的稀疏度不是2:4: {block.sum().item()}/4")
                return False
        
        print("✅ 确认使用了正确的2by4-pretrain-acc-examples实现")
        print(f"   - soft_threshold24_triton类型: {type(soft_threshold24_triton)}")
        print(f"   - MVUE24_approx_triton类型: {type(MVUE24_approx_triton)}")
        print(f"   - matmul类型: {type(matmul)}")
        return True
        
    except Exception as e:
        print(f"❌ 测试sparse实现时出错: {e}")
        return False


if __name__ == "__main__":
    check_sparse_implementation() 