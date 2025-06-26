import torch
from torch import autograd, nn
from torch.cuda.amp import custom_fwd, custom_bwd

from .triton_ops import MVUE24_approx_triton


class MyLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super(MyLinear, self).__init__(in_features, out_features, bias=bias, **kwargs)

    def forward(self, x):
        x = my_linear.apply(x, self.weight, self.bias)
        return x


class my_linear(autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = torch.mm(input, weight.t())
        if bias is None:
            return output.view(*ctx.shape[:-1], -1)
        else:
            return output.view(*ctx.shape[:-1], -1) + bias

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = torch.mm(grad_output, weight).view(ctx.shape)
        if ctx.needs_input_grad[1]:
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = torch.mm(grad_output.t(), input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class SparseLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super(SparseLinear, self).__init__(in_features, out_features, bias=bias, **kwargs)

    def forward(self, x):
        x = sparse_linear.apply(x, self.weight, self.bias)
        return x


class sparse_linear(autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = torch.mm(input, weight.t())
        if bias is None:
            return output.view(*ctx.shape[:-1], -1)
        else:
            return output.view(*ctx.shape[:-1], -1) + bias

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1])
            
            # Convert bfloat16 to float16 for Triton compatibility
            weight_temp = weight.t()
            original_dtype = weight_temp.dtype
            if weight_temp.dtype == torch.bfloat16:
                weight_temp = weight_temp.to(torch.float16)
                
            weight_mvue = MVUE24_approx_triton(weight_temp)
            
            # Convert back to original dtype
            if original_dtype == torch.bfloat16:
                weight_mvue = weight_mvue.to(torch.bfloat16)
                
            grad_input = torch.mm(grad_output, weight_mvue.t()).view(ctx.shape)
        if ctx.needs_input_grad[1]:
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = torch.mm(grad_output.t(), input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias
