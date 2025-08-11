"""
Integration module for fused GEMM with sparsity computation.
This module provides drop-in replacements for existing split_gemm operations
that compute sparsity statistics in the GEMM epilogue without extra overhead.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from triton_fused_gemm import triton_matmul_with_sparsity, fused_gemm_sparsity


class SparsityTracker:
    """Global tracker for sparsity statistics computed during forward pass."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance
    
    def reset(self):
        """Reset all tracked statistics."""
        self.forward_sparsity = {}
        self.forward_masks = {}
        self.step_count = 0
    
    def update_step(self):
        """Increment step counter."""
        self.step_count += 1
    
    def store_sparsity(self, layer_id: str, sparsity: torch.Tensor, mask: torch.Tensor):
        """Store sparsity stats for a layer."""
        self.forward_sparsity[layer_id] = sparsity.detach()
        self.forward_masks[layer_id] = mask.detach()
    
    def get_sparsity(self, layer_id: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve sparsity stats for a layer."""
        return self.forward_sparsity.get(layer_id), self.forward_masks.get(layer_id)


# Global tracker instance
sparsity_tracker = SparsityTracker()


def fused_gemm_forward_with_sparsity(
    x: torch.Tensor,
    weight: torch.Tensor,
    layer_id: str,
    activation_relu2: bool = False,
    compute_2to4: bool = False,
    sparsity_threshold: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward GEMM with fused sparsity computation.
    
    This function computes Y = X @ W and column sparsity in a single pass,
    storing the sparsity stats for use in backward pass.
    
    Args:
        x: Input tensor [batch*seq, in_features]
        weight: Weight tensor [in_features, out_features]
        layer_id: Unique identifier for this layer
        activation_relu2: Whether to apply ReLU² activation
        compute_2to4: Whether to compute 2:4 sparse version
        sparsity_threshold: Threshold for determining sparse columns
    
    Returns:
        output: Result of GEMM
        sparse_output: 2:4 sparse version if requested, else same as output
    """
    # Compute GEMM with sparsity in single fused kernel
    activation = 'relu2' if activation_relu2 else 'none'
    y, col_sparsity = triton_matmul_with_sparsity(
        x, weight, 
        activation=activation,
        track_sparsity=True
    )
    y_sparse = y if compute_2to4 else None  # TODO: implement actual 2:4 sparsification
    
    # Determine sparse/dense split based on sparsity
    num_features = col_sparsity.shape[0]
    num_sparse = int(sparsity_threshold * num_features)
    
    # Get indices of most sparse columns
    if num_sparse > 0:
        sparse_values, sparse_indices = torch.topk(col_sparsity, num_sparse)
        sparse_mask = torch.zeros(num_features, dtype=torch.bool, device=x.device)
        sparse_mask[sparse_indices] = True
    else:
        sparse_mask = torch.zeros(num_features, dtype=torch.bool, device=x.device)
    
    # Store sparsity info for backward pass
    sparsity_tracker.store_sparsity(layer_id, col_sparsity, sparse_mask)
    
    # Return appropriate output
    if compute_2to4 and y_sparse is not None:
        return y, y_sparse
    else:
        return y, y


def compute_split_gemm_dw_with_cached_sparsity(
    activation: torch.Tensor,
    grad_output: torch.Tensor,
    layer_id: str,
    use_2to4: bool = True,
    transpose_result: bool = False
) -> torch.Tensor:
    """
    Compute weight gradient using cached sparsity from forward pass.
    
    This avoids recomputing sparsity in backward, using the stats
    computed during forward pass instead.
    
    Args:
        activation: Activation tensor from forward pass [batch, features]
        grad_output: Gradient from next layer [batch, out_features]
        layer_id: Unique identifier for cached sparsity lookup
        use_2to4: Whether to apply 2:4 sparsity pattern
        transpose_result: Whether to transpose the final result
    
    Returns:
        Weight gradient in appropriate shape
    """
    col_sparsity, sparse_mask = sparsity_tracker.get_sparsity(layer_id)
    
    if sparse_mask is None:
        # Compute without cached sparsity
        result = torch.mm(activation.T, grad_output)
        return result.T if transpose_result else result
    
    # Get dimensions
    batch_size, in_features = activation.shape
    batch_size2, out_features = grad_output.shape
    assert batch_size == batch_size2, "Batch size mismatch"
    
    # Initialize gradient - we compute activation.T @ grad_output
    grad_weight = torch.zeros(in_features, out_features, device=activation.device, dtype=activation.dtype)
    
    dense_mask = ~sparse_mask
    
    if use_2to4 and sparse_mask.any():
        # Apply 2:4 sparsity to sparse columns
        from sparse_fullrank_linear import fake_fp8_mm
        
        # Sparse part with 2:4 pattern
        activation_sparse = activation[:, sparse_mask]
        activation_sparse_2to4 = apply_feature_wise_2to4(activation_sparse)
        grad_weight[sparse_mask, :] = fake_fp8_mm(
            activation_sparse_2to4.T, 
            grad_output, 
            torch.float8_e4m3fn
        )
    
    # Dense part
    if dense_mask.any():
        activation_dense = activation[:, dense_mask]
        grad_weight[dense_mask, :] = torch.mm(activation_dense.T, grad_output)
    
    # Return with optional transpose
    return grad_weight.T if transpose_result else grad_weight


def apply_feature_wise_2to4(tensor: torch.Tensor) -> torch.Tensor:
    """Apply 2:4 sparsity pattern feature-wise."""
    # Reshape to ensure we have groups of 4
    orig_shape = tensor.shape
    batch_size = orig_shape[0]
    num_features = orig_shape[1]
    
    # Pad if necessary
    pad_features = (4 - num_features % 4) % 4
    if pad_features > 0:
        tensor = F.pad(tensor, (0, pad_features))
    
    # Reshape to groups of 4
    tensor = tensor.view(batch_size, -1, 4)
    
    # Keep top 2 values in each group
    values, indices = torch.topk(tensor.abs(), k=2, dim=2)
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    mask.scatter_(2, indices, True)
    
    # Apply mask
    tensor = tensor * mask
    
    # Reshape back and remove padding
    tensor = tensor.view(batch_size, -1)
    if pad_features > 0:
        tensor = tensor[:, :num_features]
    
    return tensor


class FusedSparseLinear(torch.nn.Module):
    """
    Drop-in replacement for Linear layer with fused sparsity computation.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        bias: bool = True,
        activation_relu2: bool = False,
        sparsity_threshold: float = 0.95,
        layer_id: Optional[str] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) / (in_features ** 0.5))
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None
        self.activation_relu2 = activation_relu2
        self.sparsity_threshold = sparsity_threshold
        self.layer_id = layer_id or f"linear_{id(self)}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use fused GEMM with sparsity computation
        output, _ = fused_gemm_forward_with_sparsity(
            x, 
            self.weight.T,  # Transpose for correct dimensions
            self.layer_id,
            activation_relu2=self.activation_relu2,
            sparsity_threshold=self.sparsity_threshold
        )
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


# Integration with existing ActivationSparse2to4Function
class ActivationSparse2to4FunctionOptimized(torch.autograd.Function):
    """
    Optimized version using fused sparsity computation.
    """
    
    @staticmethod
    def forward(
        ctx,
        input,
        weight1,
        weight2,
        bias1,
        bias2,
        perm,
        inv_perm,
        enable_permute,
        sparsity_method,
        warmup_steps,
        dx_direct_sparse,
        dynamic_steps,
        calibration_samples
    ):
        batch_size, seq_len, hidden_size = input.shape
        layer_id = f"mlp_{id(ctx)}"
        
        # Apply permutation if needed
        if enable_permute and perm is not None:
            input_permuted = input[:, perm, :]
        else:
            input_permuted = input
        
        # First GEMM with fused sparsity computation
        y1, _ = fused_gemm_forward_with_sparsity(
            input_permuted.view(-1, hidden_size),
            weight1.T,
            f"{layer_id}_w1",
            activation_relu2=False,
            sparsity_threshold=0.95
        )
        
        if bias1 is not None:
            y1 = y1 + bias1
        
        # ReLU² activation
        y2 = F.relu(y1) ** 2
        
        # Second GEMM with 2:4 sparsity
        current_step = getattr(ActivationSparse2to4FunctionOptimized, '_global_training_step', 0)
        is_warmup = current_step < warmup_steps
        
        if is_warmup:
            # Dense computation during warmup
            y3 = torch.mm(y2, weight2.T)
        else:
            # Use fused kernel with 2:4 sparsity
            y3, y2_sparse = fused_gemm_forward_with_sparsity(
                y2,
                weight2.T,
                f"{layer_id}_w2",
                activation_relu2=False,
                compute_2to4=True,
                sparsity_threshold=0.95
            )
        
        if bias2 is not None:
            y3 = y3 + bias2
        
        # Save for backward
        ctx.save_for_backward(
            input_permuted, weight1, weight2, bias1, bias2,
            y1, y2, y2_sparse if not is_warmup else y2
        )
        ctx.layer_id = layer_id
        ctx.perm = perm
        ctx.inv_perm = inv_perm
        ctx.is_warmup = is_warmup
        ctx.dx_direct_sparse = dx_direct_sparse
        
        # Reshape and apply inverse permutation
        output = y3.view(batch_size, seq_len, -1)
        if enable_permute and inv_perm is not None:
            output = output[:, inv_perm, :]
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input_permuted, weight1, weight2, bias1, bias2, y1, y2, y2_sparse = ctx.saved_tensors
        layer_id = ctx.layer_id
        
        # Reshape grad_output
        batch_size, seq_len, hidden_size = grad_output.shape
        dy3 = grad_output.view(-1, hidden_size)
        
        # Apply permutation to gradient if needed
        if ctx.perm is not None:
            grad_output_permuted = grad_output[:, ctx.perm, :]
            dy3 = grad_output_permuted.view(-1, hidden_size)
        
        # Gradient through second layer
        dy2 = torch.mm(dy3, weight2)
        
        # Gradient through ReLU²
        relu_y1 = torch.where(y1 > 0, y1, torch.zeros_like(y1))
        dy1 = 2 * dy2 * relu_y1
        
        # Compute gradients using cached sparsity
        grad_input = grad_weight1 = grad_weight2 = grad_bias1 = grad_bias2 = None
        
        if ctx.needs_input_grad[0]:
            grad_input_2d = torch.mm(dy1, weight1)
            grad_input_permuted = grad_input_2d.view(batch_size, seq_len, -1)
            if ctx.inv_perm is not None:
                grad_input = grad_input_permuted[:, ctx.inv_perm, :]
            else:
                grad_input = grad_input_permuted
        
        if ctx.needs_input_grad[1]:
            if ctx.is_warmup:
                grad_weight1 = torch.mm(dy1.T, input_permuted.view(-1, input_permuted.shape[-1]))
            else:
                # Use cached sparsity from forward pass
                grad_weight1 = compute_split_gemm_dw_with_cached_sparsity(
                    input_permuted.view(-1, input_permuted.shape[-1]),
                    dy1,
                    f"{layer_id}_w1",
                    use_2to4=(ctx.dx_direct_sparse != 3)
                )
        
        if ctx.needs_input_grad[2]:
            if ctx.is_warmup:
                grad_weight2 = torch.mm(dy3.T, y2)
            else:
                # Use cached sparsity from forward pass
                grad_weight2 = compute_split_gemm_dw_with_cached_sparsity(
                    y2,
                    dy3,
                    f"{layer_id}_w2",
                    use_2to4=(ctx.dx_direct_sparse != 3)
                )
        
        if ctx.needs_input_grad[3] and bias1 is not None:
            grad_bias1 = dy1.sum(0)
        
        if ctx.needs_input_grad[4] and bias2 is not None:
            grad_bias2 = dy3.sum(0)
        
        return grad_input, grad_weight1, grad_weight2, grad_bias1, grad_bias2, None, None, None, None, None, None, None, None


# Export optimized functions
__all__ = [
    'fused_gemm_forward_with_sparsity',
    'compute_split_gemm_dw_with_cached_sparsity',
    'FusedSparseLinear',
    'ActivationSparse2to4FunctionOptimized',
    'sparsity_tracker'
]