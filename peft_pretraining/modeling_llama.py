# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import os
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.cuda.amp import custom_fwd, custom_bwd
from torch import autograd

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

# Add custom activation functions to ACT2FN

# Register custom activation functions
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.llama.configuration_llama import LlamaConfig
from loro_torch.lowrank_module import LowRankLinear
# Import 2:4 sparse functions
try:
    from sparse import matmul, MVUE24_approx_triton, soft_threshold24_triton
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False
    raise RuntimeError(
        "❌ CRITICAL: sparse package with triton kernels is required for activation 2:4 sparsity!\n"
        "Please ensure you have the correct sparse package installed.\n"
        "No fallback implementations are allowed as per user requirements."
    )

# Import fake_fp8_mm from sparse_fullrank_linear.py
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from sparse_fullrank_linear import fake_fp8_mm
except ImportError:
    raise RuntimeError(
        "❌ CRITICAL: Cannot import fake_fp8_mm from sparse_fullrank_linear.py!\n"
        "This function is required for activation 2:4 sparsity."
    )

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def compute_split_gemm_dw2_lowrank(y2, d_intermediate_2, y2_forward, weight_in2):
    """
    计算低秩层的 weight_in2 梯度使用 Split-GEMM 策略
    grad_weight_in2 = y2.T @ d_intermediate_2，但使用95%/5%特征分割
    """
    batch_seq_len, intermediate_size = y2.shape
    rank2 = d_intermediate_2.shape[1]
    
    # 分析特征稀疏性
    feature_sparsity = torch.mean((y2_forward != 0).float(), dim=0)  # [intermediate_size]
    
    # 选择95%的特征作为稀疏特征
    num_sparse_features = int(0.95 * intermediate_size)
    _, sparse_indices = torch.topk(feature_sparsity, num_sparse_features)
    
    # 创建稀疏和稠密特征的mask
    sparse_mask = torch.zeros(intermediate_size, dtype=torch.bool, device=y2.device)
    sparse_mask[sparse_indices] = True
    dense_mask = ~sparse_mask
    
    # Split-GEMM计算
    grad_weight_in2 = torch.zeros(intermediate_size, rank2, device=y2.device, dtype=y2.dtype)
    
    # 稀疏部分：使用2:4稀疏化的y2
    if sparse_mask.any():
        y2_sparse_part = y2[:, sparse_mask]
        y2_sparse_2to4 = apply_naive_2to4_sparsity_featurewise(y2_sparse_part.t()).t()
        grad_weight_in2[sparse_mask, :] = torch.mm(y2_sparse_2to4.T, d_intermediate_2)
    
    # 稠密部分：使用原始y2
    if dense_mask.any():
        y2_dense_part = y2[:, dense_mask]
        grad_weight_in2[dense_mask, :] = torch.mm(y2_dense_part.T, d_intermediate_2)
    
    return grad_weight_in2


def compute_split_gemm_dw2(y2, dy3, y2_forward):
    """
    计算 dw2 使用 Split-GEMM 策略
    dw2 = y2.T @ dy3，但使用95%/5%特征分割
    """
    batch_seq_len, intermediate_size = y2.shape
    hidden_size = dy3.shape[1]
    
    # 分析特征稀疏性
    feature_sparsity = torch.mean((y2_forward != 0).float(), dim=0)  # [intermediate_size]
    
    # 选择95%的特征作为稀疏特征
    num_sparse_features = int(0.95 * intermediate_size)
    _, sparse_indices = torch.topk(feature_sparsity, num_sparse_features)
    
    # 创建稀疏和稠密特征的mask
    sparse_mask = torch.zeros(intermediate_size, dtype=torch.bool, device=y2.device)
    sparse_mask[sparse_indices] = True
    dense_mask = ~sparse_mask
    
    # Split-GEMM计算
    dw2 = torch.zeros(hidden_size, intermediate_size, device=y2.device, dtype=y2.dtype)
    
    # 稀疏部分：使用2:4稀疏化的y2
    if sparse_mask.any():
        y2_sparse_part = y2[:, sparse_mask]
        y2_sparse_2to4 = apply_naive_2to4_sparsity_featurewise(y2_sparse_part.t()).t()
        dw2[:, sparse_mask] = torch.mm(dy3.t(), y2_sparse_2to4)
    
    # 稠密部分：使用原始y2
    if dense_mask.any():
        y2_dense_part = y2[:, dense_mask]
        dw2[:, dense_mask] = torch.mm(dy3.t(), y2_dense_part)
    
    return dw2


def compute_split_gemm_dx(dy1, weight1, forward_mask):
    """
    计算 dx 使用 Split-GEMM 策略 (正确实现)
    dx = dy1 @ w1.T，使用95%/5%特征分割
    
    原理：
    1. 分析dy1的特征稀疏性，将特征分为稀疏(95%)和稠密(5%)两组
    2. 稀疏组：feature-wise 2:4稀疏化 → 2:4 sparse GEMM  
    3. 稠密组：保持原始值 → dense GEMM
    4. 合并结果
    """
    batch_seq_len, hidden_size = dy1.shape
    weight_out_dim, weight_in_dim = weight1.shape  # weight1: [hidden_size, intermediate_size] for dx computation
    
    # Step 1: 分析特征稀疏性 (feature-wise sparsity analysis)
    feature_sparsity = torch.mean((dy1 != 0).float(), dim=0)  # [hidden_size]
    
    # Step 2: 选择95%的特征作为稀疏特征 (select 95% features for sparsification)
    num_sparse_features = int(0.95 * hidden_size)
    sparsity_threshold = 0.75  # 稀疏度阈值：>75%稀疏的特征才能被2:4稀疏化
    
    # 找到足够稀疏的特征
    sparse_enough_mask = feature_sparsity > sparsity_threshold
    
    if sparse_enough_mask.sum() >= num_sparse_features:
        # 如果有足够的稀疏特征，选择最稀疏的num_sparse_features个
        _, sparse_indices = torch.topk(feature_sparsity, num_sparse_features)
    else:
        # 如果稀疏特征不够，选择所有足够稀疏的特征
        sparse_indices = torch.where(sparse_enough_mask)[0]
    
    # Step 3: 创建稀疏和稠密特征的mask
    sparse_mask = torch.zeros(hidden_size, dtype=torch.bool, device=dy1.device)
    if len(sparse_indices) > 0:
        sparse_mask[sparse_indices] = True
    dense_mask = ~sparse_mask
    
    # Step 4: Split-GEMM计算
    dx = torch.zeros(batch_seq_len, weight_in_dim, device=dy1.device, dtype=dy1.dtype)
    
    # 稀疏部分：feature-wise 2:4稀疏化 + sparse GEMM
    if sparse_mask.any():
        dy1_sparse_part = dy1[:, sparse_mask]  # [batch_seq_len, num_sparse_features]
        
        # Feature-wise 2:4 sparsification (transpose for feature-wise operation)
        dy1_sparse_part_t = dy1_sparse_part.t()  # [num_sparse_features, batch_seq_len]
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_part_t)
        dy1_sparse_2to4 = dy1_sparse_2to4_t.t()  # [batch_seq_len, num_sparse_features]
        
        # 2:4 Sparse GEMM: dx_sparse = dy1_sparse_2to4 @ w1[sparse_mask, :].T
        weight1_sparse = weight1[sparse_mask, :]  # [num_sparse_features, intermediate_size]
        dx += torch.mm(dy1_sparse_2to4, weight1_sparse)
    
    # 稠密部分：dense GEMM 
    if dense_mask.any():
        dy1_dense_part = dy1[:, dense_mask]  # [batch_seq_len, num_dense_features]
        weight1_dense = weight1[dense_mask, :]  # [num_dense_features, intermediate_size]
        dx += torch.mm(dy1_dense_part, weight1_dense)
    
    return dx


def compute_split_gemm_dw1(input_2d, dy1, forward_mask):
    """
    计算 dw1 使用 Split-GEMM 策略 (正确实现)
    dw1 = dy1.T @ input，使用95%/5%特征分割
    
    原理同上，但计算的是权重梯度
    """
    batch_seq_len, input_dim = input_2d.shape
    _, hidden_size = dy1.shape
    
    # Step 1: 分析特征稀疏性
    feature_sparsity = torch.mean((dy1 != 0).float(), dim=0)  # [hidden_size]
    
    # Step 2: 选择95%的特征作为稀疏特征
    num_sparse_features = int(0.95 * hidden_size)
    sparsity_threshold = 0.75
    
    sparse_enough_mask = feature_sparsity > sparsity_threshold
    
    if sparse_enough_mask.sum() >= num_sparse_features:
        _, sparse_indices = torch.topk(feature_sparsity, num_sparse_features)
    else:
        sparse_indices = torch.where(sparse_enough_mask)[0]
    
    # Step 3: 创建mask
    sparse_mask = torch.zeros(hidden_size, dtype=torch.bool, device=dy1.device)
    if len(sparse_indices) > 0:
        sparse_mask[sparse_indices] = True
    dense_mask = ~sparse_mask
    
    # Step 4: Split-GEMM计算
    dw1 = torch.zeros(hidden_size, input_dim, device=dy1.device, dtype=dy1.dtype)
    
    # 稀疏部分：feature-wise 2:4稀疏化
    if sparse_mask.any():
        dy1_sparse_part = dy1[:, sparse_mask]  # [batch_seq_len, num_sparse_features]
        
        # Feature-wise 2:4 sparsification
        dy1_sparse_part_t = dy1_sparse_part.t()  # [num_sparse_features, batch_seq_len]
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_part_t)
        dy1_sparse_2to4 = dy1_sparse_2to4_t.t()  # [batch_seq_len, num_sparse_features]
        
        # Compute gradient: dw1_sparse = dy1_sparse_2to4.T @ input
        dw1[sparse_mask, :] = torch.mm(dy1_sparse_2to4.t(), input_2d)
    
    # 稠密部分：dense GEMM
    if dense_mask.any():
        dy1_dense_part = dy1[:, dense_mask]  # [batch_seq_len, num_dense_features]
        dw1[dense_mask, :] = torch.mm(dy1_dense_part.t(), input_2d)
    
    return dw1


def apply_naive_2to4_sparsity_featurewise(input_tensor):
    """
    Apply 2:4 sparsity along the second dimension (feature-wise)
    input_tensor: [num_features, batch_seq_len]
    """
    num_features, batch_seq_len = input_tensor.shape
    
    # Ensure batch_seq_len is divisible by 4
    if batch_seq_len % 4 != 0:
        pad_size = 4 - (batch_seq_len % 4)
        input_padded = F.pad(input_tensor, (0, pad_size), value=0)
        batch_seq_len_padded = batch_seq_len + pad_size
    else:
        input_padded = input_tensor
        batch_seq_len_padded = batch_seq_len
    
    # Reshape to groups of 4 along the batch dimension
    input_reshaped = input_padded.view(num_features, -1, 4)
    
    # Find top 2 values in each group
    abs_values = torch.abs(input_reshaped)
    _, top_indices = torch.topk(abs_values, 2, dim=-1)
    
    # Create mask
    mask = torch.zeros_like(input_reshaped)
    mask.scatter_(-1, top_indices, 1.0)
    
    # Apply mask
    output_reshaped = input_reshaped * mask
    output_padded = output_reshaped.view(num_features, batch_seq_len_padded)
    
    # Remove padding if it was added
    if batch_seq_len % 4 != 0:
        output = output_padded[:, :batch_seq_len]
    else:
        output = output_padded
    
    return output





class ActivationSparse2to4LowRankFunction(autograd.Function):
    """
    Low-rank版本的Activation 2:4 Sparsity FFN实现
    
    处理LowRankLinear层的情况，其中每个linear层由两个低秩矩阵组成：
    - up_proj: x @ weight_in1 @ weight_out1.T
    - down_proj: x @ weight_in2 @ weight_out2.T
    
    Forward Pass:
    1. 输入置换 (Input Permutation)
    2. 第一个低秩全连接层: y1 = x @ weight_in1 @ weight_out1.T
    3. 平方ReLU激活函数: y2 = ReLU²(y1)
    4. 第二个低秩全连接层: y3 = sparsified(y2) @ weight_in2 @ weight_out2.T
    5. 逆向置换 (Inverse Permutation)
    """
    
    # Class-level storage for token permutation (fixed across training)
    _token_permutation = {}
    _inverse_permutation = {}
    
    # Training step counter for dense warmup
    _training_step = 0
    _warmup_steps = 1000  # Default value, can be overridden
    

    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight_in1, weight_out1, weight_in2, weight_out2, bias1=None, bias2=None, sparsity_method="mvue", warmup_steps=None, dx_direct_sparse=False, dynamic_steps=10, calibration_samples=100):
        """
        低秩FFN Forward Pass
        
        Args:
            input: Input tensor (batch_size, seq_len, hidden_size)
            weight_in1: First layer input weight (hidden_size, rank1)
            weight_out1: First layer output weight (intermediate_size, rank1)
            weight_in2: Second layer input weight (intermediate_size, rank2)
            weight_out2: Second layer output weight (hidden_size, rank2)
            bias1: First layer bias (optional)
            bias2: Second layer bias (optional)
            sparsity_method: "naive", "mvue", "soft_threshold_weights", or "soft_dynamic"
            warmup_steps: Number of steps for dense warmup
            dx_direct_sparse: If True, use direct naive sparse for dx computation
        """
        ctx.sparsity_method = sparsity_method
        ctx.input_shape = input.shape
        ctx.dx_direct_sparse = dx_direct_sparse
        ctx.dynamic_steps = dynamic_steps
        ctx.calibration_samples = calibration_samples
        
        # Update warmup steps if provided
        if warmup_steps is not None:
            ActivationSparse2to4LowRankFunction._warmup_steps = warmup_steps
        
        batch_size, seq_len, hidden_size = input.shape
        
        # Step 1: 输入置换 (Input Permutation)
        perm_key = f"{seq_len}_{input.device}"
        
        if perm_key not in ActivationSparse2to4LowRankFunction._token_permutation:
            # Create fixed permutation for this sequence length
            perm = torch.randperm(seq_len, device=input.device)
            inv_perm = torch.argsort(perm)
            ActivationSparse2to4LowRankFunction._token_permutation[perm_key] = perm
            ActivationSparse2to4LowRankFunction._inverse_permutation[perm_key] = inv_perm
        
        perm = ActivationSparse2to4LowRankFunction._token_permutation[perm_key]
        inv_perm = ActivationSparse2to4LowRankFunction._inverse_permutation[perm_key]
        
        # Apply permutation: [batch_size, seq_len, hidden_size]
        input_permuted = input[:, perm, :]
        
        # Step 2: 第一个低秩全连接层 (First Low-rank Linear Layer)
        # y1 = x @ weight_in1 @ weight_out1.T
        input_2d = input_permuted.view(-1, input_permuted.shape[-1])  # [batch*seq, hidden_size]
        intermediate_1 = torch.mm(input_2d, weight_in1)  # [batch*seq, rank1]
        y1 = torch.mm(intermediate_1, weight_out1.T)  # [batch*seq, intermediate_size]
        if bias1 is not None:
            y1 = y1 + bias1
        
        # Step 3: 平方ReLU激活函数 (Squared-ReLU Activation)
        # y2 = ReLU²(y1)
        y2 = torch.where(y1 > 0, y1 * y1, torch.zeros_like(y1))
        
        # Record sparsity statistics if enabled (check via config)
        if hasattr(ActivationSparse2to4LowRankFunction, '_wandb_sparsityrelu_enabled') and ActivationSparse2to4LowRankFunction._wandb_sparsityrelu_enabled:
            # Record sparsity for this layer
            ActivationSparse2to4LowRankFunction._record_activation_sparsity_static(y2)
        

        
        # Dense warmup for first N iterations
        if ActivationSparse2to4LowRankFunction._training_step < ActivationSparse2to4LowRankFunction._warmup_steps:
            # During warmup, use dense computation
            # y3 = y2 @ weight_in2 @ weight_out2.T
            intermediate_2 = torch.mm(y2, weight_in2)  # [batch*seq, rank2]
            y3 = torch.mm(intermediate_2, weight_out2.T)  # [batch*seq, hidden_size]
            if bias2 is not None:
                y3 = y3 + bias2
            
            # Store variables for backward pass
            ctx.save_for_backward(input_permuted, weight_in1, weight_out1, weight_in2, weight_out2, bias1, bias2, y1, y2, y2, intermediate_1, intermediate_2)
            ctx.perm = perm
            ctx.inv_perm = inv_perm
            ctx.is_warmup = True
        else:
            # Step 4: 第二个低秩全连接层 (Second Low-rank Linear Layer) - Sparse GEMM
            # Apply 2:4 sparsity to y2 (token-wise/row-wise)
            if sparsity_method == "naive":
                y2_sparse = apply_naive_2to4_sparsity(y2)
            elif sparsity_method == "mvue":
                y2_sparse = apply_mvue_2to4_sparsity(y2)
            elif sparsity_method == "soft_threshold_weights":
                y2_sparse = apply_soft_threshold_weights_2to4_sparsity(y2, scale=1.0)
            elif sparsity_method == "soft_dynamic":
                # 获取层ID和当前步数
                layer_id = getattr(ActivationSoftThresholdManager, '_current_layer_id', 0) % 12
                current_step = getattr(ActivationSparse2to4LowRankFunction, '_global_training_step', 0)
                # 从config获取calibration_samples
                calibration_samples = getattr(ctx, 'calibration_samples', 100)
                # dynamic_steps已经作为参数传入
                y2_sparse = apply_soft_threshold_dynamic_activation_2to4_sparsity(y2, layer_id, current_step, dynamic_steps, calibration_samples)
                ActivationSoftThresholdManager._current_layer_id = getattr(ActivationSoftThresholdManager, '_current_layer_id', 0) + 1
            else:
                raise ValueError(f"Unknown sparsity method: {sparsity_method}")
            
            # y3 = sparsified(y2) @ weight_in2 @ weight_out2.T
            intermediate_2 = torch.mm(y2_sparse, weight_in2)  # [batch*seq, rank2]
            y3 = torch.mm(intermediate_2, weight_out2.T)  # [batch*seq, hidden_size]
            if bias2 is not None:
                y3 = y3 + bias2
            
            # Store variables for backward pass
            ctx.save_for_backward(input_permuted, weight_in1, weight_out1, weight_in2, weight_out2, bias1, bias2, y1, y2, y2_sparse, intermediate_1, intermediate_2)
            ctx.perm = perm
            ctx.inv_perm = inv_perm
            ctx.is_warmup = False
        
        # Step 5: 逆向置换 (Inverse Permutation)
        y3_reshaped = y3.view(batch_size, seq_len, hidden_size)
        output = y3_reshaped[:, inv_perm, :]
        
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        低秩FFN Backward Pass
        """
        input_permuted, weight_in1, weight_out1, weight_in2, weight_out2, bias1, bias2, y1, y2, y2_forward, intermediate_1, intermediate_2 = ctx.saved_tensors
        perm = ctx.perm
        inv_perm = ctx.inv_perm
        is_warmup = ctx.is_warmup
        dx_direct_sparse = ctx.dx_direct_sparse
        
        batch_size, seq_len, hidden_size = grad_output.shape
        
        # Step 1: 梯度置换 (Gradient Permutation)
        grad_output_permuted = grad_output[:, perm, :]
        dy3 = grad_output_permuted.view(-1, grad_output_permuted.shape[-1])  # [batch*seq, hidden_size]
        
        # Step 2: 计算第二个低秩层的梯度
        # dy3 -> d_intermediate_2 -> dy2
        # y3 = intermediate_2 @ weight_out2.T
        # dy3 = d_intermediate_2 @ weight_out2.T
        d_intermediate_2 = torch.mm(dy3, weight_out2)  # [batch*seq, rank2]
        # intermediate_2 = y2 @ weight_in2
        # d_intermediate_2 = dy2 @ weight_in2
        dy2 = torch.mm(d_intermediate_2, weight_in2.T)  # [batch*seq, intermediate_size]
        
        # Step 3: 反向通过激活函数 (Backprop through Activation)
        # dy1 = 2 * dy2 * ReLU(y1)
        relu_y1 = torch.where(y1 > 0, y1, torch.zeros_like(y1))
        dy1 = 2 * dy2 * relu_y1
        
        # Initialize gradients
        grad_input = grad_weight_in1 = grad_weight_out1 = grad_weight_in2 = grad_weight_out2 = grad_bias1 = grad_bias2 = None
        
        if is_warmup:
            # Dense warmup: standard gradient computation
            if ctx.needs_input_grad[0]:
                # dx = dy1 @ weight_out1 @ weight_in1.T
                d_intermediate_1 = torch.mm(dy1, weight_out1)  # [batch*seq, rank1]
                grad_input_2d = torch.mm(d_intermediate_1, weight_in1.T)  # [batch*seq, hidden_size]
                grad_input_permuted = grad_input_2d.view(batch_size, seq_len, hidden_size)
                grad_input = grad_input_permuted[:, inv_perm, :]
            
            # 第一个低秩层的梯度
            if ctx.needs_input_grad[1]:  # weight_in1
                grad_weight_in1 = torch.mm(input_permuted.view(-1, input_permuted.shape[-1]).T, torch.mm(dy1, weight_out1))
            
            if ctx.needs_input_grad[2]:  # weight_out1
                grad_weight_out1 = torch.mm(dy1.T, intermediate_1)
            
            # 第二个低秩层的梯度
            if ctx.needs_input_grad[3]:  # weight_in2
                grad_weight_in2 = torch.mm(y2.T, d_intermediate_2)
            
            if ctx.needs_input_grad[4]:  # weight_out2
                grad_weight_out2 = torch.mm(dy3.T, intermediate_2)
            
            if ctx.needs_input_grad[5] and bias1 is not None:
                grad_bias1 = dy1.sum(0)
            
            if ctx.needs_input_grad[6] and bias2 is not None:
                grad_bias2 = dy3.sum(0)
        else:
            # Sparse training: Split-GEMM策略
            if ctx.needs_input_grad[0]:
                forward_mask = (y2_forward != 0).float()
                
                if dx_direct_sparse:
                    # Direct naive sparse
                    dy1_naive_sparse = apply_naive_2to4_sparsity(dy1)
                    d_intermediate_1 = torch.mm(dy1_naive_sparse, weight_out1)
                    grad_input_2d = torch.mm(d_intermediate_1, weight_in1.T)
                else:
                    # Split-GEMM strategy: 使用95%/5%特征分割策略
                    # 对于低秩层：dx = dy1 @ weight_out1 @ weight_in1.T
                    # 使用Split-GEMM策略计算 dy1 @ weight_out1
                    d_intermediate_1 = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, forward_mask)
                    grad_input_2d = torch.mm(d_intermediate_1, weight_in1.T)
                
                grad_input_permuted = grad_input_2d.view(batch_size, seq_len, hidden_size)
                grad_input = grad_input_permuted[:, inv_perm, :]
            
            # 第一个低秋层的梯度 (使用Split-GEMM策略)
            if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                 forward_mask = (y2_forward != 0).float()
                 
                 if ctx.needs_input_grad[1]:  # weight_in1
                     # For low-rank: grad_weight_in1 = input.T @ (dy1 @ weight_out1)
                     if dx_direct_sparse:
                         # Direct sparse method: 简单的token-wise稀疏化
                         dy1_sparse = apply_naive_2to4_sparsity(dy1)
                         d_intermediate_1_for_w_in1 = torch.mm(dy1_sparse, weight_out1)
                     else:
                         # Split-GEMM strategy: 95%/5%特征分割策略
                         d_intermediate_1_for_w_in1 = compute_split_gemm_lowrank_intermediate(dy1, weight_out1, forward_mask)
                     grad_weight_in1 = torch.mm(input_permuted.view(-1, input_permuted.shape[-1]).T, d_intermediate_1_for_w_in1)
                 
                 if ctx.needs_input_grad[2]:  # weight_out1  
                     # For low-rank: grad_weight_out1 = dy1.T @ intermediate_1
                     if dx_direct_sparse:
                         # Direct sparse method: 简单的token-wise稀疏化
                         dy1_sparse = apply_naive_2to4_sparsity(dy1)
                         grad_weight_out1 = torch.mm(dy1_sparse.T, intermediate_1)
                     else:
                         # Split-GEMM strategy: 95%/5%特征分割策略
                         dy1_split_gemm = apply_split_gemm_to_dy1(dy1, forward_mask)
                         grad_weight_out1 = torch.mm(dy1_split_gemm.T, intermediate_1)
            
            # 第二个低秩层的梯度 (使用Split-GEMM策略)
            if ctx.needs_input_grad[3]:  # weight_in2
                grad_weight_in2 = compute_split_gemm_dw2_lowrank(y2, d_intermediate_2, y2_forward, weight_in2)
            
            if ctx.needs_input_grad[4]:  # weight_out2
                grad_weight_out2 = torch.mm(dy3.T, intermediate_2)
            
            if ctx.needs_input_grad[5] and bias1 is not None:
                grad_bias1 = dy1.sum(0)
            
            if ctx.needs_input_grad[6] and bias2 is not None:
                grad_bias2 = dy3.sum(0)
        
        # Return gradients for all input parameters (12 total to match forward signature)
        return grad_input, grad_weight_in1, grad_weight_out1, grad_weight_in2, grad_weight_out2, grad_bias1, grad_bias2, None, None, None, None, None
    
    @staticmethod
    def increment_step():
        """Increment training step counter for dense warmup"""
        ActivationSparse2to4LowRankFunction._training_step += 1
    
    @staticmethod
    def get_training_step():
        """Get current training step"""
        return ActivationSparse2to4LowRankFunction._training_step
    
    @staticmethod
    def set_warmup_steps(steps):
        """Set the number of warmup steps"""
        ActivationSparse2to4LowRankFunction._warmup_steps = steps
    
    @staticmethod
    def get_warmup_steps():
        """Get the number of warmup steps"""
        return ActivationSparse2to4LowRankFunction._warmup_steps
    
    @staticmethod
    def _record_activation_sparsity_static(activated_tensor, layer_id=None):
        """
        Static method to record activation sparsity (called from forward pass) - LowRank version
        """
        try:
            # Get current training step from the low-rank class variable set by main training loop
            current_step = getattr(ActivationSparse2to4LowRankFunction, '_global_training_step', 0)
        except Exception as e:
            current_step = 0
        
        # Initialize recording state for this step if needed
        if not hasattr(ActivationSparse2to4LowRankFunction, '_last_recorded_step'):
            ActivationSparse2to4LowRankFunction._last_recorded_step = -1
            ActivationSparse2to4LowRankFunction._current_step_layer_count = 0
        
        # Reset layer counter for new step
        if ActivationSparse2to4LowRankFunction._last_recorded_step != current_step:
            ActivationSparse2to4LowRankFunction._last_recorded_step = current_step
            ActivationSparse2to4LowRankFunction._current_step_layer_count = 0
            
            # Clear previous step's stats when starting a new step
            if hasattr(LlamaMLP, '_sparsity_stats'):
                LlamaMLP._sparsity_stats.clear()
        
        # For low-rank layers, we also need to reset layer registry when step changes
        if hasattr(LlamaMLP, '_layer_registry') and current_step != getattr(LlamaMLP, '_last_step_processed', -1):
            LlamaMLP._layer_registry.clear()
            LlamaMLP._last_step_processed = current_step
        
        # Use the layer count for this step as layer_id (0-11 for 12 layers)
        if layer_id is None:
            layer_id = ActivationSparse2to4LowRankFunction._current_step_layer_count
        
        ActivationSparse2to4LowRankFunction._current_step_layer_count += 1
        
        with torch.no_grad():
            # Calculate sparsity (percentage of zero values)
            total_elements = activated_tensor.numel()
            zero_elements = (activated_tensor == 0).sum().item()
            sparsity = zero_elements / total_elements
            
            # Store in the global sparsity stats (will be uploaded by main loop)
            if not hasattr(LlamaMLP, '_sparsity_stats'):
                LlamaMLP._sparsity_stats = {}
                
            LlamaMLP._sparsity_stats[f'sparsity_relu/layer_{layer_id}'] = sparsity


class ActivationSparse2to4Function(autograd.Function):
    """
    完整的Activation 2:4 Sparsity FFN实现，严格按照论文流程
    
    Forward Pass:
    1. 输入置换 (Input Permutation)
    2. 第一个全连接层 (Dense GEMM): y1 = x @ w1
    3. 平方ReLU激活函数: y2 = ReLU²(y1)
    4. 第二个全连接层 (Sparse GEMM): y3 = sparsified(y2) @ w2
    5. 逆向置换 (Inverse Permutation)
    
    Backward Pass:
    1. 梯度置换 (Gradient Permutation)
    2. 计算 dy2: dy2 = dy3 @ w2.T
    3. 反向通过激活函数: dy1 = 2 * dy2 * ReLU(y1)
    4. 计算 W2 的梯度 (dw2): Split-GEMM策略
    5. 计算 W1 的梯度 (dw1) 和 X 的梯度 (dx): Split-GEMM策略
    6. 梯度逆向置换 (Inverse Gradient Permutation)
    """
    
    # Class-level storage for token permutation (fixed across training)
    _token_permutation = {}
    _inverse_permutation = {}
    
    # Training step counter for dense warmup
    _training_step = 0
    _warmup_steps = 1000  # Default value, can be overridden
    

    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight1, weight2, bias1=None, bias2=None, sparsity_method="mvue", warmup_steps=None, dx_direct_sparse=False, dynamic_steps=10, calibration_samples=100):
        """
        完整的FFN Forward Pass
        
        Args:
            input: Input tensor (batch_size, seq_len, hidden_size)
            weight1: First linear layer weight (hidden_size, intermediate_size)
            weight2: Second linear layer weight (intermediate_size, hidden_size)
            bias1: First linear layer bias (optional)
            bias2: Second linear layer bias (optional)
            sparsity_method: "naive", "mvue", "soft_threshold_weights", or "soft_dynamic"
            warmup_steps: Number of steps for dense warmup
            dx_direct_sparse: If True, use direct naive sparse for dx computation instead of split-GEMM
        """
        ctx.sparsity_method = sparsity_method
        ctx.input_shape = input.shape
        ctx.dx_direct_sparse = dx_direct_sparse
        ctx.dynamic_steps = dynamic_steps
        ctx.calibration_samples = calibration_samples
        
        # Update warmup steps if provided
        if warmup_steps is not None:
            ActivationSparse2to4Function._warmup_steps = warmup_steps
        
        batch_size, seq_len, hidden_size = input.shape
        
        # Step 1: 输入置换 (Input Permutation) - Optimization 2
        perm_key = f"{seq_len}_{input.device}"
        
        if perm_key not in ActivationSparse2to4Function._token_permutation:
            # Create fixed permutation for this sequence length
            perm = torch.randperm(seq_len, device=input.device)
            inv_perm = torch.argsort(perm)
            ActivationSparse2to4Function._token_permutation[perm_key] = perm
            ActivationSparse2to4Function._inverse_permutation[perm_key] = inv_perm
        
        perm = ActivationSparse2to4Function._token_permutation[perm_key]
        inv_perm = ActivationSparse2to4Function._inverse_permutation[perm_key]
        
        # Apply permutation: [batch_size, seq_len, hidden_size]
        input_permuted = input[:, perm, :]
        
        # Step 2: 第一个全连接层 (First Linear Layer) - Dense GEMM
        # y1 = x @ w1
        input_2d = input_permuted.view(-1, input_permuted.shape[-1])  # [batch*seq, hidden_size]
        y1 = torch.mm(input_2d, weight1.T)  # [batch*seq, intermediate_size]
        if bias1 is not None:
            y1 = y1 + bias1
        
        # Step 3: 平方ReLU激活函数 (Squared-ReLU Activation)
        # y2 = ReLU²(y1)
        y2 = torch.where(y1 > 0, y1 * y1, torch.zeros_like(y1))
        
        # Record sparsity statistics if enabled (check via config)
        # We need to check if wandb_sparsityrelu is enabled in the model config
        # Since we don't have direct access to config here, we'll check via a class variable
        wandb_enabled = hasattr(ActivationSparse2to4Function, '_wandb_sparsityrelu_enabled') and ActivationSparse2to4Function._wandb_sparsityrelu_enabled
        if wandb_enabled:
            # Record sparsity statistics directly inline (working version)
            try:
                # Get current training step
                current_step = getattr(ActivationSparse2to4Function, '_global_training_step', 0)
                
                # Initialize recording state for this step if needed
                if not hasattr(ActivationSparse2to4Function, '_last_recorded_step'):
                    ActivationSparse2to4Function._last_recorded_step = -1
                    ActivationSparse2to4Function._current_step_layer_count = 0
                
                # Reset layer counter for new step
                if ActivationSparse2to4Function._last_recorded_step != current_step:
                    ActivationSparse2to4Function._last_recorded_step = current_step
                    ActivationSparse2to4Function._current_step_layer_count = 0
                    
                    # Clear previous step's stats when starting a new step
                    if hasattr(LlamaMLP, '_sparsity_stats'):
                        LlamaMLP._sparsity_stats.clear()
                
                # Get layer ID
                layer_id = ActivationSparse2to4Function._current_step_layer_count
                ActivationSparse2to4Function._current_step_layer_count += 1
                
                # Calculate sparsity (percentage of zero values)
                with torch.no_grad():
                    total_elements = y2.numel()
                    zero_elements = (y2 == 0).sum().item()
                    sparsity = zero_elements / total_elements
                    
                    # Store in the global sparsity stats
                    if not hasattr(LlamaMLP, '_sparsity_stats'):
                        LlamaMLP._sparsity_stats = {}
                        
                    LlamaMLP._sparsity_stats[f'sparsity_relu/layer_{layer_id}'] = sparsity
                    
            except Exception as e:
                print(f"❌ 稀疏度记录失败: {e}")
                import traceback
                traceback.print_exc()
        

        
        # # 数值稳定性处理：如果y2的值太大，进行缩放
        # max_val = y2.abs().max().item()
        # if max_val > 100:  # 经验阈值
        #     # 计算缩放因子
        #     scale_factor = 100.0 / max_val
        #     y2_scaled = y2 * scale_factor
        #     print(f"🔧 Scaling y2 by {scale_factor:.4f} to prevent numerical instability (max_val={max_val:.2f})")
        # else:
        #     y2_scaled = y2
        #     scale_factor = 1.0
        
        # Dense warmup for first N iterations
        if ActivationSparse2to4Function._training_step < ActivationSparse2to4Function._warmup_steps:
            # During warmup, use dense computation
            # y3 = torch.mm(y2_scaled, weight2)  # Dense GEMM with scaled y2
            y3 = torch.mm(y2, weight2.T)  # Dense GEMM
            if bias2 is not None:
                y3 = y3 + bias2
            
            # # Restore scaling in output
            # y3 = y3 / scale_factor
            
            # Store variables for backward pass
            ctx.save_for_backward(input_permuted, weight1, weight2, bias1, bias2, y1, y2, y2)  # y2 twice for forward_mask
            ctx.perm = perm
            ctx.inv_perm = inv_perm
            ctx.is_warmup = True
            # ctx.scale_factor = scale_factor
        else:
            # Step 4: 第二个全连接层 (Second Linear Layer) - Sparse GEMM
            # Apply 2:4 sparsity to y2 (token-wise/row-wise)
            if sparsity_method == "naive":
                y2_sparse = apply_naive_2to4_sparsity(y2)
            elif sparsity_method == "mvue":
                y2_sparse = apply_mvue_2to4_sparsity(y2)
            elif sparsity_method == "soft_threshold_weights":
                y2_sparse = apply_soft_threshold_weights_2to4_sparsity(y2, scale=1.0)
            elif sparsity_method == "soft_dynamic":
                # 获取层ID和当前步数
                layer_id = getattr(ActivationSoftThresholdManager, '_current_layer_id_standard', 0) % 12
                current_step = getattr(ActivationSparse2to4Function, '_global_training_step', 0)
                # 从config获取calibration_samples
                calibration_samples = getattr(ctx, 'calibration_samples', 100)
                # dynamic_steps已经作为参数传入
                y2_sparse = apply_soft_threshold_dynamic_activation_2to4_sparsity(y2, layer_id, current_step, dynamic_steps, calibration_samples)
                ActivationSoftThresholdManager._current_layer_id_standard = getattr(ActivationSoftThresholdManager, '_current_layer_id_standard', 0) + 1
            else:
                raise ValueError(f"Unknown sparsity method: {sparsity_method}")
            
            # y3 = sparsified(y2) @ w2
            y3 = torch.mm(y2_sparse, weight2.T)
            if bias2 is not None:
                y3 = y3 + bias2
            
            # # Restore scaling in output
            # y3 = y3 / scale_factor
            
            # Store variables for backward pass
            ctx.save_for_backward(input_permuted, weight1, weight2, bias1, bias2, y1, y2, y2_sparse)
            ctx.perm = perm
            ctx.inv_perm = inv_perm
            ctx.is_warmup = False
            # ctx.scale_factor = scale_factor
        
        # Step 5: 逆向置换 (Inverse Permutation)
        y3_reshaped = y3.view(batch_size, seq_len, hidden_size)
        output = y3_reshaped[:, inv_perm, :]
        
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        完整的FFN Backward Pass，实现论文的Split-GEMM策略
        """
        input_permuted, weight1, weight2, bias1, bias2, y1, y2, y2_forward = ctx.saved_tensors
        perm = ctx.perm
        inv_perm = ctx.inv_perm
        is_warmup = ctx.is_warmup
        dx_direct_sparse = ctx.dx_direct_sparse
        
        batch_size, seq_len, hidden_size = grad_output.shape
        
        # Step 1: 梯度置换 (Gradient Permutation)
        grad_output_permuted = grad_output[:, perm, :]
        dy3 = grad_output_permuted.view(-1, grad_output_permuted.shape[-1])  # [batch*seq, hidden_size]
        
        # Step 2: 计算 dy2
        # dy2 = dy3 @ w2.T
        dy2 = torch.mm(dy3, weight2)  # [batch*seq, intermediate_size]
        # # 由于前向传播中我们缩放了y2，所以dy2也需要相应缩放
        # dy2 = dy2 * scale_factor
        
        # Step 3: 反向通过激活函数 (Backprop through Activation)
        # dy1 = 2 * dy2 * ReLU(y1)
        relu_y1 = torch.where(y1 > 0, y1, torch.zeros_like(y1))
        dy1 = 2 * dy2 * relu_y1
        
        # Initialize gradients
        grad_input = grad_weight1 = grad_weight2 = grad_bias1 = grad_bias2 = None
        
        if is_warmup:
            # Dense warmup: standard gradient computation
            if ctx.needs_input_grad[0]:
                grad_input_2d = torch.mm(dy1, weight1)
                grad_input_permuted = grad_input_2d.view(batch_size, seq_len, hidden_size)
                grad_input = grad_input_permuted[:, inv_perm, :]
            
            if ctx.needs_input_grad[1]:
                grad_weight1 = torch.mm(dy1.t(), input_permuted.view(-1, input_permuted.shape[-1]))
            
            if ctx.needs_input_grad[2]:
                # 对于weight2的梯度，我们需要使用缩放后的y2
                # y2_scaled = y2 * scale_factor
                grad_weight2 = torch.mm(dy3.t(), y2)
            
            if ctx.needs_input_grad[3] and bias1 is not None:
                grad_bias1 = dy1.sum(0)
            
            if ctx.needs_input_grad[4] and bias2 is not None:
                grad_bias2 = dy3.sum(0)
        else:
            # Step 4: 计算 W2 的梯度 (dw2) - Split-GEMM策略
            if ctx.needs_input_grad[2]:
    
                grad_weight2 = compute_split_gemm_dw2(y2, dy3, y2_forward)
            
            # Step 5: 计算 W1 的梯度 (dw1) 和 X 的梯度 (dx) - Split-GEMM策略
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                forward_mask = (y2_forward != 0).float()
                
                if ctx.needs_input_grad[0]:
                    if dx_direct_sparse:
                        # Direct naive sparse: dx = dy1_naive_sparse @ w1.T
                        dy1_naive_sparse = apply_naive_2to4_sparsity(dy1)
                        grad_input_2d = torch.mm(dy1_naive_sparse, weight1)
                    else:
                        # Split-GEMM strategy: 95% sparse + 5% dense
                        grad_input_2d = compute_split_gemm_dx(dy1, weight1, forward_mask)
                    grad_input_permuted = grad_input_2d.view(batch_size, seq_len, hidden_size)
                    grad_input = grad_input_permuted[:, inv_perm, :]
                
                if ctx.needs_input_grad[1]:
                    # Split-GEMM strategy for weight gradient
                    grad_weight1 = compute_split_gemm_dw1(input_permuted.view(-1, input_permuted.shape[-1]), dy1, forward_mask)
                    # if dx_direct_sparse:
                    #     # Direct sparse method: 使用简单的token-wise稀疏化
                    #     dy1_sparse = apply_naive_2to4_sparsity(dy1)
                    #     grad_weight1 = torch.mm(dy1_sparse.t(), input_permuted.view(-1, input_permuted.shape[-1]))
                    # else:
                    #     # Split-GEMM strategy: 使用95%/5%特征分割策略
                    #     grad_weight1 = compute_split_gemm_dw1(input_permuted.view(-1, input_permuted.shape[-1]), dy1, forward_mask)
            
            # Bias gradients
            if ctx.needs_input_grad[3] and bias1 is not None:
                grad_bias1 = dy1.sum(0)
            
            if ctx.needs_input_grad[4] and bias2 is not None:
                grad_bias2 = dy3.sum(0)
        
        # Return gradients for all input parameters (10 total to match forward signature)
        return grad_input, grad_weight1, grad_weight2, grad_bias1, grad_bias2, None, None, None, None, None
    
    @staticmethod
    def increment_step():
        """Increment training step counter for dense warmup"""
        ActivationSparse2to4Function._training_step += 1
    
    @staticmethod
    def get_training_step():
        """Get current training step"""
        return ActivationSparse2to4Function._training_step
    
    @staticmethod
    def set_warmup_steps(steps):
        """Set the number of warmup steps"""
        ActivationSparse2to4Function._warmup_steps = steps
    
    @staticmethod
    def get_warmup_steps():
        """Get the number of warmup steps"""
        return ActivationSparse2to4Function._warmup_steps
    
    @staticmethod
    def _record_activation_sparsity_static(activated_tensor, layer_id=None):
        """
        Static method to record activation sparsity statistics
        """
        try:
            # Get current training step from the class variable set by main training loop
            current_step = getattr(ActivationSparse2to4Function, '_global_training_step', 0)
        except Exception as e:
            current_step = 0
        
        # Initialize recording state for this step if needed
        if not hasattr(ActivationSparse2to4Function, '_last_recorded_step'):
            ActivationSparse2to4Function._last_recorded_step = -1
            ActivationSparse2to4Function._current_step_layer_count = 0
        
        # Reset layer counter for new step
        if ActivationSparse2to4Function._last_recorded_step != current_step:
            ActivationSparse2to4Function._last_recorded_step = current_step
            ActivationSparse2to4Function._current_step_layer_count = 0
            
            # Clear previous step's stats when starting a new step
            if hasattr(LlamaMLP, '_sparsity_stats'):
                LlamaMLP._sparsity_stats.clear()
        
        # For standard MLP layers, we also need to reset layer registry when step changes
        if hasattr(LlamaMLP, '_layer_registry') and current_step != getattr(LlamaMLP, '_last_step_processed', -1):
            LlamaMLP._layer_registry.clear()
            LlamaMLP._last_step_processed = current_step
        
        # Use the layer count for this step as layer_id (0-11 for 12 layers)
        if layer_id is None:
            layer_id = ActivationSparse2to4Function._current_step_layer_count
        
        ActivationSparse2to4Function._current_step_layer_count += 1
        
        with torch.no_grad():
            # Calculate sparsity (percentage of zero values)
            total_elements = activated_tensor.numel()
            zero_elements = (activated_tensor == 0).sum().item()
            sparsity = zero_elements / total_elements
            
            # Store in the global sparsity stats (will be uploaded by main loop)
            if not hasattr(LlamaMLP, '_sparsity_stats'):
                LlamaMLP._sparsity_stats = {}
                
            LlamaMLP._sparsity_stats[f'sparsity_relu/layer_{layer_id}'] = sparsity


def apply_naive_2to4_sparsity(input_tensor):
    """
    Apply naive 2:4 sparsity: keep top 2 values in each group of 4
    """
    batch_size, hidden_size = input_tensor.shape
    
    # Ensure hidden_size is divisible by 4
    if hidden_size % 4 != 0:
        # Pad to make it divisible by 4
        pad_size = 4 - (hidden_size % 4)
        input_padded = F.pad(input_tensor, (0, pad_size), value=0)
        hidden_size_padded = hidden_size + pad_size
    else:
        input_padded = input_tensor
        hidden_size_padded = hidden_size
    
    # Reshape to groups of 4
    input_reshaped = input_padded.view(batch_size, -1, 4)
    
    # Find top 2 values in each group
    abs_values = torch.abs(input_reshaped)
    _, top_indices = torch.topk(abs_values, 2, dim=-1)
    
    # Create mask
    mask = torch.zeros_like(input_reshaped)
    mask.scatter_(-1, top_indices, 1.0)
    
    # Apply mask
    output_reshaped = input_reshaped * mask
    output_padded = output_reshaped.view(batch_size, hidden_size_padded)
    
    # Remove padding if it was added
    if hidden_size % 4 != 0:
        output = output_padded[:, :hidden_size]
    else:
        output = output_padded
    
    return output


def apply_mvue_2to4_sparsity(input_tensor):
    """
    Apply MVUE 2:4 sparsity using triton kernel - strict implementation without fallback
    """
    # Convert bfloat16 to float16 for Triton compatibility
    original_dtype = input_tensor.dtype
    if input_tensor.dtype == torch.bfloat16:
        input_temp = input_tensor.to(torch.float16)
    else:
        input_temp = input_tensor
    
    # Apply MVUE 2:4 sparsity - no fallback, strict implementation
    output_temp = MVUE24_approx_triton(input_temp)
    
    # Convert back to original dtype
    if original_dtype == torch.bfloat16:
        output = output_temp.to(torch.bfloat16)
    else:
        output = output_temp
    
    return output


def apply_soft_threshold_weights_2to4_sparsity(input_tensor, scale):
    """
    Apply soft threshold 2:4 sparsity using triton kernel with weight-based scaling - strict implementation without fallback
    """
    # Convert bfloat16 to float16 for Triton compatibility
    original_dtype = input_tensor.dtype
    if input_tensor.dtype == torch.bfloat16:
        input_temp = input_tensor.to(torch.float16)
    else:
        input_temp = input_tensor
    
    # Apply soft threshold 2:4 sparsity - no fallback, strict implementation
    output_temp, _ = soft_threshold24_triton(input_temp)
    
    # Convert back to original dtype and apply scale
    if original_dtype == torch.bfloat16:
        output = output_temp.to(torch.bfloat16) * scale
    else:
        output = output_temp * scale
    
    return output


# Global storage for activation-based soft threshold scaling factors
class ActivationSoftThresholdManager:
    """
    管理基于激活的软阈值缩放因子
    """
    _activation_scales = {}  # 存储固定的激活缩放因子 {layer_id: scale}
    _dynamic_scales = {}     # 存储动态的激活缩放因子 {layer_id: scale}
    _last_update_step = {}   # 记录每层最后更新的步数 {layer_id: step}
    _calibration_data = {}   # 存储校准数据 {layer_id: [activations]}
    _is_calibrated = False   # 是否已完成固定模式的校准
    

    
    @classmethod
    def _compute_optimal_scale(cls, activations):
        """
        计算最优的缩放因子，使得2:4稀疏化前后的MSE最小
        """
        # 应用naive 2:4稀疏化
        sparse_activations = apply_naive_2to4_sparsity(activations)
        
        # 寻找最优缩放因子
        scales = torch.linspace(0.5, 2.0, 100, device=activations.device)
        best_scale = 1.0
        best_mse = float('inf')
        
        for scale in scales:
            scaled_sparse = sparse_activations * scale
            mse = torch.mean((activations - scaled_sparse) ** 2).item()
            if mse < best_mse:
                best_mse = mse
                best_scale = scale.item()
        
        return best_scale
    
    @classmethod
    def get_dynamic_activation_scale(cls, layer_id, current_step, dynamic_steps=10, activations=None, calibration_samples=100):
        """
        获取动态激活缩放因子
        """
        # 动态模式：检查是否需要更新
        if (dynamic_steps == 0 or  # 每步都更新
            layer_id not in cls._last_update_step or 
            current_step - cls._last_update_step[layer_id] >= dynamic_steps):
            
            # 从当前激活中采样固定数量的样本
            if activations.shape[0] > calibration_samples:
                # 随机采样
                indices = torch.randperm(activations.shape[0])[:calibration_samples]
                sampled_activations = activations[indices]
            else:
                sampled_activations = activations
            
            # 更新该层的缩放因子
            scale = cls._compute_optimal_scale(sampled_activations)
            cls._dynamic_scales[layer_id] = scale
            cls._last_update_step[layer_id] = current_step
            
            if current_step % 100 == 0:  # 每100步打印一次
                print(f"🔄 Layer {layer_id} dynamic scale updated at step {current_step}: {scale:.6f} (using {sampled_activations.shape[0]} samples)")
        
        return cls._dynamic_scales.get(layer_id, 1.0)
    




def apply_soft_threshold_dynamic_activation_2to4_sparsity(input_tensor, layer_id=0, current_step=0, dynamic_steps=10, calibration_samples=100):
    """
    Apply soft threshold 2:4 sparsity with dynamic activation-based scaling
    """
    # 获取动态缩放因子
    scale = ActivationSoftThresholdManager.get_dynamic_activation_scale(
        layer_id, current_step, dynamic_steps, input_tensor, calibration_samples
    )
    
    # 应用naive 2:4稀疏化并缩放
    sparse_tensor = apply_naive_2to4_sparsity(input_tensor)
    return sparse_tensor * scale


def compute_split_gemm_lowrank_intermediate(dy1, weight_out1, forward_mask):
    """
    为低秩层计算 dy1 @ weight_out1，使用Split-GEMM策略
    """
    batch_seq_len, hidden_size = dy1.shape
    _, rank1 = weight_out1.shape
    
    # 分析特征稀疏性
    feature_sparsity = torch.mean((dy1 != 0).float(), dim=0)
    num_sparse_features = int(0.95 * hidden_size)
    sparsity_threshold = 0.75
    
    sparse_enough_mask = feature_sparsity > sparsity_threshold
    
    if sparse_enough_mask.sum() >= num_sparse_features:
        _, sparse_indices = torch.topk(feature_sparsity, num_sparse_features)
    else:
        sparse_indices = torch.where(sparse_enough_mask)[0]
    
    sparse_mask = torch.zeros(hidden_size, dtype=torch.bool, device=dy1.device)
    if len(sparse_indices) > 0:
        sparse_mask[sparse_indices] = True
    dense_mask = ~sparse_mask
    
    # Split-GEMM计算
    result = torch.zeros(batch_seq_len, rank1, device=dy1.device, dtype=dy1.dtype)
    
    # 稀疏部分
    if sparse_mask.any():
        dy1_sparse_part = dy1[:, sparse_mask]
        dy1_sparse_part_t = dy1_sparse_part.t()
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_part_t)
        dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
        
        weight_out1_sparse = weight_out1[sparse_mask, :]
        result += torch.mm(dy1_sparse_2to4, weight_out1_sparse)
    
    # 稠密部分
    if dense_mask.any():
        dy1_dense_part = dy1[:, dense_mask]
        weight_out1_dense = weight_out1[dense_mask, :]
        result += torch.mm(dy1_dense_part, weight_out1_dense)
    
    return result


def apply_split_gemm_to_dy1(dy1, forward_mask):
    """
    对dy1应用Split-GEMM策略的稀疏化
    """
    batch_seq_len, hidden_size = dy1.shape
    
    # 分析特征稀疏性
    feature_sparsity = torch.mean((dy1 != 0).float(), dim=0)
    num_sparse_features = int(0.95 * hidden_size)
    sparsity_threshold = 0.75
    
    sparse_enough_mask = feature_sparsity > sparsity_threshold
    
    if sparse_enough_mask.sum() >= num_sparse_features:
        _, sparse_indices = torch.topk(feature_sparsity, num_sparse_features)
    else:
        sparse_indices = torch.where(sparse_enough_mask)[0]
    
    sparse_mask = torch.zeros(hidden_size, dtype=torch.bool, device=dy1.device)
    if len(sparse_indices) > 0:
        sparse_mask[sparse_indices] = True
    dense_mask = ~sparse_mask
    
    # 应用Split-GEMM策略
    dy1_result = dy1.clone()
    
    # 对稀疏部分应用feature-wise 2:4稀疏化
    if sparse_mask.any():
        dy1_sparse_part = dy1[:, sparse_mask]
        dy1_sparse_part_t = dy1_sparse_part.t()
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_part_t)
        dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
        dy1_result[:, sparse_mask] = dy1_sparse_2to4
    
    # 稠密部分保持不变
    
    return dy1_result


def compute_split_gemm_lowrank_intermediate(dy1, weight_out1, forward_mask):
    """
    为低秩层计算 dy1 @ weight_out1，使用Split-GEMM策略
    """
    batch_seq_len, hidden_size = dy1.shape
    _, rank1 = weight_out1.shape
    
    # 分析特征稀疏性
    feature_sparsity = torch.mean((dy1 != 0).float(), dim=0)
    num_sparse_features = int(0.95 * hidden_size)
    sparsity_threshold = 0.75
    
    sparse_enough_mask = feature_sparsity > sparsity_threshold
    
    if sparse_enough_mask.sum() >= num_sparse_features:
        _, sparse_indices = torch.topk(feature_sparsity, num_sparse_features)
    else:
        sparse_indices = torch.where(sparse_enough_mask)[0]
    
    sparse_mask = torch.zeros(hidden_size, dtype=torch.bool, device=dy1.device)
    if len(sparse_indices) > 0:
        sparse_mask[sparse_indices] = True
    dense_mask = ~sparse_mask
    
    # Split-GEMM计算
    result = torch.zeros(batch_seq_len, rank1, device=dy1.device, dtype=dy1.dtype)
    
    # 稀疏部分
    if sparse_mask.any():
        dy1_sparse_part = dy1[:, sparse_mask]
        dy1_sparse_part_t = dy1_sparse_part.t()
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_part_t)
        dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
        
        weight_out1_sparse = weight_out1[sparse_mask, :]
        result += torch.mm(dy1_sparse_2to4, weight_out1_sparse)
    
    # 稠密部分
    if dense_mask.any():
        dy1_dense_part = dy1[:, dense_mask]
        weight_out1_dense = weight_out1[dense_mask, :]
        result += torch.mm(dy1_dense_part, weight_out1_dense)
    
    return result


def apply_split_gemm_to_dy1(dy1, forward_mask):
    """
    对dy1应用Split-GEMM策略的稀疏化
    """
    batch_seq_len, hidden_size = dy1.shape
    
    # 分析特征稀疏性
    feature_sparsity = torch.mean((dy1 != 0).float(), dim=0)
    num_sparse_features = int(0.95 * hidden_size)
    sparsity_threshold = 0.75
    
    sparse_enough_mask = feature_sparsity > sparsity_threshold
    
    if sparse_enough_mask.sum() >= num_sparse_features:
        _, sparse_indices = torch.topk(feature_sparsity, num_sparse_features)
    else:
        sparse_indices = torch.where(sparse_enough_mask)[0]
    
    sparse_mask = torch.zeros(hidden_size, dtype=torch.bool, device=dy1.device)
    if len(sparse_indices) > 0:
        sparse_mask[sparse_indices] = True
    dense_mask = ~sparse_mask
    
    # 应用Split-GEMM策略
    dy1_result = dy1.clone()
    
    # 对稀疏部分应用feature-wise 2:4稀疏化
    if sparse_mask.any():
        dy1_sparse_part = dy1[:, sparse_mask]
        dy1_sparse_part_t = dy1_sparse_part.t()
        dy1_sparse_2to4_t = apply_naive_2to4_sparsity_featurewise(dy1_sparse_part_t)
        dy1_sparse_2to4 = dy1_sparse_2to4_t.t()
        dy1_result[:, sparse_mask] = dy1_sparse_2to4
    
    # 稠密部分保持不变
    
    return dy1_result


def apply_feature_wise_2to4_sparsity(grad_tensor, forward_mask):
    """
    Apply feature-wise 2:4 sparsity for backward pass as described in the paper
    
    Implementation of the paper's approach:
    1. Split features into sparse (95%) and dense (5%) groups
    2. Apply 2:4 sparsity feature-wise to the sparse group
    3. Keep dense group unchanged
    4. Maintain forward sparsity mask
    """
    batch_seq_len, hidden_size = grad_tensor.shape
    
    # Step 1: Analyze feature sparsity to determine which features can be 2:4 sparsified
    feature_sparsity = torch.mean((grad_tensor != 0).float(), dim=0)  # [hidden_size]
    feature_sparsity = feature_sparsity.to(grad_tensor.dtype)  # Ensure same dtype
    
    # Step 2: Sort features by sparsity level
    sparsity_threshold = 0.75  # Features with >75% sparsity can be 2:4 sparsified
    sparse_features_mask = feature_sparsity > sparsity_threshold
    
    # Ensure we have at least 95% of features as sparse (as mentioned in paper)
    num_sparse_features = max(int(0.95 * hidden_size), sparse_features_mask.sum().item())
    
    if sparse_features_mask.sum() < num_sparse_features:
        # If not enough naturally sparse features, select top sparsest features
        _, sparse_indices = torch.topk(feature_sparsity, num_sparse_features)
        sparse_features_mask = torch.zeros(hidden_size, dtype=torch.bool, device=grad_tensor.device)
        sparse_features_mask[sparse_indices] = True
    
    # Step 3: Apply feature-wise 2:4 sparsity
    grad_output = grad_tensor.clone()
    original_dtype = grad_tensor.dtype
    
    # For sparse features: apply feature-wise 2:4 sparsity
    if sparse_features_mask.any():
        sparse_grad = grad_tensor[:, sparse_features_mask]  # [batch_seq_len, num_sparse_features]
        
        # Apply 2:4 sparsity along the batch dimension (feature-wise)
        sparse_grad_t = sparse_grad.t()  # [num_sparse_features, batch_seq_len]
        sparse_grad_2to4 = apply_naive_2to4_sparsity_featurewise(sparse_grad_t)
        sparse_grad_final = sparse_grad_2to4.t()  # [batch_seq_len, num_sparse_features]
        
        grad_output[:, sparse_features_mask] = sparse_grad_final
    
    # Step 4: Apply forward mask to maintain consistency
    # This ensures that values dropped in forward pass don't reappear in backward
    grad_output = grad_output * forward_mask
    
    # Ensure output maintains original dtype
    return grad_output.to(original_dtype)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        config=None,
    ):
        super().__init__()
        
        # Store config for later use
        self.config = config
        
        # Get activation function type from config
        activation_type = getattr(config, 'squ_relu', 'silu') if config is not None else 'silu'
        
        if activation_type == 'silu':
            # Original SwiGLU architecture: gate_proj * silu(up_proj) -> down_proj
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.act_fn = ACT2FN[hidden_act]
            self.new_intermediate_size = intermediate_size
            self.architecture_type = 'silu'
            print(f"🔧 Using original SwiGLU architecture: hidden_size={hidden_size}, intermediate_size={intermediate_size}")
        else:
            # New architecture: GPT-2 style MLP without gate projection
            # Remove gate_proj, keep only up_proj and down_proj
            # Adjust dimensions to maintain same parameter count
            
            # Original SwiGLU has 3 linear layers: gate_proj, up_proj, down_proj
            # Each with dimensions: hidden_size -> intermediate_size (gate & up) and intermediate_size -> hidden_size (down)
            # Total params: 2 * (hidden_size * intermediate_size) + (intermediate_size * hidden_size) = 3 * hidden_size * intermediate_size
            
            # New architecture has 2 linear layers: up_proj and down_proj
            # To maintain same param count: 2 * (hidden_size * new_intermediate_size) = 3 * hidden_size * intermediate_size
            # Therefore: new_intermediate_size = 1.5 * intermediate_size
            
            self.new_intermediate_size = int(1.5 * intermediate_size)
            self.up_proj = nn.Linear(hidden_size, self.new_intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.new_intermediate_size, hidden_size, bias=False)
            self.gate_proj = None  # No gate projection in new architecture
            self.architecture_type = activation_type
            
            if activation_type == 'relu':
                self.act_fn = ACT2FN["relu"]
                print(f"🔧 Using ReLU MLP architecture: hidden_size={hidden_size}, new_intermediate_size={self.new_intermediate_size}")
            elif activation_type == 'relu2':
                self.act_fn = ACT2FN["relu2"]
                print(f"🔧 Using squared ReLU MLP architecture: hidden_size={hidden_size}, new_intermediate_size={self.new_intermediate_size}")
            
            print(f"🔧 Parameter count maintained: original={3 * hidden_size * intermediate_size}, new={2 * hidden_size * self.new_intermediate_size}")

    def record_activation_sparsity(self, activated_tensor):
        """
        Record sparsity statistics of activated tensor (called every 10 steps)
        
        Args:
            activated_tensor: Tensor after activation function [batch*seq, intermediate_size]
        """
        with torch.no_grad():
            # Calculate sparsity (percentage of zero values)
            total_elements = activated_tensor.numel()
            zero_elements = (activated_tensor == 0).sum().item()
            sparsity = zero_elements / total_elements
            
            # Get layer index using a simple registry
            if not hasattr(LlamaMLP, '_layer_registry'):
                LlamaMLP._layer_registry = {}
            
            layer_id = id(self)
            if layer_id not in LlamaMLP._layer_registry:
                LlamaMLP._layer_registry[layer_id] = len(LlamaMLP._layer_registry)
            
            layer_idx = LlamaMLP._layer_registry[layer_id]
            
            # Store in the global sparsity stats (same format as ActivationSparse2to4Function)
            if not hasattr(LlamaMLP, '_sparsity_stats'):
                LlamaMLP._sparsity_stats = {}
                
            LlamaMLP._sparsity_stats[f'sparsity_relu/layer_{layer_idx}'] = sparsity
            
            # Debug info
            current_step = getattr(LlamaMLP, '_current_training_step', 0)
            print(f"🔍 Standard MLP: Step {current_step}, Layer {layer_idx}, Sparsity: {sparsity:.4f}")

    @classmethod
    def get_sparsity_stats(cls):
        """
        Get current sparsity statistics for all layers
        Returns dict suitable for wandb logging
        """
        if not hasattr(cls, '_sparsity_stats') or not cls._sparsity_stats:
            return {}
        
        # Start with the stats we have
        wandb_dict = cls._sparsity_stats.copy()
        
        # Calculate aggregated statistics
        all_sparsities = []
        for key, value in cls._sparsity_stats.items():
            if key.startswith("sparsity_relu/layer_"):
                all_sparsities.append(value)
        
        # Add mean statistics
        if all_sparsities:
            wandb_dict["sparsity_relu/mean_across_layers"] = sum(all_sparsities) / len(all_sparsities)
            
            # Clear the stats after reading them
            cls._sparsity_stats.clear()
        
        return wandb_dict

    @classmethod
    def clear_sparsity_stats(cls):
        """Clear sparsity statistics"""
        if hasattr(cls, '_sparsity_stats'):
            cls._sparsity_stats.clear()

    def forward(self, x):
        if self.architecture_type == 'silu':
            # Original SwiGLU architecture: gate_proj * silu(up_proj) -> down_proj
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            # New architecture without gate projection
            config = getattr(self, 'config', None)
            use_activation_2by4 = getattr(config, 'activation_2by4', False) if config is not None else False
            

            
            if use_activation_2by4 and self.architecture_type == 'relu2':
                # Use activation 2:4 sparsity with squared ReLU
                # Complete FFN flow: input -> up_proj -> squared_relu -> [2:4 sparsify] -> down_proj
                if config is not None:
                    sparsity_method = getattr(config, 'activation_sparse_method', 'mvue')
                    warmup_steps = getattr(config, 'activation_dense_warmup_steps', 1000)
                    dx_direct_sparse = getattr(config, 'dx_direct_sparse', False)
                    dynamic_steps = getattr(config, 'dynamic_activation_steps', 10)
                    calibration_samples = getattr(config, 'activation_calibration_samples', 100)
                else:
                    # Fallback values if no config
                    sparsity_method = 'mvue'
                    warmup_steps = 1000
                    dx_direct_sparse = False
                    dynamic_steps = 10
                    calibration_samples = 100
                
                # Check if we're using LowRankLinear layers
                
                is_lowrank = isinstance(self.up_proj, LowRankLinear) and isinstance(self.down_proj, LowRankLinear)
                

                
                if is_lowrank:
                    # Use low-rank version of activation 2:4 sparsity
                    return ActivationSparse2to4LowRankFunction.apply(
                        x,                          # input
                        self.up_proj.weight_in,     # weight_in1
                        self.up_proj.weight_out,    # weight_out1
                        self.down_proj.weight_in,   # weight_in2
                        self.down_proj.weight_out,  # weight_out2
                        self.up_proj.bias,          # bias1
                        self.down_proj.bias,        # bias2
                        sparsity_method,            # sparsity method
                        warmup_steps,               # warmup steps
                        dx_direct_sparse,           # dx computation method
                        dynamic_steps,              # dynamic activation steps
                        calibration_samples         # calibration samples
                    )
                else:
                    # Use standard version for full-rank layers
                    return ActivationSparse2to4Function.apply(
                        x,                    # input
                        self.up_proj.weight,  # weight1 (first linear layer)
                        self.down_proj.weight, # weight2 (second linear layer)
                        None,                 # bias1 (up_proj has no bias)
                        None,                 # bias2 (down_proj has no bias)
                        sparsity_method,      # sparsity method
                        warmup_steps,         # warmup steps
                        dx_direct_sparse,     # dx computation method
                        dynamic_steps,        # dynamic activation steps
                        calibration_samples   # calibration samples
                    )
            else:
                # Standard forward pass without activation 2:4 sparsity
                # input -> up_proj -> activation -> down_proj
                up_output = self.up_proj(x)
                activated = self.act_fn(up_output)
                
                # Record sparsity statistics if enabled (will be uploaded later by main loop)
                config = getattr(self, 'config', None)
                if config is not None and getattr(config, 'wandb_sparsityrelu', False):
                    # Record sparsity for every forward pass during training
                    # (frequency control is handled in the main training loop)
                    self.record_activation_sparsity(activated)
                
                return self.down_proj(activated)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # WARNING: padding mask is ignored, causal is always applied
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, dropout_p=0.0, is_causal=True,
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            config=config,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # NOTE: big optimization could be done here (?)
            # maybe the copy operation that you saw in the debugger was happening here

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
