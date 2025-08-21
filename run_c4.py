import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import wandb
import datetime
import re
import inspect

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM, LlamaMLP

import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
from loro_torch.loro_optim import LOROAdamW
from loro_torch.lowrank_module import LowRankLinear

# Import activation sparse functions (moved to top to avoid UnboundLocalError)
from peft_pretraining.modeling_llama import ActivationSparse2to4Function, ActivationSparse2to4LowRankFunction

from sparse_fullrank_linear import (
    apply_sparse2to4_to_model,
    apply_activation_sparse2to4_to_model, 
    Sparse2to4Linear,
    ActivationSparse2to4Linear,
    enable_flip_rate_tracking_for_model,
    calculate_model_flip_rate
)

transformers.logging.set_verbosity_error()

# ============================================================================
# CoLA and LoST Integration - HybridSparseLinear class
# ============================================================================

from transformers.activations import ACT2FN
import torch.nn.functional as F

# 已弃用：CoLALoSTActivationSparse2to4Function
# 统一改为 peft_pretraining/modeling_llama.py 中的
# ActivationSparse2to4LowRankFunctionSingle（别名 _cola/_lost）实现。
# 这里删除旧实现以避免重复与行为不一致。


def reset_optimizer_momentum(optimizer):
    """
    Reset momentum buffers for Adam-based optimizers
    Works for AdamW, Adam, and other Adam variants
    对所有优化器类型都有效
    """
    if not hasattr(optimizer, 'state'):
        return 0
    
    reset_count = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                # Reset exp_avg (momentum) and exp_avg_sq (RMSprop-like term) 
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                    reset_count += 1
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()
                # Reset step counter for Adam (must be tensor in newer PyTorch versions)
                if 'step' in state:
                    if isinstance(state['step'], torch.Tensor):
                        state['step'].zero_()  # For tensor version
                    else:
                        state['step'] = 0      # For integer version
    
    return reset_count

class HybridSparseLinear(nn.Module):
    """
    Hybrid sparse linear layer that combines low-rank and column-wise sparse components
    Supports both CoLA (with SiLU activation) and LoST (column-wise sparsity) modes
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            original_weight: torch.Tensor,
            lowrank_module,
            sparsity: float = 0.05,
            sparse_method: str = "random",
            sparse_svd_rank: int = None,
            sparse_svd_inverse: bool = False,
            rank: int = 128,
            gamma: float = 0.5,
            bias: bool = True,
            cola_silu: bool = False,
            cola_sparse_method: str = "cola_init",
            more_activation_relu2: bool = False,
            activation_sparse_method: str = "mvue",
            activation_dense_warmup_steps: int = 1000,
            activation_2by4_permute: bool = True,
            module_name: str = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lowrank_module = lowrank_module
        self.cola_silu = cola_silu
        self.cola_sparse_method = cola_sparse_method
        self.module_name = module_name  # Store module name for logging
        
        # More activation ReLU2 configuration
        self.more_activation_relu2 = more_activation_relu2
        self.activation_sparse_method = activation_sparse_method
        self.activation_dense_warmup_steps = activation_dense_warmup_steps
        self.activation_2by4_permute = activation_2by4_permute
        
        # 临时使用original_weight，初始化后会删除
        self.register_buffer('original_weight', original_weight.clone())
        
        # 存储形状和参数
        self.shape = (out_features, in_features)
        self.sparsity = sparsity
        self.sparse_method = sparse_method
        self.sparse_svd_rank = sparse_svd_rank
        self.sparse_svd_inverse = sparse_svd_inverse
        self.rank = rank
        
        # Gamma blending coefficient
        self.register_buffer('gamma', torch.tensor(gamma))
        
        if self.cola_silu:
            self.lr_act = ACT2FN["silu"]
        
        # Bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=original_weight.device))
        else:
            self.register_parameter('bias', None)
        
        self.mask_initialized = False

    def initialize_mask(self, gradient=None):
        """Initialize column-wise sparse mask and CoLA initialization if enabled"""
        with torch.no_grad():
            if self.cola_sparse_method == "svd":
                # Initialize low-rank matrices using SVD decomposition
                weight_float = self.original_weight.to(torch.float32)
                U, S, Vh = torch.linalg.svd(weight_float, full_matrices=False)
                
                rank = self.lowrank_module.rank
                # Take the top 'rank' singular values/vectors
                U_k = U[:, :rank]
                S_k = S[:rank]
                Vh_k = Vh[:rank, :]
                
                # Initialize weight_in as U_k * sqrt(S_k)
                # Initialize weight_out as sqrt(S_k) * Vh_k
                S_sqrt = torch.sqrt(S_k)
                self.lowrank_module.weight_in.data.copy_((U_k * S_sqrt).T.to(self.lowrank_module.weight_in.dtype))
                self.lowrank_module.weight_out.data.copy_((S_sqrt.unsqueeze(1) * Vh_k).to(self.lowrank_module.weight_out.dtype))
            elif self.cola_sparse_method == "cola_init":
                # Initialize low-rank matrices using CoLA style initialization
                target_sdv = (self.in_features + self.out_features) ** (-1 / 2)
                rank = self.lowrank_module.rank
                scale_factor = rank ** (-1 / 4) * target_sdv ** (1 / 2)
                self.lowrank_module.weight_in.data.copy_(torch.randn_like(self.lowrank_module.weight_in) * scale_factor)
                self.lowrank_module.weight_out.data.copy_(torch.randn_like(self.lowrank_module.weight_out) * scale_factor)
            
            # Column-wise sparsity for LoST
            num_cols_to_keep = max(1, int(self.sparsity * self.in_features))
            
            if self.sparse_method == "svd":
                weight_float = self.original_weight.to(torch.float32)
                U, S, Vh = torch.linalg.svd(weight_float.t(), full_matrices=False)
                
                lowrank_k = int(self.lowrank_module.rank)
                
                if self.sparse_svd_rank and self.sparse_svd_rank > 0:
                    k = min(self.sparse_svd_rank, len(S) - lowrank_k)
                    
                    if self.sparse_svd_inverse:
                        # Use inverse order (smallest singular values first)
                        # Start from the end and go backwards
                        start_idx = max(0, len(S) - k)
                        end_idx = len(S)
                    else:
                        # Use normal order (largest singular values after lowrank)
                        start_idx = lowrank_k
                        end_idx = start_idx + k
                    
                    U_k = U[:, start_idx:end_idx]
                    S_k = S[start_idx:end_idx]
                    Vh_k = Vh[start_idx:end_idx, :]
                    
                    reconstructed = U_k @ torch.diag(S_k) @ Vh_k
                    # 使用 L2 范数筛选列
                    col_norms = torch.norm(reconstructed.t(), dim=0)  # [in]
                    _, topk_cols = torch.topk(col_norms, num_cols_to_keep, largest=True)
                else:
                    # 直接使用原权重的列范数
                    col_norms = torch.norm(self.original_weight, dim=0)  # [in]
                    _, topk_cols = torch.topk(col_norms, num_cols_to_keep, largest=True)
            elif self.sparse_method == "random":
                topk_cols = torch.randperm(self.in_features)[:num_cols_to_keep]
            else:  # gradient or other methods
                # For simplicity, fall back to random
                topk_cols = torch.randperm(self.in_features)[:num_cols_to_keep]
            
            # 排序方便 forward 使用
            topk_cols, _ = torch.sort(topk_cols)
            
            # 保存被选中的列 index
            self.register_buffer("selected_col_indices", topk_cols)
            
            # 创建稀疏参数，只有这些列是参数
            sparse_weight = self.original_weight[:, topk_cols]  # [out, kept_in]
            # 确保稀疏权重使用与低秩模块相同的dtype和device
            target_dtype = self.lowrank_module.weight_in.dtype
            target_device = self.lowrank_module.weight_in.device
            self.values = nn.Parameter(sparse_weight.clone().to(dtype=target_dtype, device=target_device))
            
            # 删除原始大矩阵
            self.register_buffer('original_weight', None)
            self.mask_initialized = True
            torch.cuda.empty_cache()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 计算低秩部分 - 使用正确的属性名 weight_in, weight_out
        if self.cola_silu:
            if self.more_activation_relu2:
                # CoLA mode with ReLU² + activation 2:4 sparsity using unified single-layer function
                from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunction_cola
                base_output = ActivationSparse2to4LowRankFunction_cola.apply(
                    input, self.lowrank_module.weight_in, self.lowrank_module.weight_out,
                    getattr(self.lowrank_module, 'bias', None),
                    getattr(self, 'activation_sparse_method', 'naive'),
                    getattr(self, 'activation_dense_warmup_steps', 1000),
                    getattr(self, 'dx_direct_sparse', False),
                    getattr(self, 'dynamic_activation_steps', 10),
                    getattr(self, 'activation_calibration_samples', 100),
                    getattr(self, 'activation_2by4_permute', True),
                    getattr(self, 'module_name', None),  # Pass module name
                )
            else:
                # Default CoLA mode: x @ weight_in -> SiLU -> @ weight_out.T
                temp = input @ self.lowrank_module.weight_in    # [batch, seq, rank]
                temp = self.lr_act(temp)                       # SiLU activation  
                base_output = temp @ self.lowrank_module.weight_out.T  # [batch, seq, out_dim]
            
            # For CoLA, we primarily use the low-rank part with SiLU activation
            # Optionally blend with sparse part if sparse is enabled
            if hasattr(self, 'values') and self.values is not None:
                # Blend with sparse part
                input_selected = input[..., self.selected_col_indices] 
                sparse_output = input_selected @ self.values.T
                output = self.gamma * base_output + (1 - self.gamma) * sparse_output
            else:
                # Pure CoLA: only low-rank with SiLU
                output = base_output
        else:
            # Standard low-rank computation for LoST: x @ weight_in @ weight_out.T
            if self.more_activation_relu2:
                # LoST mode with ReLU² + activation 2:4 sparsity using low-rank split-GEMM aware function
                from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunction_lost
                base_output = ActivationSparse2to4LowRankFunction_lost.apply(
                    input, self.lowrank_module.weight_in, self.lowrank_module.weight_out,
                    getattr(self.lowrank_module, 'bias', None),
                    getattr(self, 'activation_sparse_method', 'mvue'),
                    getattr(self, 'activation_dense_warmup_steps', 1000),
                    getattr(self, 'dx_direct_sparse', False),
                    getattr(self, 'dynamic_activation_steps', 10),
                    getattr(self, 'activation_calibration_samples', 100),
                    getattr(self, 'activation_2by4_permute', True),
                    getattr(self, 'module_name', None),  # Pass module name
                )
            else:
                # Default LoST: simple matmul
                base_output = input @ self.lowrank_module.weight_in @ self.lowrank_module.weight_out.T
            
            # Compute sparse part (column-wise sparse) 
            if hasattr(self, 'values') and self.values is not None:
                input_selected = input[..., self.selected_col_indices]
                sparse_output = input_selected @ self.values.T  # [batch, seq, out_dim]
            else:
                # Fallback to zero sparse output if mask not initialized
                sparse_output = torch.zeros_like(base_output)
            
            # 组合低秩和稀疏部分
            output = self.gamma * base_output + (1 - self.gamma) * sparse_output
        
        if self.bias is not None:
            output = output + self.bias
            
        return output

class CoLALowRankLinear(nn.Module):
    """
    CoLA implementation - adds SiLU activation between low-rank matrices weight_in and weight_out
    Correct implementation: input @ weight_in -> SiLU -> @ weight_out.T
    
    With more_activation_relu2=True: input @ weight_in -> ReLU² + 2:4 sparse -> @ weight_out.T
    """
    def __init__(self, original_module, more_activation_relu2=False, activation_sparse_method="mvue", 
                 activation_dense_warmup_steps=1000, activation_2by4_permute=True,
                 dx_direct_sparse: int = 1,
                 dynamic_activation_steps: int = 10,
                 activation_calibration_samples: int = 100,
                 cola_sparse_method: str = "cola_init",
                 original_weight=None,
                 module_name: str = None):
        super().__init__()
        # Copy all attributes from original LowRankLinear module
        self.in_dim = original_module.in_dim
        self.out_dim = original_module.out_dim
        self.init = original_module.init
        self.weight_in = original_module.weight_in  # [in_dim, rank]
        self.weight_out = original_module.weight_out  # [out_dim, rank] 
        self.bias = original_module.bias
        self.module_name = module_name  # Store the module name for logging
        
        # CoLA activation configuration
        self.more_activation_relu2 = more_activation_relu2
        if more_activation_relu2:
            # Use ReLU2 + activation 2:4 sparsity
            self.activation_sparse_method = activation_sparse_method
            self.activation_dense_warmup_steps = activation_dense_warmup_steps
            self.activation_2by4_permute = activation_2by4_permute
            self.dx_direct_sparse = dx_direct_sparse
            self.dynamic_activation_steps = dynamic_activation_steps
            self.activation_calibration_samples = activation_calibration_samples
        else:
            # Default CoLA: use SiLU
            self.lr_act = ACT2FN["silu"]

        # CoLA 初始化（对低秩矩阵 A、B 做初始化）
        with torch.no_grad():
            if cola_sparse_method == "svd" and original_weight is not None:
                # SVD-based initialization
                weight_float = original_weight.to(torch.float32)
                U, S, Vh = torch.linalg.svd(weight_float, full_matrices=False)
                
                rank = min(self.weight_in.shape[1], self.weight_out.shape[1])
                # Take the top 'rank' singular values/vectors
                U_k = U[:, :rank]
                S_k = S[:rank]
                Vh_k = Vh[:rank, :]
                
                # Initialize weight_in and weight_out from SVD decomposition
                # original_weight: [out_dim, in_dim]
                # weight_in: [in_dim, rank], weight_out: [out_dim, rank]
                # SVD: original_weight = U @ S @ Vh where U:[out_dim, k], S:[k], Vh:[k, in_dim]
                S_sqrt = torch.sqrt(S_k)
                
                # weight_in should be Vh.T * sqrt(S) -> [in_dim, rank]
                self.weight_in.data.copy_((Vh_k.T * S_sqrt).to(self.weight_in.dtype))
                # weight_out should be U * sqrt(S) -> [out_dim, rank]  
                self.weight_out.data.copy_((U_k * S_sqrt).to(self.weight_out.dtype))
            elif cola_sparse_method == "cola_init":
                # Original CoLA-style initialization
                target_sdv = (self.in_dim + self.out_dim) ** (-0.5)
                rank = min(self.weight_in.shape[1], self.weight_out.shape[1])
                
                # Adjust scale factor based on activation function
                if more_activation_relu2:
                    # ReLU² has stronger gradients, use smaller initialization
                    scale_factor = (rank ** (-0.25)) * (target_sdv ** 0.5) * 0.5
                else:
                    # SiLU activation, use original scale
                    scale_factor = (rank ** (-0.25)) * (target_sdv ** 0.5)
                
                dtype_in = self.weight_in.dtype
                dtype_out = self.weight_out.dtype
                device = self.weight_in.device
                self.weight_in.data.copy_(torch.randn_like(self.weight_in.to(torch.float32)).mul_(scale_factor).to(dtype_in).to(device))
                self.weight_out.data.copy_(torch.randn_like(self.weight_out.to(torch.float32)).mul_(scale_factor).to(dtype_out).to(device))
        
    @property 
    def rank(self):
        return min(min(self.weight_in.shape), min(self.weight_out.shape))
    
    def forward(self, x):
        if self.more_activation_relu2:
            # CoLA with ReLU² + activation 2:4 sparsity using low-rank split-GEMM aware function
            from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunction_cola
            return ActivationSparse2to4LowRankFunction_cola.apply(
                x, self.weight_in, self.weight_out, self.bias,
                self.activation_sparse_method,
                self.activation_dense_warmup_steps,
                self.dx_direct_sparse,
                self.dynamic_activation_steps,
                self.activation_calibration_samples,
                self.activation_2by4_permute,
                self.module_name,  # Pass module name for logging
            )
        else:
            # Default CoLA: x @ weight_in -> SiLU -> @ weight_out.T  
            temp = x @ self.weight_in          # [batch, seq, rank]
            temp = self.lr_act(temp)           # SiLU activation
            out = temp @ self.weight_out.T     # [batch, seq, out_dim]
            
            if self.bias is not None:
                out = out + self.bias
                
            return out

def create_hybrid_sparse_lowrank_model(base_model, lowrank_model, args):
    """
    Create hybrid sparse+lowrank model for CoLA and LoST optimizers
    """
    sparse_layers = []
    
    for base_name, base_module in base_model.named_modules():
        if isinstance(base_module, nn.Linear):
            if any(t in base_name for t in ["attn", "mlp"]):
                # Get corresponding lowrank module
                lowrank_module = get_module_by_name(lowrank_model, base_name)
                
                if hasattr(lowrank_module, 'weight_in') and hasattr(lowrank_module, 'weight_out'):
                    hybrid_layer = HybridSparseLinear(
                        base_module.in_features,
                        base_module.out_features,
                        base_module.weight.data,
                        lowrank_module,
                        sparsity=args.lost_sparsity,
                        sparse_method=args.lost_sparse_method,
                        sparse_svd_rank=args.lost_sparse_svd_rank,
                        gamma=args.lost_gamma,
                        bias=base_module.bias is not None,
                        cola_silu=args.cola_silu,
                        cola_init=args.cola_init
                    )
                    
                    sparse_layers.append(hybrid_layer)
                    
                    # Replace the module in lowrank_model
                    parts = base_name.split('.')
                    current = lowrank_model
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    setattr(current, parts[-1], hybrid_layer)
    
    return lowrank_model, sparse_layers

def get_module_by_name(model, name):
    """Get module by name"""
    names = name.split('.')
    module = model
    for n in names:
        module = getattr(module, n)
    return module

def extract_size_and_type(config_path):
    size_match = re.search(r'(\d+[mb])', config_path.lower())
    size = size_match.group(1) if size_match else "unknown"

    if "small" in config_path:
        type = "small"
    elif "delta" in config_path:
        type = "delta"
    elif "normal" in config_path:
        type = "normal"
    else:
        return f"{size}"

    return f"{size}_{type}"

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    elif value.lower() in {"false", "0", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--flip_rate",
        type=str_to_bool, default=False,
        help="Enable flip rate calculation and logging to track mask stability in 2:4 sparse training."
    )
    parser.add_argument("--attn_2by4", type=str_to_bool, default=False)
    parser.add_argument("--mlp_2by4", type=str_to_bool, default=True)

    parser.add_argument(
        "--enable_2to4_sparse",
        type=str_to_bool,
        default=False,
                help="Enable 2:4 sparse training on low-rank matrices"
    )
    parser.add_argument(
        "--sparse_init_scale",
        type=float,
        default=1.0,
        help="Initial scale for sparse weights (will be overwritten by computed values)"
    )
    parser.add_argument(
        "--activation_2by4",
        type=str_to_bool,
        default=False,
        help="Enable 2:4 sparsity on input activations in addition to weights"
    )
    parser.add_argument(
        "--activation_soft_threshold",
        type=str_to_bool,
        default=False,
        help="Use soft threshold method for activation 2:4 sparsity (True) or MVUE method (False). Only effective when --activation_2by4 is True"
    )
    parser.add_argument(
        "--squ_relu",
        type=str,
        default="silu",
        choices=["silu", "relu", "relu2"],
        help="MLP activation function: silu (original SwiGLU), relu (standard ReLU without gate), relu2 (squared ReLU without gate)"
    )
    parser.add_argument(
        "--activation_sparse_method",
        type=str,
        default="naive",
        choices=["naive", "mvue", "soft_threshold_weights", "soft_dynamic"],
        help="Method for activation 2:4 sparsification: naive (default, top-2), mvue, soft_threshold_weights (weight MSE), soft_dynamic (dynamic activation MSE)"
    )
    parser.add_argument(
        "--permute_2by4",
        type=str_to_bool,
        default=False,
        help="Enable input permutation during activation 2:4 sparsity training. Set to False to disable permutation for better warmup consistency."
    )
    parser.add_argument(
        "--dynamic_activation_steps",
        type=int,
        default=10,
        help="Update interval for soft_dynamic method (default: 10 steps)"
    )
    parser.add_argument(
        "--activation_calibration_samples",
        type=int,
        default=100,
        help="Number of samples to use for activation-based soft threshold calibration (default: 100)"
    )

    parser.add_argument(
        "--activation_dense_warmup_steps",
        type=int,
        default=1000,
        help="Number of training steps to use dense training before enabling activation 2:4 sparsity (paper uses 1000)"
    )
    parser.add_argument(
        "--dx_direct_sparse",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help=(
            "Backward sparse strategy: 1=full split_gemm, 2=naive split_gemm, 3=dense chain rule. "
            "Enable_permute 控制是否做输入/梯度 permutation。"
        )
    )
    parser.add_argument(
        "--wandb_sparsityrelu",
        type=str_to_bool,
        default=False,
        help="If True, log sparsity statistics of activations after ReLU/ReLU² to wandb for each layer"
    )
    parser.add_argument("--c4_local", type=str_to_bool, default=True)
    parser.add_argument("--train_data_path", type=str, default="en/c4-train.*.json.gz",
                        help="Path to C4 training data files")
    parser.add_argument("--val_data_path", type=str, default="en/c4-validation.*.json.gz",
                        help="Path to C4 validation data files")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        # choices=["linear", "cosine", "cosine_restart"],
    )
    parser.add_argument("--cosine_restart_freq", type=int, default=None)
    parser.add_argument("--cosine_restart_warmup", type=int, default=5)
    parser.add_argument("--lr_jag_after_warmup", default=False, action="store_true")
    parser.add_argument(
        "--lr_adjust_steps", type=int, default=0, help="lr schedule displacement"
    )

    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=10_000,
        help="Number of **update steps** to train for."
        "Notice that gradient accumulation is taken into account.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=training_utils.max_train_tokens_to_number,
        default=None,
        help="Number of tokens to train on. Overwrites num_training_steps. "
        "You can use M and B suffixes, e.g. 100M or 1B.",
    )
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_ckpt", type=str_to_bool, default=False)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)

    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # CoLA and LoST parameters for adamw_cola and adamw_lost optimizers
    parser.add_argument("--cola_silu", type=str_to_bool, default=False,
                        help="Whether to add SiLU activation between A and B matrices in low-rank structures")
    parser.add_argument("--cola_sparse_method", type=str, default="cola_init",
                        choices=["cola_init", "svd"],
                        help="Initialization method for CoLA low-rank matrices: cola_init (original) or svd")
    parser.add_argument("--lost_sparsity", type=float, default=0.05,
                        help="Column-wise sparsity ratio for LoST optimizer")
    parser.add_argument("--lost_sparse_method", type=str, default="random",
                        choices=["random", "gradient", "svd"],
                        help="Method to initialize the sparse mask for LoST")
    parser.add_argument("--lost_sparse_svd_rank", type=int, default=256,
                        help="Rank to use for SVD-based mask initialization in LoST")
    parser.add_argument("--lost_gamma", type=float, default=0.5,
                        help="Blending coefficient between low-rank and sparse parts in LoST")
    parser.add_argument("--lost_sparse_svd_inverse", action="store_true",
                        help="Use inverse order (smallest singular values first) for sparse SVD selection")

    # Momentum reset functionality  
    parser.add_argument("--momentum_reset_steps", type=int, default=0,
                        help="Reset Adam optimizer momentum every N steps. 0 means no reset. Works for all optimizer types.")

    # More activation ReLU2 for CoLA/LoST
    parser.add_argument("--more_activation_relu2", type=str_to_bool, default=False,
                        help="Use ReLU2 activation with activation sparsity (2:4) for CoLA/LoST low-rank matrices instead of default activations")

    # LORO parameters
    parser.add_argument(
        "--loro_type",
        type=str,
        default="loro",
        help="eucl | loro",
    )
    parser.add_argument(
        "--loro_freq",
        type=int,
        default=1,
        help="frequency of using LORO exact update",
    )
    parser.add_argument(
        "--loro_fallback_freq",
        type=int,
        default=None,
        help="fallback frequency if numerical issues occur (auto-adjust loro_freq)",
    )
    parser.add_argument(
        "--loro_refresh",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--loro_refresh_freq",
        type=int,
        default=0,
        help="frequency of refreshing GeomLowRank states",
    )
    parser.add_argument(
        "--loro_attn_rank",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--loro_mlp_rank",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--loro_init",
        type=str,
        default="xavier",
        help="auto | xavier | kaiming | orth | randn_x.x | const_x.x",
    )
    parser.add_argument(
        "--loro_scope",
        type=str,
        default=None,
        help="all | mlp | attn | None",
    )
    parser.add_argument(
        "--loro_lr_scaler",
        type=str,
        default="1.0",
        help="lr down scaling for lazy LORO update, -1 | float | auto",
    )
    parser.add_argument(
        "--loro_mlp_dense",
        action="store_true",
        default=False,
    )

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")


    args = parser.parse_args()

    if args.save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.save_dir = os.path.join(script_dir, "ckpt")
        logger.info(f"No save_dir specified, will use {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    optimizer_desc = args.optimizer.lower()
    if args.optimizer.lower() == "loro_adamw":
        optimizer_desc += f"_{args.loro_type}"

        if args.loro_scope:
            optimizer_desc += f"_freq_{args.loro_freq}"
            optimizer_desc += f"_r_attn{args.loro_attn_rank}_mlp{args.loro_mlp_rank}_{args.loro_scope}"
            optimizer_desc += f"_init_lrk_{args.loro_init}"
            optimizer_desc += f"_rs_{args.loro_lr_scaler}"
            if args.enable_2to4_sparse:
                optimizer_desc += f"_sparse2to4"
        else:
            args.loro_type = "eucl"
            optimizer_desc += f"_fullrank"

        if args.loro_refresh:
            optimizer_desc += f"_rfsh_{args.loro_refresh}_{args.loro_refresh_freq}"
        if args.loro_mlp_dense:
            optimizer_desc += "_mlpdense"

    scheduler_desc = args.scheduler
    if args.scheduler in ["cosine_restart", "cosine_restart_zero"]:
        assert args.cosine_restart_freq is not None
        assert args.cosine_restart_warmup is not None
        scheduler_desc += f"_cyc{args.cosine_restart_freq}_wp{args.cosine_restart_warmup}_adj{args.lr_adjust_steps}"

    args.desc = f"{optimizer_desc}_lr_{args.lr}_gc{args.grad_clipping}_{scheduler_desc}_{args.warmup_steps}_{args.min_lr_ratio}"

    args = args_utils.check_args_torchrun_main(args)
    print(f"\n\nExperiment = {args.desc}\n\n")
    return args


@torch.no_grad()
def evaluate_model(
    model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size, c4_local
):
    _time = time.time()
    if args.c4_local:
        val_data = datasets.load_dataset('arrow', data_files=args.val_data_path, split="train", streaming=True,
                                         trust_remote_code=True)
    else:
        val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True, trust_remote_code=True)

    # val_data = datasets.load_dataset(
    #     "allenai/c4",
    #     "en",
    #     split="validation",
    #     streaming=True,
    #     trust_remote_code=True,
    #     # cache_dir=f"{args.data_dir}/c4",
    # )
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(
            val_data, rank=global_rank, world_size=world_size
        )

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(
        val_data_mapped, batch_size
    )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")
    eval_time = time.time()

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    torch.cuda.synchronize()
    eval_time = time.time() - eval_time

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size
    perplexity = np.exp(total_loss)

    return total_loss, evaluated_on_tokens, eval_time, perplexity


def measure_inference_speed(model, dataloader, tokenizer, device, num_batches=100):
    model.eval()  # 设置为评估模式

    # 预热几个批次
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3:  # 预热3个批次
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)

    # 开始测量
    total_tokens = 0
    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)

            # 计算此批次中的令牌数量
            tokens_in_batch = (batch["input_ids"] != tokenizer.pad_token_id).sum().item()
            total_tokens += tokens_in_batch

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 计算每秒处理的令牌数
    tokens_per_second = total_tokens / elapsed_time

    return {
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_tokens,
        "elapsed_time": elapsed_time,
        "batches_processed": num_batches
    }


def get_weight_sparsity_stats(model):
    """
    计算模型中所有权重矩阵的稀疏度
    
    Args:
        model: LlamaForCausalLM模型
        
    Returns:
        dict: 包含所有权重矩阵稀疏度的字典
    """
    sparsity_stats = {}
    
    with torch.no_grad():
        # 获取原始模型 (处理DistributedDataParallel包装)
        if hasattr(model, 'module'):
            # 分布式训练模式
            actual_model = model.module
        else:
            # 单GPU模式
            actual_model = model
        
        # 遍历所有transformer layers
        for layer_idx, layer in enumerate(actual_model.model.layers):
            # Attention权重
            attention = layer.self_attn
            
            # 检查是否是LowRankLinear
            
            # Q, K, V, O projections
            for proj_name, proj_module in [
                ('q_proj', attention.q_proj),
                ('k_proj', attention.k_proj), 
                ('v_proj', attention.v_proj),
                ('o_proj', attention.o_proj)
            ]:
                if hasattr(proj_module, 'weight_in') and hasattr(proj_module, 'weight_out'):
                    # LowRank/CoLA: 两个矩阵 weight_in 和 weight_out
                    weight_in_sparsity = calculate_sparsity(proj_module.weight_in)
                    weight_out_sparsity = calculate_sparsity(proj_module.weight_out)
                    sparsity_stats[f'weight_sparsity/layer_{layer_idx}/attn_{proj_name}_in'] = weight_in_sparsity
                    sparsity_stats[f'weight_sparsity/layer_{layer_idx}/attn_{proj_name}_out'] = weight_out_sparsity
                elif hasattr(proj_module, 'weight'):
                    # Full-rank: 一个权重矩阵
                    weight_sparsity = calculate_sparsity(proj_module.weight)
                    sparsity_stats[f'weight_sparsity/layer_{layer_idx}/attn_{proj_name}'] = weight_sparsity
            
            # MLP权重
            mlp = layer.mlp
            
            # Up和Down projections (以及Gate如果存在)
            mlp_projs = [('up_proj', mlp.up_proj), ('down_proj', mlp.down_proj)]
            if hasattr(mlp, 'gate_proj') and mlp.gate_proj is not None:
                mlp_projs.append(('gate_proj', mlp.gate_proj))
            
            for proj_name, proj_module in mlp_projs:
                if hasattr(proj_module, 'weight_in') and hasattr(proj_module, 'weight_out'):
                    # LowRank/CoLA: 两个矩阵
                    weight_in_sparsity = calculate_sparsity(proj_module.weight_in)
                    weight_out_sparsity = calculate_sparsity(proj_module.weight_out)
                    sparsity_stats[f'weight_sparsity/layer_{layer_idx}/mlp_{proj_name}_in'] = weight_in_sparsity
                    sparsity_stats[f'weight_sparsity/layer_{layer_idx}/mlp_{proj_name}_out'] = weight_out_sparsity
                elif hasattr(proj_module, 'weight'):
                    # Full-rank: 一个权重矩阵
                    weight_sparsity = calculate_sparsity(proj_module.weight)
                    sparsity_stats[f'weight_sparsity/layer_{layer_idx}/mlp_{proj_name}'] = weight_sparsity
    
    return sparsity_stats


def calculate_sparsity(weight_tensor):
    """
    计算权重张量的稀疏度
    
    Args:
        weight_tensor: 权重张量
        
    Returns:
        float: 稀疏度 (0到1之间)
    """
    if weight_tensor is None:
        return 0.0
    
    total_elements = weight_tensor.numel()
    zero_elements = (weight_tensor == 0).sum().item()
    return zero_elements / total_elements if total_elements > 0 else 0.0



def main(args):
    Warning(f"\nSave_ckpt = {args.save_ckpt}, Save_dir = {args.save_dir}.\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.add(sys.stdout, level="DEBUG", colorize=False, backtrace=True, diagnose=True)
    logger.info(
        f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
    )

    # dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    dist.init_process_group(
        backend="nccl",
        rank=global_rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=7200)  # 延长超时时间为2小时
    )

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    print(f"Rank {global_rank} using device {device}")

    args.num_cuda = torch.cuda.device_count()
    print(f"\n\n# CUDA visible devices: {args.num_cuda}\n\n")

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert (
                args.total_batch_size % world_size == 0
            ), "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (
                args.batch_size * world_size
            )
            assert (
                args.gradient_accumulation > 0
            ), "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0:
        logger.remove()

    # initialize wandb without config (it is passed later)


    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    if args.c4_local:
        data = datasets.load_dataset('arrow', data_files=args.train_data_path, split="train", streaming=True)
    else:
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
    # data = datasets.load_dataset(
    #     "allenai/c4",
    #     "en",
    #     split="train",
    #     streaming=True,
    #     trust_remote_code=True,
    #     # cache_dir=f"{args.data_dir}/c4",
    # )

    seed_for_shuffle = 42
    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data,
            rank=global_rank,
            world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base",
        model_max_length=args.max_length,
        # cache_dir=f"{args.model_dir}/t5-base-tokenizer",
    )
    
    # Ensure tokenizer has a valid pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(
        data,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers,
    )

    model_config = AutoConfig.from_pretrained(args.model_config)
    
    # Add activation function and sparsity parameters to model config
    model_config.squ_relu = args.squ_relu
    model_config.activation_2by4 = args.activation_2by4
    model_config.wandb_sparsityrelu = args.wandb_sparsityrelu
    
    # Enable sparsity logging if requested (set early to ensure it's available)
    if args.wandb_sparsityrelu:
        from peft_pretraining.modeling_llama import ActivationSparse2to4Function, ActivationSparse2to4LowRankFunction
        ActivationSparse2to4Function._enable_sparsity_logging = True
        ActivationSparse2to4LowRankFunction._enable_sparsity_logging = True
        logger.info("📊 Activation sparsity logging enabled for wandb")
    
    if args.squ_relu != "silu":
        logger.info(f"🔧 Using {args.squ_relu} activation in MLP layers (no gate projection)")
        
    if args.activation_2by4:
        logger.info(f"🔧 Activation 2:4 sparsity enabled with method: {args.activation_sparse_method}")
        logger.info(f"🔧 Dense warmup for first {args.activation_dense_warmup_steps} steps, then activation 2:4 sparsity")
        mode_desc = {1: 'full split_gemm', 2: 'naive split_gemm', 3: 'dense chain rule'}.get(args.dx_direct_sparse, 'unknown')
        logger.info(f"🔧 dx_direct_sparse = {args.dx_direct_sparse} ({mode_desc})")
        
        # Configure activation 2:4 sparsity parameters
        model_config.activation_sparse_method = args.activation_sparse_method
        model_config.activation_dense_warmup_steps = args.activation_dense_warmup_steps
        model_config.dx_direct_sparse = int(args.dx_direct_sparse)
        model_config.dynamic_activation_steps = args.dynamic_activation_steps
        model_config.activation_calibration_samples = args.activation_calibration_samples
        model_config.permute_2by4 = args.permute_2by4  # Must use getattr for names starting with digits
    
    if "geomlrk" in args.optimizer and args.loro_mlp_dense:
        mlp_rank = min(model_config.intermediate_size, args.loro_mlp_rank)
        model_config.intermediate_size = mlp_rank
        print(
            "\n\n"
            f"Warning: using dense MLP for LowRank weights, \n"
            f"MLP reshaped ({model_config.intermediate_size}, {model_config.hidden_size}) --> ({mlp_rank}, {mlp_rank})"
            f"Lowrank parameterization scope: {args.loro_scope} --> attn successfully!\n"
            "\n\n"
        )
        args.loro_mlp_rank = mlp_rank
        args.loro_scope = "attn"

    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    # Initialize sparsity recording if enabled
    if args.wandb_sparsityrelu and ActivationSparse2to4Function is not None:
        print(f"🔧 Enabling sparsity recording for wandb_sparsityrelu")
        # Enable sparsity logging in both activation sparse functions
        ActivationSparse2to4Function._wandb_sparsityrelu_enabled = True
        ActivationSparse2to4LowRankFunction._wandb_sparsityrelu_enabled = True
        # Also enable for single low-rank function (CoLA/LoST)
        from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunctionSingle
        ActivationSparse2to4LowRankFunctionSingle._wandb_sparsityrelu_enabled = True
        # Initialize global training step counter for all versions
        ActivationSparse2to4Function._global_training_step = 0
        ActivationSparse2to4LowRankFunction._global_training_step = 0
        ActivationSparse2to4LowRankFunctionSingle._global_training_step = 0

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    
    # Initialize CoLA/LoST activation sparse settings if using more_activation_relu2
    if hasattr(args, 'more_activation_relu2') and args.more_activation_relu2:
        from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunctionSingle
        ActivationSparse2to4LowRankFunctionSingle._training_step = 0
        ActivationSparse2to4LowRankFunctionSingle._warmup_steps = args.activation_dense_warmup_steps
        logger.info(f"🔧 Initialized CoLA/LoST activation sparse: warmup_steps={args.activation_dense_warmup_steps}")

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update(
        {
            "max_lr": run_config.pop(
                "lr"
            ),  # rename lr to max_lr to avoid conflicts with scheduler _ar{args.loro_attn_rank}loty{args.loro_type}fr{args.loro_freq}ls{args.loro_lr_scaler}sc{args.scheduler}crfr{args.cosine_restart_freq}as{args.lr_adjust_steps}ra{args.loro_refresh}rf{args.loro_refresh_freq}sc{args.loro_scope}_ini{args.loro_init}
            "total_params_M": n_total_params / 1_000_000,
            "dataset": "c4",
            "model": model_config.to_dict(),
            "world_size": world_size,
            "device": str(device),
        }
    )
    if global_rank == 0:
        model_size = extract_size_and_type(args.model_config)
        runname = f"{time.strftime('%m%d_%H%M%S')}gc{args.grad_clipping}w{args.weight_decay}s{args.num_training_steps}" \
                  f"m{model_size}_op{args.optimizer}mlr{args.min_lr_ratio}lr{args.lr}bs{args.batch_size}" \
                  f"tb{args.total_batch_size}_se{args.save_every}_ee{args.eval_every}_24{args.enable_2to4_sparse}a{args.attn_2by4}m{args.mlp_2by4}_" \
                  f"sa{args.save_ckpt}_ac{args.activation_2by4}_sf{args.activation_soft_threshold}_ac{args.squ_relu}_wb{args.wandb_sparsityrelu}_am{args.activation_sparse_method}_s{args.dynamic_activation_steps}_s{args.activation_calibration_samples}_w{args.activation_dense_warmup_steps}_dx{args.dx_direct_sparse}"
        print(f"runname= {runname}")
        runname_dir = os.path.join(args.save_dir, runname)
        os.makedirs(runname_dir, exist_ok=True)
        wandb.init(
            project="2by4",
            name=runname,
        )

    if global_rank == 0:
        if wandb is not None:
            wandb.config.update(run_config, allow_val_change=True)
            wandb.save(os.path.abspath(__file__), policy="now")  # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(
            total=args.num_training_steps - update_step, desc="Update steps", ncols=80
        )

    # apply GaLore parameterization
    if args.optimizer.lower().startswith("galore"):
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print("enable GaLore for weights in module: ", module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [
            p for p in model.parameters() if id(p) not in id_galore_params
        ]
        # then call galore_adamw
        param_groups = [
            {"params": regular_params},
            {
                "params": galore_params,
                "rank": args.rank,
                "update_proj_gap": args.update_proj_gap,
                "scale": args.galore_scale,
                "proj_type": args.proj_type,
            },
        ]

    layer_wise_flag = (
        False  # flag of using layer-wise optimizer like galore_adamw8bit_per_layer
    )

    if args.optimizer.lower() == "adam":
        # 🔧 Use proper parameter grouping for weight decay (Adam)
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': args.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.Adam(optim_groups, lr=args.lr)

    elif args.optimizer.lower() == "adamw":
        # Full-rank training with optional 2:4 sparsity support
        if args.enable_2to4_sparse:
            logger.info("🔧 Full-rank + 2:4 Sparse Training Mode")
            logger.info("📌 将在普通full-rank linear层上应用2:4稀疏训练")
            
            # Build target modules list based on attn_2by4 and mlp_2by4 flags
            target_modules = []
            
            # Attention modules
            attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            
            if args.attn_2by4:
                target_modules.extend(attn_modules)
                logger.info("📌 将对注意力模块应用2:4稀疏: " + str(attn_modules))
            
            if args.mlp_2by4:
                if args.squ_relu == "silu":
                    # SwiGLU架构：对所有MLP模块应用2:4稀疏 (gate_proj, up_proj, down_proj)
                    mlp_modules = ["gate_proj", "up_proj", "down_proj"]
                    logger.info("📌 将对SwiGLU MLP模块应用2:4稀疏: " + str(mlp_modules))
                else:
                    # 非SwiGLU架构（relu/relu2）：只有up_proj和down_proj，没有gate_proj
                    mlp_modules = ["up_proj", "down_proj"]
                    logger.info("📌 将对非SwiGLU MLP模块应用2:4稀疏: " + str(mlp_modules) + " (无gate_proj)")
                target_modules.extend(mlp_modules)

            logger.info(f"🎯 最终目标模块列表: {target_modules}")

            # Choose sparsity mode based on activation_2by4 parameter
            if args.activation_2by4:
                # Apply activation 2:4 sparsity (weights remain dense)
                logger.info("🔧 Mode: 激活稀疏化 (权重保持dense)")
                model = apply_activation_sparse2to4_to_model(
                    model,
                    target_modules=target_modules,
                    activation_2by4=True,
                    activation_soft_threshold=args.activation_soft_threshold,
                )
                method = "soft threshold" if args.activation_soft_threshold else "MVUE"
                logger.info(f"✅ Full-rank linear layers replaced with ActivationSparse2to4Linear!")
                logger.info(f"🎯 激活2:4化已启用: 使用 {method} 方法，权重保持dense")
            else:
                # Apply weight 2:4 sparsity (traditional mode)
                logger.info("🔧 Mode: 权重稀疏化 (传统2:4模式)")
                model = apply_sparse2to4_to_model(
                    model,
                    target_modules=target_modules,
                )
                logger.info("✅ Full-rank linear layers replaced with Sparse2to4Linear!")
                logger.info("🎯 权重2:4化已启用，激活保持dense")

            # 🔧 Use proper parameter grouping for weight decay for 2:4 sparse training
            param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

            optim_groups = [
                {'params': decay_params, 'weight_decay': args.weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]

            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            logger.info(f"📊 (2:4 Sparse) Weight decay applied to {len(decay_params)} tensors ({num_decay_params:,} parameters)")
            logger.info(f"📊 (2:4 Sparse) Weight decay NOT applied to {len(nodecay_params)} tensors ({num_nodecay_params:,} parameters)")

            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), **extra_args)
            print(f"using fused AdamW: {use_fused}")
            # ‼️ CRITICAL FIX: Use bnb.optim.AdamW for correct weight decay with sparse autograd.Function
            # logger.info("‼️ 使用 bnb.optim.AdamW 来确保 weight_decay 在2:4稀疏训练中正确生效 (L2 Regularization)")
            # optimizer = bnb.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))

        else:
            logger.info("🔧 Standard Full-rank AdamW Training Mode")
            
            # 🔧 Improved: Use proper parameter grouping for weight decay
            # Following best practices: only apply weight decay to 2D parameters (weights)
            # Don't apply weight decay to 1D parameters (bias, LayerNorm, etc.)
            param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            
            optim_groups = [
                {'params': decay_params, 'weight_decay': args.weight_decay},
                {'params': nodecay_params, 'weight_decay': args.weight_decay}
            ]
            
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            logger.info(f"📊 Weight decay applied to {len(decay_params)} tensors ({num_decay_params:,} parameters)")
            logger.info(f"📊 Weight decay NOT applied to {len(nodecay_params)} tensors ({num_nodecay_params:,} parameters)")
            
            optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)

    elif args.optimizer.lower() == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )

    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.beta1,
        )

    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )

    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )

    elif args.optimizer.lower() == "galore_adamw8bit_per_layer":
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit(
                        [
                            {
                                "params": [p],
                                "rank": args.rank,
                                "update_proj_gap": args.update_proj_gap * 2,
                                "scale": args.galore_scale,
                                "proj_type": args.proj_type,
                            }
                        ],
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                    )
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit(
                        [p], lr=args.lr, weight_decay=args.weight_decay
                    )

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheduler(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        layer_wise_flag = True

    ## NOTE: CoLA and LoST optimizers - create models before LORO optimizer
    elif args.optimizer.lower() in ["cola_adamw", "lost_adamw"]:
        # Apply low-rank parameterization with CoLA/LoST features
        from loro_torch.lowrank_module import apply_lowrank_param, get_lowrank_param
        
        # Fix scheduler compatibility for CoLA/LoST optimizers
        if args.lr_adjust_steps != 0 and args.scheduler not in ["cosine_restart", "cosine_restart_zero"]:
            print(f"⚠️  Warning: Setting lr_adjust_steps=0 for {args.scheduler} scheduler compatibility")
            args.lr_adjust_steps = 0
        
        if args.loro_scope is not None:
            # Step 1: Apply standard low-rank parameterization first
            apply_lowrank_param(
                model,
                model_config,
                model_type="llama",
                scope=args.loro_scope,
                attn_rank=args.loro_attn_rank,
                mlp_rank=args.loro_mlp_rank,
                init=args.loro_init,
            )
            print("✅ LORO low-rank parameterization applied!")
            
            # Step 2: Apply CoLA or LoST specific modifications
            if args.optimizer.lower() in ["adamw_cola", "cola_adamw"]:
                # CoLA: Replace LowRankLinear modules with CoLA versions (SiLU activation)
                print("🔧 Applying CoLA SiLU activation to LowRankLinear modules")
                cola_count = 0
                
                for name, module in model.named_modules():
                    if hasattr(module, 'weight_in') and hasattr(module, 'weight_out'):
                        # This is a LowRankLinear module - replace with CoLA version (with more_activation_relu2 support)
                        # Construct original weight if needed for SVD initialization
                        original_weight = None
                        if args.cola_sparse_method == "svd":
                            # Try to reconstruct the original weight from low-rank decomposition
                            # Note: This is an approximation as we only have the low-rank representation
                            # weight_in: [in_features, rank], weight_out: [out_features, rank]
                            # Forward uses: x @ weight_in @ weight_out.T
                            # So effective weight is: weight_in @ weight_out.T = [in_features, out_features]
                            # We need [out_features, in_features] for standard linear layer, so transpose
                            original_weight = (module.weight_in @ module.weight_out.T).T
                        
                        cola_module = CoLALowRankLinear(
                            module,
                            more_activation_relu2=args.more_activation_relu2,
                            activation_sparse_method=args.activation_sparse_method,
                            activation_dense_warmup_steps=args.activation_dense_warmup_steps,
                            activation_2by4_permute=args.permute_2by4,  # Must use getattr for names starting with digits
                            dx_direct_sparse=args.dx_direct_sparse,
                            dynamic_activation_steps=args.dynamic_activation_steps,
                            activation_calibration_samples=args.activation_calibration_samples,
                            cola_sparse_method=args.cola_sparse_method,
                            original_weight=original_weight,
                            module_name=name,  # Pass the full module path name
                        )
                        
                        # Replace the module in the model
                        parts = name.split('.')
                        current = model
                        for part in parts[:-1]:
                            current = getattr(current, part)
                        setattr(current, parts[-1], cola_module)
                        cola_count += 1
                
                if args.more_activation_relu2:
                    print(f"✅ Applied CoLA ReLU² + activation 2:4 sparsity to {cola_count} LowRankLinear modules")
                else:
                    print(f"✅ Applied CoLA SiLU activation to {cola_count} LowRankLinear modules")
                # 若 CoLA 也需要初始化列稀疏mask（与 LoST 同步需求）
                # 为 CoLALowRankLinear 不创建稀疏分支，仅按需执行 CoLA 初始化（已在模块构造时支持 cola_init）
                
            elif args.optimizer.lower() in ["adamw_lost", "lost_adamw"]:
                # LoST: Replace LowRankLinear modules with hybrid sparse+low-rank versions  
                print(f"🔧 LoST features enabled: column-wise sparsity ({args.lost_sparsity})")
                
                lost_count = 0
                for name, module in model.named_modules():
                    if hasattr(module, 'weight_in') and hasattr(module, 'weight_out'):
                        # This is a LowRankLinear module - replace with LoST version
                        # Create a dummy original_weight from the low-rank decomposition with correct dtype/device
                        with torch.no_grad():
                            # Forward uses: x @ weight_in @ weight_out.T, so effective weight is weight_in @ weight_out.T
                            # We need [out_features, in_features] for standard linear layer, so transpose
                            dummy_weight = (module.weight_in @ module.weight_out.T).T.detach().clone()  # Reconstruct full weight
                        
                        lost_module = HybridSparseLinear(
                            in_features=module.in_dim,
                            out_features=module.out_dim,
                            original_weight=dummy_weight,
                            lowrank_module=module,
                            sparsity=args.lost_sparsity,
                            sparse_method=args.lost_sparse_method,
                            sparse_svd_rank=args.lost_sparse_svd_rank,
                            sparse_svd_inverse=args.lost_sparse_svd_inverse,
                            rank=args.rank,
                            gamma=args.lost_gamma,
                            cola_silu=False,  # LoST doesn't use CoLA SiLU
                            more_activation_relu2=args.more_activation_relu2,
                            activation_sparse_method=args.activation_sparse_method,
                            activation_dense_warmup_steps=args.activation_dense_warmup_steps,
                            activation_2by4_permute=args.permute_2by4,
                            module_name=name  # Pass module name
                        )
                        
                        # Initialize the sparse mask (LoST 列稀疏 + 可选 CoLA 初始化)
                        lost_module.initialize_mask()
                        
                        # Replace the module in the model
                        parts = name.split('.')
                        current = model
                        for part in parts[:-1]:
                            current = getattr(current, part)
                        setattr(current, parts[-1], lost_module)
                        lost_count += 1
                
                if args.more_activation_relu2:
                    print(f"✅ Applied LoST hybrid processing + ReLU² activation 2:4 sparsity to {lost_count} LowRankLinear modules")
                else:
                    print(f"✅ Applied LoST hybrid processing to {lost_count} LowRankLinear modules")
        
        # 重要：在CoLA/LoST模块替换后，再次确保整个模型使用正确的dtype
        if args.dtype in ["bf16", "bfloat16"]:
            model = model.to(device=device, dtype=torch.bfloat16)
            print("✅ 在CoLA/LoST模块替换后，重新转换模型为bfloat16")
        else:
            Warning(f"\nUsing full-rank model for {args.optimizer} ...\n")
        
        # Get parameter groups
        param_groups = get_lowrank_param(model, model_config, args.loro_lr_scaler, args.weight_decay)
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )

    ## NOTE: LORO optimizer
    elif args.optimizer.lower() == "loro_adamw":
        # Always use standard LORO implementation as the base
        from loro_torch.lowrank_module import apply_lowrank_param, get_lowrank_param
        
        # Step 1: Apply LORO low-rank parameterization first
        if args.loro_scope is not None:
            if args.loro_mlp_dense:
                assert (
                    args.loro_scope == "attn" and args.loro_mlp_rank == mlp_rank
                ), "Only support dense MLP for attn"

            logger.info("🔧 Step 1: Applying LORO low-rank parameterization...")
            apply_lowrank_param(
                model,
                model_config,
                model_type="llama",
                scope=args.loro_scope,
                attn_rank=args.loro_attn_rank,
                mlp_rank=args.loro_mlp_rank,
                init=args.loro_init,
            )
            logger.info("✅ LORO low-rank parameterization applied successfully!")
        else:
            Warning(f"\nUsing full-rank model ...\n")

        # Step 2: Apply 2:4 sparse on top of LORO (if enabled)
        if args.enable_2to4_sparse:
            logger.info("🔧 Step 2: Applying 2:4 sparse parameterization on LORO parameters...")
            
            # Build target modules list based on attn_2by4 and mlp_2by4 flags
            target_modules = []
            
            # Attention modules
            attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            
            if args.attn_2by4:
                target_modules.extend(attn_modules)
                logger.info("📌 将对注意力模块应用2:4稀疏: " + str(attn_modules))
            
            if args.mlp_2by4:
                if args.squ_relu == "silu":
                    # SwiGLU架构：对所有MLP模块应用2:4稀疏 (gate_proj, up_proj, down_proj)
                    mlp_modules = ["gate_proj", "up_proj", "down_proj"]
                    logger.info("📌 将对SwiGLU MLP模块应用2:4稀疏: " + str(mlp_modules))
                else:
                    # 非SwiGLU架构（relu/relu2）：只有up_proj和down_proj，没有gate_proj
                    mlp_modules = ["up_proj", "down_proj"]
                    logger.info("📌 将对非SwiGLU MLP模块应用2:4稀疏: " + str(mlp_modules) + " (无gate_proj)")
                target_modules.extend(mlp_modules)
            
            if not target_modules:
                logger.warning("⚠️ 启用了2:4稀疏但没有选择任何目标模块！请检查 --attn_2by4 和 --mlp_2by4 参数")
            else:
                logger.info(f"🎯 最终目标模块列表: {target_modules}")
                
                from loro_torch.sparse_overlay import apply_sparse_overlay_on_loro
                apply_sparse_overlay_on_loro(
                    model,
                    sparse_init_scale=args.sparse_init_scale,
                    target_modules=target_modules
                )
                logger.info("✅ 2:4 sparse overlay applied on LORO parameters!")

        # Get base parameter groups for LORO
        param_groups = get_lowrank_param(model, model_config, args.loro_lr_scaler, args.weight_decay)
        
        # Note: Sparse scale parameters are now fixed buffers, not learnable parameters
        if args.enable_2to4_sparse:
            logger.info("📊 Sparse scale parameters are fixed (not learnable) - computed once and then kept constant")

        optimizer = LOROAdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loro_type=args.loro_type,
            model=model,
        )

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    logger.info(f"\n{model}\n")
    logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M"
    )
    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    )
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    if not layer_wise_flag:

        if args.scheduler.lower() in ["cosine_restart", "cosine_restart_zero"]:
            cycle_length = args.cosine_restart_freq
            restart_warmup_steps = args.cosine_restart_warmup
            lr_adjust_steps = args.lr_adjust_steps
            lr_jag_after_warmup = args.lr_jag_after_warmup
            Warning(
                f"\nUsing jagged cosine lr schedule, "
                f"n_cycle = {cycle_length}, n_warmup = {restart_warmup_steps}, "
                f"lr_displacement = {lr_adjust_steps}, after_warmup = {lr_jag_after_warmup}.\n"
            )
        else:
            cycle_length = None
            restart_warmup_steps = None
            lr_adjust_steps = 0  # Must be 0, not None, to avoid scheduler error
            lr_jag_after_warmup = None
            Warning(f"\nUsing normal {args.scheduler} lr schedule ...\n")

        scheduler = training_utils.get_scheduler(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            # only apply to [cosine_restart, cosine_restart_zero]
            cycle_length=cycle_length,
            restart_warmup_steps=restart_warmup_steps,
            lr_adjust_steps=lr_adjust_steps,
            lr_jag_after_warmup=lr_jag_after_warmup,
        )

    # NOTE: when resuming training, the dataloader might repeat the same data
    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu"), strict=True
        )
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}"
            )
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(
                f"Will train for {args.num_training_steps - update_step} update steps"
            )
        else:
            logger.warning(
                f"Did not find training state in {args.continue_from}, global step will start from zero"
            )
        logger.info("*" * 40)

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    df_train_all = pd.DataFrame([])
    df_eval_all = pd.DataFrame([])

    # 启用flip rate跟踪（如果请求）
    if args.flip_rate:
        logger.info("🔧 启用flip rate跟踪...")
        # For pure Sparse2to4Linear (full-rank + 2:4 sparse mode)
        from loro_torch.sparse_overlay import enable_flip_rate_tracking_for_sparse_overlay
        enable_flip_rate_tracking_for_sparse_overlay(model, enabled=True)
        logger.info("✅ Flip rate tracking enabled for Sparse2to4Linear modules")
    else:
        logger.info("ℹ️ Flip rate tracking disabled")

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1
        
        # Set current training step for sparsity logging
        LlamaMLP._current_training_step = global_step
        # Also update the global step for activation sparse functions
        if args.wandb_sparsityrelu and ActivationSparse2to4Function is not None:
            ActivationSparse2to4Function._global_training_step = global_step
            # Also set for low-rank version
            if ActivationSparse2to4LowRankFunction is not None:
                ActivationSparse2to4LowRankFunction._global_training_step = global_step
            # Also set for single low-rank version (CoLA/LoST)
            from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunctionSingle
            ActivationSparse2to4LowRankFunctionSingle._global_training_step = global_step

        if update_step > args.num_training_steps:
            logger.info(
                f"Reached max number of update steps (f{args.num_training_steps}). Stopping training."
            )
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        # 先尝试正常forward
        loss = model(**batch, labels=labels).loss
        
        # 如果检测到NaN，使用增强的NaN追踪器进行详细分析
        if torch.isnan(loss):
            print("=" * 80)
            print(f"[CRITICAL] Loss is NaN detected at step {global_step}, update_step {update_step}")
            print("=" * 80)
            
            # 使用根本原因检测器
            from nan_root_cause_detector import NaNRootCauseDetector, analyze_split_gemm_root_cause
            from nan_detection_enhanced import NaNTracker, debug_split_gemm
            
            root_cause_detector = NaNRootCauseDetector(model)
            nan_tracker = NaNTracker(model, verbose=True)
            
            print("\n[Starting ROOT CAUSE analysis for NaN...]")
            
            # Keep old variables for compatibility
            hooks = []
            nan_found_at = []
            
            def check_tensor_for_nan(tensor, name=""):
                if tensor is None:
                    return False, "None"
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                
                # Skip integer tensors - they can't have NaN/Inf
                if not torch.is_floating_point(tensor) and not torch.is_complex(tensor):
                    return False, {
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape),
                        "is_integer": True
                    }
                
                has_nan = torch.isnan(tensor).any().item()
                has_inf = torch.isinf(tensor).any().item()
                
                # Only compute statistics if tensor is floating point
                if has_nan or has_inf:
                    stats = {
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                        "min": "NaN/Inf present",
                        "max": "NaN/Inf present",
                        "mean": "NaN/Inf present",
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype)
                    }
                else:
                    stats = {
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                        "min": tensor.min().item(),
                        "max": tensor.max().item(),
                        "mean": tensor.mean().item(),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype)
                    }
                return has_nan or has_inf, stats
            
            def forward_hook(module, input, output, module_name):
                # 检查输入
                for i, inp in enumerate(input):
                    if torch.is_tensor(inp):
                        has_issue, stats = check_tensor_for_nan(inp, f"input_{i}")
                        if has_issue:
                            nan_found_at.append({
                                "layer": module_name,
                                "type": "input",
                                "index": i,
                                "stats": stats
                            })
                            print(f"\n❌ NaN/Inf found in {module_name} input[{i}]:")
                            print(f"   Shape: {stats['shape']}")
                            print(f"   Has NaN: {stats['has_nan']}, Has Inf: {stats['has_inf']}")
                            if not stats['has_nan']:
                                print(f"   Min: {stats['min']:.6f}, Max: {stats['max']:.6f}, Mean: {stats['mean']:.6f}")
                
                # 检查输出
                has_issue, stats = check_tensor_for_nan(output, "output")
                if has_issue:
                    nan_found_at.append({
                        "layer": module_name,
                        "type": "output",
                        "stats": stats
                    })
                    print(f"\n❌ NaN/Inf found in {module_name} output:")
                    print(f"   Shape: {stats['shape']}")
                    print(f"   Has NaN: {stats['has_nan']}, Has Inf: {stats['has_inf']}")
                    if not stats['has_nan']:
                        print(f"   Min: {stats['min']:.6f}, Max: {stats['max']:.6f}, Mean: {stats['mean']:.6f}")
                    
                    # 检查各种矩阵的状态
                    # Check LowRankLinear A and B matrices
                    if hasattr(module, 'A') and hasattr(module, 'B'):
                        a_has_issue, a_stats = check_tensor_for_nan(module.A, "A")
                        b_has_issue, b_stats = check_tensor_for_nan(module.B, "B")
                        if a_has_issue:
                            print(f"   A matrix has NaN/Inf! Shape: {a_stats['shape']}")
                        if b_has_issue:
                            print(f"   B matrix has NaN/Inf! Shape: {b_stats['shape']}")
                    
                    # Check weight_in and weight_out for low-rank modules
                    if hasattr(module, 'weight_in') and hasattr(module, 'weight_out'):
                        win_has_issue, win_stats = check_tensor_for_nan(module.weight_in, "weight_in")
                        wout_has_issue, wout_stats = check_tensor_for_nan(module.weight_out, "weight_out")
                        if win_has_issue:
                            print(f"   weight_in has NaN/Inf! Stats: {win_stats}")
                        if wout_has_issue:
                            print(f"   weight_out has NaN/Inf! Stats: {wout_stats}")
                        
                        # Check gradients if available
                        if module.weight_in.grad is not None:
                            grad_has_issue, grad_stats = check_tensor_for_nan(module.weight_in.grad, "weight_in.grad")
                            if grad_has_issue:
                                print(f"   weight_in.grad has NaN/Inf! Stats: {grad_stats}")
                        if module.weight_out.grad is not None:
                            grad_has_issue, grad_stats = check_tensor_for_nan(module.weight_out.grad, "weight_out.grad")
                            if grad_has_issue:
                                print(f"   weight_out.grad has NaN/Inf! Stats: {grad_stats}")
                    
                    # 如果是activation 2:4相关层，输出更多信息
                    if "mlp" in module_name.lower() and args.activation_2by4:
                        print(f"   [Activation 2:4 active] dx_direct_sparse={args.dx_direct_sparse}")
                        if hasattr(module, 'scale_factor'):
                            print(f"   Scale factor: {module.scale_factor}")
            
            # 注册hooks到所有层
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # 只在叶子节点注册
                    hook = module.register_forward_hook(
                        lambda m, i, o, n=name: forward_hook(m, i, o, n)
                    )
                    hooks.append(hook)
            
            # First, use root cause detector to find exact operation
            print("\n[PHASE 1: Finding exact operation that creates NaN...]")
            first_nan_op = root_cause_detector.analyze_forward_pass(batch, labels)
            
            # Then use enhanced tracker for detailed layer analysis
            print("\n[PHASE 2: Layer-by-layer tracking...]")
            try:
                with torch.no_grad():
                    # Track the forward and backward pass with detailed analysis
                    loss_rerun, first_nan = nan_tracker.track_forward_backward(batch, labels)
                    
                    # Check model parameters
                    print("\n[Checking model parameters for NaN/Inf...]")
                    param_issues = nan_tracker.check_model_parameters()
                    if param_issues:
                        print(f"Found {len(param_issues)} parameters with NaN/Inf")
                        for name, info in list(param_issues.items())[:5]:
                            print(f"  {name}: NaN={info.has_nan}, Inf={info.has_inf}, Shape={info.shape}")
                    
                    # Special analysis for split_gemm if we're using activation 2:4
                    if args.activation_2by4 and args.dx_direct_sparse == 1:
                        print("\n[Analyzing Split-GEMM specific issues...]")
                        # Find problematic layers
                        for name, module in model.named_modules():
                            if hasattr(module, 'weight_in') and hasattr(module, 'weight_out'):
                                # Check if this is where NaN occurred
                                if first_nan and name in first_nan.get('module', ''):
                                    print(f"\n🔍 Detailed analysis of problematic layer: {name}")
                                    
                                    # Check weights
                                    win_nan = torch.isnan(module.weight_in).any().item()
                                    wout_nan = torch.isnan(module.weight_out).any().item()
                                    if win_nan:
                                        print(f"  ⚠️ weight_in contains NaN")
                                        print(f"     Shape: {module.weight_in.shape}")
                                        print(f"     NaN count: {torch.isnan(module.weight_in).sum().item()}")
                                    if wout_nan:
                                        print(f"  ⚠️ weight_out contains NaN")
                                        print(f"     Shape: {module.weight_out.shape}")
                                        print(f"     NaN count: {torch.isnan(module.weight_out).sum().item()}")
                                    
                                    # Check gradients if available
                                    if hasattr(module.weight_in, 'grad') and module.weight_in.grad is not None:
                                        win_grad_nan = torch.isnan(module.weight_in.grad).any().item()
                                        if win_grad_nan:
                                            print(f"  ⚠️ weight_in.grad contains NaN")
                                            print(f"     Shape: {module.weight_in.grad.shape}")
                                            print(f"     NaN count: {torch.isnan(module.weight_in.grad).sum().item()}")
                                            # Debug the split_gemm computation
                                            layer_id = f"layer_{id(module)}"
                                            debug_split_gemm(module.weight_in.grad, module.weight_out, layer_id)
            except Exception as e:
                print(f"\nException during re-run: {e}")
                import traceback
                traceback.print_exc()
            
            # Remove hooks if any were created
            for hook in hooks:
                hook.remove()
            
            # Summary from enhanced tracker
            if nan_found_at:
                print("\n" + "=" * 80)
                print("[NaN/Inf First Occurrence Summary from old detector]")
                first_nan_old = nan_found_at[0]
                print(f"First NaN/Inf detected at: {first_nan_old['layer']}")
                print(f"Type: {first_nan_old['type']}")
                print(f"Stats: {first_nan_old['stats']}")
                
                if len(nan_found_at) > 1:
                    print(f"\nTotal {len(nan_found_at)} NaN/Inf occurrences found")
                    print("Propagation path:")
                    for i, occurrence in enumerate(nan_found_at[:5]):  # 只显示前5个
                        print(f"  {i+1}. {occurrence['layer']} ({occurrence['type']})")
            
            # Enhanced tracker summary
            nan_tracker.summarize_nan_propagation()
            
            # 特别检查split_gemm相关
            if args.activation_2by4 and args.dx_direct_sparse == 1:
                print("\n[Split GEMM Specific Checks]")
                print("  Split GEMM is active (95% sparse + 5% dense)")
                print(f"  Activation sparse method: {args.activation_sparse_method}")
                print(f"  Dense warmup steps: {args.activation_dense_warmup_steps}")
                print(f"  Current step: {global_step}")
                
                # Check if we're past warmup
                if global_step >= args.activation_dense_warmup_steps:
                    print(f"  ✅ Activation 2:4 sparsity is ACTIVE (past warmup)")
                else:
                    print(f"  ⏳ Still in dense warmup phase")
                
                # Try to get debug info from the custom functions
                try:
                    from peft_pretraining.modeling_llama import (
                        ActivationSparse2to4Function, 
                        ActivationSparse2to4LowRankFunction,
                        ActivationSparse2to4LowRankFunctionSingle
                    )
                    
                    # Check all possible activation sparse functions
                    for func_class in [ActivationSparse2to4Function, 
                                       ActivationSparse2to4LowRankFunction,
                                       ActivationSparse2to4LowRankFunctionSingle]:
                        if func_class and hasattr(func_class, '_debug_info'):
                            debug_info = func_class._debug_info
                            if debug_info:
                                print(f"\n  Debug info from {func_class.__name__}:")
                                for key, val in debug_info.items():
                                    print(f"    {key}: {val}")
                except Exception as e:
                    print(f"  Could not retrieve debug info: {e}")
                
                # Check for specific modules with activation sparsity
                print("\n  Checking activation sparse modules:")
                for name, module in model.named_modules():
                    if hasattr(module, 'activation_sparse_method'):
                        print(f"    {name}: method={module.activation_sparse_method}")
                        if hasattr(module, 'scale_factor'):
                            scale_val = module.scale_factor
                            if torch.is_tensor(scale_val):
                                print(f"      scale_factor: {scale_val.item() if scale_val.numel() == 1 else scale_val.shape}")
                        if hasattr(module, 'sparsity_tracker'):
                            print(f"      sparsity: {module.sparsity_tracker}")
            
            print("\n" + "=" * 80)
            print("Stopping training due to NaN loss.")
            print("=" * 80)
            exit()


        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        grad_norm = sum(
            [
                torch.norm(p.grad.clone().detach().cpu())
                for p in model.parameters()
                if p.grad is not None
            ]
        ).item()  # Convert tensor to Python scalar for JSON serialization

        if global_rank == 0:
            pbar.update(1)

        if not layer_wise_flag:
            lr_tmp = optimizer.param_groups[0]["lr"]
            if lr_tmp == 0.0:  # avoid zero lr
                scheduler.step()

            # LORO optimizer update
            if args.optimizer.lower() == "loro_adamw":

                use_exact_loro = (update_step + 1) % args.loro_freq == 0
                optimizer.step(use_exact_loro=use_exact_loro)

                if (
                    args.loro_refresh is not None
                    and (update_step + 1) % args.loro_refresh_freq == 0
                ):
                    optimizer.refresh_states(args.loro_refresh)
                    print("LORO optim states reset successfully.")

            else:
                use_exact_loro = None
                optimizer.step()

            # Momentum reset functionality (works for all optimizer types)
            if args.momentum_reset_steps > 0 and update_step % args.momentum_reset_steps == 0:
                reset_count = reset_optimizer_momentum(optimizer)
                if reset_count > 0:
                    print(f"🔄 Step {update_step}: Reset momentum for {reset_count} parameters")

            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        
        # Momentum reset for layer-wise optimizers 
        if layer_wise_flag and args.momentum_reset_steps > 0 and update_step % args.momentum_reset_steps == 0:
            total_reset_count = 0
            if 'optimizer_dict' in locals():
                for param, opt in optimizer_dict.items():
                    reset_count = reset_optimizer_momentum(opt)
                    total_reset_count += reset_count
            if total_reset_count > 0:
                print(f"🔄 Step {update_step}: Reset momentum for {total_reset_count} parameters (layer-wise)")
        
        # Update CoLA/LoST activation sparse step counter for dense warmup
        if hasattr(args, 'more_activation_relu2') and args.more_activation_relu2:
            from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunctionSingle
            ActivationSparse2to4LowRankFunctionSingle._training_step = update_step
            ActivationSparse2to4LowRankFunctionSingle._global_update_step = update_step  # For sparsity recording control
            # Log warmup status for CoLA/LoST
            if update_step == args.activation_dense_warmup_steps:
                logger.info(f"🔧 CoLA/LoST dense warmup completed at step {update_step}. Activation 2:4 sparsity now enabled.")
            elif update_step < args.activation_dense_warmup_steps and update_step % 100 == 0:
                logger.info(f"🔧 CoLA/LoST dense warmup progress: {update_step}/{args.activation_dense_warmup_steps} steps")
        
        # Update activation sparse step counter for dense warmup
        if args.activation_2by4:
            from peft_pretraining.modeling_llama import ActivationSparse2to4Function, ActivationSparse2to4LowRankFunction
            if ActivationSparse2to4Function is not None and ActivationSparse2to4LowRankFunction is not None:
                ActivationSparse2to4Function.increment_step()
                ActivationSparse2to4LowRankFunction.increment_step()
            
            # Log warmup status
            current_step = ActivationSparse2to4Function.get_training_step()
            if current_step == args.activation_dense_warmup_steps:
                logger.info(f"🔧 Dense warmup completed at step {current_step}. Activation 2:4 sparsity now enabled.")
            elif current_step < args.activation_dense_warmup_steps and current_step % 100 == 0:
                logger.info(f"🔧 Dense warmup progress: {current_step}/{args.activation_dense_warmup_steps} steps")
        
        update_time = time.time() - update_time

        # verbose logging
        torch.cuda.synchronize()
        max_memory_GB = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_max_memory_allocated()
        # print(max_memory_GB)

        if update_step % 1000 == 0 or update_step < 10:
            print(
                f"Iter = {update_step}, global step = {global_step}, "
                f"Total loss = {loss.item()}, "
                f"lr = {lr_tmp}, Time = {update_time} sec, max_memory_GB = {max_memory_GB:.2f}"
            )


        # save checkpoint by save_every
        if update_step % args.save_every == 0:
            current_model_directory = f"{args.save_dir}/{runname}/model_{update_step}"
            if global_rank == 0 and not os.path.exists(current_model_directory):
                logger.info(
                    f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
                )
                os.makedirs(current_model_directory, exist_ok=True)
                
                # Fix generation_config pad_token_id before saving
                model_to_save = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                if hasattr(model_to_save, 'generation_config') and model_to_save.generation_config is not None:
                    if model_to_save.generation_config.pad_token_id == -1:
                        model_to_save.generation_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    model.module.save_pretrained(current_model_directory)
                else:
                    model.save_pretrained(current_model_directory)

                optimizer_checkpoint = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "wandb": wandb.run.dir if wandb is not None else None,
                    "dtype": args.dtype,
                }
                torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                    "gradnorm": grad_norm,
                    "max_memory_GB": max_memory_GB,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)

                print(f"\nModel saved at {current_model_directory} successfully.\n")

        # 计算tokens和batches统计信息
        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]

        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        # save wandb related info - 移出save_every条件，每步都记录
        if wandb is not None:
            wandb_dict = {
                "global_step": global_step,
                "update_step": update_step,
                "loss": loss.item(),
                "lr": lr_tmp,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                "gradnorm": grad_norm,
                "max_memory_GB": max_memory_GB,
            }
            
            # 计算并添加flip rate指标
            if args.flip_rate:
                if hasattr(args, 'enable_2to4_sparse') and args.enable_2to4_sparse:
                    # 根据优化器类型决定计算哪种flip rate
                    if args.optimizer.lower() == "loro_adamw":
                        # LORO + 2:4 sparse组合的flip rate
                        try:
                            from loro_torch.sparse_overlay import calculate_sparse_overlay_flip_rate
                            flip_rates = calculate_sparse_overlay_flip_rate(model)
                            wandb_dict.update(flip_rates)
                        except ImportError:
                            # 如果LORO sparse overlay函数不可用，返回默认值
                            wandb_dict.update({
                                "flip_rate/mean": 0.0,
                                "flip_rate/max": 0.0,
                                "flip_rate/min": 0.0,
                                "flip_rate/total": 0.0
                            })
                    else:
                        # 普通AdamW + Sparse2to4Linear模块的flip rate
                        flip_rates = calculate_model_flip_rate(model)
                        wandb_dict.update(flip_rates)
                else:
                    # 没有启用2:4稀疏训练，返回0
                    wandb_dict.update({
                        "flip_rate/mean": 0.0,
                        "flip_rate/max": 0.0,
                        "flip_rate/min": 0.0,
                        "flip_rate/total": 0.0
                    })

            # Add sparsity statistics if enabled (every 10 steps to reduce logging overhead)
            if args.wandb_sparsityrelu and update_step % 10 == 0:
                sparsity_stats = LlamaMLP.get_sparsity_stats()
                if sparsity_stats:
                    wandb_dict.update(sparsity_stats)
                
                # Add low-rank activation sparsity statistics (CoLA/LoST)
                if args.more_activation_relu2:
                    from peft_pretraining.modeling_llama import ActivationSparse2to4LowRankFunctionSingle
                    lowrank_sparsity_stats = ActivationSparse2to4LowRankFunctionSingle.get_lowrank_sparsity_stats()
                    if lowrank_sparsity_stats:
                        wandb_dict.update(lowrank_sparsity_stats)
            
            # Add weight sparsity statistics for all network layers
            weight_sparsity_stats = get_weight_sparsity_stats(model)
            if weight_sparsity_stats:
                wandb_dict.update(weight_sparsity_stats)
            
            wandb.log(wandb_dict, step=global_step)

            # track training stats - 确保所有值都是Python标量，避免BFloat16类型问题
            df_train_tmp = {}
            for k, v in wandb_dict.items():
                if hasattr(v, 'item'):  # 如果是torch tensor，转换为Python标量
                    df_train_tmp[k] = [float(v.item())]
                else:
                    df_train_tmp[k] = [v]
            df_train_tmp["use_exact_loro"] = [use_exact_loro]
            df_train_tmp["opt_step"] = [scheduler.last_epoch]
            df_train_tmp["update_time"] = [update_time]
            df_train_tmp = pd.DataFrame(df_train_tmp)
            df_train_all = pd.concat([df_train_all, df_train_tmp], ignore_index=True)
            df_train_all.to_csv(
                f"{args.save_dir}/{runname}/train_stats_{args.timestamp}.csv", index=False
            )

        update_time = time.time()

        # Check gradient health


        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens, eval_time, perplexity = evaluate_model(
                model,
                preprocess_batched,
                pad_idx,
                global_rank,
                world_size,
                device,
                args.batch_size,
                args.c4_local
            )
            if global_rank == 0 and wandb is not None:
                wandb.log(
                    {
                        "final_eval_loss": total_loss,
                        "final_eval_tokens": evaluated_on_tokens,
                        "eval_times": eval_time,
                        "perplexity_val_set": perplexity,
                    },
                    step=global_step,
                )

            # track evaluation stats - 确保所有值都是Python标量，避免BFloat16类型问题
            df_eval_tmp = {
                "global_step": [global_step],
                "update_step": [update_step],
                "eval_loss": [float(total_loss.item()) if hasattr(total_loss, 'item') else total_loss],
                "eval_tokens": [evaluated_on_tokens],
                "eval_time": [eval_time],
            }
            df_eval_tmp = pd.DataFrame(df_eval_tmp)
            df_eval_all = pd.concat([df_eval_all, df_eval_tmp], ignore_index=True)
            df_eval_all.to_csv(
                f"{args.save_dir}/{runname}/eval_stats_{args.timestamp}.csv", index=False
            )

            logger.info(f"Eval loss at step {update_step}: {total_loss}")

    if torch.isnan(loss):
        print("=" * 80)
        print(f"[CRITICAL] Loss is NaN detected at end of training loop")
        print(f"Final step: {global_step}, update_step: {update_step}")
        print("=" * 80)
        print("Stopping training due to NaN loss.")
        exit()  # 或者 break，看你是在函数里还是主循环里
    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0:
        pbar.close()

    current_model_directory = f"{args.save_dir}/{runname}/model_{update_step}"
    if (
        global_rank == 0
        and not os.path.exists(current_model_directory)
        and args.save_ckpt
    ):
        logger.info(
            f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
        )
        os.makedirs(current_model_directory, exist_ok=True)
        
        # Fix generation_config pad_token_id before saving
        model_to_save = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        if hasattr(model_to_save, 'generation_config') and model_to_save.generation_config is not None:
            if model_to_save.generation_config.pad_token_id == -1:
                model_to_save.generation_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.save_pretrained(current_model_directory)
        else:
            model.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir if wandb is not None else None,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

        print(f"\nFinal model saved at {current_model_directory} successfully.\n")

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens, eval_time, perplexity = evaluate_model(
        model,
        preprocess_batched,
        pad_idx,
        global_rank,
        world_size,
        device,
        args.batch_size,
        args.c4_local
    )

    if global_rank == 0 and wandb is not None:
        wandb.log(
            {
                "final_eval_loss": total_loss,
                "final_eval_tokens": evaluated_on_tokens,
                "eval_times": eval_time,
                "perplexity": perplexity,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    # track evaluation stats - 确保所有值都是Python标量，避免BFloat16类型问题
    df_eval_tmp = {
        "global_step": [global_step],
        "update_step": [update_step],
        "eval_loss": [float(total_loss.item()) if hasattr(total_loss, 'item') else total_loss],
        "eval_tokens": [evaluated_on_tokens],
        "eval_time": [eval_time],
    }
    df_eval_tmp = pd.DataFrame(df_eval_tmp)
    df_eval_all = pd.concat([df_eval_all, df_eval_tmp], ignore_index=True)
    df_eval_all.to_csv(f"{args.save_dir}/{runname}/eval_stats_{args.timestamp}.csv", index=False)

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")
    print("perplexity", perplexity)


if __name__ == "__main__":
    print("Starting script")
    args = parse_args()
    main(args)