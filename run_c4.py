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
from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

from loro_torch.loro_optim import LOROAdamW

from sparse_fullrank_linear import (
    apply_sparse2to4_to_model, 
    enable_flip_rate_tracking_for_model,
    calculate_model_flip_rate
)

transformers.logging.set_verbosity_error()

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
    parser.add_argument("--mlp_up_down", type=str_to_bool, default=False,
                        help="Apply 2:4 sparsity only to MLP up_proj and down_proj modules (excludes gate_proj)")
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

def check_gradient_health(model, step):
    """Enhanced gradient health check with early warning system"""
    if step % 500 != 0:  # 🔧 REDUCED: Every 500 steps instead of 100
        return None
        
    print(f"\n🩺 Gradient Health Check at Step {step}")
    print("=" * 60)
    
    total_params = 0
    zero_grad_params = 0
    small_grad_params = 0
    large_grad_params = 0
    nan_grad_params = 0
    inf_grad_params = 0
    gradient_norms = []
    suspicious_params = []
    param_details = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            grad_max = torch.max(torch.abs(param.grad)).item()
            grad_min = torch.min(torch.abs(param.grad)).item()
            
            gradient_norms.append(grad_norm)
            total_params += 1
            
            # Check for suspicious gradients
            has_nan = torch.isnan(param.grad).any()
            has_inf = torch.isinf(param.grad).any()
            
            if has_nan:
                nan_grad_params += 1
                suspicious_params.append(f"NaN: {name}")
                
            if has_inf:
                inf_grad_params += 1
                suspicious_params.append(f"Inf: {name}")
                
            if grad_norm == 0:
                zero_grad_params += 1
                suspicious_params.append(f"Zero: {name}")
            elif grad_norm < 1e-8:
                small_grad_params += 1
                suspicious_params.append(f"Tiny: {name} (norm={grad_norm:.2e})")
            elif grad_norm > 1e3:
                large_grad_params += 1
                suspicious_params.append(f"Large: {name} (norm={grad_norm:.2e})")
            
            # Store detailed info for key parameters
            if any(key in name for key in ['weight_in', 'weight_out', 'q_proj', 'k_proj', 'v_proj']):
                param_details.append({
                    'name': name,
                    'norm': grad_norm,
                    'max': grad_max,
                    'min': grad_min,
                    'shape': list(param.grad.shape),
                    'has_nan': has_nan,
                    'has_inf': has_inf
                })
    
    # Calculate statistics
    if gradient_norms:
        avg_norm = sum(gradient_norms) / len(gradient_norms)
        max_norm = max(gradient_norms)
        min_norm = min(gradient_norms)
        zero_ratio = zero_grad_params / total_params
        small_ratio = small_grad_params / total_params
        large_ratio = large_grad_params / total_params
    else:
        avg_norm = max_norm = min_norm = 0
        zero_ratio = small_ratio = large_ratio = 0
    
    # Print summary
    print(f"📊 Gradient Statistics:")
    print(f"   Total parameters with gradients: {total_params}")
    print(f"   Average gradient norm: {avg_norm:.2e}")
    print(f"   Max gradient norm: {max_norm:.2e}")
    print(f"   Min gradient norm: {min_norm:.2e}")
    print(f"   Zero gradients: {zero_grad_params} ({zero_ratio*100:.1f}%)")
    print(f"   Small gradients (<1e-8): {small_grad_params} ({small_ratio*100:.1f}%)")
    print(f"   Large gradients (>1e3): {large_grad_params} ({large_ratio*100:.1f}%)")
    print(f"   NaN gradients: {nan_grad_params}")
    print(f"   Inf gradients: {inf_grad_params}")
    
    # Print detailed info for key parameters
    if param_details:
        print(f"\n🔍 Key Parameter Details:")
        for detail in param_details[:10]:  # Show top 10
            print(f"   {detail['name'][:50]:50s} | norm: {detail['norm']:.2e} | max: {detail['max']:.2e} | shape: {detail['shape']}")
            if detail['has_nan'] or detail['has_inf']:
                print(f"      ⚠️  Contains NaN: {detail['has_nan']}, Inf: {detail['has_inf']}")
    
    # Print suspicious parameters
    if suspicious_params:
        print(f"\n⚠️  Suspicious Parameters ({len(suspicious_params)}):")
        for param_info in suspicious_params[:15]:  # Show top 15
            print(f"   {param_info}")
        if len(suspicious_params) > 15:
            print(f"   ... and {len(suspicious_params) - 15} more")
    
    # Health status
    health_status = {
        'total_params': total_params,
        'avg_norm': avg_norm,
        'max_norm': max_norm,
        'zero_ratio': zero_ratio,
        'suspicious': nan_grad_params + inf_grad_params,
        'large_gradients': large_grad_params
    }
    
    # Determine overall health
    if nan_grad_params > 0 or inf_grad_params > 0:
        health_color = "🔴"
        health_desc = "CRITICAL"
    elif zero_ratio > 0.5 or large_grad_params > 5:
        health_color = "🟡"
        health_desc = "WARNING"
    else:
        health_color = "🟢"
        health_desc = "HEALTHY"
    
    print(f"\n{health_color} Overall Gradient Health: {health_desc}")
    print("=" * 60)
    
    return health_status

def check_model_weights_health(model, step):
    """Check model weights for NaN/Inf issues"""
    if step % 200 != 0:
        return
        
    print(f"\n🔧 Model Weights Health Check at Step {step}")
    
    nan_weights = 0
    inf_weights = 0
    zero_weights = 0
    total_weight_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_weight_params += 1
            
            if torch.isnan(param.data).any():
                nan_weights += 1
                print(f"🚨 NaN in weight: {name}")
                
            if torch.isinf(param.data).any():
                inf_weights += 1
                print(f"🚨 Inf in weight: {name}")
                
            weight_norm = torch.norm(param.data).item()
            if weight_norm == 0:
                zero_weights += 1
                print(f"⚠️  Zero weight: {name}")
    
    if nan_weights == 0 and inf_weights == 0 and zero_weights == 0:
        print("✅ All weights are healthy")
    else:
        print(f"⚠️  Weight issues: {nan_weights} NaN, {inf_weights} Inf, {zero_weights} Zero out of {total_weight_params} total")

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

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

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
            ),  # rename lr to max_lr to avoid conflicts with scheduler
            "total_params_M": n_total_params / 1_000_000,
            "dataset": "c4",
            "model": model_config.to_dict(),
            "world_size": world_size,
            "device": str(device),
        }
    )
    if global_rank == 0:
        model_size = extract_size_and_type(args.model_config)
        runname = f"{time.strftime('%m%d_%H%M%S')}_gc{args.grad_clipping}_step{args.num_training_steps}_" \
                  f"model{model_size}_ar{args.loro_attn_rank}_loty{args.loro_type}_fr{args.loro_freq}_ls_{args.loro_lr_scaler}_sc{args.scheduler}_crfr{args.cosine_restart_freq}_as{args.lr_adjust_steps}_ra{args.loro_refresh}_rf{args.loro_refresh_freq}_sc_{args.loro_scope}_ini_{args.loro_init}_op_{args.optimizer}_mlr{args.min_lr_ratio}_lr{args.lr}_bs{args.batch_size}_" \
                  f"tbs{args.total_batch_size}_severy_{args.save_every}_eevery_{args.eval_every}_2by4{args.enable_2to4_sparse}_a2by4{args.attn_2by4}_m2by4{args.mlp_2by4}_mlpupdown{args.mlp_up_down}_" \
                  f"save{args.save_ckpt}"
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
                if args.mlp_up_down:
                    # 只对up_proj和down_proj应用2:4稀疏，排除gate_proj
                    mlp_modules = ["up_proj", "down_proj"]
                    logger.info("📌 将对MLP up/down模块应用2:4稀疏: " + str(mlp_modules) + " (排除gate_proj)")
                else:
                    # 对所有MLP模块应用2:4稀疏
                    mlp_modules = ["gate_proj", "up_proj", "down_proj"]
                    logger.info("📌 将对MLP模块应用2:4稀疏: " + str(mlp_modules))
                target_modules.extend(mlp_modules)
            
            if not target_modules:
                logger.warning("⚠️ 启用了2:4稀疏但没有选择任何目标模块！请检查 --attn_2by4 和 --mlp_2by4 参数")
                logger.info("🔄 回退到普通full-rank AdamW训练")
                
                # 🔧 Use proper parameter grouping for weight decay (same as standard case)
                param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
                decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
                nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
                
                optim_groups = [
                    {'params': decay_params, 'weight_decay': args.weight_decay},
                    {'params': nodecay_params, 'weight_decay': 0.0}
                ]
                
                optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)
            else:
                logger.info(f"🎯 最终目标模块列表: {target_modules}")
                
                # Apply 2:4 sparsity to full-rank linear layers
                from sparse_fullrank_linear import apply_sparse2to4_to_model
                model = apply_sparse2to4_to_model(
                    model,
                    target_modules=target_modules,
                )
                logger.info("✅ Full-rank linear layers replaced with Sparse2to4Linear!")
                logger.info("🔬 使用与LORO+2:4完全相同的实现: SparseOverlayFunction、MVUE、scaling等")
                
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
                
                # ‼️ CRITICAL FIX: Use bnb.optim.AdamW for correct weight decay with sparse autograd.Function
                logger.info("‼️ 使用 bnb.optim.AdamW 来确保 weight_decay 在2:4稀疏训练中正确生效 (L2 Regularization)")
                optimizer = bnb.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))

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
                {'params': nodecay_params, 'weight_decay': 0.0}
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
                if args.mlp_up_down:
                    # 只对up_proj和down_proj应用2:4稀疏，排除gate_proj
                    mlp_modules = ["up_proj", "down_proj"]
                    logger.info("📌 将对MLP up/down模块应用2:4稀疏: " + str(mlp_modules) + " (排除gate_proj)")
                else:
                    # 对所有MLP模块应用2:4稀疏
                    mlp_modules = ["gate_proj", "up_proj", "down_proj"]
                    logger.info("📌 将对MLP模块应用2:4稀疏: " + str(mlp_modules))
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
            lr_adjust_steps = None
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
        if hasattr(args, 'enable_2to4_sparse') and args.enable_2to4_sparse:
            logger.info("🔧 启用flip rate跟踪...")
            # For pure Sparse2to4Linear (full-rank + 2:4 sparse mode)
            enable_flip_rate_tracking_for_model(model, enabled=True)
            logger.info("✅ Flip rate tracking enabled for Sparse2to4Linear modules")
            
            # Also enable for LORO + 2:4 sparse combination if applicable
            try:
                from loro_torch.sparse_overlay import enable_flip_rate_tracking_for_sparse_overlay
                enable_flip_rate_tracking_for_sparse_overlay(model, enabled=True)
                logger.info("✅ Flip rate tracking also enabled for LORO SparseOverlay modules")
            except ImportError:
                pass  # LORO sparse overlay functions may not be available
        else:
            logger.warning("⚠️ Flip rate requested but no 2:4 sparse training enabled.")
            logger.info("ℹ️ Flip rate只适用于2:4稀疏训练。当前模式下flip rate将始终为0。")
            logger.info("ℹ️ 要启用flip rate跟踪，请设置 --enable_2to4_sparse True")
    else:
        logger.info("ℹ️ Flip rate tracking disabled")

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1

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

        loss = model(**batch, labels=labels).loss
        
        # 🔍 Enhanced NaN Detection and Early Stopping
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ CRITICAL: NaN/Inf loss detected at step {global_step}! Loss: {loss}")
            print(f"🛑 This indicates numerical instability. Stopping training to prevent corruption.")
            
            # Detailed diagnosis
            nan_params = []
            large_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params.append(name)
                elif param.abs().max() > 1e6:
                    large_params.append((name, param.abs().max().item()))
            
            if nan_params:
                print(f"❌ Parameters with NaN: {nan_params[:10]}")
            if large_params:
                print(f"⚠️  Parameters with large values: {large_params[:5]}")
            
            # Save debug checkpoint
            if global_rank == 0:
                debug_path = os.path.join(args.save_dir, f"debug_nan_step_{global_step}")
                os.makedirs(debug_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(debug_path, "model_state_with_nan.bin"))
                print(f"💾 Debug checkpoint saved to {debug_path}")
            
            # Exit immediately to prevent further corruption
            exit(1)

        # 🔍 Debug: Check loss for NaN/Inf before backward
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"🚨 CRITICAL: NaN/Inf detected in LOSS at step {global_step}!")
            print(f"   Loss value: {loss.item()}")
            print(f"   Loss has NaN: {torch.isnan(loss).any()}")
            print(f"   Loss has Inf: {torch.isinf(loss).any()}")
            
            # Save emergency checkpoint before crash
            if global_rank == 0:
                emergency_path = os.path.join(args.save_dir, f"emergency_nan_loss_step_{global_step}")
                os.makedirs(emergency_path, exist_ok=True)
                torch.save({
                    'model': model.state_dict() if not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.module.state_dict(),
                    'step': global_step,
                    'loss': loss.item()
                }, os.path.join(emergency_path, "emergency_state.pt"))
                print(f"💾 Emergency checkpoint saved to: {emergency_path}")
            break
        
        # 🔍 Debug: Track loss trends and problems
        if global_step % 200 == 0:  # 🔧 REDUCED: Every 200 steps instead of 50
            current_loss = loss.item()
            print(f"📈 Step {global_step}: Loss = {current_loss:.4f}")
            
            # Check for problematic loss trends
            if hasattr(main, 'prev_loss'):
                loss_change = current_loss - main.prev_loss
                if abs(loss_change) > 1.0:  # 🚨 Large loss jump
                    print(f"🚨 Large loss change: {loss_change:+.4f}")
                elif current_loss > 20:  # 🚨 Very high loss
                    print(f"🚨 High loss detected: {current_loss:.4f}")
                elif current_loss < 0.1:  # 🚨 Loss too low too fast
                    print(f"🚨 Loss dropping very fast: {current_loss:.4f}")
            main.prev_loss = current_loss
            
            # 🔍 Track training health status
            if not hasattr(main, 'mvue_skip_count'):
                main.mvue_skip_count = 0
                main.grad_vanish_count = 0
                main.mvue_inf_count = 0
                main.problem_start_step = None
            
            # 🔍 Report accumulated issues every 1000 steps
            if global_step % 1000 == 0 and global_step > 0:
                print(f"📊 Training Health Summary (Steps {global_step-1000} to {global_step}):")
                print(f"   MVUE inf amplifications: {main.mvue_inf_count}")
                print(f"   Gradient vanishing cases: {main.grad_vanish_count}")
                if main.problem_start_step:
                    print(f"   First major issue detected at step: {main.problem_start_step}")
                
                # Reset counters
                main.mvue_inf_count = 0
                main.grad_vanish_count = 0

        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        ## NOTE: The below code is only executed during the update step

        # Check gradient health after backward pass
        health_status = check_gradient_health(model, global_step)
        
        # Check model weights health
        check_model_weights_health(model, global_step)
        
        # 🔧 NEW: Early stopping based on gradient health
        if health_status and (health_status['suspicious'] > 0 or health_status['zero_ratio'] > 0.8):
            print(f"🛑 EARLY STOPPING: Gradient health deteriorated beyond recovery threshold!")
            print(f"   Suspicious params: {health_status['suspicious']}")
            print(f"   Zero gradient ratio: {health_status['zero_ratio']*100:.1f}%")
            print(f"   Average gradient norm: {health_status['avg_norm']:.2e}")
            
            # Save emergency checkpoint
            if global_rank == 0:
                emergency_path = os.path.join(args.save_dir, f"emergency_stop_step_{global_step}")
                os.makedirs(emergency_path, exist_ok=True)
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model.module.save_pretrained(emergency_path, max_shard_size="100GB")
                else:
                    model.save_pretrained(emergency_path, max_shard_size="100GB")
                print(f"💾 Emergency checkpoint saved to {emergency_path}")
            
            # Exit gracefully
            break

        # 🔍 Debug: Gradient clipping analysis
        if args.grad_clipping != 0.0:
            total_norm_before = torch.nn.utils.clip_grad_norm_(trainable_params, float('inf'))
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
            
            if global_step % 100 == 0:
                print(f"🔧 Gradient Clipping Analysis (Step {global_step}):")
                print(f"   Total gradient norm before clipping: {total_norm_before:.6f}")
                print(f"   Clipping threshold: {args.grad_clipping}")
                if total_norm_before > args.grad_clipping:
                    print(f"   ✂️  Gradients were clipped (ratio: {args.grad_clipping/total_norm_before:.4f})")
                else:
                    print(f"   ✅ No clipping needed")
                    
                # Track extreme gradient norms
                if total_norm_before > 100:
                    print(f"⚠️  Very large gradient norm detected: {total_norm_before:.2f}")
                elif total_norm_before < 1e-6:
                    print(f"⚠️  Very small gradient norm detected: {total_norm_before:.2e}")

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

            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
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
            
        # 🔍 Debug: Periodic health check every 100 steps
        if update_step % 100 == 0:
            param_health = {"healthy": 0, "nan": 0, "inf": 0, "high_norm": 0}
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    param_health["nan"] += 1
                elif torch.isinf(param).any():
                    param_health["inf"] += 1
                elif torch.norm(param) > 1000:
                    param_health["high_norm"] += 1
                else:
                    param_health["healthy"] += 1
                    
            if param_health["nan"] > 0 or param_health["inf"] > 0:
                print(f"🚨 HEALTH CHECK @ step {update_step}: {param_health}")
            elif update_step % 500 == 0:  # Only print healthy status every 500 steps
                print(f"✅ HEALTH CHECK @ step {update_step}: {param_health}")

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
        health_status = check_gradient_health(model, global_step)

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