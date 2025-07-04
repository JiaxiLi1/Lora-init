import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
# import wandb

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

transformers.logging.set_verbosity_error()

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
    parser.add_argument("--save_ckpt", default=False, action="store_true")
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

    optimizer_desc = args.optimizer.lower()
    if args.optimizer.lower() == "loro_adamw":
        optimizer_desc += f"_{args.loro_type}"

        if args.loro_scope:
            optimizer_desc += f"_freq_{args.loro_freq}"
            optimizer_desc += f"_r_attn{args.loro_attn_rank}_mlp{args.loro_mlp_rank}_{args.loro_scope}"
            optimizer_desc += f"_init_lrk_{args.loro_init}"
            optimizer_desc += f"_rs_{args.loro_lr_scaler}"
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

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

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
    # if global_rank == 0:
    #     wandb.init(
    #         project="galore-c4",
    #         name=args.desc,
    #     )

    if global_rank == 0:
        # if wandb is not None:
        #     wandb.config.update(run_config, allow_val_change=True)
        #     wandb.save(os.path.abspath(__file__), policy="now")  # save current script
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
        optimizer = torch.optim.Adam(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )

    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )

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
    if args.optimizer.lower() == "loro_adamw":
        from loro_torch.lowrank_module import apply_lowrank_param, get_lowrank_param

        # apply lowrank parameterization
        if args.loro_scope is not None:
            if args.loro_mlp_dense:
                assert (
                    args.loro_scope == "attn" and args.loro_mlp_rank == mlp_rank
                ), "Only support dense MLP for attn"

            apply_lowrank_param(
                model,
                model_config,
                model_type="llama",
                scope=args.loro_scope,
                attn_rank=args.loro_attn_rank,
                mlp_rank=args.loro_mlp_rank,
                init=args.loro_init,
            )
        else:
            Warning(f"\nUsing full-rank model ...\n")

        param_groups = get_lowrank_param(model, model_config, args.loro_lr_scaler)

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

        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        ## NOTE: The below code is only executed during the update step

        if args.grad_clipping != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

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

        # if update_step % 1000 == 0 or update_step < 10:
        #     print(
        #         f"Iter = {update_step}, global step = {global_step}, "
        #         f"Total loss = {loss.item()}, "
        #         f"lr = {lr_tmp}, Time = {update_time} sec, max_memory_GB = {max_memory_GB:.2f}"
        #     )

        # save checkpoint by save_every
        if (
            local_step > args.gradient_accumulation
            and update_step % args.save_every == 0
            and global_rank == 0
        ):
            if args.save_ckpt:
                current_model_directory = f"{args.save_dir}/model_{update_step}"
                logger.info(
                    f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
                )
                os.makedirs(args.save_dir, exist_ok=True)
                os.makedirs(current_model_directory, exist_ok=True)
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model.module.save_pretrained(
                        current_model_directory, max_shard_size="100GB"
                    )
                else:
                    model.save_pretrained(
                        current_model_directory, max_shard_size="100GB"
                    )

                optimizer_checkpoint = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "dtype": args.dtype,
                }
                torch.save(
                    optimizer_checkpoint, f"{current_model_directory}/optimizer.pt"
                )

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)

                print(f"\nModel saved at {current_model_directory} successfully.\n")

            # save wandb related info
            # if wandb is not None:
            #     wandb_info = {
            #         "wandb_id": wandb.run.id,
            #     }
            #     with open(f"{args.save_dir}/wandb.json", "w") as f:
            #         json.dump(wandb_info, f, indent=4)

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
            if global_rank == 0 :
                print(f"global_step {global_step} update_step {update_step} eval_loss: {total_loss:.4f}, eval_tokens: {evaluated_on_tokens}, eval_times: {eval_time:.2f}s, perplexity_val_set: {perplexity:.4f}")


            # track evaluation stats
            df_eval_tmp = {
                "global_step": [global_step],
                "update_step": [update_step],
                "eval_loss": [total_loss],
                "eval_tokens": [evaluated_on_tokens],
                "eval_time": [eval_time],
            }
            df_eval_tmp = pd.DataFrame(df_eval_tmp)
            df_eval_all = pd.concat([df_eval_all, df_eval_tmp], ignore_index=True)
            df_eval_all.to_csv(
                f"{args.save_dir}/eval_stats_{args.timestamp}.csv", index=False
            )

            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]

        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        # log to wandb
        # if global_rank == 0:
        #     wandb_dict = {
        #         "global_step": global_step,
        #         "update_step": update_step,
        #         "loss": loss.item(),
        #         "lr": lr,
        #         "tokens_seen": tokens_seen,
        #         "throughput_tokens": tokens_in_update / update_time,
        #         "throughput_examples": args.total_batch_size / update_time,
        #         "throughput_batches": batches_in_update / update_time,
        #         "max_memory_GB": max_memory_GB,
        #     }
        #
        #     if wandb is not None:
        #         wandb.log(wandb_dict, step=global_step)
        #
        #     # track training stats
        #     df_train_tmp = {k: [v] for k, v in wandb_dict.items()}
        #     df_train_tmp["use_exact_loro"] = [use_exact_loro]
        #     df_train_tmp["opt_step"] = [scheduler.last_epoch]
        #     df_train_tmp["update_time"] = [update_time]
        #     df_train_tmp = pd.DataFrame(df_train_tmp)
        #     df_train_all = pd.concat([df_train_all, df_train_tmp], ignore_index=True)
        #     df_train_all.to_csv(
        #         f"{args.save_dir}/train_stats_{args.timestamp}.csv", index=False
        #     )

        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0:
        pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if (
        global_rank == 0
        and not os.path.exists(current_model_directory)
        and args.save_ckpt
    ):
        logger.info(
            f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
        )
        os.makedirs(current_model_directory, exist_ok=True)
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

    if global_rank == 0:
        print(
            f"[step {global_step}] final_eval_loss: {total_loss:.4f}, final_eval_tokens: {evaluated_on_tokens}, eval_times: {eval_time:.2f}s, final_perplexity_val_set: {perplexity:.4f}")

        # wandb.log(
        #     {
        #         "final_eval_loss": total_loss,
        #         "final_eval_tokens": evaluated_on_tokens,
        #         "eval_times": eval_time,
        #         "perplexity": perplexity,
        #     },
        #     step=global_step,
        # )
        logger.info(f"Final perplexity: {perplexity}")

    # track evaluation stats
    df_eval_tmp = {
        "global_step": [global_step],
        "update_step": [update_step],
        "eval_loss": [total_loss],
        "eval_tokens": [evaluated_on_tokens],
        "eval_time": [eval_time],
    }
    df_eval_tmp = pd.DataFrame(df_eval_tmp)
    df_eval_all = pd.concat([df_eval_all, df_eval_tmp], ignore_index=True)
    df_eval_all.to_csv(f"{args.save_dir}/eval_stats_{args.timestamp}.csv", index=False)

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args()
    main(args)
