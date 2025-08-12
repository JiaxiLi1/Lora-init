INFO:    Setting 'NVIDIA_VISIBLE_DEVICES=all' to emulate legacy GPU binding.
INFO:    Setting --writable-tmpfs (required by nvidia-container-cli)
WARNING: Not mounting current directory: user bind control is disabled by system administrator
2025-08-12 01:00:06.630 | INFO     | __main__:parse_args:744 - No save_dir specified, will use /mnt/fast/nobackup/users/xw0006/Lora-init/ckpt
2025-08-12 01:00:06.631 | INFO     | peft_pretraining.args_utils:check_args_torchrun_main:36 - save_dir not specified, using /mnt/fast/nobackup/users/xw0006/Lora-init/ckpt

2025-08-12 01:00:06.631 | INFO     | peft_pretraining.args_utils:check_args_torchrun_main:40 - Logging to /mnt/fast/nobackup/users/xw0006/Lora-init/ckpt/log_2025-08-12_01-00-06.txt

2025-08-12 01:00:06.634 | INFO     | __main__:main:1153 - Global rank 0, local rank 0, device: 0
2025-08-12 01:00:06.637 | INFO     | __main__:main:1165 - Process group initialized
2025-08-12 01:00:06.638 | INFO     | __main__:main:1197 - Using dist with rank 0 (only rank 0 will log)
2025-08-12 01:00:06.638 | INFO     | __main__:main:1198 - ****************************************
2025-08-12 01:00:06.638 | INFO     | __main__:main:1199 - Starting training with the arguments
2025-08-12 01:00:06.639 | INFO     | __main__:main:1201 - flip_rate                      True
2025-08-12 01:00:06.639 | INFO     | __main__:main:1201 - attn_2by4                      False
2025-08-12 01:00:06.640 | INFO     | __main__:main:1201 - mlp_2by4                       False
2025-08-12 01:00:06.640 | INFO     | __main__:main:1201 - enable_2to4_sparse             False
2025-08-12 01:00:06.640 | INFO     | __main__:main:1201 - sparse_init_scale              1.0
2025-08-12 01:00:06.641 | INFO     | __main__:main:1201 - activation_2by4                True
2025-08-12 01:00:06.641 | INFO     | __main__:main:1201 - activation_soft_threshold      False
2025-08-12 01:00:06.642 | INFO     | __main__:main:1201 - squ_relu                       relu2
2025-08-12 01:00:06.642 | INFO     | __main__:main:1201 - activation_sparse_method       naive
2025-08-12 01:00:06.642 | INFO     | __main__:main:1201 - permute_2by4                   False
2025-08-12 01:00:06.643 | INFO     | __main__:main:1201 - dynamic_activation_steps       200
2025-08-12 01:00:06.643 | INFO     | __main__:main:1201 - activation_calibration_samples 256
2025-08-12 01:00:06.643 | INFO     | __main__:main:1201 - activation_dense_warmup_steps  1000
2025-08-12 01:00:06.644 | INFO     | __main__:main:1201 - dx_direct_sparse               3
2025-08-12 01:00:06.644 | INFO     | __main__:main:1201 - wandb_sparsityrelu             True
2025-08-12 01:00:06.644 | INFO     | __main__:main:1201 - c4_local                       True
2025-08-12 01:00:06.645 | INFO     | __main__:main:1201 - train_data_path                /mnt/fast/nobackup/scratch4weeks/yl02855/hf_cache/datasets/allenai___c4/en/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/c4-train-*.arrow
2025-08-12 01:00:06.645 | INFO     | __main__:main:1201 - val_data_path                  /mnt/fast/nobackup/scratch4weeks/yl02855/hf_cache/datasets/allenai___c4/en/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/c4-validation-*.arrow
2025-08-12 01:00:06.646 | INFO     | __main__:main:1201 - model_config                   configs/llama_130m.json
2025-08-12 01:00:06.646 | INFO     | __main__:main:1201 - use_hf_model                   False
2025-08-12 01:00:06.646 | INFO     | __main__:main:1201 - continue_from                  None
2025-08-12 01:00:06.647 | INFO     | __main__:main:1201 - batch_size                     64
2025-08-12 01:00:06.647 | INFO     | __main__:main:1201 - gradient_accumulation          8
2025-08-12 01:00:06.647 | INFO     | __main__:main:1201 - total_batch_size               512
2025-08-12 01:00:06.648 | INFO     | __main__:main:1201 - max_length                     256
2025-08-12 01:00:06.648 | INFO     | __main__:main:1201 - optimizer                      cola_adamw
2025-08-12 01:00:06.649 | INFO     | __main__:main:1201 - lr                             0.001
2025-08-12 01:00:06.649 | INFO     | __main__:main:1201 - scheduler                      cosine_restart
2025-08-12 01:00:06.649 | INFO     | __main__:main:1201 - cosine_restart_freq            500
2025-08-12 01:00:06.650 | INFO     | __main__:main:1201 - cosine_restart_warmup          5
2025-08-12 01:00:06.652 | INFO     | __main__:main:1201 - lr_jag_after_warmup            False
2025-08-12 01:00:06.653 | INFO     | __main__:main:1201 - lr_adjust_steps                -2000
2025-08-12 01:00:06.653 | INFO     | __main__:main:1201 - min_lr_ratio                   0.1
2025-08-12 01:00:06.653 | INFO     | __main__:main:1201 - activation_checkpointing       False
2025-08-12 01:00:06.654 | INFO     | __main__:main:1201 - weight_decay                   0.0
2025-08-12 01:00:06.654 | INFO     | __main__:main:1201 - warmup_steps                   2000
2025-08-12 01:00:06.654 | INFO     | __main__:main:1201 - eval_every                     1000
2025-08-12 01:00:06.655 | INFO     | __main__:main:1201 - num_training_steps             20000
2025-08-12 01:00:06.655 | INFO     | __main__:main:1201 - max_train_tokens               None
2025-08-12 01:00:06.655 | INFO     | __main__:main:1201 - save_every                     5000
2025-08-12 01:00:06.656 | INFO     | __main__:main:1201 - save_ckpt                      True
2025-08-12 01:00:06.656 | INFO     | __main__:main:1201 - save_dir                       /mnt/fast/nobackup/users/xw0006/Lora-init/ckpt
2025-08-12 01:00:06.656 | INFO     | __main__:main:1201 - tags                           None
2025-08-12 01:00:06.657 | INFO     | __main__:main:1201 - dtype                          bfloat16
2025-08-12 01:00:06.657 | INFO     | __main__:main:1201 - workers                        8
2025-08-12 01:00:06.658 | INFO     | __main__:main:1201 - seed                           43
2025-08-12 01:00:06.658 | INFO     | __main__:main:1201 - name                           test
2025-08-12 01:00:06.658 | INFO     | __main__:main:1201 - grad_clipping                  0.5
2025-08-12 01:00:06.659 | INFO     | __main__:main:1201 - beta1                          0.0
2025-08-12 01:00:06.659 | INFO     | __main__:main:1201 - rank                           128
2025-08-12 01:00:06.659 | INFO     | __main__:main:1201 - update_proj_gap                50
2025-08-12 01:00:06.660 | INFO     | __main__:main:1201 - galore_scale                   1.0
2025-08-12 01:00:06.660 | INFO     | __main__:main:1201 - proj_type                      std
2025-08-12 01:00:06.661 | INFO     | __main__:main:1201 - cola_silu                      False
2025-08-12 01:00:06.661 | INFO     | __main__:main:1201 - cola_sparse_method             svd
2025-08-12 01:00:06.661 | INFO     | __main__:main:1201 - lost_sparsity                  0.05
2025-08-12 01:00:06.664 | INFO     | __main__:main:1201 - lost_sparse_method             random
2025-08-12 01:00:06.664 | INFO     | __main__:main:1201 - lost_sparse_svd_rank           256
2025-08-12 01:00:06.664 | INFO     | __main__:main:1201 - lost_gamma                     0.5
2025-08-12 01:00:06.665 | INFO     | __main__:main:1201 - lost_sparse_svd_inverse        False
2025-08-12 01:00:06.665 | INFO     | __main__:main:1201 - momentum_reset_steps           500
2025-08-12 01:00:06.666 | INFO     | __main__:main:1201 - more_activation_relu2          True
2025-08-12 01:00:06.666 | INFO     | __main__:main:1201 - loro_type                      loro
2025-08-12 01:00:06.666 | INFO     | __main__:main:1201 - loro_freq                      500
2025-08-12 01:00:06.670 | INFO     | __main__:main:1201 - loro_fallback_freq             None
2025-08-12 01:00:06.670 | INFO     | __main__:main:1201 - loro_refresh                   all
2025-08-12 01:00:06.671 | INFO     | __main__:main:1201 - loro_refresh_freq              500
2025-08-12 01:00:06.671 | INFO     | __main__:main:1201 - loro_attn_rank                 256
2025-08-12 01:00:06.671 | INFO     | __main__:main:1201 - loro_mlp_rank                  256
2025-08-12 01:00:06.672 | INFO     | __main__:main:1201 - loro_init                      xavier
2025-08-12 01:00:06.672 | INFO     | __main__:main:1201 - loro_scope                     all
2025-08-12 01:00:06.673 | INFO     | __main__:main:1201 - loro_lr_scaler                 -1
2025-08-12 01:00:06.673 | INFO     | __main__:main:1201 - loro_mlp_dense                 False
2025-08-12 01:00:06.673 | INFO     | __main__:main:1201 - single_gpu                     False
2025-08-12 01:00:06.674 | INFO     | __main__:main:1201 - desc                           cola_adamw_lr_0.001_gc0.5_cosine_restart_cyc500_wp5_adj-2000_2000_0.1
2025-08-12 01:00:06.674 | INFO     | __main__:main:1201 - timestamp                      2025-08-12_01-00-06
2025-08-12 01:00:06.674 | INFO     | __main__:main:1201 - log_path                       /mnt/fast/nobackup/users/xw0006/Lora-init/ckpt/log_2025-08-12_01-00-06.txt
2025-08-12 01:00:06.677 | INFO     | __main__:main:1201 - num_cuda                       1
2025-08-12 01:00:06.678 | INFO     | __main__:main:1202 - ****************************************
2025-08-12 01:00:07.514 | INFO     | __main__:main:1218 - Shuffling data with seed 42
/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/utils/data/dataloader.py:557: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
2025-08-12 01:00:08.036 | INFO     | __main__:main:1273 - 📊 Activation sparsity logging enabled for wandb
2025-08-12 01:00:08.037 | INFO     | __main__:main:1276 - 🔧 Using relu2 activation in MLP layers (no gate projection)
2025-08-12 01:00:08.037 | INFO     | __main__:main:1279 - 🔧 Activation 2:4 sparsity enabled with method: naive
2025-08-12 01:00:08.037 | INFO     | __main__:main:1280 - 🔧 Dense warmup for first 1000 steps, then activation 2:4 sparsity
2025-08-12 01:00:08.038 | INFO     | __main__:main:1282 - 🔧 dx_direct_sparse = 3 (dense chain rule)
/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/transformers/generation/configuration_utils.py:568: UserWarning: `pad_token_id` should be positive but got -1. This will cause errors when batch generating, if there is padding. Please set `pad_token_id` explicitly as `model.generation_config.pad_token_id=PAD_TOKEN_ID` to avoid errors in generation
  warnings.warn(
2025-08-12 01:00:09.675 | INFO     | __main__:main:1338 - 🔧 Initialized CoLA/LoST activation sparse: warmup_steps=1000
wandb: Currently logged in as: ljx923721867 (ljx923721867-university-of-surrey) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.20.0
wandb: Run data is saved locally in /mnt/fast/nobackup/users/xw0006/Lora-init/wandb/run-20250812_010010-ylqpc08r
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run 0812_010010gc0.5w0.0s20000m130m_opcola_adamwmlr0.1lr0.001bs64tb512_se5000_ee1000_24FalseaFalsemFalse_saTrue_acTrue_sfFalse_acrelu2_wbTrue_amnaive_s200_s256_w1000_dx3
wandb:  View project at https://wandb.ai/ljx923721867-university-of-surrey/2by4
wandb:  View run at https://wandb.ai/ljx923721867-university-of-surrey/2by4/runs/ylqpc08r
wandb: WARNING Saving files without folders. If you want to preserve subdirectories pass base_path to wandb.save, i.e. wandb.save("/mnt/folder/file.h5", base_path="/mnt")

Update steps:   0%|                                   | 0/20000 [00:00<?, ?it/s]Traceback (most recent call last):
  File "run_c4.py", line 2525, in <module>
    main(args)
  File "run_c4.py", line 1681, in main
    cola_module = CoLALowRankLinear(
  File "run_c4.py", line 366, in __init__
    self.weight_in.data.copy_((U_k * S_sqrt).to(self.weight_in.dtype))
RuntimeError: The size of tensor a (768) must match the size of tensor b (3072) at non-singleton dimension 0
[2025-08-12 01:00:25,401] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: 1) local_rank: 0 (pid: 2484092) of binary: /opt/conda/envs/loro_2by4/bin/python
Traceback (most recent call last):
  File "/opt/conda/envs/loro_2by4/bin/torchrun", line 33, in <module>
    sys.exit(load_entry_point('torch==2.1.0', 'console_scripts', 'torchrun')())
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 346, in wrapper
    return f(*args, **kwargs)
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/distributed/run.py", line 806, in main
    run(args)
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/distributed/run.py", line 797, in run
    elastic_launch(
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 264, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
run_c4.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-08-12_01:00:25
  host      : aisurrey14.surrey.ac.uk
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 2484092)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================