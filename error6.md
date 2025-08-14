/home/rtx3090/miniconda3/envs/loro_2by4/bin/python /home/rtx3090/miniconda3/envs/loro_2by4/bin/torchrun --nproc_per_node 1 run_c4.py --model_config configs/llama_130m.json --dtype bfloat16 --batch_size 64 --total_batch_size 512 --num_training_steps 20000 --save_every 1000 --eval_every 1000 --lr 0.01 --scheduler cosine_restart --warmup_steps 2000 --min_lr_ratio 0.1 --cosine_restart_freq 500 --lr_adjust_steps -2000 --weight_decay 0.1 --optimizer loro_adamw --loro_refresh all --loro_refresh_freq 500 --loro_scope all --loro_init xavier --loro_attn_rank 256 --loro_mlp_rank 256 --loro_type loro --loro_freq 500 --loro_lr_scaler -1 --c4_local False --enable_2to4_sparse False --save_ckpt True --attn_2by4 True --mlp_2by4 True --seed 43 --flip_rate True --activation_2by4 True --activation_soft_threshold True --squ_relu relu2 --activation_sparse_method soft_dynamic --dynamic_activation_steps 10 --activation_calibration_samples 50 --activation_dense_warmup_steps 0 --dx_direct_sparse 1 --wandb_sparsityrelu True --permute_2by4 True --momentum_reset_steps 200 --more_activation_relu2 False --cola_sparse_method cola_init 
Starting script


Experiment = loro_adamw_loro_freq_500_r_attn256_mlp256_all_init_lrk_xavier_rs_-1_rfsh_all_500_lr_0.01_gc0.0_cosine_restart_cyc500_wp5_adj-2000_2000_0.1


2025-08-14 22:40:47.576 | INFO     | __main__:main:1165 - Global rank 0, local rank 0, device: 0
2025-08-14 22:40:47.577 | INFO     | __main__:main:1177 - Process group initialized
Rank 0 using device cuda:0


# CUDA visible devices: 1


2025-08-14 22:40:47.577 | INFO     | __main__:main:1209 - Using dist with rank 0 (only rank 0 will log)
2025-08-14 22:40:47.577 | INFO     | __main__:main:1210 - ****************************************
2025-08-14 22:40:47.577 | INFO     | __main__:main:1211 - Starting training with the arguments
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - flip_rate                      True
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - attn_2by4                      True
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - mlp_2by4                       True
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - enable_2to4_sparse             False
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - sparse_init_scale              1.0
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - activation_2by4                True
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - activation_soft_threshold      True
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - squ_relu                       relu2
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - activation_sparse_method       soft_dynamic
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - permute_2by4                   True
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - dynamic_activation_steps       10
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - activation_calibration_samples 50
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - activation_dense_warmup_steps  0
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - dx_direct_sparse               1
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - wandb_sparsityrelu             True
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - c4_local                       False
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - train_data_path                en/c4-train.*.json.gz
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - val_data_path                  en/c4-validation.*.json.gz
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - model_config                   configs/llama_130m.json
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - use_hf_model                   False
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - continue_from                  None
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - batch_size                     64
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - gradient_accumulation          8
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - total_batch_size               512
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - max_length                     256
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - optimizer                      loro_adamw
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - lr                             0.01
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - scheduler                      cosine_restart
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - cosine_restart_freq            500
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - cosine_restart_warmup          5
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - lr_jag_after_warmup            False
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - lr_adjust_steps                -2000
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - min_lr_ratio                   0.1
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - activation_checkpointing       False
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - weight_decay                   0.1
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - warmup_steps                   2000
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - eval_every                     1000
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - num_training_steps             20000
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - max_train_tokens               None
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - save_every                     1000
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - save_ckpt                      True
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - save_dir                       /home/rtx3090/code_jiaxi/LORO-main_temp/ckpt
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - tags                           None
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - dtype                          bfloat16
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - workers                        8
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - seed                           43
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - name                           test
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - grad_clipping                  0.0
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - beta1                          0.0
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - rank                           128
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - update_proj_gap                50
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - galore_scale                   1.0
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - proj_type                      std
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - cola_silu                      False
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - cola_sparse_method             cola_init
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - lost_sparsity                  0.05
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - lost_sparse_method             random
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - lost_sparse_svd_rank           256
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - lost_gamma                     0.5
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - lost_sparse_svd_inverse        False
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - momentum_reset_steps           200
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - more_activation_relu2          False
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_type                      loro
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_freq                      500
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_fallback_freq             None
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_refresh                   all
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_refresh_freq              500
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_attn_rank                 256
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_mlp_rank                  256
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_init                      xavier
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - loro_scope                     all
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - loro_lr_scaler                 -1
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - loro_mlp_dense                 False
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - single_gpu                     False
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - desc                           loro_adamw_loro_freq_500_r_attn256_mlp256_all_init_lrk_xavier_rs_-1_rfsh_all_500_lr_0.01_gc0.0_cosine_restart_cyc500_wp5_adj-2000_2000_0.1
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - timestamp                      2025-08-14_22-40-47
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - log_path                       /home/rtx3090/code_jiaxi/LORO-main_temp/ckpt/log_2025-08-14_22-40-47.txt
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - num_cuda                       1
2025-08-14 22:40:47.582 | INFO     | __main__:main:1214 - ****************************************
2025-08-14 22:40:47.574 | INFO     | __main__:parse_args:756 - No save_dir specified, will use /home/rtx3090/code_jiaxi/LORO-main_temp/ckpt
2025-08-14 22:40:47.575 | INFO     | peft_pretraining.args_utils:check_args_torchrun_main:36 - save_dir not specified, using /home/rtx3090/code_jiaxi/LORO-main_temp/ckpt

2025-08-14 22:40:47.575 | INFO     | peft_pretraining.args_utils:check_args_torchrun_main:40 - Logging to /home/rtx3090/code_jiaxi/LORO-main_temp/ckpt/log_2025-08-14_22-40-47.txt

2025-08-14 22:40:47.576 | INFO     | __main__:main:1165 - Global rank 0, local rank 0, device: 0
2025-08-14 22:40:47.577 | INFO     | __main__:main:1177 - Process group initialized
2025-08-14 22:40:47.577 | INFO     | __main__:main:1209 - Using dist with rank 0 (only rank 0 will log)
2025-08-14 22:40:47.577 | INFO     | __main__:main:1210 - ****************************************
2025-08-14 22:40:47.577 | INFO     | __main__:main:1211 - Starting training with the arguments
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - flip_rate                      True
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - attn_2by4                      True
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - mlp_2by4                       True
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - enable_2to4_sparse             False
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - sparse_init_scale              1.0
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - activation_2by4                True
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - activation_soft_threshold      True
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - squ_relu                       relu2
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - activation_sparse_method       soft_dynamic
2025-08-14 22:40:47.577 | INFO     | __main__:main:1213 - permute_2by4                   True
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - dynamic_activation_steps       10
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - activation_calibration_samples 50
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - activation_dense_warmup_steps  0
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - dx_direct_sparse               1
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - wandb_sparsityrelu             True
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - c4_local                       False
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - train_data_path                en/c4-train.*.json.gz
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - val_data_path                  en/c4-validation.*.json.gz
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - model_config                   configs/llama_130m.json
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - use_hf_model                   False
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - continue_from                  None
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - batch_size                     64
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - gradient_accumulation          8
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - total_batch_size               512
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - max_length                     256
2025-08-14 22:40:47.578 | INFO     | __main__:main:1213 - optimizer                      loro_adamw
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - lr                             0.01
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - scheduler                      cosine_restart
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - cosine_restart_freq            500
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - cosine_restart_warmup          5
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - lr_jag_after_warmup            False
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - lr_adjust_steps                -2000
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - min_lr_ratio                   0.1
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - activation_checkpointing       False
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - weight_decay                   0.1
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - warmup_steps                   2000
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - eval_every                     1000
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - num_training_steps             20000
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - max_train_tokens               None
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - save_every                     1000
2025-08-14 22:40:47.579 | INFO     | __main__:main:1213 - save_ckpt                      True
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - save_dir                       /home/rtx3090/code_jiaxi/LORO-main_temp/ckpt
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - tags                           None
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - dtype                          bfloat16
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - workers                        8
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - seed                           43
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - name                           test
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - grad_clipping                  0.0
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - beta1                          0.0
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - rank                           128
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - update_proj_gap                50
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - galore_scale                   1.0
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - proj_type                      std
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - cola_silu                      False
2025-08-14 22:40:47.580 | INFO     | __main__:main:1213 - cola_sparse_method             cola_init
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - lost_sparsity                  0.05
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - lost_sparse_method             random
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - lost_sparse_svd_rank           256
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - lost_gamma                     0.5
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - lost_sparse_svd_inverse        False
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - momentum_reset_steps           200
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - more_activation_relu2          False
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_type                      loro
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_freq                      500
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_fallback_freq             None
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_refresh                   all
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_refresh_freq              500
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_attn_rank                 256
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_mlp_rank                  256
2025-08-14 22:40:47.581 | INFO     | __main__:main:1213 - loro_init                      xavier
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - loro_scope                     all
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - loro_lr_scaler                 -1
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - loro_mlp_dense                 False
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - single_gpu                     False
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - desc                           loro_adamw_loro_freq_500_r_attn256_mlp256_all_init_lrk_xavier_rs_-1_rfsh_all_500_lr_0.01_gc0.0_cosine_restart_cyc500_wp5_adj-2000_2000_0.1
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - timestamp                      2025-08-14_22-40-47
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - log_path                       /home/rtx3090/code_jiaxi/LORO-main_temp/ckpt/log_2025-08-14_22-40-47.txt
2025-08-14 22:40:47.582 | INFO     | __main__:main:1213 - num_cuda                       1
2025-08-14 22:40:47.582 | INFO     | __main__:main:1214 - ****************************************
2025-08-14 22:41:01.360 | INFO     | __main__:main:1230 - Shuffling data with seed 42
2025-08-14 22:41:01.360 | INFO     | __main__:main:1230 - Shuffling data with seed 42
2025-08-14 22:41:01.840 | INFO     | __main__:main:1285 - 📊 Activation sparsity logging enabled for wandb
2025-08-14 22:41:01.841 | INFO     | __main__:main:1288 - 🔧 Using relu2 activation in MLP layers (no gate projection)
2025-08-14 22:41:01.841 | INFO     | __main__:main:1291 - 🔧 Activation 2:4 sparsity enabled with method: soft_dynamic
2025-08-14 22:41:01.841 | INFO     | __main__:main:1292 - 🔧 Dense warmup for first 0 steps, then activation 2:4 sparsity
2025-08-14 22:41:01.841 | INFO     | __main__:main:1294 - 🔧 dx_direct_sparse = 1 (full split_gemm)
/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/transformers/generation/configuration_utils.py:568: UserWarning: `pad_token_id` should be positive but got -1. This will cause errors when batch generating, if there is padding. Please set `pad_token_id` explicitly as `model.generation_config.pad_token_id=PAD_TOKEN_ID` to avoid errors in generation
  warnings.warn(
2025-08-14 22:41:01.840 | INFO     | __main__:main:1285 - 📊 Activation sparsity logging enabled for wandb
2025-08-14 22:41:01.841 | INFO     | __main__:main:1288 - 🔧 Using relu2 activation in MLP layers (no gate projection)
2025-08-14 22:41:01.841 | INFO     | __main__:main:1291 - 🔧 Activation 2:4 sparsity enabled with method: soft_dynamic
2025-08-14 22:41:01.841 | INFO     | __main__:main:1292 - 🔧 Dense warmup for first 0 steps, then activation 2:4 sparsity
2025-08-14 22:41:01.841 | INFO     | __main__:main:1294 - 🔧 dx_direct_sparse = 1 (full split_gemm)
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Using squared ReLU MLP architecture: hidden_size=768, new_intermediate_size=3072
🔧 Parameter count maintained: original=4718592, new=4718592
🔧 Enabling sparsity recording for wandb_sparsityrelu
runname= 0814_224103gc0.0w0.1s20000m130m_oploro_adamwmlr0.1lr0.01bs64tb512_se1000_ee1000_24FalseaTruemTrue_saTrue_acTrue_sfTrue_acrelu2_wbTrue_amsoft_dynamic_s10_s50_w0_dx1
wandb: Currently logged in as: ljx923721867 (ljx923721867-university-of-surrey) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.20.0
wandb: Run data is saved locally in /home/rtx3090/code_jiaxi/LORO-main_temp/wandb/run-20250814_224103-dbtxa5kc
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run 0814_224103gc0.0w0.1s20000m130m_oploro_adamwmlr0.1lr0.01bs64tb512_se1000_ee1000_24FalseaTruemTrue_saTrue_acTrue_sfTrue_acrelu2_wbTrue_amsoft_dynamic_s10_s50_w0_dx1
wandb:  View project at https://wandb.ai/ljx923721867-university-of-surrey/2by4
wandb:  View run at https://wandb.ai/ljx923721867-university-of-surrey/2by4/runs/dbtxa5kc
2025-08-14 22:41:04.375 | INFO     | __main__:main:1800 - 🔧 Step 1: Applying LORO low-rank parameterization...
layer.0.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
wandb: WARNING Saving files without folders. If you want to preserve subdirectories pass base_path to wandb.save, i.e. wandb.save("/mnt/folder/file.h5", base_path="/mnt")
Update steps:   0%|                                   | 0/20000 [00:00<?, ?it/s]2025-08-14 22:41:04.375 | INFO     | __main__:main:1800 - 🔧 Step 1: Applying LORO low-rank parameterization...
layer.0.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.0.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.0.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.0.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.0.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.1.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.1.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.1.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.1.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.1.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.1.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.2.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.2.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.2.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.2.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.2.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.2.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.3.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.3.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.3.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.3.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.3.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.3.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.4.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.4.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.4.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.4.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.4.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.4.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.5.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.5.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.5.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.5.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.5.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.5.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.6.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.6.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.6.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.6.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.6.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.6.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.7.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.7.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.7.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.7.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.7.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.7.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.8.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.8.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.8.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.8.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.8.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.8.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.9.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.9.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.9.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.9.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.9.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.9.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.10.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.10.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.10.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.10.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.10.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.10.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
layer.11.self_attn.q_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.11.self_attn.k_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.11.self_attn.v_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.11.self_attn.o_proj: Linear(in_features=768, out_features=768, bias=False) --> LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
layer.11.mlp.up_proj: Linear(in_features=768, out_features=3072, bias=False) --> LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
layer.11.mlp.down_proj: Linear(in_features=3072, out_features=768, bias=False) --> LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
2025-08-14 22:41:10.134 | INFO     | __main__:main:1810 - ✅ LORO low-rank parameterization applied successfully!
/home/rtx3090/code_jiaxi/LORO-main_temp/loro_torch/loro_optim.py:217: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
2025-08-14 22:41:10.137 | INFO     | __main__:main:1870 - 
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 768, padding_idx=31999)
    (layers): ModuleList(
      (0-11): 12 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
          (k_proj): LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
          (v_proj): LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
          (o_proj): LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (up_proj): LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
          (down_proj): LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
          (act_fn): ReLUSquaredActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=768, out_features=32000, bias=False)
)

2025-08-14 22:41:10.138 | INFO     | __main__:main:1871 - Total params: 91.64M
2025-08-14 22:41:10.138 | INFO     | __main__:main:1874 - Trainable params: 91.64M
2025-08-14 22:41:10.139 | INFO     | __main__:main:1877 - Saving model to /home/rtx3090/code_jiaxi/LORO-main_temp/ckpt every 1000 update steps

LowRankLinear modules are set successfully!
Self-attention rank: 256, MLP rank: 256.
Full param: 134.105856 M = 0.134105856 G
Low-rank param: 91.638528 M = 0.091638528 G
Cprs rate: 68.33298%

2025-08-14 22:41:10.134 | INFO     | __main__:main:1810 - ✅ LORO low-rank parameterization applied successfully!
2025-08-14 22:41:10.137 | INFO     | __main__:main:1870 - 
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 768, padding_idx=31999)
    (layers): ModuleList(
      (0-11): 12 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
          (k_proj): LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
          (v_proj): LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
          (o_proj): LowRankLinear(in_dim=768, out_dim=768, rank=256, init=xavier)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (up_proj): LowRankLinear(in_dim=768, out_dim=3072, rank=256, init=xavier)
          (down_proj): LowRankLinear(in_dim=3072, out_dim=768, rank=256, init=xavier)
          (act_fn): ReLUSquaredActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=768, out_features=32000, bias=False)
)

2025-08-14 22:41:10.138 | INFO     | __main__:main:1871 - Total params: 91.64M
2025-08-14 22:41:10.138 | INFO     | __main__:main:1874 - Trainable params: 91.64M
2025-08-14 22:41:10.139 | INFO     | __main__:main:1877 - Saving model to /home/rtx3090/code_jiaxi/LORO-main_temp/ckpt every 1000 update steps
2025-08-14 22:41:10.320 | INFO     | __main__:main:1967 - 🔧 启用flip rate跟踪...
2025-08-14 22:41:10.321 | INFO     | __main__:main:1971 - ✅ Flip rate tracking enabled for Sparse2to4Linear modules
2025-08-14 22:41:10.320 | INFO     | __main__:main:1967 - 🔧 启用flip rate跟踪...
启用 0 个LORO SparseOverlay模块的flip rate跟踪
2025-08-14 22:41:10.321 | INFO     | __main__:main:1971 - ✅ Flip rate tracking enabled for Sparse2to4Linear modules
Kernel execution failed with shape [3072, 16384], grid (24, 512)
dy1_work dtype: torch.bfloat16, device: cuda:0
sparse_mask_int dtype: torch.int32, shape: torch.Size([3072])
Traceback (most recent call last):
  File "run_c4.py", line 2393, in <module>
    main(args)
  File "run_c4.py", line 2008, in main
    scaled_loss.backward()
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/_tensor.py", line 492, in backward
    torch.autograd.backward(
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/autograd/__init__.py", line 251, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/autograd/function.py", line 288, in apply
    return user_fn(self, *args)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/cuda/amp/autocast_mode.py", line 140, in decorate_bwd
    return bwd(*args, **kwargs)
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/peft_pretraining/modeling_llama.py", line 617, in backward
    grad_weight_in2 = compute_split_gemm_dw2_lowrank(y2, d_intermediate_2, y2_forward, weight_in2, ctx.layer_id_y2)
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/peft_pretraining/modeling_llama.py", line 111, in compute_split_gemm_dw2_lowrank
    result = split_gemm_nocopy(y2_t, d_intermediate_2, sparse_mask)
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/triton_split_gemm_nocopy.py", line 154, in split_gemm_nocopy
    raise e
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/triton_split_gemm_nocopy.py", line 149, in split_gemm_nocopy
    torch.cuda.synchronize()
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/cuda/__init__.py", line 783, in synchronize
    return torch._C._cuda_synchronize()
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

[E ProcessGroupNCCL.cpp:915] [Rank 0] NCCL watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /opt/conda/conda-bld/pytorch_1695392036766/work/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x57 (0x78ed04918617 in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x64 (0x78ed048d398d in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #2: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x118 (0x78ed049d49f8 in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libc10_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x80 (0x78ec7e746790 in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x58 (0x78ec7e74a5b8 in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::workCleanupLoop() + 0x24b (0x78ec7e760dfb in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #6: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x78 (0x78ec7e761108 in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #7: <unknown function> + 0xecdb4 (0x78ed0eeecdb4 in /lib/x86_64-linux-gnu/libstdc++.so.6)
frame #8: <unknown function> + 0x9ca94 (0x78ed2549ca94 in /lib/x86_64-linux-gnu/libc.so.6)
frame #9: <unknown function> + 0x129c3c (0x78ed25529c3c in /lib/x86_64-linux-gnu/libc.so.6)

terminate called after throwing an instance of 'std::runtime_error'
  what():  [Rank 0] NCCL watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /opt/conda/conda-bld/pytorch_1695392036766/work/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x57 (0x78ed04918617 in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x64 (0x78ed048d398d in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #2: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x118 (0x78ed049d49f8 in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libc10_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x80 (0x78ec7e746790 in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x58 (0x78ec7e74a5b8 in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::workCleanupLoop() + 0x24b (0x78ec7e760dfb in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #6: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x78 (0x78ec7e761108 in /home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #7: <unknown function> + 0xecdb4 (0x78ed0eeecdb4 in /lib/x86_64-linux-gnu/libstdc++.so.6)
frame #8: <unknown function> + 0x9ca94 (0x78ed2549ca94 in /lib/x86_64-linux-gnu/libc.so.6)
frame #9: <unknown function> + 0x129c3c (0x78ed25529c3c in /lib/x86_64-linux-gnu/libc.so.6)

[2025-08-14 22:41:30,275] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: -6) local_rank: 0 (pid: 1675255) of binary: /home/rtx3090/miniconda3/envs/loro_2by4/bin/python
Traceback (most recent call last):
  File "/home/rtx3090/miniconda3/envs/loro_2by4/bin/torchrun", line 33, in <module>
    sys.exit(load_entry_point('torch==2.1.0', 'console_scripts', 'torchrun')())
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 346, in wrapper
    return f(*args, **kwargs)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/distributed/run.py", line 806, in main
    run(args)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/distributed/run.py", line 797, in run
    elastic_launch(
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 264, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
========================================================
run_c4.py FAILED
--------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
--------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-08-14_22:41:30
  host      : 3090
  rank      : 0 (local_rank: 0)
  exitcode  : -6 (pid: 1675255)
  error_file: <N/A>
  traceback : Signal 6 (SIGABRT) received by PID 1675255
========================================================

Process finished with exit code 1
