torchrun --nproc_per_node 1 run_c4.py --model_config configs/llama_130m.json --dtype bfloat16 --batch_size 64 --total_batch_size 512 --num_training_steps 20000 --save_every 1000 --eval_every 1000 --lr 0.0001 --scheduler cosine_restart --warmup_steps 2000 --min_lr_ratio 0.1 --cosine_restart_freq 500 --lr_adjust_steps -2000 --weight_decay 0.1 --optimizer cola_adamw --loro_refresh all --loro_refresh_freq 500 --loro_scope all --loro_init xavier --loro_attn_rank 256 --loro_mlp_rank 256 --loro_type loro --loro_freq 500 --loro_lr_scaler -1 --c4_local False --enable_2to4_sparse False --save_ckpt True --attn_2by4 True --mlp_2by4 True --seed 43 --flip_rate True --activation_2by4 True --activation_soft_threshold True --squ_relu silu --activation_sparse_method soft_threshold_weights --dynamic_activation_steps 10 --activation_calibration_samples 50 --activation_dense_warmup_steps 0 --dx_direct_sparse 3 --wandb_sparsityrelu True --permute_2by4 False --momentum_reset_steps 200 --more_activation_relu2 True --cola_sparse_method cola_init

2025-08-12 14:06:26.312 | INFO     | __main__:main:1971 - ✅ Flip rate tracking enabled for Sparse2to4Linear modules
2025-08-12 14:06:26.311 | INFO     | __main__:main:1967 - 🔧 启用flip rate跟踪...
2025-08-12 14:06:26.312 | INFO     | __main__:main:1971 - ✅ Flip rate tracking enabled for Sparse2to4Linear modules
Traceback (most recent call last):
  File "run_c4.py", line 2393, in <module>
    main(args)
  File "run_c4.py", line 2004, in main
    loss = model(**batch, labels=labels).loss
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/parallel/distributed.py", line 1519, in forward
    else self._run_ddp_forward(*inputs, **kwargs)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/parallel/distributed.py", line 1355, in _run_ddp_forward
    return self.module(*inputs, **kwargs)  # type: ignore[index]
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/peft_pretraining/modeling_llama.py", line 2551, in forward
    outputs = self.model(
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/peft_pretraining/modeling_llama.py", line 2441, in forward
    layer_outputs = decoder_layer(
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/peft_pretraining/modeling_llama.py", line 2156, in forward
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/peft_pretraining/modeling_llama.py", line 2068, in forward
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "run_c4.py", line 406, in forward
    return ActivationSparse2to4LowRankFunction_cola.apply(
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/autograd/function.py", line 539, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/home/rtx3090/miniconda3/envs/loro_2by4/lib/python3.8/site-packages/torch/cuda/amp/autocast_mode.py", line 113, in decorate_fwd
    return fwd(*args, **kwargs)
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/peft_pretraining/modeling_llama.py", line 1120, in forward
    ActivationSparse2to4LowRankFunctionSingle._record_activation_sparsity_static(y2)
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/peft_pretraining/modeling_llama.py", line 1314, in _record_activation_sparsity_static
    if module_name:
NameError: name 'module_name' is not defined