 2025-08-21 21:07:56
   Has NaN: True, Has Inf: False
2025-08-21 21:07:56

2025-08-21 21:07:56
❌ NaN/Inf found in module.model.layers.11.self_attn.q_proj input[0]:
2025-08-21 21:07:56
   Shape: [64, 256, 768]
2025-08-21 21:07:56
   Has NaN: True, Has Inf: False
2025-08-21 21:07:56

2025-08-21 21:07:56
❌ NaN/Inf found in module.model.layers.11.self_attn.q_proj output:
2025-08-21 21:07:56
   Shape: [64, 256, 768]
2025-08-21 21:07:56
   Has NaN: True, Has Inf: False
2025-08-21 21:07:56

2025-08-21 21:07:56
❌ NaN/Inf found in module.model.layers.11.self_attn.k_proj input[0]:
2025-08-21 21:07:56
   Shape: [64, 256, 768]
2025-08-21 21:07:56
   Has NaN: True, Has Inf: False
2025-08-21 21:07:56

2025-08-21 21:07:56
❌ NaN/Inf found in module.model.layers.11.self_attn.k_proj output:
2025-08-21 21:07:56
   Shape: [64, 256, 768]
2025-08-21 21:07:56
   Has NaN: True, Has Inf: False
2025-08-21 21:07:56

2025-08-21 21:07:56
❌ NaN/Inf found in module.model.layers.11.self_attn.v_proj input[0]:
2025-08-21 21:07:56
   Shape: [64, 256, 768]
2025-08-21 21:07:56
   Has NaN: True, Has Inf: False
2025-08-21 21:07:56

2025-08-21 21:07:56
❌ NaN/Inf found in module.model.layers.11.self_attn.v_proj output:
2025-08-21 21:07:56
   Shape: [64, 256, 768]
2025-08-21 21:07:56
   Has NaN: True, Has Inf: False
2025-08-21 21:07:56

2025-08-21 21:07:56
❌ NaN/Inf found in module.model.layers.11.self_attn.rotary_emb input[0]:
2025-08-21 21:07:56
   Shape: [64, 12, 256, 64]
2025-08-21 21:07:56
   Has NaN: True, Has Inf: False
2025-08-21 21:07:56


 File "run_c4.py", line 1986, in main
2025-08-21 21:07:56
    first_nan_op = root_cause_detector.analyze_forward_pass(batch, labels)
2025-08-21 21:07:56
  File "/mnt/fast/nobackup/users/xw0006/Lora-init/nan_root_cause_detector.py", line 390, in analyze_forward_pass
2025-08-21 21:07:56
    output = self.model(**batch, labels=labels)
2025-08-21 21:07:56
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
2025-08-21 21:07:56
    return self._call_impl(*args, **kwargs)
2025-08-21 21:07:56
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
2025-08-21 21:07:56
    return forward_call(*args, **kwargs)
2025-08-21 21:07:56
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/parallel/distributed.py", line 1519, in forward
2025-08-21 21:07:56
    else self._run_ddp_forward(*inputs, **kwargs)
2025-08-21 21:07:56
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/parallel/distributed.py", line 1355, in _run_ddp_forward
2025-08-21 21:07:56
    return self.module(*inputs, **kwargs)  # type: ignore[index]
2025-08-21 21:07:56
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
2025-08-21 21:07:56
    return self._call_impl(*args, **kwargs)
2025-08-21 21:07:56
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
2025-08-21 21:07:56
    return forward_call(*args, **kwargs)
2025-08-21 21:07:56
  File "/mnt/fast/nobackup/users/xw0006/Lora-init/peft_pretraining/modeling_llama.py", line 3224, in forward
2025-08-21 21:07:56
    logits = self.lm_head(hidden_states)
2025-08-21 21:07:56
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
2025-08-21 21:07:56
    return self._call_impl(*args, **kwargs)
2025-08-21 21:07:56
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1568, in _call_impl
2025-08-21 21:07:56
    result = forward_call(*args, **kwargs)
2025-08-21 21:07:56
  File "/opt/conda/envs/loro_2by4/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 114, in forward
2025-08-21 21:07:56
    return F.linear(input, self.weight, self.bias)
2025-08-21 21:07:56
  File "/mnt/fast/nobackup/users/xw0006/Lora-init/nan_operation_tracker.py", line 81, in wrapped
2025-08-21 21:07:56
    'first_nan_idx': torch.where(torch.isnan(result))[0][0].item() if has_nan and result.numel() > 0 else None
2025-08-21 21:07:56
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 11.72 GiB. GPU 0 has a total capacty of 23.59 GiB of which 9.18 GiB is free. Including non-PyTorch memory, this process has 14.40 GiB memory in use. Of the allocated memory 13.08 GiB is allocated by PyTorch, and 837.40 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF