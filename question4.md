现在问题是split_gemm我觉得可能实现有问题，目前的split_gemm是在forward里，参考了原文章说的《The dense GEMM in the forward pass, Y1 = XW1 can compute the column level sparsity in the epilogue for free》，在forward阶段用fused_gemm_forward_with_sparsity这个用triton kernel写的函数在进行矩阵乘法同时统计sparsity，然后在backward时直接用，结果出现了这个报错:
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
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/peft_pretraining/modeling_llama.py", line 576, in backward
    grad_weight_in2 = compute_split_gemm_dw2_lowrank(y2, d_intermediate_2, y2_forward, weight_in2)
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/peft_pretraining/modeling_llama.py", line 96, in compute_split_gemm_dw2_lowrank
    return compute_split_gemm_dw_with_cached_sparsity(
  File "/home/rtx3090/code_jiaxi/LORO-main_temp/fused_sparsity_ops.py", line 143, in compute_split_gemm_dw_with_cached_sparsity
    dense_mask = ~sparse_mask
TypeError: bad operand type for unary ~: 'NoneType'
这是怎么回事？请你检查，修改，并检查还有没有其他代码bug，比如针对lowrank的forward和backward不对什么的。