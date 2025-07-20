# 稀疏度记录功能修复总结

## 问题描述
原始代码在使用 `--wandb_sparsityrelu True` 启动训练时，无法正确记录每层MLP中ReLU²激活后的稀疏度数据到wandb。

## 根本原因
1. **两个不兼容的记录系统**：标准MLP使用 `_current_sparsities`，而 `ActivationSparse2to4Function` 使用 `_sparsity_stats`
2. **函数调用问题**：`_record_activation_sparsity_static` 函数存在模块缓存或执行问题
3. **变量作用域问题**：重复的import语句导致UnboundLocalError
4. **训练步数追踪缺失**：缺少全局训练步数的同步机制

## 修复内容

### 1. 统一稀疏度记录系统
- 将标准MLP和激活稀疏函数都统一使用 `_sparsity_stats` 存储格式
- 统一数据结构：`sparsity_relu/layer_{id}` (仅记录稀疏度值)

### 2. 修复ActivationSparse2to4Function中的稀疏度记录
- 用内联代码替换有问题的 `_record_activation_sparsity_static` 函数调用
- 添加正确的层计数逻辑，确保每个训练步骤的层编号从0开始
- 实现步骤间的状态重置，避免层编号累积

### 3. 修复变量作用域和导入问题
- 移除重复的import语句
- 确保 `ActivationSparse2to4Function` 在全局作用域内可用
- 添加空值检查，提高代码健壮性

### 4. 添加训练步数同步机制
- 在训练循环中添加 `ActivationSparse2to4Function._global_training_step` 的更新
- 确保稀疏度记录系统能正确追踪当前训练步数

## 关键代码修改

### 在 `peft_pretraining/modeling_llama.py` 中：
1. **ActivationSparse2to4Function.forward()**: 用内联稀疏度记录代码替换函数调用
2. **LlamaMLP.record_activation_sparsity()**: 统一使用 `_sparsity_stats` 格式
3. **LlamaMLP.get_sparsity_stats()**: 保持现有逻辑，添加调试支持

### 在 `run_c4.py` 中：
1. **模型初始化后**: 设置 `_wandb_sparsityrelu_enabled` 标志
2. **训练循环中**: 同步 `_global_training_step` 变量
3. **wandb记录**: 简化稀疏度统计的上传逻辑

## 验证结果
测试显示修复后的代码能够：
- ✅ 正确记录每个transformer layer的MLP稀疏度
- ✅ 为12层模型生成13个wandb图表（12个层级 + 1个平均值）
- ✅ 按训练步骤正确更新和清理统计数据
- ✅ 正常启动和运行训练

## 使用方法
使用原始命令行参数启动训练，稀疏度记录功能将自动工作：

```bash
torchrun --nproc_per_node 1 run_c4.py \
  --model_config configs/llama_130m.json \
  --squ_relu relu2 \
  --activation_2by4 True \
  --wandb_sparsityrelu True \
  [其他参数...]
```

在wandb中将看到以下指标：

**激活稀疏度 (ReLU²后):**
- `sparsity_relu/layer_0` 到 `sparsity_relu/layer_11`: 各层激活稀疏度 (12个图表)
- `sparsity_relu/mean_across_layers`: 所有层的平均激活稀疏度 (1个图表)

**权重稀疏度 (所有网络层):**
- Full-rank模式: 每层6个权重矩阵 (4个attention + 2个MLP)
  - `weight_sparsity/layer_X/attn_q_proj`, `attn_k_proj`, `attn_v_proj`, `attn_o_proj`
  - `weight_sparsity/layer_X/mlp_up_proj`, `mlp_down_proj`
- LowRank模式(loro_adamw): 每层12个权重矩阵 (每个层有两个矩阵weight_in/weight_out)
  - `weight_sparsity/layer_X/attn_q_proj_in`, `attn_q_proj_out` (等等)

对于12层模型，总共约85-145个权重稀疏度图表 + 13个激活稀疏度图表。 