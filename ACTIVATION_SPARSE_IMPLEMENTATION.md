# Activation 2:4 Sparsity Implementation - Paper Compliance

## 概述

本实现严格按照论文要求实现了activation 2:4 sparsity，包含以下关键组件：

## 1. 核心实现组件

### 1.1 ActivationSparse2to4Function (autograd.Function)
- **Forward pass**: 实现三种2:4稀疏化方法 (naive, MVUE, soft_threshold)
- **Backward pass**: 实现feature-wise 2:4稀疏化
- **Token permutation**: 实现固定token排列优化 (Optimization 2)
- **Dense warmup**: 可配置的dense训练阶段

### 1.2 支持函数
- `apply_naive_2to4_sparsity()`: 简单的top-2选择
- `apply_mvue_2to4_sparsity()`: 使用MVUE24_approx_triton kernel
- `apply_soft_threshold_2to4_sparsity()`: 使用soft_threshold24_triton kernel
- `apply_feature_wise_2to4_sparsity()`: 实现论文的backward pass策略
- `apply_naive_2to4_sparsity_featurewise()`: Feature-wise的2:4稀疏化

## 2. 论文要求的实现细节

### 2.1 Forward Pass
```
Y2 = ReLU²(X1W1)  # Squared ReLU activation
Y2_sparse = 2to4_sparsify(Y2)  # Apply 2:4 sparsity
Y3 = Y2_sparse * W2  # Second linear layer
```

### 2.2 Backward Pass (Feature-wise 2:4 Sparsity)
论文描述的优化策略：

#### Optimization 1: 95%/5% 特征分割
- **95%的特征**: 可以进行feature-wise 2:4稀疏化
- **5%的特征**: 保持dense (不够稀疏的特征)
- **实现**: `apply_feature_wise_2to4_sparsity()`函数

#### Optimization 2: Token Permutation
- **目的**: 利用连续token间的特征相关性
- **实现**: 在进入FFN前对token进行固定排列，处理后重新排列回来
- **存储**: 使用类级别的`_token_permutation`和`_inverse_permutation`字典

#### Optimization 3: Forward Mask维护
- **目的**: 防止forward pass中丢弃的值在backward pass中重新出现
- **实现**: 在backward pass中应用forward pass的稀疏化mask

### 2.3 Dense Warmup
- **论文要求**: 前1000个iteration使用dense训练
- **实现**: 可配置的warmup steps参数 (`--activation_dense_warmup_steps`)
- **原因**: 初始化时稀疏度约50%，需要几个训练步骤增加到90%+

## 3. 配置参数

### 3.1 运行时参数
```bash
--squ_relu True                           # 启用squared ReLU + 自动activation 2:4稀疏
--activation_sparse_method mvue            # 稀疏化方法: naive/mvue/soft_threshold
--activation_dense_warmup_steps 1000      # Dense warmup的步数 (可配置)
```

### 3.2 自动配置
当`--squ_relu True`时：
- 自动移除gate_proj (GPT-2风格架构)
- 自动启用activation 2:4稀疏化
- 自动配置稀疏化方法和warmup步数

## 4. 架构变化

### 4.1 原始SwiGLU架构
```
x -> gate_proj -> SiLU -> * -> down_proj -> output
  -> up_proj   ---------> 
```

### 4.2 新的Squared ReLU架构
```
x -> up_proj -> ReLU² -> 2:4_sparsify -> down_proj -> output
```

参数数量保持不变：
- 原始: 3 × hidden_size × intermediate_size
- 新架构: 2 × hidden_size × (1.5 × intermediate_size)

## 5. 技术细节

### 5.1 Scale计算 (Soft Threshold方法)
- **计算一次**: `scale = dot(input, sparse_output) / dot(sparse_output, sparse_output)`
- **固定使用**: 计算后存储在类级别字典中，不再重新计算
- **与现有代码一致**: 遵循相同的scale管理模式

### 5.2 Triton Kernel兼容性
- **数据类型转换**: bfloat16 ↔ float16 for triton compatibility
- **错误处理**: 当triton kernels不可用时的fallback机制

### 5.3 Feature-wise稀疏化策略
```python
# 1. 分析特征稀疏度
feature_sparsity = torch.mean((grad_tensor != 0).float(), dim=0)

# 2. 选择95%最稀疏的特征
sparse_features_mask = select_top_sparse_features(feature_sparsity, 0.95)

# 3. 对选中特征应用feature-wise 2:4稀疏化
sparse_grad = apply_2to4_along_batch_dim(grad_tensor[:, sparse_features_mask])

# 4. 应用forward mask维护一致性
grad_output = grad_output * forward_mask
```

## 6. 测试验证

### 6.1 测试覆盖
- ✅ 2:4稀疏化模式验证
- ✅ Token permutation一致性
- ✅ Dense warmup功能
- ✅ Feature-wise backward pass
- ✅ Autograd集成
- ✅ MLP集成测试

### 6.2 运行测试
```bash
cd LORO-main
python test_activation_sparse.py
```

## 7. 使用示例

### 7.1 训练命令
```bash
torchrun --nproc_per_node 1 run_c4.py \
    --model_config configs/llama_130m.json \
    --dtype bfloat16 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --save_every 1000 \
    --eval_every 1000 \
    --lr 0.001 \
    --scheduler cosine_restart \
    --warmup_steps 2000 \
    --min_lr_ratio 0.1 \
    --cosine_restart_freq 500 \
    --lr_adjust_steps -2000 \
    --weight_decay 0.1 \
    --optimizer adamw \
    --loro_refresh all \
    --loro_refresh_freq 500 \
    --loro_scope all \
    --loro_init xavier \
    --loro_attn_rank 256 \
    --loro_mlp_rank 256 \
    --loro_type loro \
    --loro_freq 500 \
    --loro_lr_scaler -1 \
    --c4_local False \
    --enable_2to4_sparse False \
    --save_ckpt True \
    --attn_2by4 True \
    --mlp_2by4 False \
    --seed 43 \
    --flip_rate True \
    --squ_relu True \
    --activation_sparse_method mvue \
    --activation_dense_warmup_steps 1000
```

## 8. 论文合规性检查

| 论文要求 | 实现状态 | 说明 |
|---------|---------|------|
| Forward 2:4 sparsity | ✅ | 三种方法: naive/MVUE/soft_threshold |
| Feature-wise backward sparsity | ✅ | 95%/5%分割策略 |
| Token permutation | ✅ | 固定排列优化 |
| Forward mask维护 | ✅ | 防止backward中值重新出现 |
| Dense warmup | ✅ | 可配置步数，默认1000 |
| Scale固定 | ✅ | 计算一次后保持不变 |
| Squared ReLU | ✅ | 替代SwiGLU激活 |
| 参数数量维持 | ✅ | 通过调整intermediate_size |

## 9. 性能特点

- **前向推理加速**: 在compute-bound场景下提供加速
- **内存效率**: 通过activation稀疏化减少内存使用
- **训练稳定性**: 通过dense warmup和forward mask维护确保稳定性
- **硬件兼容**: 支持NVIDIA GPU的2:4稀疏化加速

## 10. 注意事项

1. **环境要求**: 需要loro_2by4 conda环境和相应的triton kernels
2. **GPU要求**: 需要支持2:4稀疏化的NVIDIA GPU
3. **参数配置**: 当启用`--squ_relu True`时，其他activation相关参数会自动配置
4. **向后兼容**: 不启用squ_relu时，保持原有架构不变 