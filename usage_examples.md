# CoLA 和 LoST 优化器使用示例

本文档展示如何使用新集成的 `adamw_cola` 和 `adamw_lost` 优化器。

## CoLA 优化器使用示例

CoLA 优化器在低秩矩阵间添加 SiLU 激活函数，并使用 CoLA 风格的初始化。

### 基本使用命令

```bash
torchrun --nproc_per_node 1 run_c4.py \
    --model_config configs/llama_130m.json \
    --dtype bfloat16 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --save_every 1000 \
    --eval_every 1000 \
    --lr 0.0001 \
    --scheduler cosine_restart \
    --warmup_steps 2000 \
    --min_lr_ratio 0.1 \
    --cosine_restart_freq 500 \
    --lr_adjust_steps -2000 \
    --weight_decay 0.1 \
    --optimizer adamw_cola \
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
    --save_ckpt True \
    --seed 43 \
    --cola_silu True \
    --cola_init True
```

### CoLA 特有参数

- `--cola_silu`: 是否在低秩矩阵间添加 SiLU 激活函数 (默认: False)
- `--cola_init`: 是否使用 CoLA 风格初始化 (默认: False)

**注意**: 当使用 `--optimizer adamw_cola` 时，`cola_silu` 会自动设为 True。

## LoST 优化器使用示例

LoST 优化器结合低秩和 column-wise 稀疏，支持可选的激活函数。

### 基本使用命令

```bash
torchrun --nproc_per_node 1 run_c4.py \
    --model_config configs/llama_130m.json \
    --dtype bfloat16 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --save_every 1000 \
    --eval_every 1000 \
    --lr 0.0001 \
    --scheduler cosine_restart \
    --warmup_steps 2000 \
    --min_lr_ratio 0.1 \
    --cosine_restart_freq 500 \
    --lr_adjust_steps -2000 \
    --weight_decay 0.1 \
    --optimizer adamw_lost \
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
    --save_ckpt True \
    --seed 43 \
    --lost_sparsity 0.01 \
    --lost_sparse_method random \
    --lost_sparse_svd_rank 256 \
    --lost_gamma 0.25 \
    --cola_silu False \
    --cola_init False
```

### LoST 特有参数

- `--lost_sparsity`: Column-wise 稀疏比率 (默认: 0.05)
- `--lost_sparse_method`: 稀疏掩码初始化方法 (默认: "random", 选项: ["random", "gradient", "svd"])
- `--lost_sparse_svd_rank`: SVD-based 掩码初始化的秩 (默认: 256)
- `--lost_gamma`: 低秩和稀疏部分的混合系数 (默认: 0.5)

## 与原始 LORO 训练的对比

### 原始 LORO 训练
```bash
--optimizer loro_adamw \
--loro_type loro \
--loro_freq 500
```

### CoLA 训练 (新增)
```bash
--optimizer adamw_cola \
--cola_silu True \
--cola_init True
```

### LoST 训练 (新增)
```bash
--optimizer adamw_lost \
--lost_sparsity 0.01 \
--lost_sparse_method svd \
--lost_gamma 0.25
```

## 关键区别说明

1. **CoLA**: 主要特点是在低秩矩阵 A 和 B 之间添加 SiLU 激活函数，使用 CoLA 风格初始化
2. **LoST**: 主要特点是结合低秩矩阵和 column-wise 稀疏，支持不同的稀疏初始化方法
3. **兼容性**: 两种优化器都与现有的 LORO 参数兼容，可以使用所有 `--loro_*` 参数

## 推荐设置

### 对于 CoLA：
- 使用较高的学习率，因为 SiLU 激活可能需要更强的优化
- `cola_init=True` 通常能提供更好的初始化

### 对于 LoST：
- `lost_sparsity` 建议从 0.01-0.05 开始调整
- `lost_gamma` 控制低秩和稀疏的平衡，0.25-0.5 通常效果较好
- 对于大模型，使用 `svd` 初始化方法可能效果更好

## 注意事项

1. 确保 `--loro_scope` 设置正确 (通常使用 "all")
2. CoLA 和 LoST 都需要先应用 LORO 低秩参数化
3. 训练时会看到相应的日志输出确认功能已启用
4. 内存使用可能与标准 LORO 有所不同，特别是 LoST