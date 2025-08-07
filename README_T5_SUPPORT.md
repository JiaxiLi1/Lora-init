# T5 Support in LORO-main

本文档说明如何在LORO-main代码库中使用T5模型进行低秩训练。

## 新增功能

已为LORO-main代码库添加了完整的T5支持，包括：

1. **T5模型架构**: 支持T5 encoder-decoder模型
2. **T5数据处理**: 实现了T5的去噪(denoising)预训练目标
3. **LORO低秩训练**: T5模型可以使用LORO低秩优化器训练
4. **自动层名匹配**: 自动识别T5的层名结构（SelfAttention, EncDecAttention, DenseReluDense等）

## 使用方法

### 基本用法

在原有命令基础上添加 `--model_type t5` 参数：

```bash
python run_c4_t5.py \
    --model_type t5 \
    --model_config configs/t5_small_normal.json \
    --optimizer loro_adamw \
    --loro_scope all \
    --loro_attn_rank 64 \
    --loro_mlp_rank 64 \
    --batch_size 8 \
    --total_batch_size 64 \
    --lr 1e-4 \
    --num_training_steps 1000
```

### 配置文件

提供了以下T5配置文件：

- `configs/t5_small_normal.json` - T5-small标准配置
- `configs/t5_small_small.json` - T5-small紧凑配置  
- `configs/t5_small_delta.json` - T5-small Delta配置
- `configs/t5_base_normal.json` - T5-base标准配置
- `configs/t5_base_small.json` - T5-base紧凑配置
- `configs/t5_base_delta.json` - T5-base Delta配置

### 支持的优化器

目前T5支持以下优化器：

- `loro_adamw` - ✅ **主要支持** (LORO低秩训练)
- `adam` - ✅ 全秩训练
- `adamw` - ✅ 全秩训练

**注意**: T5暂不支持2:4稀疏训练相关的优化器配置。

### 关键参数

- `--model_type`: 设置为 `t5` 启用T5支持
- `--model_config`: 指向T5配置文件路径
- `--optimizer`: 使用 `loro_adamw` 进行低秩训练
- `--loro_scope`: 设置低秩应用范围 (`all`, `attn`, `mlp`)
- `--loro_attn_rank`: attention层的低秩秩数
- `--loro_mlp_rank`: MLP层的低秩秩数

## T5 vs LLaMA 训练差异

### 模型架构
- **LLaMA**: Decoder-only, 因果语言模型
- **T5**: Encoder-decoder, 条件生成模型

### 训练目标
- **LLaMA**: 因果语言建模 (下一个token预测)
- **T5**: 去噪目标 (span corruption, 重建被mask的文本片段)

### 数据处理
- **LLaMA**: 直接使用input_ids作为labels
- **T5**: 使用去噪处理，创建encoder input, decoder input, 和labels

## 测试

运行测试脚本验证T5支持：

```bash
python test_t5_loro.py
```

## 示例命令

### T5-small + LORO训练

```bash
python run_c4_t5.py \
    --model_type t5 \
    --model_config configs/t5_small_normal.json \
    --optimizer loro_adamw \
    --loro_scope all \
    --loro_attn_rank 128 \
    --loro_mlp_rank 128 \
    --batch_size 16 \
    --total_batch_size 128 \
    --lr 2e-4 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --eval_every 1000
```

### T5-base + LORO训练

```bash
python run_c4_t5.py \
    --model_type t5 \
    --model_config configs/t5_base_normal.json \
    --optimizer loro_adamw \
    --loro_scope all \
    --loro_attn_rank 256 \
    --loro_mlp_rank 256 \
    --batch_size 8 \
    --total_batch_size 64 \
    --lr 1e-4 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --eval_every 2000
```

## 注意事项

1. **内存使用**: T5是encoder-decoder模型，比同等大小的decoder-only模型使用更多内存
2. **训练时间**: T5的去噪处理会增加一些数据预处理时间
3. **收敛性**: T5的去噪目标与LLaMA的因果建模不同，可能需要调整学习率和训练步数
4. **评估**: T5使用去噪目标进行评估，损失值的意义与LLaMA不同

## 文件说明

- `peft_pretraining/modeling_t5.py`: T5模型实现
- `t5_data_utils.py`: T5数据处理工具（去噪功能）
- `configs/t5_*.json`: T5模型配置文件
- `test_t5_loro.py`: T5功能测试脚本

现在您可以使用 `--model_type t5` 和 `--optimizer loro_adamw` 来训练T5模型了！