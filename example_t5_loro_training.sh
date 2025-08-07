#!/bin/bash

# T5 + LORO 训练示例脚本
# 在使用前请确保已经正确设置了环境和数据路径

echo "🚀 Starting T5 + LORO training example..."

# 检查配置文件是否存在
if [ ! -f "configs/t5_small_normal.json" ]; then
    echo "❌ Error: T5 config file not found!"
    echo "Please make sure T5 config files are in the configs/ directory"
    exit 1
fi

# T5-small + LORO 低秩训练示例
python run_c4_t5.py \
    --model_type t5 \
    --model_config configs/t5_small_normal.json \
    --optimizer loro_adamw \
    --loro_scope all \
    --loro_attn_rank 64 \
    --loro_mlp_rank 64 \
    --loro_init xavier \
    --batch_size 8 \
    --total_batch_size 32 \
    --lr 2e-4 \
    --num_training_steps 100 \
    --warmup_steps 10 \
    --eval_every 50 \
    --max_length 128 \
    --dtype bfloat16 \
    --c4_local True \
    --save_ckpt False

echo "✅ T5 + LORO training example completed!"
echo "💡 Tip: Adjust batch_size, ranks, and num_training_steps based on your hardware and requirements"