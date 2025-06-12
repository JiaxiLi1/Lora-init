#!/usr/bin/env python3

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from loro_torch.lowrank_module import apply_lowrank_param
from loro_torch.sparse_overlay import apply_sparse_overlay_on_loro

print('🔍 调试LORO模块检测...')

def debug_loro_detection():
    # 创建一个小的LLaMA模型配置
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        initializer_range=0.02,
    )
    
    print("1. 创建原始模型...")
    model = LlamaForCausalLM(config)
    
    # 检查原始模型中的Linear模块
    print("2. 原始模型中的Linear模块:")
    original_linear_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            original_linear_count += 1
            if 'q_proj' in name or 'k_proj' in name:
                print(f"   {name}: {type(module)}")
    print(f"   总共{original_linear_count}个Linear模块")
    
    print("\n3. 应用LORO...")
    apply_lowrank_param(
        model,
        config,
        model_type="llama",
        scope="all",  # 应用到所有层
        attn_rank=16,
        mlp_rank=32,
        init="xavier",
        verbose=False
    )
    
    # 检查LORO应用后的模块
    print("4. LORO应用后的模块检查:")
    loro_count = 0
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    for name, module in model.named_modules():
        module_name = name.split('.')[-1]
        if module_name in target_modules:
            print(f"   {name}:")
            print(f"     类型: {type(module)}")
            print(f"     有weight_in: {hasattr(module, 'weight_in')}")
            print(f"     有weight_out: {hasattr(module, 'weight_out')}")
            if hasattr(module, 'weight_in') and hasattr(module, 'weight_out'):
                print(f"     weight_in shape: {module.weight_in.shape}")
                print(f"     weight_out shape: {module.weight_out.shape}")
                loro_count += 1
    
    print(f"\n5. 找到{loro_count}个LORO模块")
    
    print("\n6. 尝试应用sparse overlay...")
    try:
        model = apply_sparse_overlay_on_loro(
            model,
            sparse_init_scale=1.0,
            target_modules=["q_proj", "k_proj", "v_proj"]  # 只测试几个模块
        )
        print("✅ Sparse overlay应用成功！")
    except Exception as e:
        print(f"❌ Sparse overlay应用失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 检查最终的模块类型
    print("\n7. 最终模块类型检查:")
    for name, module in model.named_modules():
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            print(f"   {name}: {type(module)}")

if __name__ == '__main__':
    debug_loro_detection() 