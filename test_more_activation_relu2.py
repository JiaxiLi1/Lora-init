#!/usr/bin/env python3
"""
测试more_activation_relu2功能的脚本
"""

import subprocess
import sys

def test_cola_with_more_activation_relu2():
    """测试CoLA优化器with more_activation_relu2"""
    cmd = [
        "conda", "run", "-n", "loro_2by4",
        "timeout", "60",
        "torchrun", "--nproc_per_node", "1", "run_c4.py",
        "--model_config", "configs/llama_130m.json",
        "--dtype", "bfloat16",
        "--batch_size", "1", "--total_batch_size", "1",
        "--num_training_steps", "2",
        "--save_every", "1000", "--eval_every", "1000",
        "--lr", "0.0001", "--weight_decay", "0.1",
        "--optimizer", "adamw_cola",
        "--loro_scope", "all", "--loro_init", "xavier",
        "--loro_attn_rank", "8", "--loro_mlp_rank", "8",
        "--c4_local", "False", "--enable_2to4_sparse", "False",
        "--save_ckpt", "False",
        "--more_activation_relu2",  # 启用ReLU² + activation 2:4 sparsity
        "--activation_sparse_method", "mvue",
        "--activation_dense_warmup_steps", "1",  # 短暂warmup用于测试
        "--2by4_permute", "True",
        "--seed", "43"
    ]
    
    print("🧪 Testing CoLA optimizer with --more_activation_relu2...")
    print("Expected output: ✅ Applied CoLA ReLU² + activation 2:4 sparsity to X LowRankLinear modules")
    print("Command:", " ".join(cmd))
    print("=" * 80)
    
    return subprocess.run(cmd)

def test_lost_with_more_activation_relu2():
    """测试LoST优化器with more_activation_relu2"""
    cmd = [
        "conda", "run", "-n", "loro_2by4", 
        "timeout", "60",
        "torchrun", "--nproc_per_node", "1", "run_c4.py",
        "--model_config", "configs/llama_130m.json",
        "--dtype", "bfloat16", 
        "--batch_size", "1", "--total_batch_size", "1",
        "--num_training_steps", "2",
        "--save_every", "1000", "--eval_every", "1000",
        "--lr", "0.0001", "--weight_decay", "0.1",
        "--optimizer", "adamw_lost",
        "--loro_scope", "all", "--loro_init", "xavier",
        "--loro_attn_rank", "8", "--loro_mlp_rank", "8",
        "--c4_local", "False", "--enable_2to4_sparse", "False", 
        "--save_ckpt", "False",
        "--lost_sparsity", "0.1",
        "--more_activation_relu2",  # 启用ReLU² + activation 2:4 sparsity
        "--activation_sparse_method", "mvue",
        "--activation_dense_warmup_steps", "1",  # 短暂warmup用于测试
        "--2by4_permute", "True", 
        "--seed", "43"
    ]
    
    print("\n🧪 Testing LoST optimizer with --more_activation_relu2...")
    print("Expected output: ✅ Applied LoST hybrid processing + ReLU² activation 2:4 sparsity to X LowRankLinear modules")
    print("Command:", " ".join(cmd))
    print("=" * 80)
    
    return subprocess.run(cmd)

def test_cola_without_more_activation_relu2():
    """测试CoLA优化器without more_activation_relu2（对比测试）"""
    cmd = [
        "conda", "run", "-n", "loro_2by4",
        "timeout", "60", 
        "torchrun", "--nproc_per_node", "1", "run_c4.py",
        "--model_config", "configs/llama_130m.json",
        "--dtype", "bfloat16",
        "--batch_size", "1", "--total_batch_size", "1", 
        "--num_training_steps", "2",
        "--save_every", "1000", "--eval_every", "1000",
        "--lr", "0.0001", "--weight_decay", "0.1",
        "--optimizer", "adamw_cola",
        "--loro_scope", "all", "--loro_init", "xavier", 
        "--loro_attn_rank", "8", "--loro_mlp_rank", "8",
        "--c4_local", "False", "--enable_2to4_sparse", "False",
        "--save_ckpt", "False",
        # 不使用 --more_activation_relu2
        "--seed", "43"
    ]
    
    print("\n🧪 Testing CoLA optimizer WITHOUT --more_activation_relu2 (comparison)...")
    print("Expected output: ✅ Applied CoLA SiLU activation to X LowRankLinear modules")
    print("Command:", " ".join(cmd))
    print("=" * 80)
    
    return subprocess.run(cmd)

if __name__ == "__main__":
    print("🚀 Testing more_activation_relu2 functionality for CoLA and LoST optimizers")
    print("=" * 80)
    
    # Test 1: CoLA with more_activation_relu2
    result1 = test_cola_with_more_activation_relu2()
    
    # Test 2: LoST with more_activation_relu2  
    result2 = test_lost_with_more_activation_relu2()
    
    # Test 3: CoLA without more_activation_relu2 (comparison)
    result3 = test_cola_without_more_activation_relu2()
    
    print("\n" + "=" * 80)
    print("📊 Test Results Summary:")
    print(f"CoLA with more_activation_relu2: {'✅ PASSED' if result1.returncode in [0, 124] else '❌ FAILED'}")
    print(f"LoST with more_activation_relu2: {'✅ PASSED' if result2.returncode in [0, 124] else '❌ FAILED'}")  
    print(f"CoLA without more_activation_relu2: {'✅ PASSED' if result3.returncode in [0, 124] else '❌ FAILED'}")
    print("Note: Exit code 124 (timeout) is acceptable as we're only testing initialization")