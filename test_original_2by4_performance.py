#!/usr/bin/env python3
"""
使用原始2by4-pretrain-acc-examples的FP8SparseLinear测试性能
复现用户记录中的成功案例
"""

import sys
import os
sys.path.insert(0, '/home/rtx3090/code_jiaxi/2by4-pretrain-acc-examples/v2/nanoGPT')

import torch
import torch.nn as nn
import time

# Import original 2by4 implementation
try:
    from sparse_ops import FP8SparseLinear
    print("✅ 成功导入原始2by4-pretrain-acc-examples的FP8SparseLinear")
except ImportError as e:
    print(f"❌ 无法导入原始FP8SparseLinear: {e}")
    sys.exit(1)

def test_original_fp8_sparse_linear():
    """测试原始的FP8SparseLinear层性能"""
    print("=== 测试原始FP8SparseLinear性能 ===")
    
    # 创建测试参数（与您记录中相同）
    batch_size = 8
    seq_len = 1024
    hidden_size = 768
    intermediate_size = hidden_size * 4  # 3072
    
    input_tensor = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    print(f"测试配置: batch_size={batch_size}, seq_len={seq_len}")
    print(f"矩阵尺寸: {hidden_size} -> {intermediate_size}")
    
    # 1. 创建原始FP8SparseLinear层
    print("\n1. 创建FP8SparseLinear层...")
    sparse_layer = FP8SparseLinear(hidden_size, intermediate_size).cuda()
    sparse_layer.init_scale()
    
    # 检查稀疏度
    sparse_weights = sparse_layer.get_sparse_weights()
    sparsity = (sparse_weights == 0).float().mean().item()
    print(f"   稀疏度: {sparsity:.1%}")
    
    # 2. 创建普通Linear层
    print("\n2. 创建普通Linear层...")
    dense_layer = nn.Linear(hidden_size, intermediate_size).cuda()
    
    def test_layer_performance(layer, name, iterations=50):
        """测试层性能"""
        layer.eval()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = layer(input_tensor)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            with torch.no_grad():
                output = layer(input_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations * 1000  # ms
        print(f"   {name}平均时间: {avg_time:.2f}ms")
        return avg_time
    
    # 3. 测试性能
    print("\n3. 性能测试...")
    sparse_time = test_layer_performance(sparse_layer, "FP8SparseLinear")
    dense_time = test_layer_performance(dense_layer, "Linear")
    
    # 4. 结果分析
    speedup = dense_time / sparse_time
    
    print(f"\n=== 原始FP8SparseLinear性能对比 ===")
    print(f"FP8SparseLinear时间: {sparse_time:.2f}ms")
    print(f"Linear时间: {dense_time:.2f}ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"实际稀疏度: {sparsity:.1%}")
    
    if sparsity < 0.4:
        print("⚠️  警告：稀疏度低于40%，2:4稀疏化可能没有正确应用")
    
    if speedup > 1.5:
        print("✅ 2:4稀疏化加速成功！")
        return True
    elif speedup > 1.0:
        print("✅ 2:4稀疏化有轻微加速")
        return True
    else:
        print("❌ 2:4稀疏化没有加速效果")
        return False


def test_pure_sparse_matmul():
    """测试纯稀疏矩阵乘法"""
    print("\n" + "="*50)
    print("测试纯稀疏矩阵乘法")
    print("="*50)
    
    # 创建测试数据
    batch_size = 8
    seq_len = 1024
    hidden_size = 768
    
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    # 创建FP8SparseLinear层并获取稀疏权重
    sparse_layer = FP8SparseLinear(hidden_size, hidden_size).cuda()
    sparse_layer.init_scale()
    
    # 获取稀疏权重（预计算）
    sparse_weight = sparse_layer.get_sparse_weights()
    bias = sparse_layer.bias
    
    # 创建密集权重
    dense_weight = torch.randn_like(sparse_weight)
    
    # 导入稀疏操作
    from sparse_ops import fp8_linear
    
    def test_matmul_performance(name, weight, use_sparse=False, iterations=100):
        # Warmup
        for _ in range(20):
            if use_sparse:
                output = fp8_linear.apply(x, weight, bias)
            else:
                output = torch.matmul(x, weight.t()) + bias
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            if use_sparse:
                output = fp8_linear.apply(x, weight, bias)
            else:
                output = torch.matmul(x, weight.t()) + bias
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations * 1000
        print(f"{name}矩阵乘法平均时间: {avg_time:.3f}ms")
        return avg_time
    
    # 测试稀疏和密集矩阵乘法
    sparse_matmul_time = test_matmul_performance("稀疏", sparse_weight, use_sparse=True)
    dense_matmul_time = test_matmul_performance("密集", dense_weight, use_sparse=False)
    
    speedup = dense_matmul_time / sparse_matmul_time
    sparsity = (sparse_weight == 0).float().mean().item()
    
    print(f"\n=== 纯矩阵乘法性能对比 ===")
    print(f"稀疏矩阵乘法: {sparse_matmul_time:.3f}ms")
    print(f"密集矩阵乘法: {dense_matmul_time:.3f}ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"稀疏度: {sparsity:.1%}")
    
    return speedup > 1.2


def main():
    print("🧪 原始2by4-pretrain-acc-examples性能测试")
    print("="*60)
    
    # 检查GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
    else:
        print("❌ CUDA不可用")
        return
    
    # 运行测试
    test1_passed = test_original_fp8_sparse_linear()
    test2_passed = test_pure_sparse_matmul()
    
    print(f"\n" + "="*60)
    print("📋 测试总结")
    print("="*60)
    print(f"FP8SparseLinear层测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"纯矩阵乘法测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 原始2by4实现确实有加速效果！")
        print("这说明RTX 3090支持2:4稀疏加速，我们的LORO实现可能需要调整")
    elif test1_passed or test2_passed:
        print("\n⚠️  部分测试显示有加速效果")
    else:
        print("\n❌ 测试未显示明显加速效果")


if __name__ == "__main__":
    main() 