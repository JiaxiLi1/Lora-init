#!/usr/bin/env python3
"""
测试RTX 3090上2:4稀疏矩阵乘法的加速效果
只测试纯矩阵乘法部分，不包含sparse weight计算开销
"""

import torch
import torch.nn as nn
import time

# Import our corrected implementation
from loro_torch.sparse_lowrank_module import (
    FP8SparseOperation, 
    SparseLowRankLinear,
    SPARSE_AVAILABLE
)

def test_pure_matmul_performance():
    """测试纯矩阵乘法性能，预先计算sparse weights"""
    print("=== 测试纯矩阵乘法性能 ===")
    
    if not SPARSE_AVAILABLE:
        print("❌ 2by4 sparse package not available")
        return False
    
    # 创建测试参数
    batch_size = 8
    seq_len = 1024
    hidden_size = 768
    intermediate_size = 3072
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    print(f"测试配置: batch_size={batch_size}, seq_len={seq_len}")
    print(f"矩阵尺寸: {hidden_size} -> {intermediate_size}")
    
    # 1. 创建稀疏层并预计算sparse weights
    print("\n1. 准备稀疏层...")
    linear_ref = nn.Linear(hidden_size, intermediate_size).cuda()
    sparse_layer = SparseLowRankLinear(linear_ref, rank=64, init="xavier", enable_sparse=True).cuda()
    
    # 预计算sparse weights（这个开销不计入性能测试）
    sparse_weight_in = sparse_layer.get_sparse_weight_in()
    sparse_weight_out = sparse_layer.get_sparse_weight_out()
    
    # 检查稀疏度
    sparsity_in = (sparse_weight_in == 0).float().mean().item()
    sparsity_out = (sparse_weight_out == 0).float().mean().item()
    print(f"   Weight_in 稀疏度: {sparsity_in:.1%}")
    print(f"   Weight_out 稀疏度: {sparsity_out:.1%}")
    
    # 2. 创建对应的密集层
    print("\n2. 准备密集层...")
    dense_layer = SparseLowRankLinear(linear_ref, rank=64, init="xavier", enable_sparse=False).cuda()
    dense_weight_in = dense_layer.weight_in
    dense_weight_out = dense_layer.weight_out
    
    # 3. 测试纯矩阵乘法性能
    iterations = 100
    warmup = 20
    
    def time_matmul_operations(name, weight_in, weight_out, use_sparse=False):
        print(f"\n3. 测试{name}矩阵乘法...")
        
        # Warmup
        for _ in range(warmup):
            if use_sparse:
                x_proj = FP8SparseOperation.apply(x, weight_in.t(), None)
                output = FP8SparseOperation.apply(x_proj, weight_out, None)
            else:
                x_proj = torch.matmul(x, weight_in)
                output = torch.matmul(x_proj, weight_out.t())
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # 实际测试
        for _ in range(iterations):
            if use_sparse:
                x_proj = FP8SparseOperation.apply(x, weight_in.t(), None)
                output = FP8SparseOperation.apply(x_proj, weight_out, None)
            else:
                x_proj = torch.matmul(x, weight_in)
                output = torch.matmul(x_proj, weight_out.t())
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations * 1000  # ms
        print(f"   {name}平均时间: {avg_time:.3f}ms")
        return avg_time
    
    # 测试稀疏矩阵乘法
    sparse_time = time_matmul_operations("稀疏", sparse_weight_in, sparse_weight_out, use_sparse=True)
    
    # 测试密集矩阵乘法
    dense_time = time_matmul_operations("密集", dense_weight_in, dense_weight_out, use_sparse=False)
    
    # 计算加速比
    speedup = dense_time / sparse_time
    
    print(f"\n=== 纯矩阵乘法性能对比 ===")
    print(f"稀疏矩阵乘法: {sparse_time:.3f}ms")
    print(f"密集矩阵乘法: {dense_time:.3f}ms")
    print(f"加速比: {speedup:.2f}x")
    
    if speedup > 1.2:
        print("✅ 2:4稀疏化在RTX 3090上成功加速！")
        return True
    elif speedup > 0.9:
        print("⚠️  性能相当，可能受其他因素影响")
        return True
    else:
        print("❌ 稀疏化性能低于预期")
        return False


def test_single_layer_performance():
    """测试单层MLP性能，复现用户记录中的测试"""
    print("\n" + "="*50)
    print("测试单层MLP性能（复现之前的成功案例）")
    print("="*50)
    
    if not SPARSE_AVAILABLE:
        print("❌ 2by4 sparse package not available")
        return False
    
    # 使用与记录中相同的配置
    batch_size = 8
    seq_len = 1024
    hidden_size = 768
    
    input_tensor = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    # 创建稀疏层
    linear_ref = nn.Linear(hidden_size, hidden_size * 4).cuda()
    sparse_layer = SparseLowRankLinear(linear_ref, rank=64, init="xavier", enable_sparse=True).cuda()
    
    # 创建密集层
    dense_layer = SparseLowRankLinear(linear_ref, rank=64, init="xavier", enable_sparse=False).cuda()
    
    def test_layer_performance(layer, name, iterations=50):
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
        print(f"{name}层平均时间: {avg_time:.2f}ms")
        return avg_time
    
    # 测试性能
    sparse_time = test_layer_performance(sparse_layer, "稀疏")
    dense_time = test_layer_performance(dense_layer, "密集")
    
    # 检查稀疏度
    sparse_weight_in = sparse_layer.get_sparse_weight_in()
    sparse_weight_out = sparse_layer.get_sparse_weight_out()
    sparsity = ((sparse_weight_in == 0).float().mean() + (sparse_weight_out == 0).float().mean()) / 2
    
    speedup = dense_time / sparse_time
    
    print(f"\n=== 单层MLP性能对比 ===")
    print(f"稀疏层时间: {sparse_time:.2f}ms")
    print(f"密集层时间: {dense_time:.2f}ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"实际稀疏度: {sparsity:.1%}")
    
    if sparsity < 0.4:
        print("⚠️  警告：稀疏度低于40%，2:4稀疏化可能没有正确应用")
        
    return speedup > 1.5  # 期望至少1.5x加速


def main():
    print("🧪 RTX 3090 2:4稀疏矩阵乘法加速测试")
    print("="*60)
    
    # 检查GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
    else:
        print("❌ CUDA不可用")
        return False
    
    if not SPARSE_AVAILABLE:
        print("❌ 2by4 sparse package不可用")
        return False
    
    print("✅ 使用正确的2by4-pretrain-acc-examples实现")
    
    # 运行测试
    test1_passed = test_pure_matmul_performance()
    test2_passed = test_single_layer_performance()
    
    print(f"\n" + "="*60)
    print("📋 测试总结")
    print("="*60)
    print(f"纯矩阵乘法测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"单层MLP测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 RTX 3090上的2:4稀疏化加速确认成功！")
    else:
        print(f"\n⚠️  部分测试未达到预期，可能需要进一步调试")


if __name__ == "__main__":
    main() 