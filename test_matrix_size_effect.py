#!/usr/bin/env python3
"""
测试不同矩阵尺寸的2:4加速效果
使用相同的稀疏函数进行公平对比
"""

import sys
import os
sys.path.insert(0, '/home/rtx3090/code_jiaxi/2by4-pretrain-acc-examples/v2/nanoGPT')

import torch
import time
from sparse_ops import fp8_linear

def create_24_sparse_weight(shape):
    """创建真正的2:4稀疏权重"""
    weight = torch.randn(shape).cuda()
    # 每4个元素保留最大的2个
    weight_flat = weight.view(-1, 4)
    _, indices = torch.topk(torch.abs(weight_flat), 2, dim=1)
    mask = torch.zeros_like(weight_flat)
    mask.scatter_(1, indices, 1)
    sparse_weight = weight_flat * mask
    return sparse_weight.view(shape)

def test_matrix_size_performance(in_features, out_features, name):
    """测试指定尺寸矩阵的性能"""
    print(f"\n=== 测试{name} {in_features}×{out_features} ===")
    
    batch_size = 8
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, in_features).cuda()
    
    # 创建稀疏权重
    sparse_weight = create_24_sparse_weight((out_features, in_features))
    sparsity = (sparse_weight == 0).float().mean().item()
    
    # 创建密集权重
    dense_weight = torch.randn(out_features, in_features).cuda()
    
    iterations = 100
    warmup = 20
    
    # 测试稀疏矩阵乘法
    print("   测试稀疏矩阵乘法...")
    for _ in range(warmup):
        _ = fp8_linear.apply(x, sparse_weight, None)
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = fp8_linear.apply(x, sparse_weight, None)
    torch.cuda.synchronize()
    sparse_time = (time.time() - start_time) / iterations * 1000
    
    # 测试密集矩阵乘法
    print("   测试密集矩阵乘法...")
    for _ in range(warmup):
        _ = torch.matmul(x, dense_weight.t())
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = torch.matmul(x, dense_weight.t())
    torch.cuda.synchronize()
    dense_time = (time.time() - start_time) / iterations * 1000
    
    speedup = dense_time / sparse_time
    
    print(f"   稀疏矩阵: {sparse_time:.3f}ms")
    print(f"   密集矩阵: {dense_time:.3f}ms") 
    print(f"   加速比: {speedup:.2f}x")
    print(f"   稀疏度: {sparsity:.1%}")
    
    return sparse_time, dense_time, speedup

def test_loro_vs_original_comparison():
    """对比LORO的两步乘法 vs 原始的单步乘法"""
    print("\n" + "="*60)
    print("LORO分解 vs 原始矩阵 性能对比")
    print("="*60)
    
    batch_size = 8
    seq_len = 1024
    hidden_size = 768
    intermediate_size = 3072
    rank = 64
    
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    # 1. 原始大矩阵 768×3072
    print("\n1. 原始矩阵 768×3072")
    original_sparse_weight = create_24_sparse_weight((intermediate_size, hidden_size))
    original_dense_weight = torch.randn(intermediate_size, hidden_size).cuda()
    
    iterations = 50
    warmup = 10
    
    # 原始稀疏
    for _ in range(warmup):
        _ = fp8_linear.apply(x, original_sparse_weight, None)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = fp8_linear.apply(x, original_sparse_weight, None)
    torch.cuda.synchronize()
    original_sparse_time = (time.time() - start) / iterations * 1000
    
    # 原始密集
    for _ in range(warmup):
        _ = torch.matmul(x, original_dense_weight.t())
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(x, original_dense_weight.t())
    torch.cuda.synchronize()
    original_dense_time = (time.time() - start) / iterations * 1000
    
    print(f"   原始稀疏: {original_sparse_time:.3f}ms")
    print(f"   原始密集: {original_dense_time:.3f}ms")
    print(f"   原始加速比: {original_dense_time/original_sparse_time:.2f}x")
    
    # 2. LORO分解：768×64 + 64×3072
    print(f"\n2. LORO分解 768×{rank} + {rank}×3072")
    
    # 创建LORO权重
    weight_in_sparse = create_24_sparse_weight((rank, hidden_size))  # 768×64
    weight_out_sparse = create_24_sparse_weight((intermediate_size, rank))  # 64×3072
    
    weight_in_dense = torch.randn(rank, hidden_size).cuda()
    weight_out_dense = torch.randn(intermediate_size, rank).cuda()
    
    # LORO稀疏两步
    for _ in range(warmup):
        x_proj = fp8_linear.apply(x, weight_in_sparse, None)
        _ = fp8_linear.apply(x_proj, weight_out_sparse, None)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        x_proj = fp8_linear.apply(x, weight_in_sparse, None)
        _ = fp8_linear.apply(x_proj, weight_out_sparse, None)
    torch.cuda.synchronize()
    loro_sparse_time = (time.time() - start) / iterations * 1000
    
    # LORO密集两步
    for _ in range(warmup):
        x_proj = torch.matmul(x, weight_in_dense.t())
        _ = torch.matmul(x_proj, weight_out_dense.t())
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        x_proj = torch.matmul(x, weight_in_dense.t())
        _ = torch.matmul(x_proj, weight_out_dense.t())
    torch.cuda.synchronize()
    loro_dense_time = (time.time() - start) / iterations * 1000
    
    print(f"   LORO稀疏: {loro_sparse_time:.3f}ms")
    print(f"   LORO密集: {loro_dense_time:.3f}ms")
    print(f"   LORO加速比: {loro_dense_time/loro_sparse_time:.2f}x")
    
    # 3. 总结对比
    print(f"\n=== 总结对比 ===")
    print(f"原始稀疏 vs 原始密集: {original_dense_time/original_sparse_time:.2f}x 加速")
    print(f"LORO稀疏 vs LORO密集: {loro_dense_time/loro_sparse_time:.2f}x 加速")
    print(f"LORO稀疏 vs 原始稀疏: {original_sparse_time/loro_sparse_time:.2f}x ({'快' if original_sparse_time < loro_sparse_time else '慢'})")
    print(f"LORO密集 vs 原始密集: {original_dense_time/loro_dense_time:.2f}x ({'快' if original_dense_time < loro_dense_time else '慢'})")

def main():
    print("🧪 测试不同矩阵尺寸的2:4加速效果")
    print("使用相同的fp8_linear函数进行公平对比")
    print("="*60)
    
    # 检查GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        print("❌ CUDA不可用")
        return
    
    # 测试不同尺寸的矩阵
    test_results = []
    
    # 小矩阵 (LORO的第一步)
    sparse_time, dense_time, speedup = test_matrix_size_performance(768, 64, "小矩阵")
    test_results.append(("768×64 (LORO第一步)", speedup))
    
    # 中小矩阵 (LORO的第二步)  
    sparse_time, dense_time, speedup = test_matrix_size_performance(64, 3072, "中小矩阵")
    test_results.append(("64×3072 (LORO第二步)", speedup))
    
    # 中等矩阵
    sparse_time, dense_time, speedup = test_matrix_size_performance(768, 256, "中等矩阵")
    test_results.append(("768×256", speedup))
    
    # 大矩阵 (原始)
    sparse_time, dense_time, speedup = test_matrix_size_performance(768, 3072, "大矩阵")
    test_results.append(("768×3072 (原始)", speedup))
    
    # 对比LORO vs 原始
    test_loro_vs_original_comparison()
    
    # 汇总结果
    print(f"\n" + "="*60)
    print("📋 不同矩阵尺寸加速效果汇总")
    print("="*60)
    for name, speedup in test_results:
        status = "✅" if speedup > 1.2 else "⚠️ " if speedup > 0.9 else "❌"
        print(f"{status} {name:20} {speedup:.2f}x")

if __name__ == "__main__":
    main() 