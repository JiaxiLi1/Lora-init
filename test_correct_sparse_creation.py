#!/usr/bin/env python3
"""
使用正确的soft_threshold24_triton创建稀疏矩阵进行测试
比较不同矩阵尺寸的2:4加速效果
"""

import sys
import os
sys.path.insert(0, '/home/rtx3090/code_jiaxi/2by4-pretrain-acc-examples/v2/nanoGPT')

import torch
import time
from sparse_ops import fp8_linear
from sparse import soft_threshold24_triton

def create_correct_24_sparse_weight(shape):
    """使用正确的soft_threshold24_triton创建2:4稀疏权重"""
    weight = torch.randn(shape).cuda()
    # 使用2by4项目的正确实现
    weight_sparse, mask = soft_threshold24_triton(weight)
    return weight_sparse

def test_sparse_creation():
    """测试稀疏矩阵创建是否正确"""
    print("🔬 验证稀疏矩阵创建")
    print("="*50)
    
    weight = torch.randn(256, 768).cuda()
    sparse_weight = create_correct_24_sparse_weight((256, 768))
    
    # 检查稀疏度
    sparsity = (sparse_weight == 0).float().mean().item()
    print(f"稀疏度: {sparsity:.1%}")
    
    # 检查2:4模式
    reshaped = sparse_weight.view(-1, 4)
    nonzero_counts = (reshaped != 0).sum(dim=1)
    perfect_24 = torch.all(nonzero_counts <= 2)
    print(f"2:4模式: {'✅ 正确' if perfect_24 else '❌ 错误'}")
    
    # 检查非零元素分布
    print(f"每4个元素的非零计数分布:")
    for i in range(3):
        count = (nonzero_counts == i).sum().item()
        print(f"  {i}个非零: {count}")
    
    return sparse_weight

def test_matrix_size_performance_correct(in_features, out_features, name):
    """使用正确的稀疏创建方法测试性能"""
    print(f"\n=== 测试{name} {in_features}×{out_features} ===")
    
    batch_size = 8
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, in_features).cuda()
    
    # 使用正确的soft_threshold创建稀疏权重
    sparse_weight = create_correct_24_sparse_weight((out_features, in_features))
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

def compare_old_vs_new_sparse_creation():
    """对比我之前错误的稀疏创建 vs 正确的soft_threshold创建"""
    print("\n" + "="*60)
    print("错误稀疏创建 vs 正确稀疏创建 对比")
    print("="*60)
    
    # 修正：确保矩阵维度正确
    in_features = 768
    out_features = 3072
    x = torch.randn(8, 1024, in_features).cuda()
    
    # 1. 我之前错误的方法
    print("\n1. 错误的稀疏创建方法 (topk)")
    weight_wrong = torch.randn(out_features, in_features).cuda()  # 修正维度
    weight_flat = weight_wrong.view(-1, 4)
    _, indices = torch.topk(torch.abs(weight_flat), 2, dim=1)
    mask = torch.zeros_like(weight_flat)
    mask.scatter_(1, indices, 1)
    sparse_weight_wrong = (weight_flat * mask).view(out_features, in_features)
    
    sparsity_wrong = (sparse_weight_wrong == 0).float().mean().item()
    print(f"   稀疏度: {sparsity_wrong:.1%}")
    
    # 2. 正确的方法
    print("\n2. 正确的稀疏创建方法 (soft_threshold24_triton)")
    sparse_weight_correct = create_correct_24_sparse_weight((out_features, in_features))
    
    sparsity_correct = (sparse_weight_correct == 0).float().mean().item()
    print(f"   稀疏度: {sparsity_correct:.1%}")
    
    # 性能测试
    iterations = 50
    warmup = 10
    
    # 错误方法性能
    for _ in range(warmup):
        _ = fp8_linear.apply(x, sparse_weight_wrong, None)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = fp8_linear.apply(x, sparse_weight_wrong, None)
    torch.cuda.synchronize()
    wrong_time = (time.time() - start) / iterations * 1000
    
    # 正确方法性能
    for _ in range(warmup):
        _ = fp8_linear.apply(x, sparse_weight_correct, None)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = fp8_linear.apply(x, sparse_weight_correct, None)
    torch.cuda.synchronize()
    correct_time = (time.time() - start) / iterations * 1000
    
    # 密集参照
    dense_weight = torch.randn(out_features, in_features).cuda()
    for _ in range(warmup):
        _ = torch.matmul(x, dense_weight.t())
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(x, dense_weight.t())
    torch.cuda.synchronize()
    dense_time = (time.time() - start) / iterations * 1000
    
    print(f"\n=== 性能对比 ===")
    print(f"错误方法: {wrong_time:.3f}ms (加速比: {dense_time/wrong_time:.2f}x)")
    print(f"正确方法: {correct_time:.3f}ms (加速比: {dense_time/correct_time:.2f}x)")
    print(f"密集方法: {dense_time:.3f}ms")
    print(f"正确 vs 错误: {wrong_time/correct_time:.2f}x ({'快' if wrong_time > correct_time else '慢'})")

def test_two_step_matmul_comparison():
    """测试连续两步矩阵乘法：密集vs稀疏"""
    print("\n" + "="*60)
    print("连续两步矩阵乘法对比")
    print("="*60)
    
    batch_size = 8
    seq_len = 1024
    hidden_size = 512
    intermediate_size = 512
    rank = 128
    
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    # 创建权重
    # 第一步：768 -> 64
    weight_in_dense = torch.randn(rank, hidden_size).cuda()
    weight_in_sparse = create_correct_24_sparse_weight((rank, hidden_size))
    
    # 第二步：64 -> 3072
    weight_out_dense = torch.randn(intermediate_size, rank).cuda()
    weight_out_sparse = create_correct_24_sparse_weight((intermediate_size, rank))
    
    iterations = 100
    warmup = 20
    
    print(f"\n测试配置:")
    print(f"  输入: {batch_size}×{seq_len}×{hidden_size}")
    print(f"  第一步: {hidden_size}→{rank}")
    print(f"  第二步: {rank}→{intermediate_size}")
    print(f"  迭代次数: {iterations}")
    
    # 1. 连续两步密集矩阵乘法
    print(f"\n1. 连续两步密集矩阵乘法")
    for _ in range(warmup):
        x_proj = torch.matmul(x, weight_in_dense.t())
        _ = torch.matmul(x_proj, weight_out_dense.t())
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        x_proj = torch.matmul(x, weight_in_dense.t())
        output = torch.matmul(x_proj, weight_out_dense.t())
    torch.cuda.synchronize()
    dense_two_step_time = (time.time() - start) / iterations * 1000
    
    print(f"   密集两步总时间: {dense_two_step_time:.3f}ms")
    
    # 2. 连续两步稀疏矩阵乘法
    print(f"\n2. 连续两步稀疏矩阵乘法")
    for _ in range(warmup):
        x_proj = fp8_linear.apply(x, weight_in_sparse, None)
        _ = fp8_linear.apply(x_proj, weight_out_sparse, None)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        x_proj = fp8_linear.apply(x, weight_in_sparse, None)
        output = fp8_linear.apply(x_proj, weight_out_sparse, None)
    torch.cuda.synchronize()
    sparse_two_step_time = (time.time() - start) / iterations * 1000
    
    print(f"   稀疏两步总时间: {sparse_two_step_time:.3f}ms")
    
    # 3. 分解对比各步
    print(f"\n3. 分解各步时间")
    
    # 第一步单独测试
    for _ in range(warmup):
        _ = torch.matmul(x, weight_in_dense.t())
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(x, weight_in_dense.t())
    torch.cuda.synchronize()
    dense_step1_time = (time.time() - start) / iterations * 1000
    
    for _ in range(warmup):
        _ = fp8_linear.apply(x, weight_in_sparse, None)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = fp8_linear.apply(x, weight_in_sparse, None)
    torch.cuda.synchronize()
    sparse_step1_time = (time.time() - start) / iterations * 1000
    
    # 第二步单独测试 (使用中间尺寸)
    x_temp = torch.randn(batch_size, seq_len, rank).cuda()
    
    for _ in range(warmup):
        _ = torch.matmul(x_temp, weight_out_dense.t())
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(x_temp, weight_out_dense.t())
    torch.cuda.synchronize()
    dense_step2_time = (time.time() - start) / iterations * 1000
    
    for _ in range(warmup):
        _ = fp8_linear.apply(x_temp, weight_out_sparse, None)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = fp8_linear.apply(x_temp, weight_out_sparse, None)
    torch.cuda.synchronize()
    sparse_step2_time = (time.time() - start) / iterations * 1000
    
    print(f"   第一步 (768→64):")
    print(f"     密集: {dense_step1_time:.3f}ms")
    print(f"     稀疏: {sparse_step1_time:.3f}ms (加速: {dense_step1_time/sparse_step1_time:.2f}x)")
    
    print(f"   第二步 (64→3072):")
    print(f"     密集: {dense_step2_time:.3f}ms")
    print(f"     稀疏: {sparse_step2_time:.3f}ms (加速: {dense_step2_time/sparse_step2_time:.2f}x)")
    
    # 4. 总结对比
    print(f"\n=== 总结对比 ===")
    print(f"连续两步总时间:")
    print(f"  密集: {dense_two_step_time:.3f}ms")
    print(f"  稀疏: {sparse_two_step_time:.3f}ms")
    print(f"  稀疏加速比: {dense_two_step_time/sparse_two_step_time:.2f}x")
    
    # 各步相加 vs 连续操作
    theoretical_dense = dense_step1_time + dense_step2_time
    theoretical_sparse = sparse_step1_time + sparse_step2_time
    
    print(f"\n理论时间 (各步相加) vs 实际时间:")
    print(f"  密集理论: {theoretical_dense:.3f}ms vs 实际: {dense_two_step_time:.3f}ms (差异: {abs(theoretical_dense-dense_two_step_time):.3f}ms)")
    print(f"  稀疏理论: {theoretical_sparse:.3f}ms vs 实际: {sparse_two_step_time:.3f}ms (差异: {abs(theoretical_sparse-sparse_two_step_time):.3f}ms)")
    
    # 检查稀疏度
    sparsity_in = (weight_in_sparse == 0).float().mean().item()
    sparsity_out = (weight_out_sparse == 0).float().mean().item()
    print(f"\n稀疏度检查:")
    print(f"  第一步权重稀疏度: {sparsity_in:.1%}")
    print(f"  第二步权重稀疏度: {sparsity_out:.1%}")
    
    return {
        'dense_two_step': dense_two_step_time,
        'sparse_two_step': sparse_two_step_time,
        'dense_step1': dense_step1_time,
        'sparse_step1': sparse_step1_time,
        'dense_step2': dense_step2_time,
        'sparse_step2': sparse_step2_time,
        'speedup': dense_two_step_time/sparse_two_step_time
    }

def main():
    print("🧪 使用正确的soft_threshold24_triton测试2:4加速效果")
    print("="*60)
    
    # 检查GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        print("❌ CUDA不可用")
        return
    
    # 验证稀疏矩阵创建
    test_sparse_creation()
    
    # 对比错误 vs 正确的稀疏创建方法
    # compare_old_vs_new_sparse_creation()
    
    # 测试不同尺寸的矩阵 (使用正确方法)
    print(f"\n" + "="*60)
    print("不同矩阵尺寸测试 (使用正确稀疏创建)")
    print("="*60)
    
    test_results = []
    
    # # 小矩阵 (LORO的第一步)
    # sparse_time, dense_time, speedup = test_matrix_size_performance_correct(768, 64, "小矩阵")
    # test_results.append(("768×64 (LORO第一步)", speedup))
    
    # # 中小矩阵 (LORO的第二步)  
    # sparse_time, dense_time, speedup = test_matrix_size_performance_correct(64, 3072, "中小矩阵")
    # test_results.append(("64×3072 (LORO第二步)", speedup))
    
    # # 中等矩阵
    # sparse_time, dense_time, speedup = test_matrix_size_performance_correct(768, 256, "中等矩阵")
    # test_results.append(("768×256", speedup))
    
    # # 大矩阵 (原始)
    # sparse_time, dense_time, speedup = test_matrix_size_performance_correct(768, 3072, "大矩阵")
    # test_results.append(("768×3072 (原始)", speedup))
    
    # 新增：测试连续两步操作
    two_step_results = test_two_step_matmul_comparison()
    
    # 汇总结果
    print(f"\n" + "="*60)
    print("📋 正确稀疏创建方法的加速效果汇总")
    print("="*60)
    for name, speedup in test_results:
        status = "✅" if speedup > 1.2 else "⚠️ " if speedup > 0.9 else "❌"
        print(f"{status} {name:20} {speedup:.2f}x")
    
    # 两步操作汇总
    speedup_two_step = two_step_results['speedup']
    status = "✅" if speedup_two_step > 1.2 else "⚠️ " if speedup_two_step > 0.9 else "❌"
    print(f"{status} 连续两步操作:        {speedup_two_step:.2f}x")

if __name__ == "__main__":
    main() 