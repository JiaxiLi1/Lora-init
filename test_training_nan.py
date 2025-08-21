#!/usr/bin/env python3
"""
测试在实际训练场景中的NaN检测
模拟split_gemm可能产生NaN的情况
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nan_root_cause_detector import NaNRootCauseDetector, analyze_split_gemm_root_cause, computation_tracker


class SimpleLowRankLayer(nn.Module):
    """模拟低秩层"""
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.weight_in = nn.Parameter(torch.randn(in_features, rank))
        self.weight_out = nn.Parameter(torch.randn(out_features, rank))
        
    def forward(self, x):
        # x @ weight_in @ weight_out.T
        intermediate = torch.mm(x.view(-1, x.shape[-1]), self.weight_in)
        output = torch.mm(intermediate, self.weight_out.T)
        return output.view(*x.shape[:-1], -1)


class TestSplitGemmModel(nn.Module):
    """测试模型，模拟split_gemm场景"""
    def __init__(self):
        super().__init__()
        self.layer1 = SimpleLowRankLayer(128, 256, 32)
        self.layer2 = SimpleLowRankLayer(256, 128, 32)
        
    def forward(self, x, simulate_nan=False):
        # 第一个低秩层
        x = self.layer1(x)
        
        # ReLU²激活
        x = F.relu(x)
        x = x * x  # Squared ReLU
        
        if simulate_nan:
            # 模拟split_gemm中的问题：极小的scale factor导致除法问题
            print(f"Before scaling: min={x.min():.6e}, max={x.max():.6e}")
            
            # 模拟错误的scale计算（可能导致0或极小值）
            scale = torch.zeros_like(x)
            scale[0, 0] = 1e-45  # 极小值，可能导致下溢
            
            # 应用缩放 - 这可能产生NaN
            x = x / scale
            print(f"After scaling: has_nan={torch.isnan(x).any().item()}")
        
        # 第二个低秩层
        x = self.layer2(x)
        
        return x


def test_bfloat16_issues():
    """测试bfloat16特有的数值问题"""
    print("\n" + "="*60)
    print("Testing bfloat16 numerical issues")
    print("="*60)
    
    if torch.cuda.is_available():
        device = 'cuda'
        
        # 1. 测试下溢
        print("\n1. Underflow in bfloat16:")
        x = torch.tensor([1e-45], dtype=torch.bfloat16, device=device)
        y = torch.tensor([1e-45], dtype=torch.bfloat16, device=device)
        result = x * y
        print(f"   {x.item():.6e} * {y.item():.6e} = {result.item()}")
        print(f"   Result is zero: {result.item() == 0}")
        
        # 2. 测试混合精度计算
        print("\n2. Mixed precision issues:")
        a = torch.tensor([1e-10], dtype=torch.bfloat16, device=device)
        b = torch.tensor([1e10], dtype=torch.bfloat16, device=device)
        c = torch.tensor([1e-10], dtype=torch.bfloat16, device=device)
        
        # (a * b) * c vs a * (b * c)
        result1 = (a * b) * c
        result2 = a * (b * c)
        print(f"   (a * b) * c = {result1.item():.6e}")
        print(f"   a * (b * c) = {result2.item():.6e}")
        print(f"   Results differ: {result1.item() != result2.item()}")
        
        # 3. 累积误差
        print("\n3. Accumulation errors:")
        x = torch.ones(1000, dtype=torch.bfloat16, device=device) * 1e-4
        sum_result = x.sum()
        expected = 1000 * 1e-4
        print(f"   Sum of 1000 * 1e-4 = {sum_result.item():.6f}")
        print(f"   Expected = {expected:.6f}")
        print(f"   Relative error: {abs(sum_result.item() - expected) / expected * 100:.2f}%")


def test_split_gemm_scenario():
    """测试split_gemm场景中的NaN"""
    print("\n" + "="*60)
    print("Testing Split-GEMM NaN scenario")
    print("="*60)
    
    # 创建测试数据
    batch_size = 4
    seq_len = 128
    hidden_size = 256
    
    # 模拟梯度张量
    dy1 = torch.randn(batch_size * seq_len, hidden_size)
    weight1 = torch.randn(hidden_size, hidden_size * 4)
    
    print(f"\nInput shapes:")
    print(f"  dy1: {dy1.shape}")
    print(f"  weight1: {weight1.shape}")
    
    # 使分析split_gemm
    analyze_split_gemm_root_cause(dy1, weight1, layer_id="test_layer")
    
    # 模拟95/5分割中的问题
    print("\n\nSimulating 95/5 split issues:")
    
    # 1. 创建稀疏特征
    sparse_features = int(0.95 * hidden_size)
    dense_features = hidden_size - sparse_features
    
    print(f"  Sparse features: {sparse_features}")
    print(f"  Dense features: {dense_features}")
    
    # 2. 模拟2:4稀疏化后的极值问题
    dy1_sparse = dy1.clone()
    # 将部分值设为极大值（模拟稀疏化后的scaling问题）
    dy1_sparse[:, :10] *= 1e10
    
    # 3. 计算可能溢出
    try:
        result = torch.mm(dy1_sparse, weight1)
        print(f"  Result shape: {result.shape}")
        print(f"  Has NaN: {torch.isnan(result).any().item()}")
        print(f"  Has Inf: {torch.isinf(result).any().item()}")
        
        if torch.isnan(result).any():
            nan_positions = torch.where(torch.isnan(result))
            print(f"  NaN positions (first 5): {[pos[:5].tolist() for pos in nan_positions]}")
    except Exception as e:
        print(f"  Error during computation: {e}")


def main():
    print("Testing NaN detection in training scenarios")
    print("="*60)
    
    # 测试bfloat16问题
    test_bfloat16_issues()
    
    # 测试split_gemm场景
    test_split_gemm_scenario()
    
    # 测试模型中的NaN
    print("\n" + "="*60)
    print("Testing model with simulated NaN")
    print("="*60)
    
    model = TestSplitGemmModel()
    detector = NaNRootCauseDetector(model)
    
    # 创建输入
    batch_size = 2
    seq_len = 64
    hidden_size = 128
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # 正常前向传播
    print("\n1. Normal forward pass:")
    with torch.no_grad():
        output = model(x, simulate_nan=False)
        print(f"   Output shape: {output.shape}")
        print(f"   Has NaN: {torch.isnan(output).any().item()}")
    
    # 模拟NaN的前向传播
    print("\n2. Forward pass with simulated NaN:")
    computation_tracker.enabled = True
    computation_tracker.operations.clear()
    computation_tracker.first_nan_op = None
    
    detector.patch_operations()
    
    try:
        with torch.no_grad():
            output = model(x, simulate_nan=True)
            print(f"   Output shape: {output.shape}")
            print(f"   Has NaN: {torch.isnan(output).any().item()}")
            
            if computation_tracker.first_nan_op:
                print(f"\n   ✅ Found NaN source: {computation_tracker.first_nan_op['op_name']}")
    finally:
        detector.unpatch_operations()
        computation_tracker.enabled = False


if __name__ == "__main__":
    main()