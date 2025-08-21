#!/usr/bin/env python3
"""
测试NaN根本原因检测器
"""

import torch
import torch.nn as nn
from nan_root_cause_detector import NaNRootCauseDetector, computation_tracker


class TestModel(nn.Module):
    """测试模型，故意产生NaN"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 10)
        
    def forward(self, x, labels=None):
        # 第一层：正常
        x = self.layer1(x)
        print(f"After layer1: has_nan={torch.isnan(x).any().item()}, range=[{x.min():.3f}, {x.max():.3f}]")
        
        # ReLU：正常
        x = torch.relu(x)
        print(f"After ReLU: has_nan={torch.isnan(x).any().item()}, range=[{x.min():.3f}, {x.max():.3f}]")
        
        # 第二层：正常
        x = self.layer2(x)
        print(f"After layer2: has_nan={torch.isnan(x).any().item()}, range=[{x.min():.3f}, {x.max():.3f}]")
        
        # 这里故意制造NaN（除以一个可能为0的值）
        # 创建一个包含0的张量
        denominator = x - x.mean(dim=1, keepdim=True)  # 中心化，可能产生0
        print(f"Denominator min: {denominator.min():.6e}, max: {denominator.max():.6e}")
        print(f"Denominator has zeros: {(denominator == 0).any().item()}")
        
        # 执行除法 - 故意创建NaN
        # 方法：创建一些0/0的情况
        zeros_mask = torch.zeros_like(denominator)
        zeros_mask[0, 0] = 1  # 将第一个元素设为将要除0
        denominator = denominator * (1 - zeros_mask)  # 将选中的位置设为0
        x = x * (1 - zeros_mask)  # 分子也设为0，创建0/0
        x = x / denominator  # 这是产生NaN的地方！
        print(f"After division: has_nan={torch.isnan(x).any().item()}")
        
        # 第三层
        x = self.layer3(x)
        return x


def test_specific_nan_cases():
    """测试特定的NaN情况"""
    print("="*60)
    print("Testing specific NaN cases")
    print("="*60)
    
    # 1. 测试 0/0
    print("\n1. Testing 0/0:")
    a = torch.tensor([0.0, 1.0, 2.0])
    b = torch.tensor([0.0, 1.0, 2.0])
    result = a / b
    print(f"   {a} / {b} = {result}")
    print(f"   Has NaN: {torch.isnan(result).any().item()}")
    
    # 2. 测试 inf - inf
    print("\n2. Testing inf - inf:")
    a = torch.tensor([float('inf'), 1.0])
    b = torch.tensor([float('inf'), 2.0])
    result = a - b
    print(f"   {a} - {b} = {result}")
    print(f"   Has NaN: {torch.isnan(result).any().item()}")
    
    # 3. 测试 sqrt(negative)
    print("\n3. Testing sqrt(negative):")
    a = torch.tensor([-1.0, 0.0, 1.0])
    result = torch.sqrt(a)
    print(f"   sqrt({a}) = {result}")
    print(f"   Has NaN: {torch.isnan(result).any().item()}")
    
    # 4. 测试大数值溢出
    print("\n4. Testing overflow in bfloat16:")
    if torch.cuda.is_available():
        a = torch.tensor([1e38], dtype=torch.bfloat16, device='cuda')
        b = torch.tensor([1e38], dtype=torch.bfloat16, device='cuda')
        result = a * b
        print(f"   {a.item():.2e} * {b.item():.2e} = {result.item()}")
        print(f"   Is inf: {torch.isinf(result).any().item()}")


def main():
    print("Testing NaN Root Cause Detector")
    print("="*60)
    
    # 测试特定情况
    test_specific_nan_cases()
    
    # 测试模型中的NaN检测
    print("\n" + "="*60)
    print("Testing NaN detection in model")
    print("="*60)
    
    model = TestModel()
    detector = NaNRootCauseDetector(model)
    
    # 创建输入
    batch = {'x': torch.randn(2, 10)}
    
    # 分析前向传播
    print("\nRunning forward pass with root cause detection:")
    print("-"*60)
    
    # 直接运行看输出
    with torch.no_grad():
        output = model(batch['x'])
        print(f"\nFinal output has NaN: {torch.isnan(output).any().item()}")
    
    # 使用检测器分析
    print("\n" + "="*60)
    print("Now with Root Cause Detector:")
    print("="*60)
    
    first_nan_op = detector.analyze_forward_pass({'x': batch['x']}, None)
    
    if first_nan_op:
        print("\n✅ Successfully identified the operation that created NaN!")
    else:
        print("\n⚠️ Could not identify the exact operation")


if __name__ == "__main__":
    main()