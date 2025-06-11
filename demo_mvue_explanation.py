import torch
import torch.nn as nn

def simple_2to4_sparse(x):
    """简单的2:4稀疏操作 - 每4个元素保留2个最大的"""
    x_reshaped = x.view(-1, 4)
    
    # 找到每组4个中最大的2个
    _, indices = torch.topk(torch.abs(x_reshaped), k=2, dim=1)
    
    # 创建mask
    mask = torch.zeros_like(x_reshaped, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    
    # 应用稀疏
    sparse_result = x_reshaped * mask.float()
    return sparse_result.view(x.shape), mask

def simple_mvue(x, mask):
    """简化的MVUE操作"""
    x_reshaped = x.view(-1, 4)
    mask_reshaped = mask.view(-1, 4)
    
    # 计算每个元素被选中的概率（简化版）
    abs_vals = torch.abs(x_reshaped) + 1e-7
    sum_abs = abs_vals.sum(dim=1, keepdim=True)
    probs = abs_vals / sum_abs  # 基于大小的选择概率
    
    # MVUE校正：除以选择概率
    mvue_result = x_reshaped / torch.clamp(probs, min=0.1, max=1.0)
    
    return mvue_result.view(x.shape)

print("=" * 60)
print("🎯 MVUE 在 2:4 稀疏训练中的作用演示")
print("=" * 60)

# === 步骤1：模拟前向传播 ===
print("\n📈 步骤1：前向传播")
print("-" * 30)

# 原始权重（密集）
weight_dense = torch.tensor([[1.0, 0.5, 1.8, 0.3],  # 第一行
                           [0.8, 1.2, 0.4, 1.5]])   # 第二行
print(f"原始密集权重:\n{weight_dense}")

# 输入激活
input_activation = torch.tensor([[2.0, 1.0, 1.5, 0.8]])  # batch_size=1
print(f"输入激活: {input_activation}")

# 标准密集前向传播
output_dense = input_activation @ weight_dense.t()
print(f"密集前向结果: {output_dense}")

# 2:4稀疏前向传播
weight_sparse, weight_mask = simple_2to4_sparse(weight_dense)
print(f"\n稀疏权重 (2:4):\n{weight_sparse}")
print(f"权重掩码:\n{weight_mask}")

output_sparse = input_activation @ weight_sparse.t()
print(f"稀疏前向结果: {output_sparse}")
print(f"前向差异: {output_sparse - output_dense}")

# === 步骤2：模拟反向传播 ===
print("\n📉 步骤2：反向传播 - 这里就需要MVUE了！")
print("-" * 50)

# 假设从后面层传来的梯度
grad_output = torch.tensor([[0.5, 0.3]])  # 对应两个输出的梯度
print(f"从后面传来的梯度: {grad_output}")

print("\n❌ 如果直接用稀疏操作计算梯度：")
# 直接用稀疏权重计算输入梯度（这是有偏的！）
grad_input_biased = grad_output @ weight_sparse  # 稀疏权重
print(f"有偏的输入梯度: {grad_input_biased}")

# 用密集权重计算输入梯度（正确的）
grad_input_correct = grad_output @ weight_dense  # 密集权重
print(f"正确的输入梯度: {grad_input_correct}")
print(f"偏差: {grad_input_biased - grad_input_correct}")

print("\n✅ 使用MVUE校正：")
# 步骤2a：对输入激活应用MVUE
input_mvue = simple_mvue(input_activation, torch.ones_like(input_activation, dtype=torch.bool))
print(f"MVUE校正后的输入: {input_mvue}")

# 步骤2b：对梯度也应用MVUE  
grad_output_mvue = simple_mvue(grad_output, torch.ones_like(grad_output, dtype=torch.bool))
print(f"MVUE校正后的梯度: {grad_output_mvue}")

# 步骤2c：用校正后的值计算权重梯度
grad_weight_mvue = input_mvue.t() @ grad_output_mvue
print(f"MVUE校正后的权重梯度:\n{grad_weight_mvue}")

# 对比：用密集值计算的正确权重梯度
grad_weight_correct = input_activation.t() @ grad_output
print(f"正确的密集权重梯度:\n{grad_weight_correct}")
print(f"MVUE误差:\n{grad_weight_mvue - grad_weight_correct}")

print("\n" + "=" * 60)
print("🔍 关键理解")
print("=" * 60)
print("1. MVUE的输入：前向传播中的激活值（不是梯度！）")
print("2. MVUE的作用：补偿因稀疏操作引入的偏差")
print("3. MVUE的原理：'如果这个值有概率p被保留，那么要乘以1/p来补偿'")
print("4. 最终目标：让优化器收到的梯度接近密集训练的梯度")

print("\n🎯 为什么需要MVUE？")
print("- 前向：用稀疏权重计算（快速）") 
print("- 反向：需要无偏梯度（准确）")
print("- MVUE：桥接两者，确保训练收敛性") 