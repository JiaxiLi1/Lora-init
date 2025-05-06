import torch
import torch.nn as nn
import numpy as np


class LowRankAdapter(nn.Module):
    """
    低秩适配器模块，将线性层重新参数化为 W + B@A.T 形式
    """

    def __init__(self, linear, rank, alpha=1.0, init="xavier"):
        super(LowRankAdapter, self).__init__()
        self.in_dim = linear.in_features
        self.out_dim = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.init = init

        # 保存原始权重并设为不可训练
        self.register_buffer("weight", linear.weight.clone())
        if hasattr(linear, "bias") and linear.bias is not None:
            self.register_buffer("bias", linear.bias.clone())
        else:
            self.bias = None

        # 创建低秩因子
        self.B = nn.Parameter(
            torch.zeros(self.out_dim, rank).to(linear.weight.device),
            requires_grad=True
        )
        self.A = nn.Parameter(
            torch.zeros(self.in_dim, rank).to(linear.weight.device),
            requires_grad=True
        )

        # 初始化低秩因子
        self._init_factors()

    def _init_factors(self):
        """初始化低秩因子"""
        if self.init == "xavier":
            nn.init.xavier_normal_(self.A)
            nn.init.xavier_normal_(self.B)

        self.weight.data -= self.alpha / self.rank * self.B @ self.A.T

        self.B.data = self.B.data.contiguous()
        self.A.data = self.A.data.contiguous()

        print(f"初始化低秩适配器: in_dim={self.in_dim}, out_dim={self.out_dim}, rank={self.rank}, alpha={self.alpha}")

    def forward(self, x):
        """前向传播: y = x @ (W + alpha * B @ A.T).T + bias"""
        # 原始权重输出
        out = x @ self.weight.T

        # 低秩适配调整
        out = out + self.alpha / self.rank * x @ self.A @ self.B.T

        # 添加偏置
        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, rank={self.rank}, alpha={self.alpha}, init={self.init}"


def apply_lowrank_adapter(model, scope, rank, alpha=1.0, init="xavier"):
    """
    为roberta-base模型应用低秩适配器

    Args:
        model: roberta-base模型
        scope: 适配器作用范围，可选"all", "qkv", "qv"
        rank: 低秩因子的秩
        alpha: 缩放因子
        init: 初始化方法

    Returns:
        trainable_params: 可训练参数数量
    """
    module_names_dict = {
        "all": ["query", "key", "value", "dense"],
        "qkv": ["query", "key", "value"],
        "qv": ["query", "value"],
    }
    module_names = module_names_dict.get(scope, ["query", "value"])  # 默认为qv

    print(f"\n为roberta-base模型应用低秩适配器，范围: {scope}, 秩: {rank}, alpha: {alpha}, 初始化: {init}")

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad_(False)

    # 收集低秩适配器参数
    lowrank_params = []

    # 遍历roberta模型的每一层
    for i, layer in enumerate(model.roberta.encoder.layer):
        target_module_dict = {
            "attention.self": layer.attention.self,
            "attention.output": layer.attention.output,
            "intermediate": layer.intermediate,
            "output": layer.output,
        }

        for m_name, module in target_module_dict.items():
            for name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear) and any(n in name for n in module_names):
                    # 替换为低秩适配器
                    low_rank_module = LowRankAdapter(sub_module, rank, alpha, init)
                    setattr(module, name, low_rank_module)

                    # 收集低秩参数
                    lowrank_params.extend([low_rank_module.A, low_rank_module.B])

                    print(
                        f"已替换 layer.{i}.{m_name}.{name}: 原始形状 {sub_module.weight.shape} -> 低秩形状 A:{low_rank_module.A.shape}, B:{low_rank_module.B.shape}")

    # 统计可训练参数
    trainable_param_count = sum(p.numel() for p in lowrank_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_ratio = trainable_param_count / total_param_count

    print(f"\n应用完成!")
    print(f"可训练参数总数: {trainable_param_count:,} ({trainable_param_count / 1e6:.2f}M)")
    print(f"模型总参数: {total_param_count:,} ({total_param_count / 1e6:.2f}M)")
    print(f"可训练参数占比: {trainable_ratio:.2%}\n")

    return trainable_param_count


def get_optimizer_param_groups(model, weight_decay=0.0, lr=0.0002):
    """
    获取优化器参数组，按不同参数类型分组

    Args:
        model: 已应用低秩适配器的模型
        weight_decay: 权重衰减值
        lr_scaler: 学习率缩放因子

    Returns:
        param_groups: 优化器参数组
    """
    # 按参数类型分组
    a_params = []  # A矩阵参数
    b_params = []  # B矩阵参数

    for name, module in model.named_modules():
        if isinstance(module, LowRankAdapter):
            a_params.append(module.A)
            b_params.append(module.B)

    param_groups = [
        {
            "params": a_params,
            "weight_decay": weight_decay,
            "lr": lr,
            "name": "lowrank_A"
        },
        {
            "params": b_params,
            "weight_decay": weight_decay,
            "lr": lr,
            "name": "lowrank_B"
        }
    ]

    return param_groups