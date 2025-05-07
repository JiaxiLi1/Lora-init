import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class HybridSparseAdapter(nn.Module):

    def __init__(self, linear, rank, sparsity=0.05, gamma=0.5, sparse_method="svd",
                 sparse_svd_rank=None, alpha=1.0, init="svd", cola_silu=False,
                 cola_init=False, svd_inverse=False):
        super(HybridSparseAdapter, self).__init__()
        self.in_dim = linear.in_features
        self.out_dim = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.init = init
        self.sparsity = sparsity
        self.sparse_method = sparse_method
        self.sparse_svd_rank = sparse_svd_rank
        self.cola_silu = cola_silu
        self.cola_init = cola_init
        self.svd_inverse = svd_inverse

        # 保存原始权重用于初始化
        self.register_buffer('original_weight', linear.weight.clone())

        # 创建低秩因子
        self.matrix_A = nn.Parameter(
            torch.zeros(self.in_dim, self.rank).to(linear.weight.device),
            requires_grad=True
        )
        self.matrix_B = nn.Parameter(
            torch.zeros(self.rank, self.out_dim).to(linear.weight.device),
            requires_grad=True
        )

        # 稀疏组件参数
        self.values = None  # 将在initialize_mask中初始化
        self.register_buffer('selected_col_indices', None)  # 将在initialize_mask中设置


        self.register_buffer('gamma', torch.tensor(gamma))

        # 偏置参数
        if hasattr(linear, "bias") and linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.clone())
        else:
            self.bias = None

        if self.cola_silu:
            self.activation = nn.SiLU()

        if self.cola_init:
            self.initialize_cola()
        else:
            self.apply_svd_init(linear.weight.data)

        # self.original_weight.data -= self.alpha / self.rank * self.matrix_B.t() @ self.matrix_A.t()

        self.initialize_mask()

    def initialize_cola(self):
        """
        使用CoLA初始化方法初始化低秩矩阵
        """
        with torch.no_grad():
            target_sdv = (self.in_dim + self.out_dim) ** (-1 / 2)
            scale_factor = self.rank ** (-1 / 4) * target_sdv ** (1 / 2)
            self.matrix_A.data.copy_(torch.randn_like(self.matrix_A) * scale_factor)
            self.matrix_B.data.copy_(torch.randn_like(self.matrix_B) * scale_factor)

    def apply_svd_init(self, weight_matrix):
        """
        使用SVD分解初始化低秩矩阵
        """
        # 转置权重矩阵以匹配维度
        weight_matrix = weight_matrix.t()

        # 执行SVD
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)

        # 选择top-k组件
        U_r = U[:, :self.rank]  # shape: [in_dim, rank]
        S_r = S[:self.rank]  # shape: [rank]
        V_r = Vh[:self.rank, :].t()  # shape: [out_dim, rank]

        # 计算matrix_A和matrix_B
        S_sqrt = torch.sqrt(S_r)
        with torch.no_grad():
            A = U_r * S_sqrt.view(1, -1)
            B = (V_r * S_sqrt.view(1, -1)).t()

            self.matrix_A.data.copy_(A)
            self.matrix_B.data.copy_(B)

    def initialize_mask(self):
        """基于选定方法初始化稀疏掩码"""
        num_cols_to_keep = max(1, int(self.sparsity * self.in_dim))

        if self.sparse_method == "svd":
            weight_float = self.original_weight.to(torch.float32)
            U, S, Vh = torch.linalg.svd(weight_float.t(), full_matrices=False)

            if self.svd_inverse:
                k = min(self.sparse_svd_rank or self.rank, len(S) - self.rank)
                start_idx = len(S) - k
                end_idx = len(S)
            else:
                k = min(self.sparse_svd_rank or self.rank, len(S) - self.rank)
                start_idx = self.rank
                end_idx = start_idx + k

            U_k = U[:, start_idx:end_idx]
            S_k = S[start_idx:end_idx]
            Vh_k = Vh[start_idx:end_idx, :]

            reconstructed = U_k @ torch.diag(S_k) @ Vh_k
            col_norms = torch.norm(reconstructed.t(), dim=0)  # [in_dim]
            _, topk_cols = torch.topk(col_norms, num_cols_to_keep, largest=True)
        elif self.sparse_method == "random":
            # 随机选择列
            topk_cols = torch.randperm(self.in_dim)[:num_cols_to_keep]

        topk_cols, _ = torch.sort(topk_cols)

        # 设置选定的列索引
        self.register_buffer("selected_col_indices", topk_cols)

        # 初始化稀疏组件的值
        sparse_weight = self.original_weight[:, topk_cols]
        self.values = nn.Parameter(sparse_weight.clone())

    def forward(self, x):
        out = F.linear(x, self.original_weight)

        if self.cola_silu:
            # 在A和B矩阵之间使用SiLU激活
            temp = torch.matmul(x, self.matrix_A)
            temp = self.activation(temp)
            low_rank_output = torch.matmul(temp, self.matrix_B)
        else:
            # 标准低秩计算
            temp = torch.matmul(x, self.matrix_A)
            low_rank_output = torch.matmul(temp, self.matrix_B)

        # 稀疏组件
        input_selected = x[:, :, self.selected_col_indices]
        sparse_output = F.linear(input_selected, self.values)

        # 使用gamma参数组合
        combined_adapter_output = self.gamma * low_rank_output + (1 - self.gamma) * sparse_output
        # combined_adapter_output = low_rank_output
        out = out + self.alpha / self.rank * combined_adapter_output

        # 如果存在则添加偏置
        if self.bias is not None:
            out = out + self.bias

        return out


def apply_hybrid_adapter(model, scope="qv", rank=8, sparsity=0.05, gamma=0.5,
                         sparse_method="svd", sparse_svd_rank=None, alpha=1.0,
                         cola_silu=False, cola_init=False, svd_inverse=False):
    """
    为Roberta或其他Transformer模型应用混合低秩+稀疏适配器

    Args:
        model: 要修改的模型
        scope: 要适配的模块 ("all", "qkv", "qv"等)
        rank: 低秩组件的秩
        sparsity: 稀疏度（保留列的百分比）
        gamma: 混合参数的初始值
        sparse_method: 选择稀疏列的方法 ("svd", "random")
        sparse_svd_rank: 基于SVD的掩码初始化要使用的秩
        alpha: 低秩组件的缩放因子
        gamma_trainable: 是否将gamma设为可训练参数
        sparse_only: 是否仅使用稀疏组件
        cola_silu: 是否在低秩组件中使用SiLU激活

    Returns:
        trainable_param_count: 可训练参数的数量
    """
    # 根据scope定义目标模块
    module_names_dict = {
        "all": ["query", "key", "value", "dense"],
        "alll": ["query", "key", "value", "dense", "intermediate", "output"],  # 所有注意力和MLP层
        "qkv": ["query", "key", "value"],
        "qv": ["query", "value"],
        "qk": ["query", "key"],  # 新增：q和k层
        # "kv": ["key", "value"],  # 新增：k和v层
        # "q": ["query"],  # 新增：仅q层
        # "k": ["key"],  # 新增：仅k层
        # "v": ["value"],  # 新增：仅v层
        "attn": ["query", "key", "value", "dense"],  # 新增：所有注意力层
        "mlp": ["intermediate", "output"],  # 新增：所有MLP层
        "self_attn": ["query", "key", "value"],  # 新增：自注意力层
        "ff": ["dense", "intermediate", "output"]  # 新增：前馈层
    }

    module_names = module_names_dict.get(scope, ["query", "value"])  # 默认为qv

    print(f"\n正在应用混合适配器到模型, scope: {scope}, rank: {rank}, sparsity: {sparsity}, "
          f"gamma: {gamma}, sparse_method: {sparse_method}, cola_init: {cola_init}, svd_inverse: {svd_inverse}")

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad_(False)

    # 收集适配器参数
    adapter_params = []

    # 处理不同模型类型（Roberta vs LLaMA）
    if hasattr(model, "roberta"):
        # RoBERTa模型
        for i, layer in enumerate(model.roberta.encoder.layer):
            target_module_dict = {
                "attention.self": layer.attention.self,
                "attention.output": layer.attention.output,
                "intermediate": layer.intermediate,
                "output": layer.output,
            }

            for m_name, module in target_module_dict.items():
                for name, sub_module in module.named_children():
                    # 检查是否为MLP层，并处理"intermediate"和"output"的情况
                    if scope in ["mlp", "alll", "ff"] and m_name in ["intermediate", "output"] and isinstance(
                            sub_module, nn.Linear):
                        hybrid_module = HybridSparseAdapter(
                            sub_module, rank, sparsity, gamma, sparse_method,
                            sparse_svd_rank, alpha, init="svd",
                            cola_silu=cola_silu, cola_init=cola_init,
                            svd_inverse=svd_inverse
                        )
                        setattr(module, name, hybrid_module)
                        adapter_params.extend([p for p in hybrid_module.parameters() if p.requires_grad])
                        print(f"已替换 layer.{i}.{m_name}.{name}: "
                              f"原始形状 {sub_module.weight.shape} -> "
                              f"低秩 A:{hybrid_module.matrix_A.shape}, "
                              f"B:{hybrid_module.matrix_B.shape}, "
                              f"稀疏列数: {len(hybrid_module.selected_col_indices)}")
                    # 处理注意力层部分
                    elif isinstance(sub_module, nn.Linear) and any(n in name for n in module_names):
                        hybrid_module = HybridSparseAdapter(
                            sub_module, rank, sparsity, gamma, sparse_method,
                            sparse_svd_rank, alpha, init="svd",
                            cola_silu=cola_silu, cola_init=cola_init,
                            svd_inverse=svd_inverse
                        )
                        setattr(module, name, hybrid_module)
                        adapter_params.extend([p for p in hybrid_module.parameters() if p.requires_grad])
                        print(f"已替换 layer.{i}.{m_name}.{name}: "
                              f"原始形状 {sub_module.weight.shape} -> "
                              f"低秩 A:{hybrid_module.matrix_A.shape}, "
                              f"B:{hybrid_module.matrix_B.shape}, "
                              f"稀疏列数: {len(hybrid_module.selected_col_indices)}")

    # 统计信息
    trainable_param_count = sum(p.numel() for p in adapter_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_ratio = trainable_param_count / total_param_count

    print(f"\n适配器应用完成!")
    print(f"可训练参数: {trainable_param_count:,} ({trainable_param_count / 1e6:.2f}M)")
    print(f"总参数: {total_param_count:,} ({total_param_count / 1e6:.2f}M)")
    print(f"可训练比例: {trainable_ratio:.2%}\n")

    return trainable_param_count


def get_optimizer_param_groups(model, weight_decay=0.0, lr=0.0002, gamma_lr=0.0001):


    a_params = []  # 低秩A矩阵
    b_params = []  # 低秩B矩阵
    sparse_params = []  # 稀疏值

    for name, module in model.named_modules():
        if isinstance(module, HybridSparseAdapter):
            a_params.append(module.matrix_A)
            b_params.append(module.matrix_B)
            sparse_params.append(module.values)

    param_groups = [
        {
            "params": a_params + b_params + sparse_params,
            "weight_decay": weight_decay,
            "lr": lr,
            "name": "adapter_params"
        }
    ]

    return param_groups