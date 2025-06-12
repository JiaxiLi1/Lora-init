#!/usr/bin/env python3

import torch
from loro_torch.sparse_overlay import apply_sparse_overlay_on_loro, SparseOverlayLinear

print('🔍 Testing LORO + 2:4 sparse detection...')

# 模拟LORO应用后的模型结构
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([MockLayer() for _ in range(1)])

class MockLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockAttn()

class MockAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟LORO应用后的模块 - 有weight_in和weight_out
        self.q_proj = torch.nn.Linear(768, 768)
        self.q_proj.weight_in = torch.nn.Parameter(torch.randn(64, 768))
        self.q_proj.weight_out = torch.nn.Parameter(torch.randn(768, 64))

def test_loro_sparse():
    model = MockModel()
    print(f'Original q_proj type: {type(model.layers[0].self_attn.q_proj)}')
    print(f'Has weight_in: {hasattr(model.layers[0].self_attn.q_proj, "weight_in")}')
    print(f'Has weight_out: {hasattr(model.layers[0].self_attn.q_proj, "weight_out")}')

    # 应用sparse overlay
    model = apply_sparse_overlay_on_loro(model, target_modules=['q_proj'])

    print(f'After sparse overlay q_proj type: {type(model.layers[0].self_attn.q_proj)}')
    print(f'Is SparseOverlayLinear: {isinstance(model.layers[0].self_attn.q_proj, SparseOverlayLinear)}')
    
    # 测试forward pass
    if isinstance(model.layers[0].self_attn.q_proj, SparseOverlayLinear):
        print('✅ SparseOverlayLinear correctly applied!')
        
        # 测试flip rate功能
        sparse_module = model.layers[0].self_attn.q_proj
        sparse_module.enable_flip_rate_tracking(True)
        
        # 初始化scales
        sparse_module.init_sparse_scales()
        print(f'Sparse scales initialized: in={sparse_module.sparse_scale_in.item():.4f}, out={sparse_module.sparse_scale_out.item():.4f}')
        
        # 测试前向传播
        x = torch.randn(2, 10, 768)
        try:
            output = sparse_module(x)
            print(f'Forward pass successful: input {x.shape} -> output {output.shape}')
            
            # 测试flip rate计算
            flip_rate, changed, total = sparse_module.calculate_flip_rate()
            print(f'Flip rate calculation: {flip_rate:.4f} ({changed}/{total})')
            
        except Exception as e:
            print(f'❌ Forward pass failed: {e}')
    else:
        print('❌ SparseOverlayLinear was NOT applied!')

if __name__ == '__main__':
    test_loro_sparse() 