"""
NaN Root Cause Detector - 精确定位NaN产生的根本原因
追踪每个计算步骤，找出从正常值到NaN的转变点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import traceback
import functools


class ComputationTracker:
    """追踪所有计算操作，找出NaN的确切来源"""
    
    def __init__(self):
        self.operations = []
        self.first_nan_op = None
        self.enabled = False
        
    def track_operation(self, op_name: str, inputs: Dict, outputs: Any, extra_info: Dict = None):
        """记录一个计算操作"""
        if not self.enabled:
            return
            
        # 检查输入
        input_has_nan = False
        input_has_inf = False
        for key, val in inputs.items():
            if torch.is_tensor(val):
                if torch.isnan(val).any():
                    input_has_nan = True
                if torch.isinf(val).any():
                    input_has_inf = True
        
        # 检查输出
        output_has_nan = False
        output_has_inf = False
        if torch.is_tensor(outputs):
            output_has_nan = torch.isnan(outputs).any().item()
            output_has_inf = torch.isinf(outputs).any().item()
        
        # 记录操作
        op_record = {
            'op_name': op_name,
            'input_has_nan': input_has_nan,
            'input_has_inf': input_has_inf,
            'output_has_nan': output_has_nan,
            'output_has_inf': output_has_inf,
            'extra_info': extra_info or {}
        }
        
        self.operations.append(op_record)
        
        # 检测NaN的首次出现（输入正常但输出有NaN）
        if not input_has_nan and not input_has_inf and (output_has_nan or output_has_inf):
            if self.first_nan_op is None:
                self.first_nan_op = {
                    'op_name': op_name,
                    'inputs': {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()},
                    'outputs': outputs.clone() if torch.is_tensor(outputs) else outputs,
                    'extra_info': extra_info
                }
                self._analyze_nan_cause(op_name, inputs, outputs)
    
    def _analyze_nan_cause(self, op_name: str, inputs: Dict, outputs: Any):
        """分析NaN产生的具体原因"""
        print("\n" + "="*80)
        print(f"🔴 NaN ROOT CAUSE DETECTED in operation: {op_name}")
        print("="*80)
        
        # 分析不同类型的操作
        if "matmul" in op_name.lower() or "mm" in op_name.lower():
            self._analyze_matmul_nan(inputs, outputs)
        elif "div" in op_name.lower() or "/" in op_name:
            self._analyze_division_nan(inputs, outputs)
        elif "sqrt" in op_name.lower():
            self._analyze_sqrt_nan(inputs, outputs)
        elif "log" in op_name.lower():
            self._analyze_log_nan(inputs, outputs)
        elif "norm" in op_name.lower():
            self._analyze_norm_nan(inputs, outputs)
        elif "softmax" in op_name.lower():
            self._analyze_softmax_nan(inputs, outputs)
        else:
            self._analyze_general_nan(inputs, outputs)
    
    def _analyze_matmul_nan(self, inputs: Dict, outputs):
        """分析矩阵乘法中的NaN"""
        print("\n📊 Matrix Multiplication NaN Analysis:")
        
        if 'a' in inputs and 'b' in inputs:
            a, b = inputs['a'], inputs['b']
            
            # 检查极值
            print(f"  Input A: shape={list(a.shape)}, dtype={a.dtype}")
            print(f"    min={a.min().item():.6e}, max={a.max().item():.6e}")
            print(f"    mean={a.mean().item():.6e}, std={a.std().item():.6e}")
            
            print(f"  Input B: shape={list(b.shape)}, dtype={b.dtype}")
            print(f"    min={b.min().item():.6e}, max={b.max().item():.6e}")
            print(f"    mean={b.mean().item():.6e}, std={b.std().item():.6e}")
            
            # 检查是否有极大值导致溢出
            if a.dtype in [torch.float16, torch.bfloat16]:
                max_val = 65504 if a.dtype == torch.float16 else 3.39e38
                if a.abs().max() > max_val * 0.1 or b.abs().max() > max_val * 0.1:
                    print(f"  ⚠️ Values approaching dtype limit ({max_val:.2e})")
            
            # 检查累积效应
            partial_result = torch.mm(a[:min(10, a.shape[0])], b[:, :min(10, b.shape[1])])
            if torch.isnan(partial_result).any():
                print("  ⚠️ NaN appears even in small partial computation")
                
                # 找出具体哪些元素相乘导致NaN
                for i in range(min(5, a.shape[0])):
                    for j in range(min(5, b.shape[1])):
                        dot_product = (a[i] * b[:, j]).sum()
                        if torch.isnan(dot_product):
                            print(f"    Row {i} × Col {j} = NaN")
                            # 检查具体哪个乘积出问题
                            products = a[i] * b[:, j]
                            nan_indices = torch.where(torch.isnan(products))[0]
                            if len(nan_indices) > 0:
                                idx = nan_indices[0].item()
                                print(f"      a[{i},{idx}]={a[i,idx].item():.6e} × b[{idx},{j}]={b[idx,j].item():.6e} = NaN")
    
    def _analyze_division_nan(self, inputs: Dict, outputs):
        """分析除法中的NaN"""
        print("\n➗ Division NaN Analysis:")
        
        if 'numerator' in inputs and 'denominator' in inputs:
            num = inputs['numerator']
            denom = inputs['denominator']
            
            # 检查除零
            zero_mask = (denom == 0)
            if zero_mask.any():
                print(f"  ⚠️ Division by zero detected!")
                print(f"    Number of zeros in denominator: {zero_mask.sum().item()}")
                
                # 找出0/0的情况（产生NaN）
                zero_over_zero = (num == 0) & (denom == 0)
                if zero_over_zero.any():
                    print(f"    0/0 cases (NaN): {zero_over_zero.sum().item()}")
                    # 显示位置
                    indices = torch.where(zero_over_zero)
                    print(f"    First few 0/0 locations: {[idx[:5].tolist() for idx in indices]}")
            
            # 检查极小值
            small_threshold = 1e-30
            small_denom = (denom.abs() < small_threshold) & (denom != 0)
            if small_denom.any():
                print(f"  ⚠️ Very small denominators detected (< {small_threshold}):")
                print(f"    Count: {small_denom.sum().item()}")
                print(f"    Min abs denominator: {denom[denom != 0].abs().min().item():.6e}")
    
    def _analyze_sqrt_nan(self, inputs: Dict, outputs):
        """分析平方根中的NaN"""
        print("\n√ Square Root NaN Analysis:")
        
        if 'x' in inputs:
            x = inputs['x']
            negative_mask = (x < 0)
            if negative_mask.any():
                print(f"  ⚠️ Square root of negative numbers!")
                print(f"    Count of negative inputs: {negative_mask.sum().item()}")
                print(f"    Min value: {x.min().item():.6e}")
                
                # 显示一些负值
                neg_values = x[negative_mask]
                print(f"    First few negative values: {neg_values[:10].tolist()}")
    
    def _analyze_log_nan(self, inputs: Dict, outputs):
        """分析对数中的NaN"""
        print("\n㏒ Logarithm NaN Analysis:")
        
        if 'x' in inputs:
            x = inputs['x']
            non_positive_mask = (x <= 0)
            if non_positive_mask.any():
                print(f"  ⚠️ Log of non-positive numbers!")
                print(f"    Count of x <= 0: {non_positive_mask.sum().item()}")
                print(f"    Min value: {x.min().item():.6e}")
                
                # 区分负数和零
                negative_mask = (x < 0)
                zero_mask = (x == 0)
                if negative_mask.any():
                    print(f"    Negative values: {negative_mask.sum().item()}")
                if zero_mask.any():
                    print(f"    Zero values: {zero_mask.sum().item()}")
    
    def _analyze_norm_nan(self, inputs: Dict, outputs):
        """分析归一化中的NaN"""
        print("\n📏 Normalization NaN Analysis:")
        
        if 'x' in inputs:
            x = inputs['x']
            
            # 计算均值和方差
            mean = x.mean()
            var = x.var()
            
            print(f"  Input stats:")
            print(f"    Mean: {mean.item():.6e}")
            print(f"    Variance: {var.item():.6e}")
            print(f"    Std: {var.sqrt().item():.6e}")
            
            # 检查方差是否为零（导致除零）
            if var == 0:
                print(f"  ⚠️ Zero variance detected! All values are identical.")
                print(f"    Unique values: {x.unique()[:5].tolist()}")
            elif var < 1e-10:
                print(f"  ⚠️ Very small variance ({var.item():.6e})")
            
            # 检查是否有极值
            if x.abs().max() > 1e10:
                print(f"  ⚠️ Large values detected: max abs = {x.abs().max().item():.6e}")
    
    def _analyze_softmax_nan(self, inputs: Dict, outputs):
        """分析Softmax中的NaN"""
        print("\n🔄 Softmax NaN Analysis:")
        
        if 'x' in inputs:
            x = inputs['x']
            
            # 检查输入的极值
            print(f"  Input range: [{x.min().item():.6e}, {x.max().item():.6e}]")
            
            # 检查是否有极大值导致exp溢出
            max_val = x.max()
            if max_val > 80:  # exp(80) 已经很大
                print(f"  ⚠️ Large input values that may cause exp() overflow")
                print(f"    Max value: {max_val.item():.6e}")
                print(f"    exp(max) would be: exp({max_val.item():.2f})")
            
            # 检查是否所有值都是-inf（导致0/0）
            if torch.isinf(x).all():
                print(f"  ⚠️ All values are infinite!")
    
    def _analyze_general_nan(self, inputs: Dict, outputs):
        """通用NaN分析"""
        print("\n🔍 General Operation NaN Analysis:")
        
        for key, val in inputs.items():
            if torch.is_tensor(val):
                print(f"  Input '{key}':")
                print(f"    Shape: {list(val.shape)}, dtype: {val.dtype}")
                if not torch.isnan(val).any() and not torch.isinf(val).any():
                    print(f"    Range: [{val.min().item():.6e}, {val.max().item():.6e}]")
                    print(f"    Mean: {val.mean().item():.6e}, Std: {val.std().item():.6e}")
                    
                    # 检查特殊值
                    if (val == 0).any():
                        print(f"    Contains zeros: {(val == 0).sum().item()}")
                    if val.abs().max() > 1e10:
                        print(f"    Contains large values (> 1e10)")
                    if val.abs().min() < 1e-10:
                        print(f"    Contains very small values (< 1e-10)")


# 全局追踪器实例
computation_tracker = ComputationTracker()


def track_computation(op_name: str):
    """装饰器：追踪函数中的计算"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not computation_tracker.enabled:
                return func(*args, **kwargs)
            
            # 记录输入
            inputs = {}
            for i, arg in enumerate(args):
                if torch.is_tensor(arg):
                    inputs[f'arg_{i}'] = arg
            for key, val in kwargs.items():
                if torch.is_tensor(val):
                    inputs[key] = val
            
            # 执行计算
            result = func(*args, **kwargs)
            
            # 记录操作
            computation_tracker.track_operation(op_name, inputs, result)
            
            return result
        return wrapper
    return decorator


class NaNRootCauseDetector:
    """NaN根本原因检测器 - 集成到模型中"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_functions = {}
        self.hooked = False
        
    def patch_operations(self):
        """替换PyTorch操作以追踪计算"""
        if self.hooked:
            return
            
        # 保存原始函数
        self.original_functions['matmul'] = torch.matmul
        self.original_functions['mm'] = torch.mm
        self.original_functions['div'] = torch.div
        self.original_functions['sqrt'] = torch.sqrt
        self.original_functions['log'] = torch.log
        self.original_functions['softmax'] = F.softmax
        
        # 替换为追踪版本
        def tracked_matmul(a, b):
            if not computation_tracker.enabled:
                return self.original_functions['matmul'](a, b)
            
            result = self.original_functions['matmul'](a, b)
            computation_tracker.track_operation('torch.matmul', {'a': a, 'b': b}, result)
            return result
        
        def tracked_mm(a, b):
            if not computation_tracker.enabled:
                return self.original_functions['mm'](a, b)
            
            result = self.original_functions['mm'](a, b)
            computation_tracker.track_operation('torch.mm', {'a': a, 'b': b}, result)
            return result
        
        def tracked_div(numerator, denominator):
            if not computation_tracker.enabled:
                return self.original_functions['div'](numerator, denominator)
            
            result = self.original_functions['div'](numerator, denominator)
            computation_tracker.track_operation('torch.div', 
                {'numerator': numerator, 'denominator': denominator}, result)
            return result
        
        # 保存原始的 __truediv__ 方法
        self.original_functions['tensor_div'] = torch.Tensor.__truediv__
        original_tensor_div = self.original_functions['tensor_div']  # 保存引用
        
        # 创建追踪版本的 __truediv__
        def tracked_tensor_div(self_tensor, other):
            if not computation_tracker.enabled:
                return original_tensor_div(self_tensor, other)
            
            result = original_tensor_div(self_tensor, other)  # 使用保存的引用
            computation_tracker.track_operation('Tensor.__truediv__', 
                {'numerator': self_tensor, 'denominator': other}, result)
            return result
        
        torch.matmul = tracked_matmul
        torch.mm = tracked_mm
        torch.div = tracked_div
        torch.Tensor.__truediv__ = tracked_tensor_div
        
        self.hooked = True
        
    def unpatch_operations(self):
        """恢复原始PyTorch操作"""
        if not self.hooked:
            return
            
        torch.matmul = self.original_functions['matmul']
        torch.mm = self.original_functions['mm']
        torch.div = self.original_functions['div']
        if 'tensor_div' in self.original_functions:
            torch.Tensor.__truediv__ = self.original_functions['tensor_div']
        
        self.hooked = False
    
    def analyze_forward_pass(self, batch, labels):
        """分析前向传播中的NaN根本原因"""
        # 启用追踪
        computation_tracker.enabled = True
        computation_tracker.operations.clear()
        computation_tracker.first_nan_op = None
        
        # Patch操作
        self.patch_operations()
        
        try:
            # 运行前向传播
            with torch.no_grad():
                output = self.model(**batch, labels=labels)
                loss = output.loss if hasattr(output, 'loss') else output
            
            # 检查结果
            has_nan = torch.isnan(loss).any().item() if torch.is_tensor(loss) else False
            if has_nan:
                print("\n" + "="*80)
                print("NaN DETECTED IN LOSS")
                print("="*80)
                
                if computation_tracker.first_nan_op:
                    print("\n🎯 Found the exact operation that created NaN!")
                    op = computation_tracker.first_nan_op
                    print(f"Operation: {op['op_name']}")
                    
                    # 显示操作统计
                    print(f"\nTotal operations tracked: {len(computation_tracker.operations)}")
                    nan_ops = sum(1 for op in computation_tracker.operations if op['output_has_nan'])
                    print(f"Operations with NaN output: {nan_ops}")
                else:
                    print("\n⚠️ NaN was present from the beginning or in untracked operations")
                    
        finally:
            # 恢复原始操作
            self.unpatch_operations()
            computation_tracker.enabled = False
        
        return computation_tracker.first_nan_op


def analyze_split_gemm_root_cause(dy1, weight1, layer_id=None):
    """深入分析split_gemm中NaN的根本原因"""
    print("\n" + "="*80)
    print("SPLIT-GEMM ROOT CAUSE ANALYSIS")
    print("="*80)
    
    # 1. 检查输入
    print("\n1. Input Tensors:")
    print(f"   dy1: shape={list(dy1.shape)}, dtype={dy1.dtype}")
    if not torch.isnan(dy1).any():
        print(f"     range=[{dy1.min():.6e}, {dy1.max():.6e}]")
        print(f"     mean={dy1.mean():.6e}, std={dy1.std():.6e}")
    else:
        print(f"     ⚠️ Contains NaN!")
        
    print(f"   weight1: shape={list(weight1.shape)}, dtype={weight1.dtype}")
    if not torch.isnan(weight1).any():
        print(f"     range=[{weight1.min():.6e}, {weight1.max():.6e}]")
        print(f"     mean={weight1.mean():.6e}, std={weight1.std():.6e}")
    else:
        print(f"     ⚠️ Contains NaN!")
    
    # 2. 分析稀疏性计算
    print("\n2. Sparsity Calculation:")
    feature_sparsity = torch.mean((dy1 != 0).float(), dim=0)
    print(f"   Feature sparsity shape: {list(feature_sparsity.shape)}")
    print(f"   Sparsity range: [{feature_sparsity.min():.4f}, {feature_sparsity.max():.4f}]")
    
    # 3. 模拟95/5分割
    print("\n3. 95/5 Split Simulation:")
    num_features = feature_sparsity.shape[0]
    num_sparse = int(0.95 * num_features)
    num_dense = num_features - num_sparse
    
    print(f"   Total features: {num_features}")
    print(f"   Sparse features (95%): {num_sparse}")
    print(f"   Dense features (5%): {num_dense}")
    
    # 4. 检查2:4稀疏化
    print("\n4. 2:4 Sparsification Check:")
    # 模拟对dy1进行2:4稀疏化
    dy1_abs = dy1.abs()
    batch_size = dy1.shape[0]
    
    for i in range(min(3, batch_size)):  # 检查前3个样本
        row = dy1[i]
        # 对每4个元素进行2:4稀疏化
        for j in range(0, row.shape[0] - 3, 4):
            group = row[j:j+4]
            sorted_indices = torch.argsort(group.abs(), descending=True)
            # 保留最大的2个，其他置零
            mask = torch.zeros_like(group, dtype=torch.bool)
            mask[sorted_indices[:2]] = True
            
            # 检查是否会产生数值问题
            if group[mask].abs().max() > 1e10:
                print(f"   ⚠️ Sample {i}, position {j}: Large values after sparsification")
                print(f"      Values: {group[mask].tolist()}")
    
    # 5. 检查矩阵乘法
    print("\n5. Matrix Multiplication Check:")
    try:
        # 尝试小规模计算
        small_dy1 = dy1[:min(10, dy1.shape[0]), :min(10, dy1.shape[1])]
        small_weight = weight1[:min(10, weight1.shape[0]), :min(10, weight1.shape[1])]
        small_result = torch.mm(small_dy1, small_weight)
        
        if torch.isnan(small_result).any():
            print("   ⚠️ NaN appears even in small-scale computation!")
            # 找出具体位置
            nan_positions = torch.where(torch.isnan(small_result))
            if len(nan_positions[0]) > 0:
                i, j = nan_positions[0][0].item(), nan_positions[1][0].item()
                print(f"   First NaN at result[{i},{j}]")
                
                # 计算这个元素
                dot_product = (small_dy1[i] * small_weight[:, j]).sum()
                print(f"   Dot product = {dot_product}")
                
                # 检查具体哪个乘积有问题
                products = small_dy1[i] * small_weight[:, j]
                for k, prod in enumerate(products):
                    if torch.isnan(prod) or torch.isinf(prod):
                        print(f"     dy1[{i},{k}]={small_dy1[i,k]:.6e} × weight[{k},{j}]={small_weight[k,j]:.6e} = {prod}")
        else:
            print("   ✅ Small-scale computation successful")
    except Exception as e:
        print(f"   ❌ Error in computation: {e}")


if __name__ == "__main__":
    print("NaN Root Cause Detector Module")
    print("This module identifies the exact computation that produces NaN")