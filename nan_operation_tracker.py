"""
NaN Operation Tracker - Find the exact operation that produces NaN from valid inputs.
This module hooks into PyTorch operations to catch the first NaN creation.
"""

import torch
import torch.nn as nn
import functools
from typing import Any, Dict, List, Optional, Tuple
import traceback


class NaNOperationTracker:
    """Track the exact operation that first produces NaN from valid inputs."""
    
    def __init__(self):
        self.original_ops = {}
        self.first_nan_operation = None
        self.operation_history = []
        self.enabled = False
        
    def wrap_operation(self, op_name: str, op_func):
        """Wrap a PyTorch operation to detect NaN production."""
        @functools.wraps(op_func)
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return op_func(*args, **kwargs)
            
            # Check inputs for NaN
            input_has_nan = False
            input_info = []
            
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and torch.is_floating_point(arg):
                    has_nan = torch.isnan(arg).any().item()
                    has_inf = torch.isinf(arg).any().item()
                    input_has_nan = input_has_nan or has_nan or has_inf
                    
                    if not has_nan and not has_inf:
                        input_info.append({
                            'arg_idx': i,
                            'shape': list(arg.shape),
                            'dtype': str(arg.dtype),
                            'min': arg.min().item(),
                            'max': arg.max().item(),
                            'mean': arg.mean().item() if arg.numel() > 0 else 0,
                            'has_zeros': (arg == 0).any().item(),
                            'has_negative': (arg < 0).any().item() if arg.dtype in [torch.float32, torch.float64, torch.bfloat16] else False
                        })
                    else:
                        input_info.append({
                            'arg_idx': i,
                            'shape': list(arg.shape),
                            'dtype': str(arg.dtype),
                            'has_nan': has_nan,
                            'has_inf': has_inf,
                            'nan_count': torch.isnan(arg).sum().item() if has_nan else 0,
                            'inf_count': torch.isinf(arg).sum().item() if has_inf else 0
                        })
            
            # Execute operation
            result = op_func(*args, **kwargs)
            
            # Check output for NaN
            output_has_nan = False
            output_info = None
            
            if isinstance(result, torch.Tensor) and torch.is_floating_point(result):
                has_nan = torch.isnan(result).any().item()
                has_inf = torch.isinf(result).any().item()
                output_has_nan = has_nan or has_inf
                
                if output_has_nan:
                    output_info = {
                        'shape': list(result.shape),
                        'dtype': str(result.dtype),
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'nan_count': torch.isnan(result).sum().item() if has_nan else 0,
                        'inf_count': torch.isinf(result).sum().item() if has_inf else 0,
                        'first_nan_idx': torch.where(torch.isnan(result))[0][0].item() if has_nan and result.numel() > 0 else None
                    }
                else:
                    output_info = {
                        'shape': list(result.shape),
                        'dtype': str(result.dtype),
                        'min': result.min().item(),
                        'max': result.max().item(),
                        'mean': result.mean().item() if result.numel() > 0 else 0
                    }
            
            # Track operation
            op_record = {
                'operation': op_name,
                'input_has_nan': input_has_nan,
                'output_has_nan': output_has_nan,
                'inputs': input_info,
                'output': output_info,
                'stack_trace': traceback.extract_stack()[-5:-1]  # Get relevant stack frames
            }
            
            self.operation_history.append(op_record)
            
            # Detect first NaN creation
            if not input_has_nan and output_has_nan and self.first_nan_operation is None:
                self.first_nan_operation = op_record
                self._analyze_nan_cause(op_name, args, kwargs, result)
            
            return result
        
        return wrapped
    
    def _analyze_nan_cause(self, op_name: str, args, kwargs, result):
        """Analyze the specific cause of NaN generation."""
        print("\n" + "="*80)
        print("🔴 FIRST NaN DETECTED - ROOT CAUSE ANALYSIS")
        print("="*80)
        print(f"\nOperation: {op_name}")
        
        # Specific analysis for common operations
        if 'div' in op_name or '/' in op_name:
            # Check for division by zero
            if len(args) >= 2:
                divisor = args[1]
                if isinstance(divisor, torch.Tensor):
                    zero_count = (divisor == 0).sum().item()
                    if zero_count > 0:
                        print(f"⚠️ CAUSE: Division by zero detected! {zero_count} zero values in divisor")
                        zero_indices = torch.where(divisor == 0)
                        print(f"   First zero at index: {[idx[0].item() for idx in zero_indices][:5]}")
        
        elif 'sqrt' in op_name:
            # Check for negative values
            if len(args) >= 1:
                input_tensor = args[0]
                if isinstance(input_tensor, torch.Tensor):
                    negative_count = (input_tensor < 0).sum().item()
                    if negative_count > 0:
                        print(f"⚠️ CAUSE: Square root of negative values! {negative_count} negative values")
                        neg_indices = torch.where(input_tensor < 0)
                        print(f"   First negative value: {input_tensor[neg_indices][0].item()}")
        
        elif 'log' in op_name:
            # Check for non-positive values
            if len(args) >= 1:
                input_tensor = args[0]
                if isinstance(input_tensor, torch.Tensor):
                    non_positive = (input_tensor <= 0).sum().item()
                    if non_positive > 0:
                        print(f"⚠️ CAUSE: Log of non-positive values! {non_positive} values <= 0")
                        bad_indices = torch.where(input_tensor <= 0)
                        print(f"   First non-positive value: {input_tensor[bad_indices][0].item()}")
        
        elif 'mm' in op_name or 'matmul' in op_name or 'bmm' in op_name:
            # Check for numerical overflow in matrix multiplication
            if len(args) >= 2:
                a, b = args[0], args[1]
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    a_max = a.abs().max().item()
                    b_max = b.abs().max().item()
                    print(f"   Input A max abs value: {a_max:.4e}")
                    print(f"   Input B max abs value: {b_max:.4e}")
                    
                    # Check if values are extremely large
                    if a_max > 1e10 or b_max > 1e10:
                        print(f"⚠️ CAUSE: Possible numerical overflow in matrix multiplication")
                        print(f"   Product magnitude estimate: {a_max * b_max:.4e}")
                    
                    # Check for inf values that could produce NaN
                    a_inf = torch.isinf(a).any().item()
                    b_inf = torch.isinf(b).any().item()
                    if a_inf or b_inf:
                        print(f"⚠️ CAUSE: Inf values in matrix multiplication inputs")
                        print(f"   Input A has inf: {a_inf}, Input B has inf: {b_inf}")
        
        elif 'pow' in op_name or '**' in str(op_name):
            # Check for invalid power operations
            if len(args) >= 2:
                base = args[0]
                exponent = args[1] if len(args) > 1 else kwargs.get('exponent', 2)
                if isinstance(base, torch.Tensor):
                    negative_base = (base < 0).any().item()
                    if negative_base and isinstance(exponent, (float, int)) and not float(exponent).is_integer():
                        print(f"⚠️ CAUSE: Negative base with non-integer exponent")
        
        # Print input details
        print("\nInput tensors:")
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and torch.is_floating_point(arg):
                print(f"  Arg {i}: shape={list(arg.shape)}, dtype={arg.dtype}")
                if not torch.isnan(arg).any() and not torch.isinf(arg).any():
                    print(f"    Range: [{arg.min().item():.4e}, {arg.max().item():.4e}]")
                    print(f"    Mean: {arg.mean().item():.4e}, Std: {arg.std().item():.4e}")
        
        # Print output details
        if isinstance(result, torch.Tensor):
            nan_mask = torch.isnan(result)
            if nan_mask.any():
                nan_indices = torch.where(nan_mask)
                print(f"\nOutput tensor: shape={list(result.shape)}, dtype={result.dtype}")
                print(f"  NaN count: {nan_mask.sum().item()} out of {result.numel()}")
                print(f"  First NaN at index: {[idx[0].item() for idx in nan_indices][:5]}")
                
                # Try to identify pattern in NaN locations
                if len(nan_indices) > 0 and len(nan_indices[0]) > 10:
                    # Check if NaNs appear in specific columns/rows
                    if len(result.shape) == 2:
                        nan_rows = torch.unique(nan_indices[0])
                        nan_cols = torch.unique(nan_indices[1])
                        if len(nan_rows) < 10:
                            print(f"  NaN in rows: {nan_rows.tolist()[:10]}")
                        if len(nan_cols) < 10:
                            print(f"  NaN in columns: {nan_cols.tolist()[:10]}")
        
        # Print stack trace
        print("\nCall stack:")
        for frame in self.first_nan_operation['stack_trace']:
            print(f"  {frame.filename}:{frame.lineno} in {frame.name}")
            if frame.line:
                print(f"    {frame.line}")
        
        print("="*80)
    
    def patch_pytorch_ops(self):
        """Patch common PyTorch operations to track NaN generation."""
        ops_to_patch = [
            ('torch.mm', torch.mm),
            ('torch.matmul', torch.matmul),
            ('torch.bmm', torch.bmm),
            ('torch.addmm', torch.addmm),
            ('torch.div', torch.div),
            ('torch.true_divide', torch.true_divide),
            ('torch.sqrt', torch.sqrt),
            ('torch.log', torch.log),
            ('torch.exp', torch.exp),
            ('torch.pow', torch.pow),
            ('torch.mean', torch.mean),
            ('torch.std', torch.std),
            ('torch.var', torch.var),
            ('F.linear', torch.nn.functional.linear),
            ('F.softmax', torch.nn.functional.softmax),
            ('F.layer_norm', torch.nn.functional.layer_norm),
        ]
        
        for op_name, op_func in ops_to_patch:
            if op_name not in self.original_ops:
                self.original_ops[op_name] = op_func
                wrapped = self.wrap_operation(op_name, op_func)
                
                # Patch the operation
                parts = op_name.split('.')
                if len(parts) == 2:
                    if parts[0] == 'torch':
                        setattr(torch, parts[1], wrapped)
                    elif parts[0] == 'F':
                        setattr(torch.nn.functional, parts[1], wrapped)
    
    def unpatch_pytorch_ops(self):
        """Restore original PyTorch operations."""
        for op_name, original_op in self.original_ops.items():
            parts = op_name.split('.')
            if len(parts) == 2:
                if parts[0] == 'torch':
                    setattr(torch, parts[1], original_op)
                elif parts[0] == 'F':
                    setattr(torch.nn.functional, parts[1], original_op)
        self.original_ops.clear()
    
    def enable(self):
        """Enable NaN tracking."""
        self.enabled = True
        self.patch_pytorch_ops()
    
    def disable(self):
        """Disable NaN tracking."""
        self.enabled = False
        self.unpatch_pytorch_ops()
    
    def reset(self):
        """Reset tracking state."""
        self.first_nan_operation = None
        self.operation_history.clear()
    
    def get_summary(self):
        """Get a summary of NaN operations."""
        summary = {
            'total_operations': len(self.operation_history),
            'first_nan_operation': self.first_nan_operation,
            'operations_with_nan_output': sum(1 for op in self.operation_history if op['output_has_nan'])
        }
        
        # Count operations by type
        op_counts = {}
        for op in self.operation_history:
            op_name = op['operation']
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
        summary['operation_counts'] = op_counts
        
        return summary


# Global tracker instance
nan_op_tracker = NaNOperationTracker()


def track_nan_operations(func):
    """Decorator to track NaN operations during a function call."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nan_op_tracker.reset()
        nan_op_tracker.enable()
        try:
            result = func(*args, **kwargs)
            if nan_op_tracker.first_nan_operation:
                print("\n⚠️ NaN was generated during this operation!")
                print(f"Summary: {nan_op_tracker.get_summary()}")
            return result
        finally:
            nan_op_tracker.disable()
    return wrapper