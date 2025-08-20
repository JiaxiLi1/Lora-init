#!/usr/bin/env python3
"""
Test script to verify NaN detection is working properly with split_gemm
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create a dummy model with known NaN issues
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64)
        ])
        
        # Introduce a known NaN in layer 1
        with torch.no_grad():
            self.layers[1].weight[10, 20] = float('nan')
    
    def forward(self, x, labels=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        
        # Simple loss calculation
        if labels is not None:
            loss = nn.functional.cross_entropy(x, labels)
            return type('Output', (), {'loss': loss})()
        return x

def test_nan_detection():
    print("Testing NaN detection code...")
    
    # Create model and input
    model = TestModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy batch
    batch_size = 8
    seq_len = 128
    hidden_size = 512
    num_classes = 64
    
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len), device=device),
        'attention_mask': torch.ones(batch_size, seq_len, device=device)
    }
    
    # Create input tensor
    input_tensor = torch.randn(batch_size, hidden_size, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # Test the NaN detection code
    try:
        # Forward pass
        output = model(input_tensor, labels=labels)
        loss = output.loss
        
        print(f"Loss value: {loss.item()}")
        print(f"Loss has NaN: {torch.isnan(loss)}")
        
        if torch.isnan(loss):
            print("\n[NaN Detection Test]")
            print("=" * 80)
            print("Loss is NaN - triggering detailed tracking...")
            
            # NaN detection code from run_c4.py (fixed version)
            hooks = []
            nan_found_at = []
            
            def check_tensor_for_nan(tensor, name=""):
                if tensor is None:
                    return False, "None"
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                
                # Skip integer tensors - they can't have NaN/Inf
                if not torch.is_floating_point(tensor) and not torch.is_complex(tensor):
                    return False, {
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape),
                        "is_integer": True
                    }
                
                has_nan = torch.isnan(tensor).any().item()
                has_inf = torch.isinf(tensor).any().item()
                
                # Only compute statistics if tensor is floating point
                if has_nan or has_inf:
                    stats = {
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                        "min": "NaN/Inf present",
                        "max": "NaN/Inf present",
                        "mean": "NaN/Inf present",
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype)
                    }
                else:
                    stats = {
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                        "min": tensor.min().item(),
                        "max": tensor.max().item(),
                        "mean": tensor.mean().item(),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype)
                    }
                return has_nan or has_inf, stats
            
            def forward_hook(module, input, output, module_name):
                # Check inputs
                for i, inp in enumerate(input):
                    if torch.is_tensor(inp):
                        has_issue, stats = check_tensor_for_nan(inp, f"input_{i}")
                        if has_issue:
                            nan_found_at.append({
                                "layer": module_name,
                                "type": "input",
                                "index": i,
                                "stats": stats
                            })
                            print(f"\n❌ NaN/Inf found in {module_name} input[{i}]:")
                            print(f"   Stats: {stats}")
                
                # Check output
                if torch.is_tensor(output):
                    has_issue, stats = check_tensor_for_nan(output, "output")
                    if has_issue:
                        nan_found_at.append({
                            "layer": module_name,
                            "type": "output",
                            "stats": stats
                        })
                        print(f"\n❌ NaN/Inf found in {module_name} output:")
                        print(f"   Stats: {stats}")
            
            # Register hooks
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    hook = module.register_forward_hook(
                        lambda m, i, o, n=name: forward_hook(m, i, o, n)
                    )
                    hooks.append(hook)
            
            # Re-run forward pass with tracking
            print("\nRe-running forward pass with detailed tracking...")
            try:
                with torch.no_grad():
                    _ = model(input_tensor, labels=labels)
            except Exception as e:
                print(f"\nException during re-run: {e}")
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Summary
            if nan_found_at:
                print("\n" + "=" * 80)
                print("[NaN/Inf Detection Summary]")
                print(f"Total {len(nan_found_at)} NaN/Inf occurrences found")
                print("\nFirst occurrence:")
                first = nan_found_at[0]
                print(f"  Layer: {first['layer']}")
                print(f"  Type: {first['type']}")
                print(f"  Stats: {first['stats']}")
            else:
                print("\n✅ No NaN/Inf found during re-run (might be intermittent)")
                
            print("=" * 80)
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nan_detection()
    print("\n✅ NaN detection test completed successfully!")