#!/usr/bin/env python3
"""
Forward Speed Benchmark for Different MLP Architectures
========================================================

Compares forward pass speed of:
1. Full-rank (普通llama swiglu)
2. LORO Low-rank (loro lowrank)
3. LORO Low-rank + 2:4 Activation Sparsity (relu2 + soft_threshold24_triton)

Tests on different model sizes: 60m, 130m, 350m, 1b, 7b
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import argparse
from transformers.models.llama.configuration_llama import LlamaConfig

# Add current directory to path to import local modules
import sys
sys.path.append(os.path.dirname(__file__))

from peft_pretraining.modeling_llama import LlamaMLP, LlamaForCausalLM
from loro_torch.lowrank_module import LowRankLinear
from transformers.activations import ACT2FN
from sparse import matmul, soft_threshold24_triton

class MockConfig:
    """Mock config to control MLP behavior"""
    def __init__(self, architecture_type='silu', activation_2by4=False):
        self.squ_relu = architecture_type
        self.activation_2by4 = activation_2by4
        self.activation_sparse_method = 'fast'  # Use fast 2:4 sparsity for testing
        self.activation_dense_warmup_steps = 0  # No warmup for benchmarking
        self.dx_direct_sparse = 1
        self.dynamic_activation_steps = 10
        self.activation_calibration_samples = 50
        self.permute_2by4 = True
        self.wandb_sparsityrelu = False

def create_fullrank_mlp(hidden_size: int, intermediate_size: int) -> nn.Module:
    """Create full-rank MLP with SwiGLU (original llama architecture)"""
    config = MockConfig(architecture_type='silu', activation_2by4=False)
    return LlamaMLP(hidden_size, intermediate_size, 'silu', config)

def create_lowrank_swiglu_mlp(hidden_size: int, intermediate_size: int, rank: int = 256) -> nn.Module:
    """Create low-rank MLP with SwiGLU (keep SwiGLU architecture but use lowrank)"""
    config = MockConfig(architecture_type='silu', activation_2by4=False)
    mlp = LlamaMLP(hidden_size, intermediate_size, 'silu', config)

    # Replace all three projections with low-rank layers
    original_gate_proj = mlp.gate_proj
    original_up_proj = mlp.up_proj
    original_down_proj = mlp.down_proj

    mlp.gate_proj = LowRankLinear(original_gate_proj, rank, 'xavier')
    mlp.up_proj = LowRankLinear(original_up_proj, rank, 'xavier')
    mlp.down_proj = LowRankLinear(original_down_proj, rank, 'xavier')

    return mlp

def create_lowrank_relu2_mlp(hidden_size: int, intermediate_size: int, rank: int = 256) -> nn.Module:
    """Create low-rank MLP with ReLU2 (no gate, FFN-style architecture)"""
    # Use relu2 without activation sparsity for pure lowrank test
    config = MockConfig(architecture_type='relu2', activation_2by4=False)
    mlp = LlamaMLP(hidden_size, intermediate_size, 'relu2', config)

    # At this point, mlp has up_proj and down_proj (no gate_proj for relu2)
    # new_intermediate_size is already 1.5x to maintain param count
    # Replace with low-rank layers
    original_up_proj = mlp.up_proj
    original_down_proj = mlp.down_proj

    mlp.up_proj = LowRankLinear(original_up_proj, rank, 'xavier')
    mlp.down_proj = LowRankLinear(original_down_proj, rank, 'xavier')

    return mlp

def create_lowrank_relu2_2to4_mlp(hidden_size: int, intermediate_size: int, rank: int = 256) -> nn.Module:
    """Create low-rank MLP with ReLU2 and 2:4 activation sparsity"""
    config = MockConfig(architecture_type='relu2', activation_2by4=True)  # Enable 2:4 activation sparsity
    mlp = LlamaMLP(hidden_size, intermediate_size, 'relu2', config)

    # Replace with low-rank layers
    original_up_proj = mlp.up_proj
    original_down_proj = mlp.down_proj

    mlp.up_proj = LowRankLinear(original_up_proj, rank, 'xavier')
    mlp.down_proj = LowRankLinear(original_down_proj, rank, 'xavier')

    return mlp

def load_model_config(config_path: str) -> Tuple[int, int, int, int]:
    """Load model configuration and return key dimensions"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    num_layers = config['num_hidden_layers']
    num_heads = config['num_attention_heads']

    return hidden_size, intermediate_size, num_layers, num_heads

def benchmark_mlp_forward(mlp: nn.Module, batch_size: int, seq_len: int, hidden_size: int,
                         num_warmup: int = 10, num_iterations: int = 100) -> float:
    """Benchmark forward pass of MLP"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp = mlp.to(device, dtype=torch.bfloat16)  # Convert model to bfloat16
    mlp.eval()

    # Check if MLP needs 3D input (for 2:4 sparsity)
    config = getattr(mlp, 'config', None)
    needs_3d = (config is not None and
                getattr(config, 'activation_2by4', False) and
                getattr(config, 'squ_relu', 'silu') == 'relu2')

    # Create random input
    if needs_3d:
        # For 2:4 sparsity, use 3D input: (batch_size, seq_len, hidden_size)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    else:
        # For standard MLP, use 2D input: (batch_size * seq_len, hidden_size)
        x = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=torch.bfloat16)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = mlp(x)

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Time forward passes
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = mlp(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms

    return avg_time

def benchmark_model_sizes():
    """Benchmark different model sizes"""
    # Model configurations
    model_configs = {
        '60m': 'configs/llama_60m.json',
        '130m': 'configs/llama_130m.json',
        '350m': 'configs/llama_350m.json',
        '1b': 'configs/llama_1b.json',
        '7b': 'configs/llama_7b.json'
    }

    # Benchmark settings
    batch_size = 8
    seq_len = 512

    # Model-specific rank values
    rank_configs = {
        '60m': 128,
        '130m': 256,
        '350m': 256,
        '1b': 512,
        '7b': 1024
    }

    results = {}

    print(f"🚀 Starting Forward Speed Benchmark")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")

    for model_name, config_path in model_configs.items():
        if not os.path.exists(config_path):
            print(f"⚠️  Config file not found: {config_path}, skipping {model_name}")
            continue

        print(f"\n📊 Benchmarking {model_name.upper()}...")

        try:
            # Load model dimensions
            hidden_size, intermediate_size, num_layers, num_heads = load_model_config(config_path)

            print(f"   Hidden size: {hidden_size}, Intermediate size: {intermediate_size}, Layers: {num_layers}")

            # Get model-specific rank
            rank = rank_configs.get(model_name, 256)
            print(f"   LoRA rank: {rank}")

            # Create different MLP types
            fullrank_mlp = create_fullrank_mlp(hidden_size, intermediate_size)
            lowrank_swiglu_mlp = create_lowrank_swiglu_mlp(hidden_size, intermediate_size, rank)
            lowrank_relu2_mlp = create_lowrank_relu2_mlp(hidden_size, intermediate_size, rank)
            lowrank_relu2_2to4_mlp = create_lowrank_relu2_2to4_mlp(hidden_size, intermediate_size, rank)

            # Benchmark each type
            methods = {
                'Full-rank (SwiGLU)': fullrank_mlp,
                'LoRO Low-rank (SwiGLU)': lowrank_swiglu_mlp,
                'LoRO Low-rank (ReLU2)': lowrank_relu2_mlp,
                'LoRO Low-rank + 2:4 (ReLU2)': lowrank_relu2_2to4_mlp
            }

            model_results = {}

            for method_name, mlp in methods.items():
                try:
                    avg_time = benchmark_mlp_forward(mlp, batch_size, seq_len, hidden_size)
                    model_results[method_name] = avg_time
                    print(f"   {method_name:25}: {avg_time:.2f} ms")
                except Exception as e:
                    print(f"   {method_name:25}: ERROR - {str(e)}")
                    model_results[method_name] = None

                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results[model_name] = {
                'config': {
                    'hidden_size': hidden_size,
                    'intermediate_size': intermediate_size,
                    'num_layers': num_layers,
                    'num_heads': num_heads
                },
                'results': model_results
            }

        except Exception as e:
            print(f"❌ Error benchmarking {model_name}: {str(e)}")
            continue

    # Print summary
    print(f"\n{'='*80}")
    print(f"🎯 SUMMARY - Forward Pass Speed (ms per forward)")
    print(f"{'='*80}")

    # Create table header
    methods = ['Full-rank (SwiGLU)', 'LoRO Low-rank (SwiGLU)', 'LoRO Low-rank (ReLU2)', 'LoRO Low-rank + 2:4 (ReLU2)']
    print(f"{'Model':<10}", end='')
    for method in methods:
        print(f"{method:<30}", end='')
    print()
    print("-" * (10 + 30 * len(methods)))

    # Print results for each model
    for model_name, data in results.items():
        print(f"{model_name:<10}", end='')
        for method in methods:
            result = data['results'].get(method)
            if result is not None:
                print(f"{result:<25.2f} ms   ", end='')
            else:
                print(f"{'ERROR':<25}     ", end='')
        print()

    # Print speedup analysis
    print(f"\n{'='*80}")
    print(f"🔥 SPEEDUP ANALYSIS (relative to Full-rank)")
    print(f"{'='*80}")

    print(f"{'Model':<10}{'LoRO (SwiGLU)':<20}{'LoRO (ReLU2)':<20}{'LoRO + 2:4 (ReLU2)':<25}")
    print("-" * 75)

    for model_name, data in results.items():
        fullrank_time = data['results'].get('Full-rank (SwiGLU)')
        lowrank_swiglu_time = data['results'].get('LoRO Low-rank (SwiGLU)')
        lowrank_relu2_time = data['results'].get('LoRO Low-rank (ReLU2)')
        lowrank_2by4_time = data['results'].get('LoRO Low-rank + 2:4 (ReLU2)')

        print(f"{model_name:<10}", end='')

        if fullrank_time and lowrank_swiglu_time:
            speedup = fullrank_time / lowrank_swiglu_time
            print(f"{speedup:<15.2f}x    ", end='')
        else:
            print(f"{'N/A':<15}     ", end='')

        if fullrank_time and lowrank_relu2_time:
            speedup = fullrank_time / lowrank_relu2_time
            print(f"{speedup:<15.2f}x    ", end='')
        else:
            print(f"{'N/A':<15}     ", end='')

        if fullrank_time and lowrank_2by4_time:
            speedup = fullrank_time / lowrank_2by4_time
            print(f"{speedup:<25.2f}x")
        else:
            print(f"{'N/A':<25}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark forward pass speeds')
    parser.add_argument('--models', nargs='+',
                       choices=['60m', '130m', '350m', '1b', '7b'],
                       default=['60m', '130m', '350m', '1b', '7b'],
                       help='Model sizes to benchmark')

    args = parser.parse_args()

    # Check if in correct environment
    if not os.path.exists('sparse_package'):
        print("❌ Please run this script from the LORO-main_temp directory")
        return

    # Run benchmarks
    results = benchmark_model_sizes()

    # Save results
    output_file = 'forward_speed_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")

if __name__ == '__main__':
    main()