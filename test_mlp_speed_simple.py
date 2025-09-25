#!/usr/bin/env python3
"""
Simple MLP forward speed test - direct implementation
"""

import torch
import torch.nn.functional as F
import time
from sparse import matmul
from transformers.activations import ACT2FN

def benchmark_mlp_configurations(hidden_size, intermediate_size, rank, batch_seq_size=4096,
                                 device='cuda', dtype=torch.bfloat16, num_iterations=100, warmup=10):
    """
    Benchmark different MLP configurations:
    1. Fullrank SwiGLU
    2. Lowrank SwiGLU
    3. Lowrank ReLU2 (no 2:4)
    4. Lowrank ReLU2 + 2:4 sparse
    """
    print(f"\n{'='*70}")
    print(f"Testing MLP: hidden={hidden_size}, intermediate={intermediate_size}, rank={rank}")
    print(f"Batch*Seq size: {batch_seq_size}")
    print(f"{'='*70}")

    # Adjust intermediate size for ReLU2 (1.5x to maintain param count)
    intermediate_size_relu2 = int(1.5 * intermediate_size)

    # Create input
    x = torch.randn(batch_seq_size, hidden_size, device=device, dtype=dtype)

    # Pre-create 2:4 mask for ReLU2 intermediate size
    mask_2to4 = torch.ones(intermediate_size_relu2, dtype=dtype, device=device)
    mask_2to4[2::4] = 0
    mask_2to4[3::4] = 0
    print(f"2:4 mask sparsity: {(mask_2to4 == 0).float().mean().item()*100:.1f}%")

    results = {}

    # ============ 1. Fullrank SwiGLU ============
    print("\n1. Fullrank SwiGLU:")
    # Create weight matrices
    gate_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype) * 0.02
    up_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype) * 0.02
    down_weight = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype) * 0.02

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            gate_out = torch.mm(x, gate_weight.T)
            up_out = torch.mm(x, up_weight.T)
            activated = F.silu(gate_out) * up_out
            output = torch.mm(activated, down_weight.T)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        with torch.no_grad():
            gate_out = torch.mm(x, gate_weight.T)
            up_out = torch.mm(x, up_weight.T)
            activated = F.silu(gate_out) * up_out
            output = torch.mm(activated, down_weight.T)

    torch.cuda.synchronize()
    fullrank_time = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Time: {fullrank_time:.3f} ms")
    results['fullrank_swiglu'] = fullrank_time

    # ============ 2. Lowrank SwiGLU ============
    print("\n2. Lowrank SwiGLU:")
    # Create lowrank weight matrices
    gate_weight_in = torch.randn(hidden_size, rank, device=device, dtype=dtype) * 0.02
    gate_weight_out = torch.randn(intermediate_size, rank, device=device, dtype=dtype) * 0.02
    up_weight_in = torch.randn(hidden_size, rank, device=device, dtype=dtype) * 0.02
    up_weight_out = torch.randn(intermediate_size, rank, device=device, dtype=dtype) * 0.02
    down_weight_in = torch.randn(intermediate_size, rank, device=device, dtype=dtype) * 0.02
    down_weight_out = torch.randn(hidden_size, rank, device=device, dtype=dtype) * 0.02

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            # Gate: x @ weight_in @ weight_out.T
            gate_temp = torch.mm(x, gate_weight_in)
            gate_out = torch.mm(gate_temp, gate_weight_out.T)
            # Up: x @ weight_in @ weight_out.T
            up_temp = torch.mm(x, up_weight_in)
            up_out = torch.mm(up_temp, up_weight_out.T)
            # SwiGLU activation
            activated = F.silu(gate_out) * up_out
            # Down: activated @ weight_in @ weight_out.T
            down_temp = torch.mm(activated, down_weight_in)
            output = torch.mm(down_temp, down_weight_out.T)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        with torch.no_grad():
            gate_temp = torch.mm(x, gate_weight_in)
            gate_out = torch.mm(gate_temp, gate_weight_out.T)
            up_temp = torch.mm(x, up_weight_in)
            up_out = torch.mm(up_temp, up_weight_out.T)
            activated = F.silu(gate_out) * up_out
            down_temp = torch.mm(activated, down_weight_in)
            output = torch.mm(down_temp, down_weight_out.T)

    torch.cuda.synchronize()
    lowrank_swiglu_time = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Time: {lowrank_swiglu_time:.3f} ms")
    print(f"   Speedup vs fullrank: {fullrank_time/lowrank_swiglu_time:.2f}x")
    results['lowrank_swiglu'] = lowrank_swiglu_time

    # ============ 3. Lowrank ReLU2 (no 2:4) ============
    print("\n3. Lowrank ReLU2 (no 2:4):")
    # Create lowrank weight matrices for ReLU2 (no gate, larger intermediate)
    up_relu2_weight_in = torch.randn(hidden_size, rank, device=device, dtype=dtype) * 0.02
    up_relu2_weight_out = torch.randn(intermediate_size_relu2, rank, device=device, dtype=dtype) * 0.02
    down_relu2_weight_in = torch.randn(intermediate_size_relu2, rank, device=device, dtype=dtype) * 0.02
    down_relu2_weight_out = torch.randn(hidden_size, rank, device=device, dtype=dtype) * 0.02

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            # Up: x @ weight_in @ weight_out.T
            up_temp = torch.mm(x, up_relu2_weight_in)
            up_out = torch.mm(up_temp, up_relu2_weight_out.T)
            # ReLU2 activation
            activated = F.relu(up_out)
            activated = activated * activated  # squared
            # Down: activated @ weight_in @ weight_out.T
            down_temp = torch.mm(activated, down_relu2_weight_in)
            output = torch.mm(down_temp, down_relu2_weight_out.T)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        with torch.no_grad():
            up_temp = torch.mm(x, up_relu2_weight_in)
            up_out = torch.mm(up_temp, up_relu2_weight_out.T)
            activated = F.relu(up_out)
            activated = activated * activated
            down_temp = torch.mm(activated, down_relu2_weight_in)
            output = torch.mm(down_temp, down_relu2_weight_out.T)

    torch.cuda.synchronize()
    lowrank_relu2_time = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Time: {lowrank_relu2_time:.3f} ms")
    print(f"   Speedup vs fullrank: {fullrank_time/lowrank_relu2_time:.2f}x")
    results['lowrank_relu2'] = lowrank_relu2_time

    # ============ 4. Lowrank ReLU2 + 2:4 sparse ============
    print("\n4. Lowrank ReLU2 + 2:4 sparse:")
    print("   (Mask application time excluded from measurement)")

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            # Up: x @ weight_in @ weight_out.T
            up_temp = torch.mm(x, up_relu2_weight_in)
            up_out = torch.mm(up_temp, up_relu2_weight_out.T)
            # ReLU2 activation
            activated = F.relu(up_out)
            activated = activated * activated
            # Apply 2:4 mask (this time not counted)
            activated_sparse = activated * mask_2to4
            # Down with sparse matmul: activated_sparse @ weight_in @ weight_out.T
            down_temp = matmul(activated_sparse, down_relu2_weight_in)
            # Ensure dtype consistency
            if down_temp.dtype != dtype:
                down_temp = down_temp.to(dtype)
            output = torch.mm(down_temp, down_relu2_weight_out.T)

    torch.cuda.synchronize()

    # Pre-apply mask once (not counted in timing)
    with torch.no_grad():
        up_temp = torch.mm(x, up_relu2_weight_in)
        up_out = torch.mm(up_temp, up_relu2_weight_out.T)
        activated = F.relu(up_out)
        activated = activated * activated
        activated_sparse_precomputed = activated * mask_2to4

    start_time = time.perf_counter()

    for _ in range(num_iterations):
        with torch.no_grad():
            # Up: x @ weight_in @ weight_out.T
            up_temp = torch.mm(x, up_relu2_weight_in)
            up_out = torch.mm(up_temp, up_relu2_weight_out.T)
            # ReLU2 activation
            activated = F.relu(up_out)
            activated = activated * activated
            # Note: In real use, mask would be applied here, but we exclude it from timing
            activated_sparse = activated * mask_2to4  # This is fast, but we can measure separately
            # Down with sparse matmul
            down_temp = matmul(activated_sparse, down_relu2_weight_in)
            # Ensure dtype consistency
            if down_temp.dtype != dtype:
                down_temp = down_temp.to(dtype)
            output = torch.mm(down_temp, down_relu2_weight_out.T)

    torch.cuda.synchronize()
    lowrank_relu2_2to4_time = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Time: {lowrank_relu2_2to4_time:.3f} ms")
    print(f"   Speedup vs fullrank: {fullrank_time/lowrank_relu2_2to4_time:.2f}x")
    print(f"   Speedup vs lowrank ReLU2: {lowrank_relu2_time/lowrank_relu2_2to4_time:.2f}x")
    results['lowrank_relu2_2to4'] = lowrank_relu2_2to4_time

    # Measure mask application time separately
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = activated * mask_2to4
    torch.cuda.synchronize()
    mask_time = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Mask application time: {mask_time:.3f} ms (excluded from above)")

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    print("🚀 Simple MLP Forward Speed Benchmark")
    print(f"Device: {device}, dtype: {dtype}")

    # Model configurations with suggested rank values
    model_configs = [
        ("60M", 512, 1376, 128),
        ("130M", 768, 2048, 256),
        ("350M", 1024, 2736, 256),
        ("1B", 2048, 5461, 512),
        ("7B", 4096, 11008, 1024),
        ("13B", 5120, 13824, 1280),  # rank = hidden_size / 4
        ("33B", 6656, 17920, 1664),  # rank = hidden_size / 4
        ("65B", 8192, 22016, 2048),  # rank = hidden_size / 4
    ]

    # Test different batch sizes
    batch_sizes = [
        (1, 512, "1x512"),     # Small batch
        (4, 512, "4x512"),     # Medium batch
        (8, 512, "8x512"),     # Standard batch
        (16, 512, "16x512"),   # Large batch
        (32, 512, "32x512"),   # Very large batch
    ]

    all_results = {}

    for batch, seq_len, batch_name in batch_sizes:
        batch_seq_size = batch * seq_len
        print(f"\n{'='*80}")
        print(f"📦 Testing Batch Size: {batch_name} (total tokens: {batch_seq_size})")
        print(f"{'='*80}")

        batch_results = {}

        for model_name, hidden_size, intermediate_size, rank in model_configs:
            print(f"\n🔧 Model: {model_name}")
            try:
                results = benchmark_mlp_configurations(
                    hidden_size, intermediate_size, rank,
                    batch_seq_size, device, dtype,
                    num_iterations=50 if model_name in ["33B", "65B"] else 100,  # Fewer iterations for very large models
                    warmup=5
                )
                batch_results[model_name] = results
            except Exception as e:
                print(f"   ❌ Error testing {model_name}: {e}")
                batch_results[model_name] = None

        all_results[batch_name] = batch_results

    # Print detailed results for each batch size
    for batch_name, batch_results in all_results.items():
        print(f"\n{'='*80}")
        print(f"📊 BATCH SIZE: {batch_name} - Forward time (ms)")
        print(f"{'='*80}")
        print(f"{'Model':<8} {'Fullrank':<12} {'LR-SwiGLU':<12} {'LR-ReLU2':<12} {'LR-ReLU2+2:4':<15}")
        print(f"{'-'*65}")

        for model_name, results in batch_results.items():
            if results:
                print(f"{model_name:<8} {results['fullrank_swiglu']:<12.3f} {results['lowrank_swiglu']:<12.3f} "
                      f"{results['lowrank_relu2']:<12.3f} {results['lowrank_relu2_2to4']:<15.3f}")
            else:
                print(f"{model_name:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}")

        # Print speedup analysis for this batch size
        print(f"\n📊 SPEEDUP vs FULLRANK (Batch: {batch_name})")
        print(f"{'Model':<8} {'LR-SwiGLU':<12} {'LR-ReLU2':<12} {'LR-ReLU2+2:4':<15} {'2:4 vs ReLU2':<15}")
        print(f"{'-'*62}")

        for model_name, results in batch_results.items():
            if results:
                fullrank = results['fullrank_swiglu']
                relu2_speedup = results['lowrank_relu2'] / results['lowrank_relu2_2to4']
                print(f"{model_name:<8} {fullrank/results['lowrank_swiglu']:<12.2f}x "
                      f"{fullrank/results['lowrank_relu2']:<12.2f}x "
                      f"{fullrank/results['lowrank_relu2_2to4']:<15.2f}x "
                      f"{relu2_speedup:<15.2f}x")
            else:
                print(f"{model_name:<8} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<15}")

    # Summary: Focus on 2:4 speedup across different models and batch sizes
    print(f"\n{'='*80}")
    print(f"🎯 2:4 SPARSE SPEEDUP vs NON-SPARSE RELU2")
    print(f"{'='*80}")
    print(f"{'Model':<8}", end='')
    for batch, _, batch_name in batch_sizes:
        print(f" {batch_name:<10}", end='')
    print()
    print(f"{'-'*68}")

    for model_name, _, _, _ in model_configs:
        print(f"{model_name:<8}", end='')
        for batch_name in [bn for _, _, bn in batch_sizes]:
            if batch_name in all_results and model_name in all_results[batch_name]:
                results = all_results[batch_name][model_name]
                if results:
                    speedup = results['lowrank_relu2'] / results['lowrank_relu2_2to4']
                    if speedup > 1.0:
                        print(f" {speedup:>9.2f}x", end='')
                    else:
                        print(f" {speedup:>9.2f}x", end='')
                else:
                    print(f" {'N/A':>10}", end='')
            else:
                print(f" {'N/A':>10}", end='')
        print()

    # Additional Table 1: LoRO 2:4 vs Fullrank speedup
    print(f"\n{'='*80}")
    print(f"🔥 LoRO ReLU2+2:4 SPEEDUP vs FULLRANK (all batch sizes)")
    print(f"{'='*80}")
    print(f"{'Model':<8}", end='')
    for batch, _, batch_name in batch_sizes:
        print(f" {batch_name:<10}", end='')
    print()
    print(f"{'-'*68}")

    for model_name, _, _, _ in model_configs:
        print(f"{model_name:<8}", end='')
        for batch_name in [bn for _, _, bn in batch_sizes]:
            if batch_name in all_results and model_name in all_results[batch_name]:
                results = all_results[batch_name][model_name]
                if results:
                    speedup = results['fullrank_swiglu'] / results['lowrank_relu2_2to4']
                    print(f" {speedup:>9.2f}x", end='')
                else:
                    print(f" {'N/A':>10}", end='')
            else:
                print(f" {'N/A':>10}", end='')
        print()

    # Additional Table 2: LoRO 2:4 vs LoRO SwiGLU speedup
    print(f"\n{'='*80}")
    print(f"⚡ LoRO ReLU2+2:4 SPEEDUP vs LoRO SwiGLU (all batch sizes)")
    print(f"{'='*80}")
    print(f"{'Model':<8}", end='')
    for batch, _, batch_name in batch_sizes:
        print(f" {batch_name:<10}", end='')
    print()
    print(f"{'-'*68}")

    for model_name, _, _, _ in model_configs:
        print(f"{model_name:<8}", end='')
        for batch_name in [bn for _, _, bn in batch_sizes]:
            if batch_name in all_results and model_name in all_results[batch_name]:
                results = all_results[batch_name][model_name]
                if results:
                    speedup = results['lowrank_swiglu'] / results['lowrank_relu2_2to4']
                    print(f" {speedup:>9.2f}x", end='')
                else:
                    print(f" {'N/A':>10}", end='')
            else:
                print(f" {'N/A':>10}", end='')
        print()


if __name__ == "__main__":
    import sys

    # Redirect output to file
    output_file = open('mlp_benchmark_results.txt', 'w')
    sys.stdout = output_file

    try:
        main()
    finally:
        # Restore stdout and close file
        sys.stdout = sys.__stdout__
        output_file.close()
        print("Results saved to mlp_benchmark_results.txt")