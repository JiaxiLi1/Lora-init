import torch
from triton_fused_gemm import triton_matmul_with_sparsity

# Small test case
M, K, N = 4, 4, 4
x = torch.randn(M, K, device='cuda', dtype=torch.float32)
w = torch.randn(K, N, device='cuda', dtype=torch.float32)

# Reference computation
y_ref = x @ w
print("Raw GEMM output (first 4x4):")
print(y_ref[:4, :4])

y_ref_relu2 = torch.where(y_ref > 0, y_ref ** 2, torch.zeros_like(y_ref))
print("\nReLU² reference (first 4x4):")
print(y_ref_relu2[:4, :4])

# Triton computation
y_triton, _ = triton_matmul_with_sparsity(x, w, activation='relu2')
print("\nTriton ReLU² output (first 4x4):")
print(y_triton[:4, :4])

# Error
error = (y_triton - y_ref_relu2).abs()
print("\nError matrix (first 4x4):")
print(error[:4, :4])
print(f"\nMax error: {error.max().item()}")