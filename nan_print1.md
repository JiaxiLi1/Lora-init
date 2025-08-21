2025-08-21 17:02:45

2025-08-21 17:02:45
================================================================================
2025-08-21 17:02:45
NaN PROPAGATION SUMMARY
2025-08-21 17:02:45
================================================================================
2025-08-21 17:02:45

2025-08-21 17:02:45
🔴 First NaN/Inf detected:
2025-08-21 17:02:45
   Module: module.model.norm
2025-08-21 17:02:45
   Stage: forward
2025-08-21 17:02:45
   Input: TensorInfo(shape=[64, 256, 768], dtype='torch.bfloat16', device='cuda:0', has_nan=True, has_inf=False, nan_count=619776, inf_count=0, min_val=None, max_val=None, mean_val=None, std_val=None, zero_ratio=0.0, grad_norm=None)
2025-08-21 17:02:45
   Output: TensorInfo(shape=[64, 256, 768], dtype='torch.bfloat16', device='cuda:0', has_nan=True, has_inf=False, nan_count=619776, inf_count=0, min_val=None, max_val=None, mean_val=None, std_val=None, zero_ratio=0.0, grad_norm=None)
2025-08-21 17:02:45

2025-08-21 17:02:45
📊 Total NaN/Inf occurrences: 4
2025-08-21 17:02:45

2025-08-21 17:02:45
📍 NaN Propagation Path:
2025-08-21 17:02:45
   1. module.model.norm_forward
2025-08-21 17:02:45
   2. module.lm_head_forward
2025-08-21 17:02:45

2025-08-21 17:02:45
[Split GEMM Specific Checks]
2025-08-21 17:02:45
  Split GEMM is active (95% sparse + 5% dense)
2025-08-21 17:02:45
  Activation sparse method: soft_threshold_weights
2025-08-21 17:02:45
  Dense warmup steps: 1000
2025-08-21 17:02:45
  Current step: 11457
2025-08-21 17:02:45
  ✅ Activation 2:4 sparsity is ACTIVE (past warmup)
2025-08-21 17:02:45

2025-08-21 17:02:45
  Checking activation sparse modules:
2025-08-21 17:02:45

2025-08-21 17:02:45
================================================================================
2025-08-21 17:02:45
Stopping training due to NaN loss.
2025-08-21 17:02:45
================================================================================
2025-08-21 17:02:45
Update steps:   7%|█▎                 | 1432/20000 [2:01:47<26:19:12,  5.10s/it]