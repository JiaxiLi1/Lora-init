2025-08-21 13:27:35
   Has NaN: True, Has Inf: False
2025-08-21 13:27:35

2025-08-21 13:27:35
❌ NaN/Inf found in module.lm_head output:
2025-08-21 13:27:35
   Shape: [64, 256, 32000]
2025-08-21 13:27:35
   Has NaN: True, Has Inf: False
2025-08-21 13:27:35

2025-08-21 13:27:35
================================================================================
2025-08-21 13:27:35
[NaN/Inf First Occurrence Summary]
2025-08-21 13:27:35
First NaN/Inf detected at: module.model.layers.11.input_layernorm
2025-08-21 13:27:35
Type: input
2025-08-21 13:27:35
Stats: {'has_nan': True, 'has_inf': False, 'min': 'NaN/Inf present', 'max': 'NaN/Inf present', 'mean': 'NaN/Inf present', 'shape': [64, 256, 768], 'dtype': 'torch.bfloat16'}
2025-08-21 13:27:35

2025-08-21 13:27:35
Total 17 NaN/Inf occurrences found
2025-08-21 13:27:35
Propagation path:
2025-08-21 13:27:35
  1. module.model.layers.11.input_layernorm (input)
2025-08-21 13:27:35
  2. module.model.layers.11.input_layernorm (output)
2025-08-21 13:27:35
  3. module.model.layers.11.self_attn.q_proj (input)
2025-08-21 13:27:35
  4. module.model.layers.11.self_attn.q_proj (output)
2025-08-21 13:27:35
  5. module.model.layers.11.self_attn.k_proj (input)
2025-08-21 13:27:35

2025-08-21 13:27:35
[Split GEMM Specific Checks]
2025-08-21 13:27:35
  Split GEMM is active (95% sparse + 5% dense)
2025-08-21 13:27:35
  Activation sparse method: soft_dynamic
2025-08-21 13:27:35
  Dense warmup steps: 0
2025-08-21 13:27:35
  Current step: 857
2025-08-21 13:27:35
  ✅ Activation 2:4 sparsity is ACTIVE (past warmup)
2025-08-21 13:27:35

2025-08-21 13:27:35
  Checking activation sparse modules:
2025-08-21 13:27:35

2025-08-21 13:27:35
================================================================================
2025-08-21 13:27:35
Stopping training due to NaN loss.
2025-08-21 13:27:35
================================================================================