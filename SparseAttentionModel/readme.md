# Summary of model-side changes for the Sparse Time Attention extension (Chronos-2)

Here, we summarizes **what was changed inside the Chronos-2 model implementation** (relative to the original “full” temporal attention path), specifically to support the **sparse time-attention** variant used in our experiments.

> Scope: This is about **model/config + attention implementation**.  
> It does **not** cover the external evaluation scripts except where they interact with model flags (e.g., forcing `time_attention_type="full"` to extract attentions).

---

## 1) New / exposed configuration knobs (model + pipeline)

We rely on Chronos-2’s configurable time-attention mechanism and expose/override the following fields through the pipeline constructor / model config:

- `time_attention_type`
  - `"full"`: standard dense temporal self-attention over all time tokens
  - `"windowed_future_global"`: sparse variant (local past + global future)
- `time_local_radius`
  - integer radius `r` used for **local past attention**
- `time_attention_chunk_size`
  - chunk size for computing attention in pieces (reduces peak memory pressure and can enable kernels)
- `time_attention_backend`
  - `"torch"` (eager/standard) or `"flash"` (if supported)
- `time_use_landmarks`
  - forced **False** in our experiments for apples-to-apples comparisons and to simplify attention semantics
- `time_reg_is_global` (optional)
  - affects the “REG token” row semantics (if the model uses a REG token):
    - if enabled, the REG query can attend globally to context keys (but still not to future keys)

**Important:** In our “attention mass” probes we explicitly set:
- `time_attention_type="full"`
- `time_use_landmarks=False`
so that the model returns full attention weights (`output_attentions=True`) without additional sparsification.

---

## 2) Temporal attention behavior change: from dense to structured sparsity

### Original (Full)
For each encoder layer, temporal self-attention computes attention for all query positions `q` over all key positions `k` in the time-token sequence:

- attention matrix shape: `[B, H, S, S]`
- all `S×S` query-key edges are allowed (subject to the usual masking)

### Modified (Sparse / `windowed_future_global`)
We change the **allowed attention pattern** along the time axis:

Let:
- `S` = number of time tokens after patching
- `num_output_patches` = number of future output patches for the prediction horizon
- `future_start = S - num_output_patches`
- `ctx_end = future_start`

Then the sparse pattern is:

#### A) Context (past) queries: **local window**
For query positions `q < ctx_end` (i.e., in the context portion):
- allowed keys `k` are restricted to a local window:
  - `k ∈ [max(0, q-r), min(ctx_end-1, q+r)]`

This reduces context attention complexity from `O(ctx_end²)` toward `O(ctx_end · r)`.

#### B) Future queries: **global**
For query positions `q ≥ future_start` (future/output patches):
- allowed keys are **global over all tokens**:
  - `k ∈ [0, S-1]`

This preserves the model’s ability to condition future decoding on the entire context.

#### C) REG token (optional policy)
If the model has a REG token and `time_reg_is_global=True`:
- the REG query row is allowed to attend **globally over context keys only**
- REG does **not** attend to future keys

---

## 3) Chunked attention computation (implementation detail)

To avoid allocating / materializing large attention matrices at once, the sparse attention path supports **chunked** computation, controlled by:

- `time_attention_chunk_size`

Behavior:
- attention is computed in blocks/chunks along the time dimension
- this reduces peak memory and sometimes improves runtime
- however, for Chronos-2’s effective token lengths after patching, the overhead may dominate (we observed speedups around ~1 or slightly <1 in many cases)

---

## 4) Backend support (torch vs flash)

The model attention layer supports different backends:

- `time_attention_backend="torch"`
  - standard PyTorch attention computation
  - easiest path for extracting attention weights
- `time_attention_backend="flash"`
  - uses a flash-attention style kernel if available
  - faster in some regimes but can be more sensitive to dtype/masking semantics

**Important note:** Some configurations can produce **NaNs** (especially under `flash` + bf16 in certain tasks/settings). In those cases:
- switch backend to `"torch"`, or
- try fp16/fp32, or
- adjust chunk size / radius

---

## 5) Attention weights extraction (for analysis)

Chronos-2 exposes temporal attention weights when called with:
- `output_attentions=True`

Output field used:
- `enc_time_self_attn_weights`: list of tensors per layer, each shaped `[B, H, S, S]`

For our “mass retained” measurement we ensured:
- the model is in `time_attention_type="full"` mode (so weights correspond to dense learned attention)
- we apply our *analysis-time mask* to those weights to estimate how much mass would be kept/dropped under sparse patterns

This is analysis-only; the actual sparse model uses the sparse pattern during forward passes and does **not** necessarily return a full `[S,S]` matrix in all backends.

---

## 6) What was not changed in the model

- No changes to:
  - tokenization / patching logic
  - group attention logic
  - quantile heads / probabilistic forecasting
  - model weights (no fine-tuning)
  - dataset processing rules

All changes are isolated to:
- temporal attention configuration
- temporal attention masking policy
- chunked computation / backend selection

---

---

## Checklist of model-side knobs used in this project

Typical sparse run:
- `time_attention_type="windowed_future_global"`
- `time_local_radius=r`
- `time_attention_chunk_size=256` (example)
- `time_attention_backend={"torch"|"flash"}`
- `time_use_landmarks=False`

Typical attention-mass probe:
- `time_attention_type="full"`
- `time_use_landmarks=False`
- `output_attentions=True`
- (optional) `time_reg_is_global=True/False`
