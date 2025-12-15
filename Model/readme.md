
---

## Project state summary (Chronos-2 sparse time attention)

### Goal implemented

Added a **memory-efficient sparse time self-attention** mode for the Chronos-2 encoder with sequence layout:

`[context patch tokens] + ([REG] optional) + [future patch tokens]`

Sparse mode name: `"windowed_future_global"`

Behavior:

* **Context queries** attend only to a **sliding window** of keys (radius = `time_local_radius`)
* Context keys restricted to **context (+ REG if present)** → **no context→future leakage**
* Optional REG behavior (`time_reg_is_global=True`):

  * REG can be included as a **global key** for every context query (if not already in the window)
  * REG can be a **global query** attending to all context keys
  * REG never attends to future
* **Future queries** (last `num_output_patches` tokens) attend **globally to all keys** (only padding masked)
* Sparse mode uses **2D padding mask** `[B,S]`; **no dense** `[B,H,S,S]` attention mask is built
* Sparse mode refuses `output_attentions=True` (to avoid forcing dense tensors)

### Files changed

#### 1) `config.py`

Added sparse-attention knobs to `Chronos2CoreConfig.__init__` and stored them:

* `time_attention_type: Literal["full", "windowed_future_global"] = "full"`
* `time_local_radius: int = 128`
* `time_attention_chunk_size: int = 32`
* `time_reg_is_global: bool = False`

#### 2) `model.py`

Threaded the metadata sparse attention needs and avoided 4D masks in sparse mode:

* In `Chronos2Model.encode()`:

  * computed `reg_token_index = num_context_patches` when inserting `[REG]`
  * passed `num_output_patches` and `reg_token_index` into `self.encoder(...)`
* In `Chronos2Encoder.__init__`:

  * added `self.config = config`
* In `Chronos2Encoder.forward()`:

  * added args: `num_output_patches`, `reg_token_index`
  * built 4D additive mask only when `time_attention_type == "full"`, else passed 2D padding mask
  * forwarded `num_output_patches` and `reg_token_index` into each encoder block
* In `Chronos2EncoderBlock.forward()`:

  * added args: `num_output_patches`, `reg_token_index`
  * passed them into `TimeSelfAttention(...)`

#### 3) `layers.py`

Implemented sparse time attention + a couple of critical stability fixes:

**A) TimeSelfAttention replaced**

* Dense path unchanged for `"full"`
* Sparse path `"windowed_future_global"`:

  * projects Q/K/V once (with RoPE)
  * context: windowed attention computed in chunks (`time_attention_chunk_size`)
  * future: global attention only for last `num_output_patches` queries
  * optional REG global key/query support
  * rejects `output_attentions=True` in sparse mode

**B) Fixed incorrect gather shape**

* Original attempt used `torch.gather` with mismatched dims → runtime error.
* Replaced with indexing-based gathering for the window:

  * `k_win = k_ctx[:, :, idx, :]`
  * `v_win = v_ctx[:, :, idx, :]`
  * `key_ok = key_pad_ctx[:, idx] & valid[None, :, :]`

**C) Fixed SDPA mask dtype mismatch (mixed precision)**

* In `MHA._sdpa_attention`, added:

  * if mask is floating and dtype != query dtype → cast mask to query dtype
* This fixes SDPA error when LayerNorm upcasts queries to fp32 but mask remains fp16.

**D) Added gradient checkpointing to prevent backward OOM**

* Backward at long context (8k) OOMed because per-chunk window tensors were being saved for autograd.
* Added checkpointing around the per-chunk context attention compute inside
  `_windowed_future_global_attention`:

  * recomputes chunk during backward instead of storing huge activations
  * fixed long-context backward memory

### Sanity tests executed (passed)

**Unit-level TimeSelfAttention tests (passed):**

* sparse mode refuses `output_attentions=True`
* no context→future leakage
* window locality holds
* future queries are global
* padding respected
* chunk size invariance (same module weights)
* dense equivalence when radius covers all context (dense/sparse weights synced)
* backward/grad flow works

**Integration tests (passed):**

* Encoder in sparse mode does **not** call `_expand_and_invert_time_attention_mask`
* REG global mode still has no context→future leakage

**Long-context perf/memory test (passed after fixes):**

* Forward at `seq_len=8192`, `d_model=512`, `num_layers=4`, `heads=8`, `radius=128`, `chunk=32`:

  * forward OK (~0.86–0.88s)
* Backward OK after checkpointing:

  * peak CUDA allocated ~1.69 GiB in the synthetic benchmark

### Known “gotchas” fixed during debugging (important)

* `torch.gather` dimension mismatch in window gather → replaced with indexing gather
* SDPA mask dtype mismatch in mixed precision → added mask cast in `_sdpa_attention`
* Backward OOM for long contexts → added checkpointing for context chunks
* Test script bug: `tensor or tensor` boolean ambiguity → used explicit `if y is None` selection

---

## How to enable sparse attention

When building the config used for the model:

```python
config.chronos_config["time_attention_type"] = "windowed_future_global"
config.chronos_config["time_local_radius"] = 128
config.chronos_config["time_attention_chunk_size"] = 32
config.chronos_config["time_reg_is_global"] = False  # or True
```

---

## Next stage (what to do next)

**Fine-tuning wiring + long-context setup**, specifically:

1. Ensure the **actual training/inference entrypoint** (pipeline/trainer) sets `time_attention_type="windowed_future_global"` at runtime (no silent fallback to `"full"`).
2. Ensure `context_length` is set correctly and not causing silent truncation in dataset/pipeline/trainer.
3. Ensure `num_output_patches` / prediction length is correct and that future tokens are indeed the **last tokens** during training.
4. Pick batch size + grad accumulation for long contexts (8k/16k) and confirm memory.
5. Optional: enable additional checkpointing at encoder-block level if full model training still gets tight on memory.

---
