# @mlxts/nn

Every learnable component extends `Module`. Public `MxArray`, `Module`, and `Module[]` fields are scanned as parameters during `parameters()`. Non-parameter state — config scalars, derived caches like `#transposedWeight`, dropout probabilities — uses JS `#` private fields.

The own-key set is captured on first scan. Public fields are assigned in the constructor before the module is first read.

Weight tying stays functional. `Embedding.asLinear(hidden)` returns the projection. Aliasing an embedding's weight onto a `Linear` module is forbidden.

`Module[]` arrays scan with string indices ("0", "1", ...) matching HuggingFace weight naming (`model.layers.0.self_attn.q_proj.weight`). Layer collections are plain arrays.

The public surface holds semantic names: `gelu`, `relu`, `silu`, `swiglu`, `crossEntropy`, `mse`, `RMSNorm`, `LayerNorm`, `RoPE`, `Linear`, `Embedding`, `Conv1d`, `Dropout`, `GroupedQueryAttention`, `LoRALinear`. Compile and shape-keyed reuse live behind these names, not in front of them.

Quantized variants (`QuantizedLinear`, `QuantizedEmbedding`) parallel their non-quantized counterparts and consume the same `Module` parameter contract. Fused quantized linears arrive through `fuseQuantizedLinears`. Mutating layer arrays in place to fuse is forbidden.

Disposable intermediates inside `forward` use `using`. Hidden disposable `MxArray` intermediates inside nested expressions are forbidden — local tensor lifetimes stay visible.

Activation gradient checkpointing for memory pressure goes through `checkpoint(module)`. Inlining a hand-rolled rematerialization wrapper is forbidden.
