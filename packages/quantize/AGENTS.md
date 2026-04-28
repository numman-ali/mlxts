# @mlxts/quantize

Checkpoint and tensor quantization. Owns: live module quantization, pre-allocated quantized layers from a checkpoint plan, mode/group_size/bits resolution, GGUF I/O delegated to `@mlxts/core`, and pretrained-config quantization metadata parsing (`mxfp4`, `compressed-tensors`, `awq`, `gptq`).

Out of scope: KV cache representation, runtime serving scheduling, engine memory policy. Those belong to `@mlxts/transformers` and `@mlxts/serve`.

New checkpoint providers register through `registerQuantizedCheckpointProvider`. Hard-coded family branching inside `resolveCheckpointQuantizationPlan` is forbidden.

Future runtime KV quantization (TurboQuant, FP8 cache) lands in `@mlxts/transformers/infrastructure/cache/` and `@mlxts/core` quantized ops, not here. The boundary is checkpoint-and-tensor only.

`QuantizedCheckpointPlan` is the contract between checkpoint providers and module setup. Path conventions translate via `translateCheckpointQuantizationPlanPaths`; the underlying plan shape stays stable.

`setupQuantizedModule` allocates quantized child slots from a plan before weight loading. Mutating a `Module` graph by replacing children at load time is forbidden — the plan declares the shape; the loader fills it.
