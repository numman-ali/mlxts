# @mlxts/lora

Generic LoRA primitives over any `Module`. `applyLoRA`, `mergeLoRA`, `removeLoRA`, `loadLoRAAdapters`, and `saveLoRAAdapters` form the public surface.

CausalLM weight-naming knowledge, PEFT adapter format, and HuggingFace target-module conventions are forbidden in this package. They live at `@mlxts/transformers/lora-adapters`. A registry of model families inside `@mlxts/lora` is forbidden.

Native I/O is one config plus one safetensors file. Multi-file or model-shaped adapter formats belong to higher layers.

Module slot traversal is canonical here in `traversal.ts`. `@mlxts/transformers` consumes the traversal API. Forking traversal logic outside this package is forbidden.

`LoRAAdapterTarget` describes structural placement — parent module, child slot, adapter shape. Model semantics do not appear in this type.

Apply/merge/remove operate on a live `Module` graph and on `applyLoRAToModule` / `mergeLoRAInModule` / `removeLoRAFromModule` for explicit subtree work. Mutating wrapped modules outside these helpers is forbidden.
