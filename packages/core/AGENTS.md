# @mlxts/core

The FFI boundary lives at `src/ffi/`. Type assertions (`as`, `!`) and `any` are forbidden outside that directory. Every layer above receives `MxArray` and consumes the `Pointer` brand only through helpers in `src/ffi/`.

ABI integrity is first. Declarations in `src/ffi/symbols.ts` are re-audited against `.reference/mlx-c/mlx/c/*.h` whenever mlx-c is upgraded or Bun FFI semantics change. The primitive layer is a product surface, not a staging area — if the primitives lie, every layer above compounds the lie.

Per-call `OutSlot` ownership at the FFI boundary. Shared reusable output buffers are forbidden. FFI result pointers are owned by exactly one call site.

Native handle temporaries — vector arrays, device handles, RNG keys — release through `try/finally`. Hope-based cleanup is forbidden.

mlx-c first. Before adding a host-side workaround, `.reference/mlx-c/mlx/c/ops.h` is checked for the operation. JS fallbacks are reserved for genuinely host-side work — small lookups, user-provided callbacks.

Compile-strategy plumbing stays under `transforms-*.ts`. The public surface holds semantic names: `mxEval`, `mxAsyncEval`, `compile`, `valueAndGrad`, `grad`, `checkpoint`. New runtime strategies do not earn top-level barrel exports until a semantic call site needs them.

`fast.ts` mirrors MLX Python's `mx.fast` namespace; the namespace re-export pattern stays intact.

Public tensor-producing primitives join the canonical tracked-op list in `scripts/check-tensor-lifetimes` when they land.

Creation symbols return a `Pointer` by value. Operation symbols take an output pointer and return `i32` status checked through `checkStatus`. Property getters return values directly.
