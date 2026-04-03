# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    nanogpt-ts                            │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │                 nanogpt (Phase 4)                 │  │
│  │  GPT model, training loop, tokenizer, generation  │  │
│  └────────────────────────┬──────────────────────────┘  │
│                           │ uses                        │
│  ┌────────────────────────┴──────────────────────────┐  │
│  │              mlx-ts nn layer (Phase 3)            │  │
│  │  Module, Linear, LayerNorm, Embedding, AdamW      │  │
│  │  (pure TypeScript — no native code)               │  │
│  └────────────────────────┬──────────────────────────┘  │
│                           │ uses                        │
│  ┌────────────────────────┴──────────────────────────┐  │
│  │           mlx-ts core bindings (Phase 1-2)        │  │
│  │  array, ops, autograd, random, eval               │  │
│  │  (TypeScript API over FFI)                        │  │
│  └────────────────────────┬──────────────────────────┘  │
│                           │ Bun FFI (dlopen + JSCallback)│
└───────────────────────────┼─────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │   mlx-c (Apple's C API)    │
              │   v0.6.0 — 580+ functions  │
              │   libmlxc.dylib            │
              └─────────────┬──────────────┘
                            │ links against
              ┌─────────────┴──────────────┐
              │     MLX (Apple's library)   │
              │  C++ core + Metal backend   │
              │  GPU kernels for Apple Si   │
              └────────────────────────────┘
```

### Why mlx-c instead of a custom C wrapper?

The original plan called for writing our own `extern "C"` wrapper around MLX's C++ API. Phase 0.5 research discovered that Apple maintains **mlx-c** (`ml-explore/mlx-c`) — a comprehensive, auto-generated C API with 580+ functions. Apple's own MLX Swift bindings use mlx-c as their bridge layer.

Using mlx-c eliminates:
- Writing and maintaining hundreds of C wrapper functions
- Tracking MLX C++ API changes ourselves
- A custom CMake build for our wrapper
- An entire layer of potential bugs

What mlx-c provides that we'd have had to build:
- All array operations, reductions, and linear algebra
- Complete autograd via the `mlx_closure` + `mlx_value_and_grad` pipeline
- Fused fast ops (SDPA, RoPE, RMS norm, layer norm) — critical for GPT performance
- Safetensors I/O
- Metal/device/stream management
- Error handling via `mlx_set_error_handler`

## Layer Responsibilities

### MLX (external dependency)
- Tensor storage and memory management
- Metal GPU kernel dispatch
- Computation graph construction and evaluation
- Automatic differentiation engine
- Lazy evaluation and JIT compilation

We don't modify MLX. We bind to it.

### mlx-c (external dependency)
- Provides pure C linkage over MLX's C++ API
- Uses opaque pointer pattern: every type is `struct { void* ctx; }`
- Manual memory management: `new` / `free` per type
- Error codes: all functions return `int` (0 = success)
- Maintained by Apple, auto-generated from C++ headers

We don't modify mlx-c. We call it via FFI.

### Core Bindings (`packages/mlx-ts/src/core/`)
- Loads libmlxc.dylib via Bun FFI (`dlopen`)
- Wraps opaque pointers in TypeScript classes
- Manages memory via FinalizationRegistry + explicit disposal
- Provides idiomatic TypeScript API for tensor operations
- Handles type conversions (JS arrays ↔ MLX arrays)
- Wraps TypeScript loss functions as `mlx_closure` via JSCallback

**Key types**:
```typescript
class MxArray implements Disposable {
  // Holds an opaque Pointer to mlx_array.ctx (Bun's branded Pointer type)
  // The Pointer type is a boundary detail — hidden behind helper functions
  readonly _ctx: Pointer

  // Lazy cached property getters — read from C on first access
  readonly shape: readonly number[]
  readonly dtype: DType
  readonly ndim: number
  readonly size: number

  toTypedArray(): TypedArray  // row-major contiguous copy
  toList(): NestedArray<number>
  item(): number  // for scalar arrays
  asType(dtype: DType): MxArray

  free(): void              // explicit cleanup
  [Symbol.dispose](): void  // `using` support
}

// Operations are free functions (matches MLX's style)
function matmul(a: MxArray, b: MxArray, stream?: Pointer): MxArray
function add(a: MxArray, b: MxArray, stream?: Pointer): MxArray
function reshape(a: MxArray, shape: number[], stream?: Pointer): MxArray
// ... etc — all ops take optional stream parameter
```

### NN Layer (`packages/mlx-ts/src/nn/`)
- Pure TypeScript (no native code)
- Calls into core bindings for all tensor operations
- Provides familiar Module-based API (like PyTorch/MLX Python)

**Key pattern**:
```typescript
abstract class Module implements Disposable {
  abstract forward(...args: MxArray[]): MxArray

  parameters(): ParameterTree      // nested tree of MxArray leaves
  trainableParameters(): ParameterTree  // excludes frozen keys
  update(params: ParameterTree): void   // recursive partial merge
  freeze(keys?: string[]): this
  train(mode?: boolean): this
  eval(): this
}

// ParameterTree = { [key: string]: MxArray | ParameterTree }
// Module scans Object.keys(this) for MxArray and Module properties.
// The key list is cached after the first scan, so public parameter and
// sub-module fields must be established during construction.
// Internal state uses #private fields (invisible to scanning).
// Shared public parameter aliases are not supported yet; Phase 4 will
// add explicit weight-tying semantics when the transformer output head lands.

class Linear extends Module {
  weight: MxArray        // scanned as parameter
  bias: MxArray | null   // null → skipped by scanner

  forward(x: MxArray): MxArray {
    // Binary ops accept number | MxArray for scalar coercion
    return add(matmul(x, transpose(this.weight)), this.bias)
  }
}
```

### nanoGPT (`packages/nanogpt/`)
- Pure TypeScript, depends only on mlx-ts
- Implements GPT-2 architecture
- Training loop, data loading, text generation
- Configurable model sizes

## Memory Management

This is the most critical cross-cutting concern.

### The problem
MLX arrays are C++ heap objects. JavaScript has GC. We need to bridge these two worlds without leaking memory or using freed pointers.

### mlx-c's memory model

mlx-c uses **manual new/free** — no reference counting at the C level:

```
mlx_array_new()          → allocates, returns empty handle
mlx_array_new_data(...)  → allocates with initial data
mlx_array_set(dst, src)  → reuses existing allocation if possible
mlx_array_free(x)        → deallocates the underlying C++ object
```

All functions return `int` (0 = success). Error details go through `mlx_set_error_handler`.

### Our memory strategy

```
┌──────────────────────────────────────────────┐
│             TypeScript (GC-managed)          │
│                                              │
│  MxArray {                                   │
│    _ctx: Pointer  ──────────────────────┐    │
│  }                                      │    │
│                                         │    │
│  FinalizationRegistry watches MxArray   │    │
│  instances — calls mlx_array_free()     │    │
│  when GC'd                              │    │
│                                         │    │
│  Explicit disposal via:                 │    │
│    using arr = ...   (Symbol.dispose)   │    │
│    arr.free()        (manual)           │    │
└─────────────────────────────────────────┼────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────┐
│      mlx-c opaque pointer (heap-allocated)   │
│                                              │
│  mlx_array { void* ctx; }                    │
│  ctx → mlx::core::array* on the C++ side     │
│                                              │
│  mlx_array_new()  → allocates (returns by    │
│                     value on ARM64)          │
│  mlx_array_free() → deallocates              │
└──────────────────────────────────────────────┘
```

**Rules**:
1. Every `new` must have a matching `free`
2. FinalizationRegistry provides a safety net, but don't rely on it for timely cleanup
3. In hot loops (training), use explicit disposal via `using` declarations
4. Native handles in temporary variables (vector arrays, key splits, device handles) must use `try/finally` for cleanup

### Bun FFI pointer boundary

Bun FFI uses a branded `Pointer` type — a compile-time brand (`{ __pointer__: null }`) distinct from plain `number`. This brand prevents accidentally mixing native addresses with regular numbers.

The Pointer brand stays at the FFI boundary. Two helpers in `src/core/ffi/pointer.ts` bridge the gap:
- `unwrapPointer(ptr, label)` — narrows `Pointer | null` to `Pointer`, throwing if null
- `sizeToNumber(value, label)` — converts `number | bigint` (from size_t returns) to `number`

Higher-level code (ops, random, transforms) works with `Pointer` directly but never needs to assert or narrow it — the helpers handle that.

Key FFI utilities (re-exported from `src/core/ffi/` with proper Pointer types):
- `ptr(typedArray)` — get a `Pointer` from a TypedArray
- `nativeSlice(ptr, offset, length)` — create an ArrayBuffer viewing native memory
- `readI32(ptr, offset)` — read a 32-bit int from native memory
- `readPtr(ptr, offset)` — read a pointer from native memory

## Autograd Design

MLX uses functional autograd (like JAX), not tape-based (like PyTorch).

```
PyTorch style (NOT what we do):
  y = model(x)       # records to tape
  loss = criterion(y) # records to tape
  loss.backward()     # replays tape

MLX/JAX style (what we do):
  lossFn = (params, x, y) => ...   # define a pure function
  gradFn = mx.grad(lossFn)         # transform it — returns a NEW function
  grads = gradFn(params, x, y)     # call the new function to get gradients
```

### The closure mechanism (mlx-c)

mlx-c exposes autograd through typed closures:

| Closure Type | Signature | Purpose |
|---|---|---|
| `mlx_closure` | `vec<array> → vec<array>` | General function (loss fn) |
| `mlx_closure_value_and_grad` | `vec<array> → (vec<array>, vec<array>)` | Result of value_and_grad |

The flow:
1. Wrap a TypeScript loss function as a `mlx_closure` via `JSCallback`
2. Call `mlx_value_and_grad(&vag, closure, argnums, num_argnums)` — returns a `mlx_closure_value_and_grad`
3. Call `mlx_closure_value_and_grad_apply(&values, &grads, vag, inputs)` — returns both loss and gradients

### Why callbacks are safe (threading)

The key insight: **MLX calls the closure synchronously on the calling thread during graph construction, not during GPU execution.** The flow is:

```
Bun main thread: calls mlx_closure_value_and_grad_apply(...)
  → mlx-c: calls the closure (still on main thread)
    → JSCallback: re-enters TypeScript (same thread — no threading issue)
      → TypeScript loss fn runs, calling mx.matmul, mx.add, etc.
        → Each op goes through FFI to mlx-c, adding to computation graph
      → Returns loss MxArray
    → JSCallback returns to mlx-c
  → mlx-c/MLX: differentiates the computation graph (no more callbacks)
  → Returns gradients
```

Since graph construction is lazy (building a graph, not executing GPU kernels), the callback happens on the same thread. `JSCallback` with `threadsafe: false` is all we need.

## Data Flow During Training

```
1. Load batch (TypeScript)
   → [tokenIds: number[]] → mx.array(tokenIds)

2. Forward pass (through FFI)
   → input MxArray → model.forward() → logits MxArray
   → cross_entropy(logits, targets) → loss MxArray

3. Backward pass (autograd through FFI)
   → mx.valueAndGrad(lossFn)(params, input, target)
   → [loss, gradients]

4. Optimizer step (TypeScript + FFI)
   → optimizer.update(model, gradients)
   → mx.eval(model.parameters())  ← forces GPU computation

5. Log and repeat
```

Steps 2-4 are mostly GPU work dispatched via Metal. The TypeScript layer orchestrates but doesn't do heavy computation.

## Fused Operations (Performance Bonus)

mlx-c exposes optimized fused ops in `mlx/c/fast.h` that are critical for GPT performance:

| Op | C Function | Why it matters |
|---|---|---|
| Scaled dot-product attention | `mlx_fast_scaled_dot_product_attention` | Core of the transformer — fused is much faster than composed |
| Layer norm | `mlx_fast_layer_norm` | Used before every attention and MLP block |
| RMS norm | `mlx_fast_rms_norm` | Alternative normalization used by some models |
| RoPE | `mlx_fast_rope` | Rotary positional embeddings — used by modern LLMs |

These fused ops bypass the need to compose from primitives in TypeScript, giving us near-native performance for the most critical transformer operations.
