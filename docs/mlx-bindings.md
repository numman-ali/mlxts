# MLX Bindings — Technical Guide

## What is MLX?

[MLX](https://github.com/ml-explore/mlx) is Apple's machine learning framework for Apple Silicon. It provides:

- **NumPy-like API** for tensor operations
- **Metal GPU acceleration** on Apple Silicon (M1/M2/M3/M4)
- **Unified memory** — CPU and GPU share the same memory, no explicit transfers
- **Lazy evaluation** — operations build a graph, compute only when needed
- **Functional autograd** — gradients via function transformation (like JAX)

MLX is written in C++ with a Python frontend. Our job is to replace that Python frontend with TypeScript.

## What is mlx-c?

[mlx-c](https://github.com/ml-explore/mlx-c) is Apple's **official C API** for MLX. It's a separate repository (`ml-explore/mlx-c`, v0.6.0) that wraps the MLX C++ API with pure C linkage. Apple's own MLX Swift bindings use mlx-c as their bridge layer.

Key properties:
- **580+ C functions** covering array ops, autograd, random, I/O, and fused ops
- **Opaque pointer pattern**: every type is `struct { void* ctx; }` — ideal for FFI
- **Manual memory**: `new` / `free` per type (no reference counting at the C level)
- **Error codes**: all functions return `int` (0 = success)
- **Auto-generated** from C++ headers — stays in sync with MLX

### Phase 0.5 Decision

We use mlx-c directly. We do **not** write a custom C wrapper. This was validated during the Phase 0.5 research spike — see PLAN.md for the full findings.

## Binding Strategy

### The three layers

```
Layer 3:  TypeScript public API     (what users write)
          mx.matmul(a, b)

Layer 2:  Bun FFI bridge            (loads .dylib, calls mlx-c functions)
          ffi.mlx_matmul(out, a._ctx, b._ctx, stream)

Layer 1:  mlx-c                     (Apple's official C API)
          libmlxc.dylib → libmlx.dylib → Metal

Layer 0:  MLX C++ core              (Apple's library, untouched)
          mlx::core::matmul(a, b)
```

### Why Bun FFI?

Bun's FFI (`bun:ffi`) provides:
- `dlopen(path, symbols)` — load a shared library and declare function signatures
- `JSCallback` — create C-callable function pointers from JavaScript functions
- `read.ptr()`, `read.i32()`, etc. — fast direct memory reads
- Branded `Pointer` type — compile-time safety for native addresses

Performance (measured on M4 Max, Bun 1.3.4):
- Basic FFI call: ~8ns (negligible vs any ML operation)
- Callback round-trip: ~35ns (negligible vs a matmul)

### Example: matmul binding

**Layer 1 — mlx-c provides:**
```c
// From mlx/c/ops.h (auto-generated)
int mlx_matmul(mlx_array* res, mlx_array a, mlx_array b, mlx_stream s);
```

Note the pattern: result is written to a pre-allocated output pointer (first arg). All ops follow this convention.

**Layer 2 — Bun FFI:**
```typescript
// src/core/ffi/symbols.ts — symbol declaration (part of the grouped constants)
mlx_matmul: { args: [P, P, P, P], returns: I32 },
```

All symbols are loaded via a single `dlopen` call with grouped constants spread in.

**Layer 3 — TypeScript API:**
```typescript
// src/core/ops/linalg.ts
export function matmul(a: MxArray, b: MxArray, stream?: S): MxArray {
  return readResultArray("matmul", (out) => {
    checkStatus(ffi.mlx_matmul(out, a._ctx, b._ctx, s(stream)), "matmul");
  });
}
```

`readResultArray()` creates a fresh `OutSlot` per call, which keeps the FFI layer reentrant and makes result ownership explicit. `checkStatus()` throws a typed `MxError` on non-zero return codes.

## mlx-c API Surface

The following catalogs the mlx-c functions we need, verified against the v0.6.0 source.

### Binding expansion policy

We expand the binding surface deliberately, not randomly.

- **Family-complete over one-off symbols.** If we decide to bring in `fast.h`, `compile.h`, or another header family, we organize it cleanly under `src/core/ffi/` and document the family as a coherent surface.
- **Official mlx-c first.** The default bridge is Apple's C API. We do not add local native shims just because they feel convenient.
- **Upstream before local shims.** If the right capability is missing or awkward in mlx-c, prefer a precise upstream issue or PR before creating local native code.
- **Local native shims are a last resort.** If we ever need them, they belong in a dedicated shim layer rather than being mixed into the ordinary FFI mapping.
- **Runtime controls matter.** `memory.h`, `fast.h`, `compile.h`, and transform/runtime controls are first-class binding families because they affect correctness, performance, and operator trust.

The mirror image of this policy is just as important: a correct FFI symbol declaration is not enough on its own. JS-side tensor lifetime discipline still matters above the FFI layer.

### Priority 1 — Required for nanoGPT

**Array lifecycle:**
| mlx-c Function | Signature | Purpose |
|---|---|---|
| `mlx_array_new_data` | `(data, shape, ndim, dtype) → mlx_array` | Create from data (returns by value) |
| `mlx_zeros` | `(res*, shape, ndim, dtype, stream) → int` | Zero-filled array |
| `mlx_ones` | `(res*, shape, ndim, dtype, stream) → int` | One-filled array |
| `mlx_full` | `(res*, shape, ndim, val, dtype, stream) → int` | Constant-filled array |
| `mlx_arange` | `(res*, start, stop, step, dtype, stream) → int` | Range array |
| `mlx_array_shape` | `(a) → const int*` | Get shape (direct return, not output pointer) |
| `mlx_array_dtype` | `(a) → mlx_dtype` | Get dtype (direct return) |
| `mlx_array_ndim` | `(a) → size_t` | Get number of dimensions (direct return) |
| `mlx_array_size` | `(a) → size_t` | Get total element count (direct return) |
| `mlx_array_eval` | `(a) → int` | Force evaluation |
| `mlx_array_free` | `(a) → int` | Deallocate |

Note the two calling conventions:
- **Creation functions** (e.g., `mlx_array_new_data`) return `mlx_array` by value — on ARM64, this is effectively a pointer return.
- **Operations** (e.g., `mlx_zeros`) write results via `mlx_array*` output pointer (first argument) and return `int` (0 = success).
- **Property getters** (e.g., `mlx_array_shape`, `mlx_array_ndim`) return values directly — no output pointer pattern.

**Arithmetic:**
| mlx-c Function | Purpose |
|---|---|
| `mlx_add` | Element-wise addition |
| `mlx_subtract` | Element-wise subtraction |
| `mlx_multiply` | Element-wise multiplication |
| `mlx_divide` | Element-wise division |
| `mlx_negative` | Negate |
| `mlx_power` | Element-wise power |
| `mlx_sqrt` | Square root |
| `mlx_exp` | Exponential |
| `mlx_log` | Natural logarithm |
| `mlx_abs` | Absolute value |
| `mlx_maximum` | Element-wise max |
| `mlx_minimum` | Element-wise min |
| `mlx_sigmoid` | Sigmoid activation |
| `mlx_erf` | Error function (for GELU) |

**Reductions:**
| mlx-c Function | Purpose |
|---|---|
| `mlx_sum` / `mlx_sum_axis` / `mlx_sum_axes` | Sum reduction |
| `mlx_mean` / `mlx_mean_axis` / `mlx_mean_axes` | Mean reduction |
| `mlx_max` / `mlx_max_axis` / `mlx_max_axes` | Max reduction |
| `mlx_min` / `mlx_min_axis` / `mlx_min_axes` | Min reduction |
| `mlx_argmax` / `mlx_argmin` | Index of max/min |
| `mlx_logsumexp` | Log-sum-exp (for cross-entropy) |

**Shape and indexing:**
| mlx-c Function | Purpose |
|---|---|
| `mlx_matmul` | Matrix multiplication |
| `mlx_transpose` / `mlx_transpose_axes` | Transpose |
| `mlx_reshape` | Reshape |
| `mlx_squeeze` | Remove size-1 dims |
| `mlx_expand_dims` | Add size-1 dim |
| `mlx_concatenate` | Join arrays |
| `mlx_split` / `mlx_split_sections` | Split array |
| `mlx_stack` | Stack arrays |
| `mlx_take` / `mlx_take_along_axis` | Gather by index |
| `mlx_where` | Conditional select |
| `mlx_equal` / `mlx_greater` / `mlx_less` | Comparisons |
| `mlx_astype` | Type casting |
| `mlx_broadcast_to` | Broadcasting |
| `mlx_stop_gradient` | Detach from gradient tracking |

**Neural network ops:**
| mlx-c Function | Purpose |
|---|---|
| `mlx_softmax` / `mlx_softmax_axis` | Softmax |
| `mlx_fast_layer_norm` | Fused layer norm |
| `mlx_fast_rms_norm` | Fused RMS norm |
| `mlx_fast_rope` | Fused rotary position embeddings |
| `mlx_fast_scaled_dot_product_attention` | Fused SDPA |

**Random:**
| mlx-c Function | Purpose |
|---|---|
| `mlx_random_key` | Create RNG key |
| `mlx_random_split` | Split RNG key |
| `mlx_random_normal` | Normal distribution |
| `mlx_random_uniform` | Uniform distribution |
| `mlx_random_bernoulli` | Bernoulli (for dropout) |

**Transforms and autograd:**
| mlx-c Function | Purpose |
|---|---|
| `mlx_eval` | Force evaluation of lazy arrays |
| `mlx_async_eval` | Async evaluation |
| `mlx_value_and_grad` | Create value+gradient transform |
| `mlx_jvp` | Forward-mode AD |
| `mlx_vjp` | Reverse-mode AD |
| `mlx_compile` | JIT compilation |
| `mlx_checkpoint` | Gradient checkpointing |

**Runtime and operational controls:**
| mlx-c Function Family | Purpose |
|---|---|
| `memory.h` | Active/cache/peak memory telemetry and allocator limits |
| `device.h` / `stream.h` | Device availability, default stream/device, explicit synchronization |
| `metal.h` | Metal availability and profiling / capture hooks |

**I/O:**
| mlx-c Function | Purpose |
|---|---|
| `mlx_load_safetensors` | Load model weights |
| `mlx_save_safetensors` | Save model weights |

Today the repo's public safetensors surface is implemented in TypeScript in `src/core/io.ts`, not through a direct `io.h` binding. That choice is deliberate: the current `mlx-c` I/O surface returns map iterators that are awkward to model directly in Bun FFI without adding a native shim layer. The canonical operator checkpoint format stays separate from safetensors either way: safetensors is for model-weight interop, not resumable training state.

### Priority 2 — Nice to have

- `mlx_conv1d`, `mlx_conv2d` — convolutions
- `mlx_fft_*` — Fourier transforms
- `mlx_linalg_*` — linear algebra (norm, svd, etc.)
- `mlx_einsum` — Einstein summation
- `mlx_quantize`, `mlx_dequantize` — quantization
- `mlx_fast_metal_kernel` — custom Metal kernels

## The Autograd Bridge

This is the most architecturally significant part of the binding.

### mlx-c closure types

mlx-c wraps function pointers in typed closures:

| Closure Type | Signature | Purpose |
|---|---|---|
| `mlx_closure` | `vec<array> → vec<array>` | General function (wrap loss fn) |
| `mlx_closure_value_and_grad` | `vec<array> → (vec<array>, vec<array>)` | Result of `mlx_value_and_grad` |
| `mlx_closure_custom` | Custom forward/backward | Custom VJP |

Each closure supports:
- `_new_func(fn_ptr)` — wrap a C function pointer
- `_new_func_payload(fn_ptr, payload, dtor)` — wrap with captured state + destructor
- `_apply(res, closure, inputs)` — invoke
- `_free(closure)` — cleanup

### The TypeScript → C → TypeScript flow

```
1. TypeScript defines a loss function:
   const lossFn = (params: MxArray[], x: MxArray, y: MxArray) => { ... }

2. We wrap it as a JSCallback with the signature mlx-c expects:
   const cb = new JSCallback(
     (outVec: Pointer | null, inVec: Pointer | null) => {
       /* unwrap inputs, call lossFn, write one result into outVec, return 0/1 */
     },
     { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.i32 }
   );

3. Create an mlx_closure from the JSCallback:
   const closure = mlx_closure_new_func(cb.ptr)

4. Create the value_and_grad transform:
   mlx_value_and_grad(&vag, closure, argnums, num_argnums)

5. Apply it to get loss + gradients:
   mlx_closure_value_and_grad_apply(&values, &grads, vag, inputs)
```

### The payload + destructor pattern

For closures that need to capture TypeScript state (e.g., model parameters), use the payload variant:

```typescript
// The payload carries a reference to our JS closure context
const payload = encodePayload({ lossFn, model });
const dtor = new JSCallback(
  (p: number) => { freePayload(p); },
  { args: [FFIType.ptr], returns: FFIType.void }
);

mlx_closure_new_func_payload(&closure, cb.ptr, payload, dtor.ptr);
```

This ensures mlx-c properly cleans up the payload when the closure is freed.

### Why `threadsafe: false` is sufficient

MLX's autograd calls the closure **synchronously on the calling thread** during graph construction. The actual GPU execution happens later during `eval()` and doesn't involve callbacks. Since graph construction is on Bun's main thread, `JSCallback` with `threadsafe: false` (the default) works correctly.

This was validated with a proof-of-concept during Phase 0.5 — see PLAN.md.

## Composing Missing Operations

Some operations needed for nanoGPT don't exist as single mlx-c functions but compose from primitives:

### Cross-entropy loss
```typescript
function crossEntropy(logits: MxArray, targets: MxArray): MxArray {
  // log_softmax = logits - logsumexp(logits, axis=-1, keepdims=true)
  const lse = mx.logsumexp(logits, -1, true);
  const logSoftmax = mx.subtract(logits, lse);

  // nll = -gather(log_softmax, targets, axis=-1)
  const gathered = mx.takeAlongAxis(logSoftmax, mx.expandDims(targets, -1), -1);
  const nll = mx.negative(mx.squeeze(gathered, -1));

  return mx.mean(nll);
}
```

### GELU activation
```typescript
function gelu(x: MxArray): MxArray {
  // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
  const scaled = mx.divide(x, mx.sqrt(mx.full([], 2.0)));
  const cdf = mx.multiply(mx.add(mx.erf(scaled), mx.full([], 1.0)), mx.full([], 0.5));
  return mx.multiply(x, cdf);
}
```

### Dropout
```typescript
function dropout(x: MxArray, p: number, key: MxArray): MxArray {
  const mask = mx.random.bernoulli(1.0 - p, x.shape, key);
  const scale = 1.0 / (1.0 - p);
  return mx.multiply(mx.multiply(x, mask), mx.full([], scale));
}
```

## Building and Linking

### Prerequisites

mlx-c builds from source via CMake and automatically fetches MLX:

```bash
# Clone mlx-c
git clone https://github.com/ml-explore/mlx-c.git
cd mlx-c

# Build (fetches MLX v0.31.1 automatically via FetchContent)
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)

# Output: libmlxc.dylib (links against libmlx.dylib)
```

### Integration with mlx-ts (implemented)

The build script (`packages/mlx-ts/scripts/build-native.ts`) automates the full pipeline:

1. Resolves the Xcode SDK path (required for Metal compiler access)
2. Runs `cmake configure` with `-DCMAKE_OSX_SYSROOT` pointing to the Xcode SDK
3. Runs `cmake --build` to compile mlx-c (which fetches MLX automatically via FetchContent)
4. Locates the built `.dylib` files in the CMake build tree
5. Copies `libmlxc.dylib` and `libmlx.dylib` to `packages/mlx-ts/native/lib/`
6. Fixes rpaths via `install_name_tool` so `libmlxc.dylib` finds `libmlx.dylib` at `@loader_path/`

```bash
# Build command
cd packages/mlx-ts && bun run build:native
```

See [docs/setup.md](./setup.md) for full prerequisites (macOS 14+, Xcode 16+, Metal Toolchain, CMake 3.24+).

### FFI loading

All symbols are loaded via a single `dlopen` in `src/core/ffi/lib.ts`:

```typescript
import { dlopen, FFIType } from "bun:ffi";
import { resolve } from "path";

const DYLIB_PATH = resolve(import.meta.dirname, "../../native/lib/libmlxc.dylib");

// Symbols are grouped by category then spread into one dlopen call
const lib = dlopen(DYLIB_PATH, {
  ...ARRAY_LIFECYCLE_SYMBOLS,
  ...ARITHMETIC_SYMBOLS,
  ...REDUCTION_SYMBOLS,
  // ... etc
});
```

### Pointer boundary discipline

Bun FFI uses a branded `Pointer` type (`{ __pointer__: null }`) that is distinct from `number` at the type level. This branded type lives at the FFI boundary only. Two helpers localize the boundary weirdness:

- `unwrapPointer(ptr, label)` — asserts a `Pointer | null` is non-null, throwing descriptively
- `sizeToNumber(value, label)` — converts `number | bigint` size_t returns to plain `number`

No code above the FFI layer needs to know about Bun's pointer branding.
