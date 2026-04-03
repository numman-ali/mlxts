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
          ffi.mlx_matmul(res, a._ptr, b._ptr)

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
- Pointers as `number` (not BigInt) — avoids allocation overhead

Performance (measured on M4 Max, Bun 1.3.4):
- Basic FFI call: ~8ns (negligible vs any ML operation)
- Callback round-trip: ~35ns (negligible vs a matmul)

### Example: matmul binding

**Layer 1 — mlx-c provides:**
```c
// From mlx/c/ops.h (auto-generated)
int mlx_matmul(mlx_array* res, mlx_array a, mlx_array b, mlx_stream s);
```

Note the pattern: result is written to a pre-allocated output pointer. All ops follow this convention.

**Layer 2 — Bun FFI:**
```typescript
// src/core/ffi.ts
import { dlopen, FFIType } from "bun:ffi";

const lib = dlopen("libmlxc.dylib", {
  mlx_matmul: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
});

export const ffi = lib.symbols;
```

**Layer 3 — TypeScript API:**
```typescript
// src/core/ops.ts
export function matmul(a: MxArray, b: MxArray): MxArray {
  const result = MxArray.empty();
  const status = ffi.mlx_matmul(result.ptr, a.ptr, b.ptr, defaultStream);
  if (status !== 0) throw new MxError("matmul", status);
  return result;
}
```

## mlx-c API Surface

The following catalogs the mlx-c functions we need, verified against the v0.6.0 source.

### Priority 1 — Required for nanoGPT

**Array lifecycle:**
| mlx-c Function | Purpose |
|---|---|
| `mlx_array_new_data(res, data, shape, ndim, dtype)` | Create from data |
| `mlx_zeros(res, shape, ndim, dtype, stream)` | Zero-filled array |
| `mlx_ones(res, shape, ndim, dtype, stream)` | One-filled array |
| `mlx_full(res, shape, ndim, val, dtype, stream)` | Constant-filled array |
| `mlx_arange(res, start, stop, step, dtype, stream)` | Range array |
| `mlx_array_shape(shape, ndim, a)` | Get shape |
| `mlx_array_dtype(dtype, a)` | Get dtype |
| `mlx_array_ndim(ndim, a)` | Get number of dimensions |
| `mlx_array_size(size, a)` | Get total element count |
| `mlx_array_eval(a)` | Force evaluation |
| `mlx_array_free(a)` | Deallocate |

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

**I/O:**
| mlx-c Function | Purpose |
|---|---|
| `mlx_load_safetensors` | Load model weights |
| `mlx_save_safetensors` | Save model weights |

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
     (inputs: Pointer) => { /* unwrap inputs, call lossFn, wrap outputs */ },
     { args: [FFIType.ptr], returns: FFIType.ptr }
   );

3. Create an mlx_closure from the JSCallback:
   mlx_closure_new_func(&closure, cb.ptr)

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

### Integration with mlx-ts

Our build script will:
1. Check if mlx-c is already built (cache the .dylib)
2. If not, clone and build mlx-c from source
3. Copy libmlxc.dylib (and libmlx.dylib if needed) to a known location
4. Bun's `dlopen` loads from that location

```typescript
// src/core/ffi.ts
import { dlopen, FFIType } from "bun:ffi";
import { resolve } from "path";

const DYLIB_PATH = resolve(__dirname, "../../native/lib/libmlxc.dylib");

const lib = dlopen(DYLIB_PATH, {
  // Array creation
  mlx_zeros: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.i32,
  },
  // ... all other symbols
});

export const ffi = lib.symbols;
```

### Finding the built libraries

After building mlx-c:
- `libmlxc.dylib` — the C API wrapper
- `libmlx.dylib` — MLX core (fetched and built by mlx-c's CMake)

Both need to be accessible at runtime. We'll either:
- Bundle them in `packages/mlx-ts/native/lib/`
- Or install them to a system location and use `@rpath`

### Homebrew alternative

If MLX or mlx-c becomes available via Homebrew in the future, we can simplify to:
```bash
brew install mlx-c
```
Until then, building from source is the reliable path.
