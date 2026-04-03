# MLX Bindings — Technical Guide

## What is MLX?

[MLX](https://github.com/ml-explore/mlx) is Apple's machine learning framework for Apple Silicon. It provides:

- **NumPy-like API** for tensor operations
- **Metal GPU acceleration** on Apple Silicon (M1/M2/M3/M4)
- **Unified memory** — CPU and GPU share the same memory, no explicit transfers
- **Lazy evaluation** — operations build a graph, compute only when needed
- **Functional autograd** — gradients via function transformation (like JAX)

MLX is written in C++ with a Python frontend. Our job is to replace that Python frontend with TypeScript.

## Binding Strategy

### The three layers

```
Layer 3:  TypeScript public API     (what users write)
          mx.matmul(a, b)

Layer 2:  Bun FFI bridge            (loads .dylib, calls C functions)
          ffi.mlx_matmul(a._ptr, b._ptr)

Layer 1:  C wrapper                 (extern "C" over C++ API)
          mlx_array* mlx_matmul(mlx_array* a, mlx_array* b)

Layer 0:  MLX C++ core              (Apple's library, untouched)
          mlx::core::matmul(a, b)
```

### Why a C wrapper?

Bun's FFI (and Node's N-API) can only call functions with C linkage. MLX's API is C++ — classes, templates, namespaces. The C wrapper provides `extern "C"` functions that:

1. Accept and return opaque pointers (`void*` cast to/from `mlx::core::array*`)
2. Convert C types to C++ types (int enums → dtype enum class, etc.)
3. Handle exceptions (C++ exceptions → error codes)

### Example: matmul binding

**Layer 1 — C wrapper:**
```c
// native/mlx_wrapper.h
extern "C" {
    mlx_array* mlx_matmul(mlx_array* a, mlx_array* b);
}

// native/mlx_wrapper.cpp
mlx_array* mlx_matmul(mlx_array* a, mlx_array* b) {
    auto result = mlx::core::matmul(
        *reinterpret_cast<mlx::core::array*>(a),
        *reinterpret_cast<mlx::core::array*>(b)
    );
    return new mlx::core::array(std::move(result));
}
```

**Layer 2 — Bun FFI:**
```typescript
// src/core/ffi.ts
import { dlopen, suffix, ptr } from "bun:ffi";

const lib = dlopen(`libmlx_wrapper.${suffix}`, {
  mlx_matmul: {
    args: ["ptr", "ptr"],
    returns: "ptr",
  },
});

export const ffi = lib.symbols;
```

**Layer 3 — TypeScript API:**
```typescript
// src/core/ops.ts
export function matmul(a: MxArray, b: MxArray): MxArray {
  const resultPtr = ffi.mlx_matmul(a.ptr, b.ptr);
  return MxArray.fromPtr(resultPtr);
}
```

## MLX C++ API Surface

The following is the complete set of C++ APIs we need to bind, grouped by priority.

### Priority 1 — Required for nanoGPT

**Array lifecycle:**
| C++ Function | Purpose |
|---|---|
| `array(data, shape, dtype)` | Create from data |
| `zeros(shape, dtype)` | Zero-filled array |
| `ones(shape, dtype)` | One-filled array |
| `full(shape, val, dtype)` | Constant-filled array |
| `arange(start, stop, step, dtype)` | Range array |
| `array::shape()` | Get shape |
| `array::dtype()` | Get dtype |
| `array::ndim()` | Get number of dimensions |
| `array::size()` | Get total element count |
| `array::eval()` | Force evaluation |

**Arithmetic:**
| C++ Function | Purpose |
|---|---|
| `add(a, b)` | Element-wise addition |
| `subtract(a, b)` | Element-wise subtraction |
| `multiply(a, b)` | Element-wise multiplication |
| `divide(a, b)` | Element-wise division |
| `negative(a)` | Negate |
| `power(a, b)` | Element-wise power |
| `sqrt(a)` | Square root |
| `exp(a)` | Exponential |
| `log(a)` | Natural logarithm |
| `abs(a)` | Absolute value |
| `maximum(a, b)` | Element-wise max |
| `minimum(a, b)` | Element-wise min |

**Reductions:**
| C++ Function | Purpose |
|---|---|
| `sum(a, axes)` | Sum reduction |
| `mean(a, axes)` | Mean reduction |
| `max(a, axes)` | Max reduction |
| `min(a, axes)` | Min reduction |
| `argmax(a, axis)` | Index of max |
| `argmin(a, axis)` | Index of min |

**Linear algebra:**
| C++ Function | Purpose |
|---|---|
| `matmul(a, b)` | Matrix multiplication |
| `transpose(a, axes)` | Transpose |
| `reshape(a, shape)` | Reshape |
| `squeeze(a, axes)` | Remove size-1 dims |
| `expand_dims(a, axis)` | Add size-1 dim |
| `concatenate(arrays, axis)` | Join arrays |
| `split(a, indices, axis)` | Split array |
| `stack(arrays, axis)` | Stack arrays |

**Comparison and selection:**
| C++ Function | Purpose |
|---|---|
| `equal(a, b)` | Element-wise equality |
| `greater(a, b)` | Element-wise greater |
| `less(a, b)` | Element-wise less |
| `where(condition, a, b)` | Conditional select |

**Indexing:**
| C++ Function | Purpose |
|---|---|
| `take(a, indices, axis)` | Gather by index |
| `take_along_axis(a, indices, axis)` | Gather along axis |

**Neural network ops:**
| C++ Function | Purpose |
|---|---|
| `softmax(a, axis)` | Softmax |

**Random:**
| C++ Function | Purpose |
|---|---|
| `random::key(seed)` | Create RNG key |
| `random::split(key)` | Split RNG key |
| `random::normal(shape, dtype, key)` | Normal distribution |
| `random::uniform(shape, dtype, key)` | Uniform distribution |

**Transforms:**
| C++ Function | Purpose |
|---|---|
| `eval(arrays)` | Force evaluation |
| `grad(fn, argnums)` | Gradient transform |
| `value_and_grad(fn, argnums)` | Loss + gradient |

### Priority 2 — Nice to have

- `conv1d`, `conv2d` — convolutions
- `fft`, `ifft` — Fourier transforms
- `linalg::norm`, `linalg::svd` — linear algebra
- `compile(fn)` — JIT compilation
- `vmap(fn)` — vectorized map
- `astype(a, dtype)` — type casting
- `broadcast_to(a, shape)` — broadcasting

## The Autograd Challenge

This is the hardest part of the binding and deserves detailed explanation.

### How MLX autograd works internally

When you call `mx.grad(fn)`, MLX:
1. Takes your function `fn`
2. Returns a new function `grad_fn`
3. When `grad_fn` is called, MLX:
   a. Runs `fn` while tracing the computation graph
   b. Performs reverse-mode autodiff on the graph
   c. Returns the gradients

The key insight: **MLX doesn't need to call back into your function repeatedly.** It calls `fn` once to build the graph, then differentiates the graph. The graph is built from MLX operations (matmul, add, etc.) which are all C++.

### Why this helps us

Since `fn` is called exactly once, the callback from C++ to TypeScript happens once per grad call. The flow is:

```
TypeScript: gradFn = mx.grad(lossFn)
TypeScript: grads = gradFn(params, x, y)
  → C++: call lossFn(params, x, y) via callback
    → TypeScript: lossFn runs, calling mx.matmul, mx.add, etc.
      → C++: each op adds to computation graph
    → TypeScript: lossFn returns loss MxArray
  → C++: differentiate the computation graph
  → C++: return gradients
TypeScript: grads is now available
```

### Implementation approach

1. Register a TypeScript callback function pointer with the C wrapper
2. The C wrapper's grad implementation calls this function pointer
3. The callback executes the TypeScript loss function
4. All tensor ops within the callback go through FFI as normal
5. MLX traces them into its computation graph automatically
6. After the callback returns, C++ differentiates and returns gradients

This is feasible with Bun's FFI callback support (`JSCallback`).

## Building and Linking

### Prerequisites
```bash
# Install MLX (provides the compiled library)
pip install mlx

# Or build from source
git clone https://github.com/ml-explore/mlx.git
cd mlx && mkdir build && cd build
cmake .. -DMLX_BUILD_METAL=ON
make -j
```

### Finding MLX headers and library
```bash
# If installed via pip
MLX_PATH=$(python3 -c "import mlx; print(mlx.__path__[0])")

# Headers are in the MLX repo
# Library is at $MLX_PATH/lib/libmlx.dylib (or similar)
```

### CMake configuration
```cmake
cmake_minimum_required(VERSION 3.20)
project(mlx_wrapper)

find_package(MLX REQUIRED)

add_library(mlx_wrapper SHARED
    mlx_wrapper.cpp
)

target_link_libraries(mlx_wrapper PRIVATE mlx)
target_include_directories(mlx_wrapper PRIVATE ${MLX_INCLUDE_DIRS})

# Output to a location Bun can find
set_target_properties(mlx_wrapper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
```
