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
│                           │ FFI calls                   │
│  ┌────────────────────────┴──────────────────────────┐  │
│  │            C wrapper (native/)                     │  │
│  │  extern "C" functions wrapping MLX C++ API        │  │
│  │  Compiled to .dylib                               │  │
│  └────────────────────────┬──────────────────────────┘  │
│                           │ links against               │
└───────────────────────────┼─────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │     MLX (Apple's library)   │
              │  C++ core + Metal backend   │
              │  GPU kernels for Apple Si   │
              └────────────────────────────┘
```

## Layer Responsibilities

### MLX (external dependency)
- Tensor storage and memory management
- Metal GPU kernel dispatch
- Computation graph construction and evaluation
- Automatic differentiation engine
- Lazy evaluation and JIT compilation

We don't modify MLX. We bind to it.

### C Wrapper (`packages/mlx-ts/native/`)
- Provides `extern "C"` linkage for MLX's C++ API
- Handles C++ object lifecycle (create/retain/release via opaque pointers)
- Converts between C types and MLX types
- Compiled to a shared library (.dylib) via CMake

**Design principle**: The C wrapper is as thin as possible. No logic, no state, no allocations beyond what MLX does internally. It's pure translation.

### Core Bindings (`packages/mlx-ts/src/core/`)
- Loads the .dylib via Bun FFI
- Wraps opaque pointers in TypeScript classes
- Manages memory via FinalizationRegistry + explicit disposal
- Provides idiomatic TypeScript API for tensor operations
- Handles type conversions (JS arrays ↔ MLX arrays)

**Key types**:
```typescript
class MxArray {
  // Wraps an opaque pointer to mlx::core::array
  readonly shape: number[]
  readonly dtype: DType
  readonly ndim: number
  readonly size: number

  toList(): NestedArray<number>
  item(): number  // for scalar arrays
  asType(dtype: DType): MxArray
}

// Operations are free functions (matches MLX's style)
function matmul(a: MxArray, b: MxArray): MxArray
function add(a: MxArray, b: MxArray): MxArray
function reshape(a: MxArray, shape: number[]): MxArray
// ... etc
```

### NN Layer (`packages/mlx-ts/src/nn/`)
- Pure TypeScript (no native code)
- Calls into core bindings for all tensor operations
- Provides familiar Module-based API (like PyTorch/MLX Python)

**Key pattern**:
```typescript
abstract class Module {
  abstract forward(...args: MxArray[]): MxArray

  parameters(): Map<string, MxArray>
  update(params: Map<string, MxArray>): void
  // ... tree traversal, serialization
}

class Linear extends Module {
  weight: MxArray
  bias?: MxArray

  forward(x: MxArray): MxArray {
    // x @ weight.T + bias — uses core bindings
    return mx.add(mx.matmul(x, mx.transpose(this.weight)), this.bias)
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

### The solution

```
┌──────────────────────────────────────────────┐
│             TypeScript (GC-managed)          │
│                                              │
│  MxArray {                                   │
│    _ptr: Pointer  ──────────────────────┐    │
│  }                                      │    │
│                                         │    │
│  FinalizationRegistry watches MxArray   │    │
│  instances — calls release() when GC'd  │    │
└─────────────────────────────────────────┼────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────┐
│             C++ (reference counted)          │
│                                              │
│  mlx::core::array* (ref count managed)       │
│                                              │
│  create()  → ref count = 1                   │
│  retain()  → ref count += 1                  │
│  release() → ref count -= 1                  │
│              if 0, free                      │
└──────────────────────────────────────────────┘
```

**Rules**:
1. Every `create` or `retain` must have a matching `release`
2. FinalizationRegistry provides a safety net, but don't rely on it for timely cleanup
3. In hot loops (training), use explicit disposal via `using` declarations
4. The C wrapper never stores references — all pointers are owned by TypeScript

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

This is more elegant and composable, but the FFI challenge is that `lossFn` is a TypeScript function that MLX's C++ core needs to trace through. The mechanism for this callback bridge is designed in Phase 2.

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
