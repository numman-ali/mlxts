# Future Backends — Vision Document

> **Status: Future vision only. Not part of the current plan.**
>
> mlxts is MLX-native. All packages use MxArray directly. There is no backend abstraction layer today.
>
> This document preserves design thinking about what multi-backend support *could* look like if demand warrants it. The interfaces below are aspirational, not implemented. See PLAN.md Phase 11 for the trigger criteria.
>
> **Key research findings from independent review (2026-04-04):**
> - libtorch has no official C API. The stable ABI is limited and "not intended to replace existing LibTorch."
> - WebGPU readback is async (`mapAsync`), which conflicts with synchronous `toTypedArray()`.
> - Bun FFI has known bugs with JSCallback and pointer handling that affect complex bindings.
> - Keras 3's multi-backend lesson: design the interface after having multiple backends working, not before.

---

# Original Design Thinking (Preserved for Reference)

How mlxts could isolate compute backends behind a single interface, and how MLX, CUDA, WebGPU, and custom backends could plug in.

## 1. Design Philosophy

mlxts is a TypeScript-native ML framework. In a multi-backend future, the framework layers -- `@mlxts/nn`, `@mlxts/train`, `@mlxts/optimizers`, model architectures -- would not call FFI directly. They would operate on an abstract tensor type through a backend interface.

**MLX-first.** The MLX backend ships first and is optimized for Apple Silicon. Other backends would extend reach but not compromise the primary path.

**The backend interface is the contract.** Framework code depends on types and behavior defined in `@mlxts/core`. A backend implements those types. If the interface is wrong, every backend inherits the mistake. Get the interface right first.

**Lazy evaluation is the default mental model.** MLX is lazy: operations build a graph, `eval()` forces computation. The interface assumes laziness. Eager backends (CUDA, WebGPU) execute immediately when an op is called, and their `eval()` is a no-op. This is fine -- laziness is a superset of eagerness from the caller's perspective.

**Native autograd is optional.** MLX provides functional autograd natively. CUDA can get it from libtorch. WebGPU cannot. The interface declares autograd as an optional capability. When the backend does not provide it, `@mlxts/core` supplies a TypeScript-level tape-based fallback.

**No lowest-common-denominator design.** Backends expose optional capabilities (fused kernels, native compile, native I/O). Framework code checks for these and uses them when available, falling back to composed primitives when not. A backend that supports `scaledDotProductAttention` natively will always be faster than one that composes it from matmul + softmax + mask.

## 2. The Backend Interface

Every backend implements `ComputeBackend`. The tensor type is opaque -- backends define their own internal representation, but all tensors are accessed through the interface.

### 2.1 Tensor Handle

```typescript
/**
 * Opaque tensor handle. Each backend defines its own internal representation.
 *
 * The framework never inspects the handle directly. All access goes through
 * the backend interface. Handles are Disposable for explicit lifetime control.
 */
export interface TensorHandle extends Disposable {
  /** Number of dimensions. */
  readonly ndim: number;
  /** Shape of the tensor. */
  readonly shape: readonly number[];
  /** Element data type. */
  readonly dtype: DType;
  /** Total number of elements. */
  readonly size: number;
  /** Bytes per element. */
  readonly itemsize: number;
  /** Total bytes in the evaluated tensor. */
  readonly nbytes: number;
  /** Whether this handle has been disposed. */
  readonly isDisposed: boolean;
  /** Force evaluation (no-op for eager backends). */
  eval(): void;
  /** Copy data to a JavaScript TypedArray. */
  toTypedArray(): SupportedTypedArray;
  /** Read a single scalar value. */
  item(): number;
  /** Cast to a different dtype. */
  asType(dtype: DType): TensorHandle;
  /** Explicitly free the underlying native resource. */
  free(): void;
}

/** Supported TypedArray types for data transfer. */
export type SupportedTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;
```

### 2.2 Core Backend Interface

```typescript
/**
 * The contract every compute backend must implement.
 *
 * Framework code (`@mlxts/nn`, `@mlxts/train`, `@mlxts/optimizers`) calls
 * only these methods. No FFI, no native pointers, no backend-specific types
 * leak above this boundary.
 */
export interface ComputeBackend {
  /** Human-readable backend name (e.g., "mlx", "cuda", "webgpu"). */
  readonly name: string;

  // ---- Tensor Creation ----

  /** Create a tensor filled with zeros. */
  zeros(shape: number[], dtype?: DType): TensorHandle;
  /** Create a tensor filled with ones. */
  ones(shape: number[], dtype?: DType): TensorHandle;
  /** Create a tensor filled with a constant value. */
  full(shape: number[], value: number, dtype?: DType): TensorHandle;
  /** Create a tensor with evenly spaced values. */
  arange(start: number, stop: number, step?: number, dtype?: DType): TensorHandle;
  /** Create a tensor from a JavaScript TypedArray. */
  fromTypedArray(
    data: SupportedTypedArray,
    shape?: number[],
    dtype?: DType,
  ): TensorHandle;
  /** Create a scalar tensor. */
  fromScalar(value: number, dtype?: DType): TensorHandle;

  // ---- Binary Ops ----

  /** Element-wise addition. */
  add(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise subtraction. */
  subtract(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise multiplication. */
  multiply(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise division. */
  divide(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Matrix multiplication. */
  matmul(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise power. */
  power(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise maximum of two tensors. */
  maximum(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise minimum of two tensors. */
  minimum(a: TensorHandle, b: TensorHandle): TensorHandle;

  // ---- Unary Ops ----

  /** Element-wise exponential (e^x). */
  exp(a: TensorHandle): TensorHandle;
  /** Element-wise natural logarithm. */
  log(a: TensorHandle): TensorHandle;
  /** Element-wise square root. */
  sqrt(a: TensorHandle): TensorHandle;
  /** Element-wise square. */
  square(a: TensorHandle): TensorHandle;
  /** Element-wise absolute value. */
  abs(a: TensorHandle): TensorHandle;
  /** Element-wise negation. */
  negative(a: TensorHandle): TensorHandle;
  /** Element-wise reciprocal (1/x). */
  reciprocal(a: TensorHandle): TensorHandle;
  /** Element-wise error function. */
  erf(a: TensorHandle): TensorHandle;
  /** Element-wise sigmoid. */
  sigmoid(a: TensorHandle): TensorHandle;
  /** Element-wise hyperbolic tangent. */
  tanh(a: TensorHandle): TensorHandle;

  // ---- Comparison Ops ----

  /** Element-wise equality. */
  equal(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise inequality. */
  notEqual(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise greater than. */
  greater(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise less than. */
  less(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise greater than or equal. */
  greaterEqual(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Element-wise less than or equal. */
  lessEqual(a: TensorHandle, b: TensorHandle): TensorHandle;
  /** Conditional select: condition ? x : y, element-wise. */
  where(
    condition: TensorHandle,
    x: TensorHandle,
    y: TensorHandle,
  ): TensorHandle;

  // ---- Reduction Ops ----

  /** Sum reduction. */
  sum(a: TensorHandle, axis?: number | number[], keepdims?: boolean): TensorHandle;
  /** Mean reduction. */
  mean(a: TensorHandle, axis?: number | number[], keepdims?: boolean): TensorHandle;
  /** Max reduction. */
  max(a: TensorHandle, axis?: number | number[], keepdims?: boolean): TensorHandle;
  /** Min reduction. */
  min(a: TensorHandle, axis?: number | number[], keepdims?: boolean): TensorHandle;
  /** Index of maximum value along an axis. */
  argmax(a: TensorHandle, axis?: number, keepdims?: boolean): TensorHandle;
  /** Index of minimum value along an axis. */
  argmin(a: TensorHandle, axis?: number, keepdims?: boolean): TensorHandle;
  /** Softmax along an axis. */
  softmax(a: TensorHandle, axis?: number): TensorHandle;
  /** Log-sum-exp reduction (numerically stable). */
  logsumexp(a: TensorHandle, axis?: number | number[], keepdims?: boolean): TensorHandle;

  // ---- Shape Ops ----

  /** Reshape a tensor. */
  reshape(a: TensorHandle, shape: number[]): TensorHandle;
  /** Transpose a tensor. Without axes: reverse dimensions. With axes: permute. */
  transpose(a: TensorHandle, axes?: number[]): TensorHandle;
  /** Broadcast a tensor to a target shape. */
  broadcastTo(a: TensorHandle, shape: number[]): TensorHandle;
  /** Add a size-1 dimension at the given axis. */
  expandDims(a: TensorHandle, axis: number): TensorHandle;
  /** Remove size-1 dimensions. */
  squeeze(a: TensorHandle, axis?: number): TensorHandle;
  /** Concatenate tensors along an axis. */
  concatenate(tensors: TensorHandle[], axis?: number): TensorHandle;
  /** Split a tensor into equal parts along an axis. */
  split(a: TensorHandle, numSplits: number, axis?: number): TensorHandle[];
  /** Stack tensors along a new axis. */
  stack(tensors: TensorHandle[], axis?: number): TensorHandle;
  /** Flatten dimensions of a tensor. */
  flatten(a: TensorHandle, startAxis?: number, endAxis?: number): TensorHandle;
  /** Gather elements along an axis by index. */
  takeAlongAxis(
    a: TensorHandle,
    indices: TensorHandle,
    axis: number,
  ): TensorHandle;
  /** Stop gradient propagation through this tensor. */
  stopGradient(a: TensorHandle): TensorHandle;
  /** Extract lower triangle. */
  tril(a: TensorHandle, k?: number): TensorHandle;
  /** Extract upper triangle. */
  triu(a: TensorHandle, k?: number): TensorHandle;

  // ---- Random ----

  /** Generate random values from a normal distribution. */
  randomNormal(
    shape: number[],
    dtype?: DType,
    loc?: number,
    scale?: number,
  ): TensorHandle;
  /** Generate random values from a uniform distribution. */
  randomUniform(
    low: number,
    high: number,
    shape: number[],
    dtype?: DType,
  ): TensorHandle;
  /** Generate random Bernoulli samples. */
  randomBernoulli(p: number, shape: number[]): TensorHandle;
  /** Sample from a categorical distribution. */
  randomCategorical(
    logits: TensorHandle,
    axis?: number,
  ): TensorHandle;
  /** Set the global random seed. */
  randomSeed(value: number): void;

  // ---- Evaluation ----

  /** Force evaluation of one or more tensors. No-op for eager backends. */
  eval(...tensors: TensorHandle[]): void;
  /** Wait for all queued computation to finish. */
  synchronize(): void;

  // ---- Memory ----

  /** Current allocator usage in bytes. */
  getActiveMemoryBytes(): number;
  /** Current cache usage in bytes. */
  getCacheMemoryBytes(): number;
  /** Release cached memory that is no longer actively used. */
  clearMemoryCache(): void;

  // ---- Device ----

  /** Get the current default device type. */
  getDefaultDevice(): DeviceType;
  /** Set the default device type. */
  setDefaultDevice(device: DeviceType): void;

  // ---- Optional Capabilities ----

  /** Native autograd support. Undefined means use the TS-level fallback. */
  readonly autograd?: AutogradCapability;
  /** Native graph compilation / fusion. */
  readonly compile?: CompileCapability;
  /** Native fused operations for hot paths. */
  readonly fast?: FastOpsCapability;
  /** Native safetensors I/O. */
  readonly io?: IOCapability;
}
```

### 2.3 Optional Capability Interfaces

```typescript
/** Native autograd. When absent, @mlxts/core provides a tape-based TS fallback. */
export interface AutogradCapability {
  /**
   * Create a function that returns both value and gradients.
   *
   * @param fn - Loss function: tensors in, scalar tensor out.
   * @param argnums - Which positional arguments to differentiate. Defaults to [0].
   * @returns A function returning [value, gradients].
   */
  valueAndGrad(
    fn: (...args: TensorHandle[]) => TensorHandle,
    argnums?: number | number[],
  ): (...args: TensorHandle[]) => [TensorHandle, TensorHandle[]];

  /**
   * Create a function that returns only gradients.
   */
  grad(
    fn: (...args: TensorHandle[]) => TensorHandle,
    argnums?: number | number[],
  ): (...args: TensorHandle[]) => TensorHandle | TensorHandle[];
}

/** Native graph compilation / kernel fusion. */
export interface CompileCapability {
  /** Compile a function for repeated execution. */
  compile(
    fn: (...args: TensorHandle[]) => TensorHandle,
    options?: { shapeless?: boolean },
  ): (...args: TensorHandle[]) => TensorHandle;

  /** Gradient checkpointing: recompute intermediates during backward pass. */
  checkpoint(
    fn: (...args: TensorHandle[]) => TensorHandle,
  ): (...args: TensorHandle[]) => TensorHandle;
}

/** Native fused operations. Each is optional within this group. */
export interface FastOpsCapability {
  /** Fused scaled dot-product attention. */
  scaledDotProductAttention?(
    queries: TensorHandle,
    keys: TensorHandle,
    values: TensorHandle,
    options: {
      scale: number;
      maskMode?: '' | 'causal' | 'array';
      maskArray?: TensorHandle;
    },
  ): TensorHandle;

  /** Fused layer normalization. */
  layerNorm?(
    x: TensorHandle,
    weight?: TensorHandle,
    bias?: TensorHandle,
    options?: { eps?: number },
  ): TensorHandle;

  /** Fused RMS normalization. */
  rmsNorm?(
    x: TensorHandle,
    weight?: TensorHandle,
    eps?: number,
  ): TensorHandle;

  /** Fused rotary position embedding. */
  rope?(
    x: TensorHandle,
    dimensions: number,
    traditional?: boolean,
    base?: number,
    scale?: number,
    offset?: number,
  ): TensorHandle;
}

/** Native tensor serialization. */
export interface IOCapability {
  /** Load tensors from a safetensors file. */
  loadSafetensors(path: string): Promise<{
    tensors: Record<string, TensorHandle>;
    metadata: Record<string, string>;
  }>;
  /** Save tensors to a safetensors file. */
  saveSafetensors(
    tensors: Record<string, TensorHandle>,
    path: string,
    metadata?: Record<string, string>,
  ): Promise<void>;
}
```

### 2.4 DType and DeviceType (Shared)

These live in `@mlxts/core` and are backend-agnostic.

```typescript
/** Supported element types. Subset available depends on backend. */
export type DType =
  | 'bool'
  | 'uint8' | 'uint16' | 'uint32' | 'uint64'
  | 'int8' | 'int16' | 'int32' | 'int64'
  | 'float16' | 'float32' | 'float64'
  | 'bfloat16'
  | 'complex64';

/** Logical device type. */
export type DeviceType = 'cpu' | 'gpu';
```

## 3. How MLX Maps to the Interface

The current extracted MLX-first package stack, centered on `packages/core`, implements every required method in the `ComputeBackend` interface, plus all four optional capabilities.

| Interface method | Current implementation | File |
|---|---|---|
| `zeros`, `ones`, `full`, `arange` | `zeros()`, `ones()`, `full()`, `arange()` | `src/core/array.ts` |
| `fromTypedArray` | `MxArray.fromData()` | `src/core/array.ts` |
| `fromScalar` | `array(number)` | `src/core/array.ts` |
| `add`, `subtract`, `multiply`, `divide` | Same-named exports | `src/core/ops/arithmetic.ts` |
| `power`, `maximum`, `minimum` | Same-named exports | `src/core/ops/arithmetic.ts` |
| `exp`, `log`, `sqrt`, `square`, `abs` | Same-named exports | `src/core/ops/arithmetic.ts` |
| `negative`, `reciprocal`, `erf`, `sigmoid`, `tanh` | Same-named exports | `src/core/ops/arithmetic.ts` |
| `matmul` | `matmul()` | `src/core/ops/linalg.ts` |
| `equal`, `notEqual`, `greater`, `less`, `greaterEqual`, `lessEqual`, `where` | Same-named exports | `src/core/ops/comparison.ts` |
| `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `softmax`, `logsumexp` | Same-named exports | `src/core/ops/reduction.ts` |
| `reshape`, `transpose`, `broadcastTo`, `expandDims`, `squeeze` | Same-named exports | `src/core/ops/shape.ts` |
| `concatenate`, `split`, `stack`, `flatten`, `takeAlongAxis`, `stopGradient` | Same-named exports | `src/core/ops/shape.ts` |
| `tril`, `triu` | Same-named exports | `src/core/ops/shape.ts` |
| `randomNormal`, `randomUniform`, `randomBernoulli`, `randomCategorical` | `random.normal()`, `random.uniform()`, etc. | `src/core/random.ts` |
| `randomSeed` | `random.seed()` | `src/core/random.ts` |
| `eval` | `mxEval()` | `src/core/transforms.ts` |
| `synchronize` | `synchronize()` | `src/core/device.ts` |
| `getActiveMemoryBytes`, `getCacheMemoryBytes`, `clearMemoryCache` | Same-named exports | `src/core/memory.ts` |
| `getDefaultDevice`, `setDefaultDevice` | Same-named exports | `src/core/device.ts` |
| `autograd.valueAndGrad`, `autograd.grad` | `valueAndGrad()`, `grad()` | `src/core/transforms.ts` |
| `compile.compile`, `compile.checkpoint` | `compile()`, `checkpoint()` | `src/core/transforms.ts` |
| `fast.scaledDotProductAttention`, `fast.layerNorm` | Same-named exports | `src/core/fast.ts` |
| `io.loadSafetensors`, `io.saveSafetensors` | Same-named exports | `src/core/io.ts` |

The MLX backend adapter wraps these existing functions. No new FFI code is needed. The adapter is a thin object that delegates each interface method to the corresponding free function, converting between `TensorHandle` and the internal `MxArray` type.

Key detail: `MxArray` already satisfies `TensorHandle`. The MLX backend can use `MxArray` directly as its `TensorHandle` implementation. The adapter layer is purely structural -- it packages the existing free functions into the `ComputeBackend` shape.

## 4. How CUDA Would Plug In

### 4.1 The Practical Path: libtorch C API via Bun FFI

The fastest route to a CUDA backend is wrapping libtorch's C API (`libtorch` ships as a `.so`/`.dylib`/`.dll` with stable C bindings via `torch/csrc/api`). This mirrors the approach used for MLX: C library with opaque pointers, Bun FFI for the binding.

```
@mlxts/cuda
  src/
    ffi/
      symbols.ts        # libtorch C ABI declarations
      lib.ts            # dlopen and symbol resolution
    tensor.ts           # CudaTensor implements TensorHandle
    backend.ts          # CudaBackend implements ComputeBackend
    autograd.ts         # Wraps torch::autograd (libtorch provides this)
```

libtorch provides:
- Tensor creation, all arithmetic/comparison/shape/reduction ops
- Native autograd (tape-based, wraps `torch::autograd`)
- CUDA kernels for every standard op
- cuDNN integration for conv, attention, normalization
- Serialization (`.pt` files, with a safetensors adapter)

### 4.2 Memory Model Differences

MLX uses unified memory -- CPU and GPU share the same address space on Apple Silicon. CUDA uses discrete GPU memory with explicit host-device transfers.

The interface hides this. `fromTypedArray()` on a CUDA backend internally performs a host-to-device copy. `toTypedArray()` performs a device-to-host copy. The user never calls `cuda.memcpy` or thinks about pinned memory.

For advanced use cases (overlapping compute with transfer), the backend could expose stream-ordered operations via the optional `compile` capability. But the base interface intentionally abstracts this away.

### 4.3 Autograd

libtorch provides tape-based autograd natively. The CUDA backend would populate `autograd.valueAndGrad` and `autograd.grad` by wrapping `torch::autograd::grad`. This is different from MLX's functional autograd (which returns a new function) but the interface abstracts both styles into the same signature.

### 4.4 What Changes in User Code

Nothing.

```typescript
import { setBackend } from '@mlxts/core';
import { cudaBackend } from '@mlxts/cuda';

setBackend(cudaBackend);

// Everything below this line is identical to MLX usage.
// @mlxts/nn, @mlxts/train, model code -- unchanged.
```

## 5. How WebGPU Would Plug In

### 5.1 Server-Side: Dawn via Bun FFI

Google's Dawn library is the reference C++ implementation of WebGPU. It exposes a C API (`webgpu.h`) that can be loaded via Bun FFI on macOS, Linux, and Windows without requiring a browser.

```
@mlxts/webgpu
  src/
    ffi/
      symbols.ts        # webgpu.h ABI declarations (Dawn)
      lib.ts            # dlopen Dawn native library
    shaders/
      arithmetic.wgsl   # WGSL compute shaders for element-wise ops
      reduction.wgsl    # WGSL compute shaders for reductions
      matmul.wgsl       # WGSL compute shader for matrix multiplication
    tensor.ts           # WebGPUTensor implements TensorHandle
    backend.ts          # WebGPUBackend implements ComputeBackend
```

### 5.2 Browser-Side: navigator.gpu

In browser environments, the same backend code targets `navigator.gpu` instead of Dawn FFI. The WGSL shaders are identical. The difference is how the GPU device is acquired and how buffers are created.

This enables a deployment story where the same model code runs:
- On Apple Silicon via `@mlxts/mlx` (fastest)
- On NVIDIA GPUs via `@mlxts/cuda`
- On any GPU (including integrated) via `@mlxts/webgpu`
- In the browser via `@mlxts/webgpu` with `navigator.gpu`

### 5.3 Autograd

WebGPU has no native autograd. The TypeScript-level tape-based fallback (see section 7) handles this. The `autograd` capability is left undefined on the WebGPU backend.

### 5.4 Performance Expectations

WebGPU will be slower than native backends for several reasons:
- WGSL shaders are general-purpose, not hand-tuned like Metal Performance Shaders or cuDNN
- No native fused kernels (attention, layer norm) unless manually written in WGSL
- Additional overhead from the WebGPU API abstraction layer
- Browser sandboxing adds latency for the browser deployment path

The tradeoff is universality. WebGPU runs on almost every modern GPU across all platforms. For inference and small-model training, this is often good enough. For large-scale training, use MLX or CUDA.

### 5.5 Optional Capabilities

The WebGPU backend would initially provide:
- No `autograd` (TS fallback)
- No `compile` (no-op)
- Possibly `fast.scaledDotProductAttention` via a hand-written WGSL kernel
- `io.loadSafetensors` and `io.saveSafetensors` via the shared TS-level implementation in `@mlxts/core`

## 6. Custom Backend / Plugin Extension

### 6.1 Implementing ComputeBackend

A third party implements the `ComputeBackend` interface for their hardware. The minimum viable backend needs only the required methods -- no optional capabilities are necessary.

```typescript
import type { ComputeBackend, TensorHandle, DType } from '@mlxts/core';

class VulkanTensor implements TensorHandle {
  // ... Vulkan buffer handle, shape, dtype, etc.
}

export const vulkanBackend: ComputeBackend = {
  name: 'vulkan',

  zeros(shape: number[], dtype: DType = 'float32'): TensorHandle {
    // Allocate Vulkan buffer, fill with zeros
    return new VulkanTensor(/* ... */);
  },

  add(a: TensorHandle, b: TensorHandle): TensorHandle {
    // Dispatch Vulkan compute shader
    return new VulkanTensor(/* ... */);
  },

  // ... implement all required methods
};
```

### 6.2 Custom FFI Bindings

Backends that wrap native libraries follow the same pattern as MLX and CUDA:

1. Write C or Rust bindings that expose a C ABI
2. Declare symbols in a `symbols.ts` file
3. Load the library with `Bun.dlopen()` (or `Deno.dlopen()` if Deno support comes later)
4. Wrap each symbol call in the `ComputeBackend` method implementations

The key constraint: the `TensorHandle` returned must fully satisfy the interface. Internal details (Vulkan descriptor sets, ROCm HIP pointers, XLA buffers) stay entirely within the backend package.

### 6.3 Extending with Custom Ops

Backends that need operations not in the standard interface can expose them as backend-specific extensions. Framework code will not call these directly, but user code at the application level can.

```typescript
import { getBackend } from '@mlxts/core';
import type { RocmBackend } from '@mlxts/rocm';

// Type-narrow to the specific backend for custom ops
const backend = getBackend() as RocmBackend;
const result = backend.rocmCustomFusedOp(a, b, c);
```

This preserves portability for framework code while allowing backend-specific optimization at the application layer.

### 6.4 Seamless Integration with @mlxts/nn and @mlxts/train

Once a backend is registered, all framework code works automatically:

```typescript
import { setBackend } from '@mlxts/core';
import { vulkanBackend } from 'my-vulkan-backend';
import { Linear, Module } from '@mlxts/nn';
import { AdamW } from '@mlxts/optimizers';

setBackend(vulkanBackend);

// Module, Linear, AdamW all call through the backend interface.
// No Vulkan-specific code needed in the model or training loop.
```

### 6.5 Realistic Backend Candidates

| Backend | FFI target | Autograd | Key use case |
|---|---|---|---|
| Vulkan | Vulkan C API + custom SPIR-V shaders | TS fallback | AMD GPUs on Linux/Windows |
| ROCm | HIP C API via Bun FFI | TS fallback or MIOpen | AMD datacenter GPUs |
| XLA/TPU | XLA C API (libtpu) | Native (XLA provides it) | Google Cloud TPUs |
| OpenCL | OpenCL C API | TS fallback | Older/embedded GPUs |
| CPU (reference) | Pure TypeScript, no FFI | TS fallback | Testing, CI, portability |

## 7. The Autograd Fallback

When a backend does not provide native autograd, `@mlxts/core` supplies a tape-based implementation in TypeScript. This is slower than native autograd but correct for any backend.

### 7.1 How Tape-Based Autograd Works

The fallback maintains a tape -- an ordered list of operations recorded during the forward pass.

**Forward pass:** Each op that flows through the autograd system is wrapped. The wrapper calls the backend's op, records the operation and its inputs on the tape, and returns the result.

**Backward pass:** The tape is replayed in reverse. Each recorded op looks up its backward function (the VJP -- vector-Jacobian product) in a registry. The backward function computes the gradient contribution for each input. Gradients accumulate via addition.

```typescript
type TapeEntry = {
  /** The backward function for this op. */
  backward: (grad: TensorHandle) => TensorHandle[];
  /** The input handles that need gradients. */
  inputs: TensorHandle[];
  /** The output handle (for graph structure). */
  output: TensorHandle;
};

class GradTape {
  private entries: TapeEntry[] = [];

  /** Record an operation on the tape. */
  record(entry: TapeEntry): void {
    this.entries.push(entry);
  }

  /**
   * Replay the tape in reverse to compute gradients.
   *
   * @param loss - The scalar output to differentiate.
   * @param targets - Which input tensors to compute gradients for.
   * @returns Gradients for each target, in the same order.
   */
  backward(
    loss: TensorHandle,
    targets: TensorHandle[],
  ): TensorHandle[] {
    // Initialize: gradient of loss with respect to itself is 1
    const grads = new Map<TensorHandle, TensorHandle>();
    grads.set(loss, backend().fromScalar(1.0));

    // Replay tape in reverse
    for (let i = this.entries.length - 1; i >= 0; i--) {
      const entry = this.entries[i]!;
      const outputGrad = grads.get(entry.output);
      if (outputGrad === undefined) continue;

      const inputGrads = entry.backward(outputGrad);
      for (let j = 0; j < entry.inputs.length; j++) {
        const input = entry.inputs[j]!;
        const inputGrad = inputGrads[j]!;
        const existing = grads.get(input);
        if (existing !== undefined) {
          const accumulated = backend().add(existing, inputGrad);
          existing.free();
          inputGrad.free();
          grads.set(input, accumulated);
        } else {
          grads.set(input, inputGrad);
        }
      }
    }

    return targets.map((target) => {
      const grad = grads.get(target);
      if (grad === undefined) {
        return backend().zeros(target.shape, target.dtype);
      }
      return grad;
    });
  }
}
```

### 7.2 The Grad Function Registry

Each op needs a registered backward function. The registry maps op names to VJP factories.

```typescript
type VJPFactory = (
  inputs: TensorHandle[],
  output: TensorHandle,
) => (grad: TensorHandle) => TensorHandle[];

const vjpRegistry = new Map<string, VJPFactory>();

// Example: register backward for addition
vjpRegistry.set('add', (inputs, _output) => {
  return (grad: TensorHandle) => {
    // d/da (a + b) = 1, d/db (a + b) = 1
    // Gradient flows through unchanged to both inputs.
    // Broadcast handling may require sum-to-shape.
    const gradA = sumToShape(grad, inputs[0]!.shape);
    const gradB = sumToShape(grad, inputs[1]!.shape);
    return [gradA, gradB];
  };
});

// Example: register backward for matmul
vjpRegistry.set('matmul', (inputs, _output) => {
  return (grad: TensorHandle) => {
    const [a, b] = inputs;
    const gradA = backend().matmul(grad, backend().transpose(b!));
    const gradB = backend().matmul(backend().transpose(a!), grad);
    return [gradA, gradB];
  };
});
```

### 7.3 Performance: Native vs. TS Fallback

Native autograd (MLX, libtorch) is significantly faster because:
- The backward pass runs in the same native framework as the forward pass
- Memory for intermediate activations is managed by the native allocator
- Graph optimizations (common subexpression elimination, dead code removal) apply to the backward graph
- No JS-to-native boundary crossing per operation during backward

The TS fallback crosses the FFI boundary on every backward op. For a transformer with hundreds of ops per layer, this adds measurable overhead. Rough expectation: 2-5x slower backward pass compared to native autograd.

This is acceptable for WebGPU (where raw throughput is already limited) and for prototyping. For production training on CUDA, the libtorch native autograd should be used.

### 7.4 When Native vs. Fallback Kicks In

The decision is automatic and based on the backend's capabilities.

```typescript
function resolveValueAndGrad(
  fn: (...args: TensorHandle[]) => TensorHandle,
  argnums?: number | number[],
): (...args: TensorHandle[]) => [TensorHandle, TensorHandle[]] {
  const backend = getActiveBackend();

  if (backend.autograd !== undefined) {
    // Backend provides native autograd -- use it directly
    return backend.autograd.valueAndGrad(fn, argnums);
  }

  // Fall back to TS-level tape-based autograd
  return tapeValueAndGrad(fn, argnums);
}
```

Framework code (`nn.valueAndGrad`, optimizer `update()`) calls `resolveValueAndGrad()`, never the backend directly. This makes the fallback transparent.

## 8. Configuration and Backend Selection

### 8.1 Setting the Backend

```typescript
import { setBackend, getBackend } from '@mlxts/core';
import { mlxBackend } from '@mlxts/mlx';

// Set the active backend. Must be called before any tensor operations.
setBackend(mlxBackend);

// All subsequent @mlxts/nn, @mlxts/train, @mlxts/optimizers code uses MLX.
const b = getBackend(); // ComputeBackend
console.log(b.name); // "mlx"
```

### 8.2 Auto-Detection (Future)

```typescript
import { autoDetectBackend } from '@mlxts/core';

// Probes available backends in priority order:
// 1. MLX (if Apple Silicon + mlx-c available)
// 2. CUDA (if NVIDIA GPU + libtorch available)
// 3. WebGPU (if Dawn available or browser environment)
// 4. CPU reference (always available)
const backend = autoDetectBackend();
setBackend(backend);
```

### 8.3 Backend from Environment Variable

```bash
BUNML_BACKEND=cuda bun run train.ts
```

```typescript
import { backendFromEnv } from '@mlxts/core';

// Reads BUNML_BACKEND, loads the corresponding package, sets it.
await backendFromEnv();
```

### 8.4 Multiple Backends in One Process

Not supported initially. A single global backend is active at any time. Mixed-backend execution (e.g., CPU preprocessing + GPU training) is a future consideration that requires tensor migration between backends.

## 9. Migration Path

How the legacy single-package MLX implementation maps onto the `@mlxts/*` ecosystem.

### 9.1 Package Split

| Current location | Destination | What moves |
|---|---|---|
| `src/core/dtype.ts` | `@mlxts/core` | `DType`, `DTYPE_BYTE_SIZE`, `isIntegerDType` |
| `src/core/device.ts` (type only) | `@mlxts/core` | `DeviceType` |
| `src/utils/tree.ts` | `@mlxts/core` | `ParameterTree`, `FlatEntry`, `treeFlatten`, `treeUnflatten`, `treeMap`, `treeLeaves` |
| `src/utils/format-shape.ts` | `@mlxts/core` | `formatShape` |
| New code | `@mlxts/core` | `ComputeBackend`, `TensorHandle`, `AutogradCapability`, etc. (interface definitions) |
| New code | `@mlxts/core` | `setBackend()`, `getBackend()`, backend resolution |
| New code | `@mlxts/core` | TS-level tape autograd fallback |
| New code | `@mlxts/core` | TS-level safetensors I/O (already in `src/core/io.ts`, just restructured) |
| `src/core/ffi/*` | `@mlxts/mlx` | All FFI symbols, pointer utilities, closure bridge |
| `src/core/array.ts` | `@mlxts/mlx` | `MxArray` (implements `TensorHandle`) |
| `src/core/ops/*` | `@mlxts/mlx` | All op implementations (delegates to mlx-c FFI) |
| `src/core/random.ts` | `@mlxts/mlx` | MLX random number generation |
| `src/core/transforms.ts` | `@mlxts/mlx` | MLX-native `valueAndGrad`, `grad`, `compile`, `checkpoint` |
| `src/core/fast.ts` | `@mlxts/mlx` | MLX-native fused ops |
| `src/core/memory.ts` | `@mlxts/mlx` | MLX allocator controls |
| `src/core/device.ts` (impl) | `@mlxts/mlx` | MLX device/stream management |
| `src/core/metal.ts` | `@mlxts/mlx` | Metal capture (MLX-specific) |
| `src/core/error.ts` | `@mlxts/mlx` | `MxError`, `checkStatus` (mlx-c specific) |
| `src/nn/module.ts` | `@mlxts/nn` | `Module` base class |
| `src/nn/linear.ts` | `@mlxts/nn` | `Linear` |
| `src/nn/embedding.ts` | `@mlxts/nn` | `Embedding` |
| `src/nn/layer-norm.ts` | `@mlxts/nn` | `LayerNorm` |
| `src/nn/dropout.ts` | `@mlxts/nn` | `Dropout` |
| `src/nn/activations.ts` | `@mlxts/nn` | `relu`, `gelu`, `silu` |
| `src/nn/losses.ts` | `@mlxts/nn` | `crossEntropy`, `mse` |
| `src/nn/value-and-grad.ts` | `@mlxts/nn` | `nn.valueAndGrad` (calls through backend) |
| `src/nn/checkpoint.ts` | `@mlxts/nn` | `nn.checkpoint` (calls through backend) |
| `src/optimizers/*` | `@mlxts/optimizers` | `Optimizer`, `SGD`, `Adam`, `AdamW` |

### 9.2 The Key Refactor: nn and Optimizers Lose Direct Imports

Today, `src/nn/linear.ts` imports directly from `../core/array` and `../core/ops/arithmetic`. After the split, it imports from `@mlxts/core` and calls through the active backend.

Before:
```typescript
// legacy monolith linear.ts
import { MxArray, zeros } from '../core/array';
import { add, matmul } from '../core/ops';

export class Linear extends Module {
  weight: MxArray;
  bias: MxArray | null;

  forward(x: MxArray): MxArray {
    const out = matmul(x, this.weight);
    if (this.bias !== null) {
      const biased = add(out, this.bias);
      out.free();
      return biased;
    }
    return out;
  }
}
```

After:
```typescript
// packages/nn/src/linear.ts
import { type TensorHandle, backend } from '@mlxts/core';

export class Linear extends Module {
  weight: TensorHandle;
  bias: TensorHandle | null;

  forward(x: TensorHandle): TensorHandle {
    const b = backend();
    const out = b.matmul(x, this.weight);
    if (this.bias !== null) {
      const biased = b.add(out, this.bias);
      out.free();
      return biased;
    }
    return out;
  }
}
```

The change is mechanical: replace free-function imports with `backend()` method calls. The logic is identical.

### 9.3 Preserving Tests

Every existing framework-level test continues to work by running with the MLX backend set:

```typescript
// packages/nn/src/linear.test.ts
import { setBackend } from '@mlxts/core';
import { mlxBackend } from '@mlxts/mlx';

beforeAll(() => {
  setBackend(mlxBackend);
});

// ... all existing test assertions unchanged
```

Backend-specific tests (e.g., testing that `MxArray._ctx` is a valid pointer) stay in `@mlxts/mlx`. Framework-level tests (e.g., `Linear` produces correct output shapes) move to `@mlxts/nn` and run against the active backend.

### 9.4 Refactor Order

1. **Define the interfaces in `@mlxts/core`.** Types, `ComputeBackend`, `TensorHandle`, backend registration. No implementation yet.
2. **Wrap the extracted MLX-first implementation as `@mlxts/mlx`.** The adapter is thin: package the existing free functions into a `ComputeBackend` object. `MxArray` implements `TensorHandle`.
3. **Move nn code to `@mlxts/nn`.** Replace direct imports with `backend()` calls. All tests must pass with `mlxBackend` set.
4. **Move optimizers to `@mlxts/optimizers`.** Same mechanical refactor.
5. **Extract I/O and autograd fallback into `@mlxts/core`.** The safetensors code in `src/core/io.ts` is already pure TypeScript. The autograd fallback is new code.
6. **Verify all tests pass at each step.** No step ships without `bun run validate` green.

This order ensures that the working MLX path is never broken. Each step is independently testable and reversible.
