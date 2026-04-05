# `@mlxts/core`

The native MLX runtime surface for mlxts.

`@mlxts/core` owns `MxArray`, core tensor ops, random utilities, transforms,
fast fused kernels, safetensors I/O, low-level quantize/dequantize primitives,
tree helpers, and the Bun FFI bridge to `mlx-c`.

```ts
import { mxEval, ones, matmul } from "@mlxts/core";

using a = ones([3, 3]);
using b = matmul(a, a);
mxEval(b);
console.log(b.toList());
```

This package is the only one that needs the native MLX build step:

```bash
cd packages/core
bun run build:native
```
