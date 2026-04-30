# `@mlxts/nn`

Neural-network primitives for mlxts.

`@mlxts/nn` builds on `@mlxts/core` and provides `Module`, layers,
transformer primitives like `RMSNorm`, `RoPE`, and `GroupedQueryAttention`,
general layers including `Linear`, `Embedding`, `LayerNorm`, `GroupNorm`,
`Conv1d`, and `Conv2d`, free-function activations such as `swiglu`, losses,
and module-aware gradient helpers.

```ts
import { Conv2d, Linear } from "@mlxts/nn";

const layer = new Linear(4, 8);
const conv = new Conv2d(3, 16, 3, 1, 1);
```
