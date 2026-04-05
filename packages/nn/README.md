# `@mlxts/nn`

Neural-network primitives for mlxts.

`@mlxts/nn` builds on `@mlxts/core` and provides `Module`, layers, activations, losses, and module-aware gradient helpers.

```ts
import { Linear } from "@mlxts/nn";

const layer = new Linear(4, 8);
```
