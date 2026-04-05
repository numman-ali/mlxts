# `@mlxts/optimizers`

Optimizer implementations for mlxts.

`@mlxts/optimizers` builds on `@mlxts/core` and `@mlxts/nn` and provides `Adam`, `AdamW`, `SGD`, and optimizer checkpoint helpers.

```ts
import { AdamW } from "@mlxts/optimizers";

const optimizer = new AdamW(3e-4, 0.1);
```
