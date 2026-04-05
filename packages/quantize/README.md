# `@mlxts/quantize`

Model quantization and quantized-checkpoint interoperability for mlxts.

`@mlxts/quantize` builds on the raw MLX quantization primitives in
`@mlxts/core` and the quantized layer forms in `@mlxts/nn`.

```ts
import { quantizeModule } from "@mlxts/quantize";

quantizeModule(model, {
  bits: 4,
  groupSize: 64,
});
```
