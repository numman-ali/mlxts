# `@mlxts/data`

Dataset and batching helpers for mlxts.

`@mlxts/data` provides canonical text-loading and token-batching helpers used by the committed nanoGPT example and regression surface.

```ts
import { prepareData } from "@mlxts/data";

const { trainTokens, valTokens } = prepareData(tokens, 0.9);
```
