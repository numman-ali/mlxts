# `@mlxts/data`

Dataset and batching helpers for mlxts.

`@mlxts/data` currently provides the canonical text-loading and token-batching helpers used by the temporary nanoGPT fixture.

```ts
import { prepareData } from "@mlxts/data";

const { trainTokens, valTokens } = prepareData(tokens, 0.9);
```
