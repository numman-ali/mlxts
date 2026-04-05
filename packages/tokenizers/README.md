# `@mlxts/tokenizers`

Tokenizer implementations for mlxts.

`@mlxts/tokenizers` currently ships the canonical character tokenizer used by the temporary nanoGPT fixture. Broader tokenizer formats are planned later.

```ts
import { CharTokenizer } from "@mlxts/tokenizers";

const tokenizer = CharTokenizer.fromText("hello");
```
