# `@mlxts/tokenizers`

Tokenizer implementations for mlxts.

`@mlxts/tokenizers` now ships the canonical character tokenizer plus the first
Phase 7 pretrained-model tokenizers: HuggingFace-compatible BPE and
SentencePiece loading.

```ts
import { CharTokenizer, loadTokenizer } from "@mlxts/tokenizers";

const tokenizer = CharTokenizer.fromText("hello");
const pretrained = loadTokenizer("/path/to/model");
```
