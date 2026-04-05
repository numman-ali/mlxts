# `@mlxts/transformers`

Pretrained decoder model loading and explicit generation for mlxts.

`@mlxts/transformers` builds on `@mlxts/hub`, `@mlxts/tokenizers`,
`@mlxts/core`, and `@mlxts/nn` to load supported decoder families from local
paths or Hugging Face snapshots.

```ts
import {
  AutoModel,
  AutoTokenizer,
  generateText,
  loadCausalLM,
  loadPretrainedTokenizer,
} from "@mlxts/transformers";

const model = await loadCausalLM("/path/to/model");
const tokenizer = await loadPretrainedTokenizer("/path/to/model");
const text = generateText(model, tokenizer, "Hello", { maxTokens: 32 });

const sameModel = await AutoModel.fromPretrained("/path/to/model");
const sameTokenizer = await AutoTokenizer.fromPretrained("/path/to/model");
```

The canonical surface is function-first: `loadCausalLM()`,
`loadPretrainedTokenizer()`, `generateStep()`, `generateTokens()`, and
`generateText()`. `AutoModel` and `AutoTokenizer` are convenience aliases over
the same loader functions.
