# `@mlxts/transformers`

Pretrained decoder model loading and explicit generation for mlxts.

`@mlxts/transformers` builds on `@mlxts/tokenizers`, `@mlxts/core`,
`@mlxts/nn`, and the official Hugging Face JS packages to load supported
decoder families from local paths or Hugging Face snapshots.

```ts
import {
  AutoModel,
  AutoTokenizer,
  generateText,
  loadCausalLM,
  loadChatTemplate,
  loadPretrainedTokenizer,
  resolvePretrainedSource,
} from "@mlxts/transformers";

const directory = await resolvePretrainedSource("google/gemma-4-E2B-it");
const model = await loadCausalLM(directory);
const tokenizer = await loadPretrainedTokenizer(directory);
const chatTemplate = await loadChatTemplate(directory);
const text = generateText(model, tokenizer, "Hello", { maxTokens: 32 });

const sameModel = await AutoModel.fromPretrained("/path/to/model");
const sameTokenizer = await AutoTokenizer.fromPretrained("/path/to/model");
```

The canonical surface is function-first: `resolvePretrainedSource()`,
`loadCausalLM()`, `loadPretrainedTokenizer()`, `generateStep()`,
`generateTokens()`, and `generateText()`. `AutoModel` and `AutoTokenizer` are
convenience aliases over the same loader functions.
