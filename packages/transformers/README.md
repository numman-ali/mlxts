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
  generateTextStream,
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  makePromptCache,
  resolvePretrainedSource,
} from "@mlxts/transformers";

const directory = await resolvePretrainedSource("google/gemma-4-E2B-it");
const model = await loadCausalLM(directory);
const tokenizer = await loadPretrainedTokenizer(directory);
const interactionProfile = await loadInteractionProfile(directory);
const cache = makePromptCache(model);
const compiled = interactionProfile.compileMessages(tokenizer, [
  { role: "user", content: "Hello" },
]);
const text = generateText(model, tokenizer, "Hello", { maxTokens: 32 });
generateTextStream(model, tokenizer, compiled.text, { maxTokens: 32, cache }, (chunk) => {
  process.stdout.write(chunk);
});

const sameModel = await AutoModel.fromPretrained("/path/to/model");
const sameTokenizer = await AutoTokenizer.fromPretrained("/path/to/model");
```

The canonical surface is function-first: `resolvePretrainedSource()`,
`loadCausalLM()`, `loadPretrainedTokenizer()`, `loadInteractionProfile()`,
`generateStep()`, `generateTokens()`, `generateText()`, and
`generateTextStream()`. `AutoModel` and `AutoTokenizer` are convenience aliases
over the same loader functions.
