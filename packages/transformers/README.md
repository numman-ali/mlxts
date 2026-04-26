# `@mlxts/transformers`

Pretrained transformer loading, multimodal preparation, and explicit generation
for mlxts.

`@mlxts/transformers` builds on `@mlxts/tokenizers`, `@mlxts/core`,
`@mlxts/nn`, and the official Hugging Face JS packages to load supported
autoregressive text and vision-language families from local paths or Hugging
Face snapshots.

```ts
import {
  AutoModel,
  AutoTokenizer,
  generatePreparedTokens,
  generateText,
  generateTextStream,
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  loadQwen3_5ForConditionalGeneration,
  loadQwen3_5VisionPreprocessor,
  makePromptCache,
  prepareQwen3_5ImageBatch,
  prepareQwen3_5ImagePrompt,
  resolvePretrainedSource,
  smartResizeQwen3_5Image,
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
`generateStep()`, `generateTokens()`, `generatePreparedTokens()`,
`generateText()`, and `generateTextStream()`. `AutoModel` and `AutoTokenizer`
are convenience aliases over the same loader functions.

## Model Regression Matrix

Use the focused matrix before commits that touch loading, quantization, model
families, generation, or benchmark surfaces:

```bash
bun run --filter '@mlxts/transformers' regression:models
```

When cached real Qwen and Gemma checkpoints are available, use the stricter
local smoke before high-risk model commits:

```bash
bun run packages/transformers/scripts/regression-model-matrix.ts --decode-smoke
```

The real path runs sequentially under the shared MLX runtime lock, checks
load-memory budgets for Qwen 3.6 and Gemma 4, and enforces decode smoke budgets
for throughput, peak memory, active-memory slope, and evals-per-token.

The repo-level Qwen/Gemma command composes this transformer matrix with the
serving matrix:

```bash
bun run regression:qwen-gemma -- --profile quick
bun run regression:qwen-gemma -- --profile real
bun run regression:qwen-gemma -- --profile substantial
```

`quick` is the normal pre-commit/substantial-edit guardrail. `real` adds cached
real-model decode and endpoint smoke. `substantial` adds longer endpoint rungs
and a 32k Qwen retrieval check with exact-match and decode-memory assertions.

Qwen 3.5 / Qwen 3.6 multimodal checkpoints also expose image-preparation
helpers through the same package surface. The package owns checkpoint loading,
chat template discovery, `preprocessor_config.json` parsing, multimodal prompt
preparation, and tensor-level image patchification. Local file decode and image
resize stay at the example/application edge.

```ts
const localSource = await resolvePretrainedSource("mlx-community/Qwen3.6-27B-4bit");
using model = await loadQwen3_5ForConditionalGeneration(localSource);
const tokenizer = await loadPretrainedTokenizer(localSource);
const preprocessor = await loadQwen3_5VisionPreprocessor(localSource);

const resized = smartResizeQwen3_5Image(768, 1024, preprocessor);
const preparedImage = prepareQwen3_5ImageBatch(
  {
    width: resized.width,
    height: resized.height,
    channels: 3,
    data: new Uint8Array(resized.width * resized.height * 3),
  },
  preprocessor,
);

try {
  const prompt = prepareQwen3_5ImagePrompt(
    model,
    tokenizer.encode("<|vision_start|><|image_pad|><|vision_end|>\nDescribe this image."),
    preparedImage.pixelValues,
    preparedImage.imageGridThw,
  );
  try {
    const result = generatePreparedTokens(model, prompt, { maxTokens: 64, temperature: 0 });
    console.log(tokenizer.decode(result.tokenIds, { skipSpecialTokens: true }));
  } finally {
    prompt.inputEmbeddings?.free();
    prompt.positionIds?.free();
  }
} finally {
  preparedImage.pixelValues.free();
  preparedImage.imageGridThw.free();
}
```
