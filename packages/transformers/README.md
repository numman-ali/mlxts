# `@mlxts/transformers`

Pretrained decoder model loading and explicit generation for mlxts.

`@mlxts/transformers` builds on `@mlxts/tokenizers`, `@mlxts/core`,
`@mlxts/nn`, and the official Hugging Face JS packages to load supported
decoder families from local paths or Hugging Face snapshots.

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

Qwen 3.5 / Qwen 3.6 multimodal checkpoints also expose image-preparation
helpers through the same package surface. The package owns checkpoint loading,
chat template discovery, `preprocessor_config.json` parsing, multimodal prompt
preparation, and tensor-level image patchification. Local file decode and image
resize stay at the example/application edge.

```ts
const localSource = await resolvePretrainedSource("mlx-community/Qwen3.6-27B-4bit");
using model = await loadCausalLM(localSource);
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
