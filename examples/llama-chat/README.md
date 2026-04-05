# `examples/llama-chat`

Minimal Bun-native interactive text-generation loop for a local or cached
pretrained decoder model.

## Run

```bash
bun run examples/llama-chat/index.ts /path/to/model
```

You can also point it at a Hugging Face repo id if the model is already cached
or if hub access is configured.

Despite the folder name, this example does not apply a model-specific chat
template. It feeds your raw prompt text directly into the tokenizer and model.

Type `/exit` or press `Ctrl-D` to quit.
