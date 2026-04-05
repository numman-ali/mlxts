# `examples/llama-chat`

Interactive Bun-native chat loop for a local or Hugging Face-hosted pretrained
decoder model.

## Run

```bash
bun run examples/llama-chat/index.ts /path/to/model
```

You can also point it at a Hugging Face repo id such as
`google/gemma-4-E2B-it`. The example will:

- resolve or download the snapshot through the official Hugging Face JS client
- show file-by-file loader progress
- load the model and tokenizer from the resolved local directory
- enable a model chat template automatically when `chat_template.jinja` or a
  tokenizer-side `chat_template` is present

Type `/exit` or press `Ctrl-D` to quit.
