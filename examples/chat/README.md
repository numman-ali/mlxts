# `examples/chat`

Interactive Bun-native chat loop for a local or Hugging Face-hosted pretrained
decoder model.

## Run

```bash
bun run examples/chat/index.ts /path/to/model
```

You can also override generation settings when you want to experiment:

```bash
bun run examples/chat/index.ts google/gemma-4-E2B-it --temperature 1 --top-k 64 --top-p 0.95
```

You can also point it at a Hugging Face repo id such as
`google/gemma-4-E2B-it`. The example will:

- resolve or download the snapshot through the official Hugging Face JS client
- show file-by-file loader progress
- load the model and tokenizer from the resolved local directory
- enable a model chat template automatically when `chat_template.jinja` or a
  tokenizer-side `chat_template` is present
- reuse the model prompt cache across turns like `mlx_lm.chat`
- stream reply text as it is generated
- apply checkpoint generation defaults automatically through the transformers
  loading surface
- let you override `temperature`, `top-k`, `top-p`, `max-tokens`, add a
  `--system-prompt`, or force greedy decoding from the command line

Type `q` or `/exit` to quit, `r` or `/reset` to reset the conversation cache,
and `h` or `/help` to print the command list again.
