# Whisper Example

This workbook composes `@mlxts/transformers` and `@mlxts/tokenizers` for a
finite local speech-to-text proof.

The supported path loads a Whisper checkpoint, reads one 16 kHz PCM or float WAV
file, prepares log-mel features, runs finite greedy decoding, and emits compact
AXI-shaped output.

```bash
bun run examples/whisper/index.ts openai/whisper-tiny \
  --local-files-only \
  --audio ./speech-16khz.wav \
  --max-tokens 64 \
  --json
```

The command uses the shared runtime lock because it loads local MLX model
weights. Progress goes to stderr; stdout is structured output. The current proof
does not implement cached decoder state, timestamp segmentation, language
detection, resampling, or long-form audio chunking.
