# Serve Protocol Benchmarking

Date: 2026-04-24

`bench:serve` now has a `--protocol` option so endpoint benchmarks can exercise
the actual wire adapters instead of inferring chat or Responses quality from
token-array completions runs.

## Scope

- `--protocol completions` remains the default exact-token throughput path over
  `/v1/completions` with token-array prompts.
- `--protocol chat` sends deterministic text prompts through
  `/v1/chat/completions`, including streaming usage chunks.
- `--protocol responses` sends deterministic text prompts through
  `/v1/responses`, including semantic Responses SSE events.
- Streaming benchmark requests now fail if the stream ends without usage or a
  finish reason, so protocol errors cannot be silently reported as zero-token
  successes.
- `--ignore-eos` is rejected for `--protocol responses` because the Responses
  benchmark does not expose that nonstandard serving extension.

## Evidence

Focused tests:

```bash
bun test packages/serve/scripts/benchmark-serve-completions.test.ts \
  packages/serve/scripts/benchmark-serve-options.test.ts \
  packages/serve/scripts/benchmark-serve.test.ts
```

Live protocol smokes on cached `mlx-community/Llama-3.2-1B-Instruct-4bit`:

```bash
bun run bench:serve --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --model-id llama-local --protocol chat --rungs 16x4@1 --trials 1 \
  --no-warmup --greedy --ignore-eos --stream \
  --report-json .tmp/llama32-chat-protocol-smoke.json \
  --max-prompt-tokens 256 --max-total-tokens 320

bun run bench:serve --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --model-id llama-local --protocol responses --rungs 16x4@1 --trials 1 \
  --no-warmup --greedy --stream \
  --report-json .tmp/llama32-responses-protocol-smoke.json \
  --max-prompt-tokens 256 --max-total-tokens 320
```

Both live smokes completed with `finish_reasons=length`, `prompt_tokens=123`,
`completion_tokens=4`, and flat active memory. The first chat smoke intentionally
used too-small prompt admission (`64`) and exposed a useful benchmark bug:
streaming protocol errors could end without usage and still report zero-token
metrics. That is now covered by focused regression tests.

## Remaining Gaps

- This is protocol-health benchmarking, not tool-quality benchmarking.
- Responses support is still text-only by design; tools, persistence,
  multimodal input, and structured output remain explicitly unsupported.
- Chat and Responses use text prompts, so do not compare their prompt token
  counts directly with completions token-array parity runs.
