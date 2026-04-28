# Runtime Review: Serve Prefill Step Size Knob

## Summary

Serving now exposes the cold prompt-prefill chunk size as a real runtime knob
instead of keeping it hardcoded inside the transformer engine adapter. The
default remains `512` tokens for interactive fairness, while single-user
serving and benchmark runs can raise it with `--prefill-step-size <n>`.

The knob is threaded through CLI parsing, multi-model source construction,
runtime strategy reporting, `/info`, memory preflight estimates, serve
benchmarks, and transformer generation options.

## Files Reviewed

- `packages/serve/src/serve-runtime-strategy.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/model-server-options.ts`
- `packages/serve/src/model-sources.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/transformers-engine-shared.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/server-info.ts`
- `packages/serve/scripts/benchmark-serve-options.ts`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/src/transformers-engine.test.ts`
- `packages/serve/src/cli.test.ts`
- `packages/serve/src/model-server.test.ts`
- `packages/serve/src/server.test.ts`
- `packages/serve/scripts/benchmark-serve-options.test.ts`
- `packages/serve/scripts/benchmark-serve.test.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `packages/transformers/src/infrastructure/generation/defaults.test.ts`
- `packages/serve/README.md`
- `docs/runtime-safety.md`
- `docs/serving-runtime-strategy.md`
- `docs/inference-optimizations.md`

## Tensor Lifetime Audit

This change does not add new MLX tensor-producing operations. It changes the
chunk size supplied to existing generation and memory-estimation paths, so the
same tensor lifetime boundaries remain visible in the transformer generation
code.

The higher chunk size can increase temporary prefill pressure, so the memory
preflight path now reads the same resolved `prefillStepSize` that generation
will use. The CLI and benchmark paths pass the knob down without adding hidden
fallback behavior.

## Memory / Performance Evidence

Focused non-heavy validation passed before the Qwen ladder:

- `bun test packages/transformers/src/infrastructure/generation/defaults.test.ts packages/serve/src/cli.test.ts packages/serve/src/model-server.test.ts packages/serve/src/server.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/scripts/benchmark-serve-options.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun test packages/serve/src/serve-runtime-strategy.test.ts packages/serve/scripts/benchmark-serve-options.test.ts`
- `bun run typecheck`

The heavy evidence for this change is the package-owned serving benchmark,
`bun run bench:serve`, run against `mlx-community/Qwen3.6-27B-4bit` with
different `--prefill-step-size` values. This used chat streaming with
model-native sampling defaults preserved, not `--greedy`:

```bash
bun run bench:serve -- \
  --model mlx-community/Qwen3.6-27B-4bit \
  --protocol chat \
  --stream \
  --rungs 8192x16@1 \
  --trials 1 \
  --no-warmup \
  --max-concurrent-requests 1 \
  --max-batch-size 1 \
  --max-prompt-tokens 262144 \
  --max-total-tokens 262144 \
  --prefill-step-size <size> \
  --request-timeout-ms 3600000 \
  --report-json .tmp/prefill-step-ladder/qwen36-prefill-step-<size>.json
```

Measured results on the local cached checkpoint:

| Prefill step | TTFT | Prompt→first token | Peak memory | Prefill events | Max silent event gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `256` | `36.0s` | `228.0 tok/s` | `16.65 GB` | `33` | `1.17s` |
| `512` | `35.4s` | `231.9 tok/s` | `17.08 GB` | `17` | `2.44s` |
| `768` | `35.0s` | `234.5 tok/s` | `17.54 GB` | `11` | `3.38s` |
| `1000` | `37.8s` | `217.6 tok/s` | `17.87 GB` | `9` | `4.72s` |
| `2048` | `35.5s` | `231.2 tok/s` | `19.22 GB` | `5` | `9.01s` |
| `4096` | `41.9s` | `196.2 tok/s` | `22.04 GB` | `3` | `21.31s` |
| `8192` | `40.2s` | `204.3 tok/s` | `27.97 GB` | `2` | `39.70s` |

The fastest single-user point in this short probe was `768`, but only by about
`0.4s` TTFT over `512` while using more peak memory and longer non-interruptible
chunks. `512` remains the safer interactive/multi-user default. `4096+` should
be treated as an explicit single-user experiment, not a better default.

Follow-up benchmark reporting clarified the prefill metric boundary. Serve now
reports `mean_server_prefill_ms` / `mean_server_prefill_tps` separately from
`mean_prompt_to_first_token_tps`; the latter is client TTFT-derived and includes
first-token/SSE overhead. Qwen3.6 27B warmed completions at `128x128@1` measured
`mean_server_prefill_tps=235.287` versus
`mean_prompt_to_first_token_tps=196.566`, and `8192x16@1` measured
`mean_server_prefill_tps=217.916`.

This is a serving/runtime knob rather than a transformer hot-path semantic
change, so it does not require `bench:generation` or `bench:generation:parity`
for this tranche.

## Independent Review

Halley reviewed the design before implementation and flagged that cold prompt
prefill and active decode-time prefill are separate controls. The review also
called out the need to wire the knob into CLI parsing, `/info`, memory
estimation, benchmarks, and multi-model source construction rather than only the
direct engine path.

Kepler reviewed model-native generation defaults separately. The local Qwen
3.6 cached checkpoint uses `do_sample: true`, `temperature: 1.0`, `top_k: 20`,
and `top_p: 0.95`; serve preserves omitted request sampling parameters so the
transformer generation layer can use those defaults.

## Remaining Risks / Follow-ups

The knob is server-level today. That is enough for a homogeneous local endpoint
or a single served model, but a production multi-model server should eventually
allow per-model runtime options because Qwen 3.6, Gemma 4, and smaller models
may have different ideal prefill and fairness tradeoffs.

Large prefill chunks may improve single-user TTFT but can reduce cancellation
responsiveness and scheduler fairness under mixed load. Any default change above
`512` should be backed by both single-user TTFT evidence and mixed long-prefill
plus short-arrival fairness evidence.
