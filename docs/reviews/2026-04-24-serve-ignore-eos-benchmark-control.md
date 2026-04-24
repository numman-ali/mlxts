# Runtime Review: Serve Ignore-EOS Benchmark Control

## Summary

`@mlxts/serve` now supports an explicit `ignore_eos` request extension so endpoint
benchmarks can ask for exact `max_tokens` decode lengths without changing normal
serving behavior. The OpenAI completions and chat adapters preserve model-native
EOS stopping unless this extension is supplied, and the serve benchmark harness
exposes the same control through `--ignore-eos`.

This matters for Qwen 3.6 parity work because a valid serving throughput ladder
must compare the same decode length as in-process parity benchmarks. Without this
control, a `1024/128` Qwen serving rung stopped after 4 generated tokens and
produced misleading throughput data.

## Files Reviewed

- `packages/serve/src/types.ts`
- `packages/serve/src/protocols/openai-completions.ts`
- `packages/serve/src/protocols/openai-chat-completions.ts`
- `packages/serve/src/transformers-engine-shared.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine.ts`

## Tensor Lifetime Audit

The changed production paths are request normalization and generation-option
plumbing only. They do not introduce new tensor-producing operations, native
handles, eval calls, cache mutation, or ownership transfers. The engine audit
checked that `ignoreEos === true` maps to `eosTokenIds: []` and suppresses only
the tokenizer-injected generated-EOS stop condition; prompt tokenization remains
unchanged so normal text prompt semantics are not silently altered.

## Memory / Performance Evidence

- `bun run --filter '@mlxts/serve' typecheck`: pass.
- `bun run lint`: pass.
- `bun test packages/serve/src/protocols/openai-completions.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/scripts/benchmark-serve-options.test.ts packages/serve/scripts/benchmark-serve-completions.test.ts`: 51 pass, 0 fail.
- Qwen endpoint check:
  `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --prompt-tokens 128,1024 --generation-tokens 128 --concurrency 1 --trials 1 --no-warmup --greedy --ignore-eos --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --max-prompt-tokens 1024 --max-total-tokens 1152 --gpu-memory-utilization 0.9`
- Qwen `128/128`: `completion_tps=25.373`, `completion_tokens=128`, `finish_reasons=length`, `peak_memory=18.481 GB`, `active_delta=0.014 GB`.
- Qwen `1024/128`: `completion_tps=15.116`, `completion_tokens=128`, `finish_reasons=length`, `peak_memory=19.934 GB`, `active_delta=0.000 GB`.

## Independent Review

Tesla reviewed the planned design as a read-only second opinion before the final
patch. The review confirmed the right integration points: protocol parsing,
generation option conversion, tokenizer EOS injection, static batch generation,
streaming generation, benchmark flags, and adapter/engine tests. Tesla also
called out that text prompt special-token insertion is separate from generated
EOS stopping; this patch intentionally leaves prompt tokenization unchanged.

## Remaining Risks / Follow-ups

`ignore_eos` is a benchmark/debug extension, not a default serving behavior.
Normal serving should continue to honor EOS so chat and completions do not run to
the length limit unnecessarily. The next serving-quality step is to run longer
endpoint ladders, including streaming ladders and concurrency rungs, after the
benchmark can produce apples-to-apples decode lengths.
