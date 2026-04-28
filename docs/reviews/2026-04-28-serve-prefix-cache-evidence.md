# Serve Prefix-Cache Evidence

This tranche turns prompt-prefix cache reuse from an endpoint behavior into a
regression-visible serving contract. Benchmark reports now expose client usage
cache tokens and server prompt-cache event counters, and the real Qwen/Gemma
protocol-health profile requires a warmed repeated message request to hit the
single-request prompt cache.

## Files Reviewed

- `packages/serve/scripts/benchmark-serve-completions.ts`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/scripts/benchmark-serve-completions.test.ts`
- `packages/serve/scripts/benchmark-serve.test.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `docs/serving-runtime-strategy.md`
- `docs/runtime-optimization-matrix.md`
- `continuity.md`
- `MEMORY.md`

## Runtime Sensitivity

Production generation, cache mutation, scheduler dispatch, and protocol
formatting code did not change. The changed scripts measure runtime behavior and
make cache evidence load-bearing in real-model regressions.

Prompt-cache counters come from `generation_prompt_cache` server events rather
than inferred route strings. Client-side cache usage is recorded separately from
server cache events because Anthropic Messages usage does not expose cached
prompt tokens on the wire.

## Evidence

- `bun test packages/serve/scripts/benchmark-serve-completions.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`: passed, `27` tests.
- `bun run typecheck`: passed.
- `bun run regression:qwen-gemma -- --profile real --report-dir .tmp/qwen-gemma-prefix-cache-live --request-timeout-ms 3600000`: passed.
- `bun run validate`: passed.

Real protocol-health cache evidence from the generated reports:

| Rung | Route | Client cached tokens | Server prompt-cache hits | Server prompt-cache read tokens |
| --- | --- | ---: | ---: | ---: |
| Qwen chat | `single:prompt_prefix_cache` | `139` | `1` | `278` |
| Qwen OpenResponses | `single:prompt_prefix_cache` | `139` | `1` | `278` |
| Qwen Anthropic Messages | `single:prompt_prefix_cache` | `0` | `1` | `278` |
| Gemma chat | `single:prompt_prefix_cache` | `138` | `1` | `276` |
| Gemma OpenResponses | `single:prompt_prefix_cache` | `138` | `1` | `276` |
| Gemma Anthropic Messages | `single:prompt_prefix_cache` | `0` | `1` | `276` |

The Anthropic client cached-token column is `0` by wire-shape limitation. The
server prompt-cache event counters are the authoritative evidence for that
protocol.

The same real profile also rechecked the existing Qwen/Gemma endpoint and mixed
long-prefill guardrails, including Qwen `32768x128 + 128x32` and Gemma
`5000x128 + 128x32`.

## Independent Review

Erdos recommended this tranche as the next bounded serving-quality step after
the architectural cleanup: make prompt-prefix cache hits explicit in
`bench:serve`, make real Qwen/Gemma protocol-health budgets require cache-hit
evidence, and keep batch/paged prefix-cache reuse as future work.

## Out-of-scope Drift Noticed

Older notes describe Qwen peak memory as the remaining main gap versus `mlx-lm`.
Fresh paired local checks before this tranche did not reproduce that as the
dominant issue:

- `1024/128`: `mlx-lm` peak `17.022 GB`, `mlxts` peak `17.184 GB`; decode
  `29.031` vs `28.792 tok/s`.
- `10000/128`: both peak `19.219 GB`; decode `27.237` vs `27.139 tok/s`.

No memory-layout changes were made here. The stale premise is recorded so the
next performance tranche starts from current evidence instead of old audit
language.

## Remaining Risks

Prompt-prefix cache reuse is still single-request message cache reuse. Batch
longest-common-prefix reuse and paged cache dedup remain separate cache-backend
work.

The report now distinguishes protocol usage cache tokens from server prompt
cache events. Future protocol adapters must preserve that separation instead of
inventing cache usage fields that the wire protocol does not expose.
