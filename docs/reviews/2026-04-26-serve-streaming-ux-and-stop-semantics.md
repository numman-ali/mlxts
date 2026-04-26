# Runtime Review: Serve Streaming UX And Stop Semantics

## Summary

Interactive transformer streaming now flushes decoded text every generated token
by default, while exposing `streamDecodeInterval` / `--stream-decode-interval`
as the explicit knob for operators who want fewer tokenizer decodes during long
throughput runs. The CLI and `/info` surface report that configured interval so
streaming cadence is visible.

Chat streaming also trims the whitespace separator that Qwen-style templates
emit after `</think>`, so visible assistant text no longer starts with a blank
line after the reasoning section. SSE stop-sequence handling now preserves
`finish_reason: "stop"` when the stop is discovered while flushing buffered
reasoning/tool-call tails at stream completion.

## Files Reviewed

- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/model-sources.ts`
- `packages/serve/src/protocols/openai-chat-completion-streaming.ts`
- `packages/serve/src/server-info.ts`
- `packages/serve/src/server-responses-streaming.ts`
- `packages/serve/src/server-stop-filter.ts`
- `packages/serve/src/server-streaming.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/transformers-engine-streaming.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`

## Tensor Lifetime Audit

This tranche does not add tensor-producing operations or retain `MxArray`
handles. The streaming decode interval only changes when generated token ids
are decoded into text for SSE output. Model execution, cache ownership, and
token tensor lifetimes remain inside the existing transformer generation and
continuous scheduler paths.

The stop-sequence changes are host-side string buffering only. The CLI and
server option changes pass integers through existing serving setup; no native
resources are created or hidden by these helpers.

## Memory / Performance Evidence

Focused checks run locally:

- `bun test packages/serve/src/server-streaming.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/server.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/cli.test.ts packages/serve/src/model-server.test.ts packages/serve/scripts/regression-serve-matrix.test.ts packages/agent/src/cli.test.ts packages/agent/src/chat-model.test.ts`
- `bun run --filter '@mlxts/serve' typecheck && bun run --filter '@mlxts/agent' typecheck && bun run lint && bun run check:file-lines && bun run check:runtime-review && bun run check:tensor-lifetimes && bun run check:assertions`
- `bun run --filter '@mlxts/serve' regression:serve`
- `bun run regression:qwen-gemma -- --profile quick`
- `bun run validate`
- `bun run build`

The focused suite passed 130 tests with 0 failures. New and updated coverage
proves per-token default streaming, explicit coalescing when
`streamDecodeInterval: 2` is configured, Qwen reasoning separator trimming, chat
final-flush stop preservation, `/info` surfacing of the streaming interval, CLI
parsing/validation, and normalized Qwen streaming route reasons in the real
serve regression budget.

Full validation passed after the focused checks: repo typecheck, lint,
assertion, file-line, tensor-lifetime, runtime-review, and coverage gates all
cleared. The workspace build also succeeded, including refreshed serve and agent
`dist` entrypoints.

## Independent Review

Explorer sub-agent Heisenberg independently audited serving and agent streaming
surfaces. Two findings directly shaped this change: streaming stop handling
could overwrite a stop discovered during final tail flush, and request/default
handling should keep model-native behavior explicit rather than relying on
hidden workarounds. Explorer sub-agent McClintock audited benchmark coverage and
flagged inconsistent Qwen streaming route reasons in the serve regression
matrix; this change normalizes those expectations before relying on real-model
serve reports.

## Remaining Risks / Follow-ups

The engine stream event contract is still text/done only. A later tranche should
promote reasoning and tool-call deltas into structured engine events so Chat and
Responses do not both reconstruct them from raw text.

`max_tokens` defaults are still protocol-level defaults before they reach the
loaded model. A later model-native defaults tranche should parse and apply
`generation_config.max_new_tokens` without weakening request-limit admission.

Per-token streaming improves interactive feel but can add tokenizer overhead on
very long outputs. Operators can set a larger stream decode interval when
benchmarking throughput; real long-output endpoint ladders should record the
configured interval in their report context.
