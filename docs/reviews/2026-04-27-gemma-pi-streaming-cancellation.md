# Runtime Review: Gemma Pi Streaming And Cancellation

## Summary

Gemma 4 local serving now handles Pi-style interactive traffic more honestly:
streaming message prompts are prepared before the model lane is held, client
disconnects abort pending stream reads, heartbeat enqueue failures cancel the
generation scope, and Gemma-native thought/tool markers are parsed into OpenAI
chat-completion reasoning and tool-call shapes instead of leaking control text.
Gemma thinking-off replay also preserves the empty thought channel in assistant
history, including assistant tool-call turns, so Pi's tool-result follow-up
shares the same prompt prefix that produced the tool call.
The Gemma/Pi request also exposed a tokenizer-level performance bug: long
sentencepiece-style BPE prompts were doing unbounded longest-match scans. That
scan is now bounded by the maximum vocabulary token length, turning the captured
Pi prompt tokenization from a CPU spin into tens of milliseconds.

The Pi local model entry for `mlx-community/gemma-4-31b-it-4bit` was also
aligned with the checkpoint and Pi's first-party Gemma 4 metadata: full
checkpoint context, `8192` max output tokens, thinking enabled, and text-only
input until the serve multimodal path is implemented end to end.

## Files Reviewed

- `packages/serve/src/protocols/openai-chat-tool-call-stream.ts`
- `packages/serve/src/protocols/openai-chat-tool-calls.ts`
- `packages/serve/src/server-anthropic-messages-streaming.ts`
- `packages/serve/src/server-anthropic-messages.ts`
- `packages/serve/src/server-responses-streaming.ts`
- `packages/serve/src/server-responses.ts`
- `packages/serve/src/server-sse-heartbeat.ts`
- `packages/serve/src/server-stream-lifecycle.ts`
- `packages/serve/src/server-stream-runtime.ts`
- `packages/serve/src/server-streaming.ts`
- `packages/serve/src/server.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/types.ts`
- `packages/tokenizers/src/bpe-base.ts`
- `packages/transformers/src/chat-template.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine-shared.ts`
- `packages/serve/src/transformers-engine-streaming.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/protocols/openai-chat-completions.ts`
- `packages/serve/src/protocols/reasoning-tags.ts`

## Tensor Lifetime Audit

The changed serving files are protocol parsing, request lifecycle, cancellation,
logging, and routing-order changes. The tokenizer and chat-template changes are
host-side string/token-id preparation. They do not introduce new MLX tensor
operations, native handles, FFI ownership, or hidden `MxArray` intermediates.

Prompt preparation still delegates to the existing transformer prompt/chat
template path, but streaming requests no longer hold the single model lane while
that host-side work happens. Cancellation remains cooperative: the normalized
request abort signal is propagated to generation, and SSE writers now also abort
the generation scope when pending reads, stream cancellation, or heartbeat
enqueue failures reveal that the client has gone away.

## Memory / Performance Evidence

Focused validation passed:

```bash
bun test packages/serve/src/server-stream-runtime.test.ts packages/serve/src/server-streaming.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/cli.test.ts packages/serve/src/transformers-engine.test.ts
bun test packages/transformers/src/chat-template.test.ts packages/tokenizers/src/bpe.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/server-streaming.test.ts
bun test packages/transformers/src/chat-template.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/server-streaming.test.ts packages/tokenizers/src/bpe.test.ts
bun run typecheck
python3 -m json.tool /Users/numman/.pi/agent/models.json >/dev/null
PI_OFFLINE=1 pi --list-models gemma
```

The focused test run covered `86` passing tests across the stream runtime,
server streaming, OpenAI chat protocol formatting/parsing, CLI logging, and
transformer engine prompt/cancellation seams. `bun run typecheck` passed across
all workspaces.

The captured Pi request for `mlx-community/gemma-4-31b-it-4bit` was `35572`
bytes with a `29987` character system prompt, two user messages, four tools, and
`chat_template_kwargs: { enable_thinking: false, preserve_thinking: true }`.
After bounding the sentencepiece-style BPE longest-match scan, the rendered
`33170` character prompt encoded in `48.76ms` to `7479` tokens in the no-model
reproduction.

`PI_OFFLINE=1 pi --list-models gemma` now shows the local mlxts entry as:
`mlx-community/gemma-4-31b-it-4bit`, context `262.1K`, max-out `8.2K`,
thinking `yes`, images `no`.

Live Pi validation after the tokenizer fix:

- Plain Gemma/Pi turn with a `7502` token prompt prepared in `66.1ms`, prefilling
  completed in `40.9s`, and Pi rendered `I am ready to help.`
- Tool turn asked Pi to read `package.json`; Gemma emitted a tool call, Pi ran
  the `read` tool, and the final visible answer was `mlxts`.
- After preserving the empty thinking channel for assistant tool-call history,
  a clean Pi run on `surface:78` showed the first tool-call request writing a
  `7474` token prompt-prefix snapshot after `40.1s`, then the tool-result
  follow-up hit that snapshot, read `7474` cached tokens, prefilled only the
  `669` token suffix, and returned in `4.5s`.
- A direct two-turn Gemma chat probe after server restart showed the first turn
  writing `15` cache tokens and the second turn reading those `15` cached tokens
  before prefilling only the new `19` token suffix.

No generation throughput benchmark was run for this tranche because the changes
do not alter model forward math, cache mutation, scheduler batch math, or MLX
decode kernels. Real endpoint validation is handled by the live Gemma/Pi smoke
after the server is restarted with the corrected model id and output cap.

## Independent Review

Sub-agent Kuhn reviewed the failure mode and found that the observed 404 came
from Pi asking for Gemma while port `8000` was still serving only Qwen. The
more important latent issue was that Pi had advertised Gemma with `32768`
output tokens, encouraging very large turns; if the server cap was raised to
match, a disconnected Pi client could leave generation running.

Kuhn also flagged two real compatibility gaps: Gemma 4's chat template emits
native `<|channel>thought` and `<|tool_call>...<tool_call|>` markers, while the
existing parser mostly expected Qwen/OpenAI-style envelopes; and cancellation
needed to abort pending iterator reads and heartbeat enqueue failures, not only
the happy-path stream loop.

Ramanujan later found one more concrete blocker: Gemma can generate
`<|tool_response>` after a tool-call envelope because it is part of the native
tool-response boundary. The parser now strips that sentinel in buffered and SSE
chat paths after structured tool-call extraction, and tests cover both shapes.

## Remaining Risks / Follow-ups

Gemma is still registered as text-only in Pi because `@mlxts/serve` does not yet
accept and route image/video inputs through the Gemma 4 multimodal preprocessing
and model path. Advertising images before that would be fake capability.

Tool-call streaming is conservative. Native model tool-call envelopes are still
buffered until a complete call can be parsed into OpenAI `delta.tool_calls`.
That is correct and safe, but full parameter-by-parameter streaming remains a
future compatibility improvement.

Raw TCP socket-close behavior is covered through request abort and stream
runtime cancellation seams, but a full low-level socket-close integration test
would further reduce risk around Bun-specific disconnect behavior.
