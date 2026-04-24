# Runtime Review: Serve long buffered timeouts

## Summary

Long non-streaming generation requests could hit Bun's per-request timeout
before `openAIRouteResponse()` returned a buffered JSON response. Streaming
requests already disabled the timeout because the route returned a stream
response immediately, but buffered completions, chat completions, and Responses
could spend minutes inside model generation first.

The fetch handler now disables Bun's request timeout before dispatching any
generation-capable OpenAI route. Streaming routes no longer apply a duplicate
timeout override after the response is created. The endpoint benchmark also has
an explicit per-request client timeout so very long buffered runs do not confuse
client patience with server/model failure.

## Files Reviewed

- `packages/serve/scripts/benchmark-serve-completions.ts`
- `packages/serve/scripts/benchmark-serve-options.ts`
- `packages/serve/src/server.ts`

## Tensor Lifetime Audit

This change is above the model runtime and does not create or retain `MxArray`
handles. It only changes HTTP request lifetime controls before an existing
generation request enters the engine. Generation cancellation, cache ownership,
and tensor disposal remain in the existing engine/generation layers.

## Memory / Performance Evidence

- Focused regression:
  `bun test packages/serve/src/server.test.ts packages/serve/src/server-streaming.test.ts`
  passes with `32` tests.
- Benchmark option coverage:
  `bun test packages/serve/scripts/benchmark-serve-options.test.ts packages/serve/scripts/benchmark-serve-completions.test.ts`
  passes.
- The new buffered regression asserts that `server.timeout(request, 0)` is
  called before the non-streaming completion engine starts work.
- Streaming regression coverage still asserts exactly one timeout override and a
  complete SSE body.

## Independent Review

The later long-context streaming run showed that server-side timeout control and
benchmark client timeout control are necessary but not sufficient for huge
buffered Qwen prompts: buffered JSON still leaves the client silent for several
minutes. The independent scheduler/serving review also reinforced that the
serving layer should not paper over engine behavior with HTTP wrappers; long
user-facing prompts should be streamed, while throughput should be handled by a
scheduler-owned engine tranche.

## Remaining Risks / Follow-ups

- This prevents Bun from timing out legitimate long local generations, but it
  does not replace admission limits. Operators still need `maxPromptTokens`,
  `maxTotalTokens`, `maxGeneratedTokens`, and memory preflight budgets.
- Very long buffered requests can still tie up a server slot for minutes; true
  user-facing responsiveness should prefer streaming plus scheduler-owned
  continuous batching.
