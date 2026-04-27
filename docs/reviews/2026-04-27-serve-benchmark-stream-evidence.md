# Runtime Review: Serve Benchmark Stream Evidence

## Summary

Serving benchmark JSON now preserves server-side stream writer evidence in each
`serverRequests[]` row. Client-observed SSE timing remains in `requests[]` and
trial-level stream averages; server-side writer timing is separately prefixed as
`serverStream*` so TTFT, chunk, byte, and terminal-result evidence do not get
mixed across clocks.

The real Qwen/Gemma streaming regression specs now require both client-side SSE
evidence and server-side stream writer evidence before accepting a streamed
serving rung.

## Files Reviewed

- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/scripts/benchmark-serve.test.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `packages/serve/README.md`
- `docs/serving-runtime-strategy.md`

## Tensor Lifetime Audit

This change does not add or move MLX tensor-producing operations. It only reads
already emitted `ServeEvent` objects from the benchmark event recorder and
serializes numeric stream timing/count fields into JSON reports.

No model forward path, cache path, tokenizer path, or native handle ownership
changed.

## Memory / Performance Evidence

Focused validation passed:

- `bun test packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`

No heavy MLX run was required for this slice because the change is host-side
benchmark evidence plumbing. The assertions intentionally do not require server
chunk counts to equal client chunk counts; Bun and transport buffering can
coalesce delivery independently of the server writer frame count.

## Independent Review

Archimedes and Fermat reviewed the serving benchmark/report path independently.
Both recommended extending the existing `serverRequests[]` evidence surface
rather than creating a new top-level event timeline or changing the client SSE
parser. They also called out the main correctness boundary: keep
client-observed TTFT/chunk metrics separate from server-side stream writer
TTFT/chunk metrics.

The implementation follows that recommendation with `serverStream*` report
fields and a separate `expectEveryServerRequestStreamed` regression budget.

## Remaining Risks / Follow-ups

The regression matrix checks stream evidence presence, terminal result, output
bytes/chunks, TTFT, and finish reason. It deliberately does not assert exact
server/client chunk-count equality.

Future serving benchmark tranches can add explicit per-rung server-stream
aggregate averages if report consumers need them, but the per-request evidence
is enough for the current Qwen/Gemma regression guardrails.
