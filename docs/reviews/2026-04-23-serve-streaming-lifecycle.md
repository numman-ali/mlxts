# Runtime Review: Serve streaming lifecycle hardening

## Summary

This slice hardens the existing `@mlxts/serve` streaming path for real local
serving without pretending it is already a continuous-batching engine.

The main serving fixes are operational rather than model-math changes.
Streaming requests now disable Bun's per-request idle timeout so long-prefill
SSE responses do not get cut off as inactive. Request lifecycle telemetry now
tracks the actual stream lifetime instead of marking a request complete as soon
as a `Response` object is returned. Client disconnects now stop the serving-side
stream bridge and surface as `client_cancelled` / `finishReason: "cancelled"`
instead of being silently invisible in the logs.

The paired architecture truth from the reference audit is important context:
this change improves the serving shell around the existing single-request engine,
but true continuous batching still requires a batch-aware cache and scheduler
seam in `@mlxts/transformers`.

## Files Reviewed

- `packages/serve/src/server.ts`
- `packages/serve/src/server-streaming.ts`

## Tensor Lifetime Audit

This change stays above the tensor-owning decode/runtime layer.

- `packages/serve/src/server.ts` only changes HTTP lifecycle ownership around an
  existing `GenerationEngine`
- `packages/serve/src/server-streaming.ts` only changes async-iterator and SSE
  control flow; it does not create or retain new `MxArray` owners
- disconnect cleanup now prefers ending the async stream bridge at the next
  yielded boundary rather than leaving the iterator unobserved

There are no new native tensor lifetimes introduced here, so the risk is around
request lifecycle correctness rather than MLX ownership.

## Memory / Performance Evidence

Validated with:

- `bun test packages/serve/src/server.test.ts packages/serve/src/server-streaming.test.ts packages/serve/src/cli.test.ts`

Focused coverage now proves:

- Bun request timeout override is applied for SSE responses
- request completion is emitted after the streamed body finishes, not when the
  `Response` object is created
- client disconnects surface as cancellation and close the serving stream bridge
- existing stop-sequence and reasoning SSE behavior still works

No decode-throughput or model-parity claim is made in this review. The change is
serving-shell hardening around the existing single-request decode path.

## Independent Review

Reference-guided sub-agent audits informed both the scope and the limits of this
change.

- `Chandrasekhar` confirmed that current `@mlxts/transformers` contracts are not
  sufficient for true continuous batching: `TransformerCache` is still a
  single-sequence timeline with one scalar `offset`, and the current serving
  layer only offers admission micro-batching rather than a token-level
  scheduler.
- `Raman` prioritized immediate serving-quality work above the scheduler/cache
  layer and specifically called out request cancellation, request/stream
  telemetry, and SSE lifecycle behavior as the next operator-facing gaps worth
  landing.
- `Kepler` confirmed that the clean future seam is still a scheduler-owned
  single-model engine under the existing `GenerationEngine` contract, not more
  wrappers around the HTTP layer.

This patch follows that guidance: improve the serving shell now, keep the
current limits explicit, and leave real continuous batching for the next lower
layer.

## Remaining Risks / Follow-ups

- Client disconnects are now visible and stop the serving bridge, but they still
  do not preempt an already in-flight token step inside the model runtime. True
  end-to-end abort still needs an engine-level cancellation contract.
- SSE responses still have no periodic keep-alive frames during long silent
  prefills. The Bun timeout override removes the most immediate failure mode,
  but keep-alives remain useful for intermediaries and operator visibility.
- True continuous batching is still blocked by the single-sequence cache
  contract in `@mlxts/transformers`; the next honest throughput tranche is a
  batch-aware cache plus scheduler-owned decode loop, not more serving wrappers.
