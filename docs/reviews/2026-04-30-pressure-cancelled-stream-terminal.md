# Pressure-Cancelled Stream Terminal Shape

## Summary

Expected lazy model-pool pressure cancellation now closes an already-started
SSE body after emitting structured server error events. Unexpected stream
failures still error the body. This keeps pressure relief observable without
letting Bun surface the expected cancellation as a transport fault.

## Files Reviewed

- `packages/serve/src/errors.ts`
- `packages/serve/src/streaming/lifecycle.ts`
- `packages/serve/src/http/server.test.ts`
- `packages/serve/scripts/regression-lazy-pool-pressure.ts`
- `packages/serve/scripts/regression-lazy-pool-pressure.test.ts`

## Tensor Lifetime Audit

The change is HTTP stream lifecycle only. It does not create, retain, dispose,
or pass through `MxArray` handles.

## Memory / Performance Evidence

- `bun test packages/serve/src/http/server.test.ts`: passed, `44` tests /
  `248` assertions.
- `bun test packages/serve/src/http/server.test.ts packages/serve/scripts/regression-lazy-pool-pressure.test.ts`:
  passed, `49` tests / `264` assertions.
- `bun run --filter '@mlxts/serve' typecheck`: passed.
- `bun run check:assertions`: passed.
- `bun run check:file-lines`: passed, `353` production source files checked.
- `bun run check:runtime-review`: passed.
- `bun run regression:lazy-pool-pressure -- --report-dir .tmp/lazy-pool-pressure-terminal`:
  passed. The active Gemma stream closed cleanly after the pressure abort
  (`done=true`, no reader error in the report), the blocked Qwen request
  completed with `35` output chars, and pressure metrics recorded `2` events,
  `1` aborted request, and `6` metric lines. The command output contained no
  pressure-cancelled stream stack.
- `bun run validate`: passed.

## Independent Review

Rawls independently traced the pressure path from
`ModelPoolPressureController.#abortPressureCandidate()` through
`readPressureAwareStream()` / pressure error mapping into
`failGenerationStream()`. The review identified `controller.error(error)` as
the Bun-visible stack seam and recommended an exact
`model_pool_memory_pressure` clean-close predicate while keeping all other
stream failures on the body-error path.

## Remaining Risks / Follow-ups

The pressure-cancelled client sees a clean end of body after any chunks already
sent. Protocol-specific terminal error frames remain out of scope for this
tranche because the internal serve event stream already records the structured
`model_pool_memory_pressure` error.

## Out-of-scope Drift Noticed

- Broader SSE terminal error framing for OpenAI, OpenResponses, and Anthropic
  protocols remains a separate compatibility design question.
