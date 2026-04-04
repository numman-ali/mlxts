# Runtime Safety

This repo treats tensor hot paths, optimizer updates, checkpoint flows, and long-run training control as runtime-sensitive code.

A green typecheck and a passing unit suite are necessary, but they are not enough. If runtime behavior is hard to read, hard to observe, or hard to stop safely, the implementation is incomplete.

## Core Rules

- Keep disposable `MxArray` lifetimes visible in local code.
- Do not hide tensor-producing calls inside other tensor-producing calls in runtime-sensitive code.
- Keep `mxEval()`, `mxAsyncEval()`, and `synchronize()` explicit and justified.
- Prefer `using` for lexical ownership and `try/finally` for non-lexical cleanup.
- Use per-call `OutSlot` result slots at the FFI boundary. Shared reusable output buffers are not allowed.
- Dispose transform-returning helpers explicitly when they are not intended to live for the whole process. Finalization is the safety net, not the plan.
- The tensor-lifetime gate is AST-based and backed by a canonical tracked-op list in `scripts/`; when new tensor-producing primitives are added, that list must be updated.
- Treat file-backed run control as concurrent protocol code: `status.json`, `control.json`, and similar shared artifacts must be written atomically.
- Treat `stalled` as a terminal operator state for acceptance and soak flows unless a human explicitly resumes or restarts the run.
- If a runtime incident happens, the fix must also add a preventive test, benchmark, rule, or gate.

## Review Protocol

Runtime-sensitive diffs must:

1. receive an independent review
2. leave a review artifact under `docs/reviews/`
3. pass `bun run check:runtime-review`
4. pass `bun run check:tensor-lifetimes`

The review artifact should record the files reviewed, the tensor-lifetime audit, the memory or performance evidence, and any remaining risks. The `Files Reviewed` section must name the exact changed runtime-sensitive files.

## Operational Ladder

Do not jump straight from fast tests to an overnight run.

Use the soak ladder:

1. short supervised smoke (`50` steps)
2. medium supervised soak (`250` steps)
3. long supervised soak (`1000` steps)
4. only then loss-targeted acceptance (`5000` steps or more)

Long unattended runs must go through the supervised `bun run run:nanogpt ...` surface.

## Bench and Soak Surfaces

- `bun run bench:memory` measures active-memory drift for the leak-prone scenarios we care about.
- `bun run soak:gpt-tiny` runs the canonical supervised soak for the tiny preset.
- `bun run soak:gpt-small` runs the canonical supervised soak for the small preset.
- `bun run acceptance:gpt-tiny` and `bun run acceptance:gpt-small` are the loss-targeted acceptance runs.

## Forward-Only Posture

We do not preserve stale runtime paths for compatibility. If a long-run or benchmark path is no longer the one we trust, delete it and update the docs in the same change.
