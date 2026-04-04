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
- Quality-oriented runs should track a real best checkpoint and use patience-based early stopping rather than assuming the latest checkpoint is the best one.
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

## External Dependency Risks

### Bun FFI

Bun's FFI (`bun:ffi`) is the foundation of our MLX bindings. Known issues that affect this project:

- **JSCallback crashes from C libraries** — [bun#17157](https://github.com/oven-sh/bun/issues/17157). Our autograd closure bridge uses JSCallback heavily. Mitigated by synchronous-only callback usage (no `threadsafe: true`).
- **Pointer handling segfaults** — [bun#17510](https://github.com/oven-sh/bun/issues/17510). We narrow pointer types only in `src/core/ffi/` via `unwrapPointer()` and `sizeToNumber()`.
- **Memory leaks with JSCallback in loops** — [bun#7582](https://github.com/oven-sh/bun/issues/7582). Mitigated by reusing `ReusableClosure` instances rather than creating per-call callbacks.
- **`bun:ffi` is incompatible with `bun build`** — limits distribution options. Not a blocker for Bun-runtime-only usage.

When upgrading Bun, run the full soak ladder before trusting the new version. FFI behavior changes are not always documented in release notes.

### mlx-c

mlx-c (Apple's C API for MLX) is pre-1.0 (currently v0.2.0). There is no published changelog, semver policy, or stability contract.

- **ABI audit required on upgrade.** When mlx-c is upgraded, `src/core/ffi/symbols.ts` must be re-verified against the new headers. This is documented in AGENTS.md as an ABI integrity rule.
- **No guarantee of backward compatibility.** The 0.1.x → 0.2.0 jump shows willingness to break.
- **Mitigation:** We vendor mlx-c source and pin to a known-good version. Upgrades are deliberate, tested, and reviewed.

If Apple deprioritizes mlx-c, we are blocked on new MLX features reaching TypeScript. This is the most significant external dependency risk.
