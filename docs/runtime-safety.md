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

Long unattended runs must go through the supervised `cd examples/nanogpt && bun run manager ...` surface.

## Generation Performance

Performance is an observable, not a review opinion. The generation hot path (`generation.ts`, `infrastructure/sampling/index.ts`, `infrastructure/cache/index.ts`, family `attention.ts`, family `model.ts`) must have measurable, comparable, regression-protected performance.

### Benchmark infrastructure

- `bun run bench:generation` runs the synthetic throughput benchmark: real cached checkpoints, synthetic random-token prompts, no downloads or network during the benchmark itself. It measures prefill tok/s, decode tok/s, peak memory, and eval-count-per-token, and acts as the low-level throughput canary.
- `bun run bench:generation:parity` runs the MLX-LM parity benchmark: the same real cached checkpoints and token counts, but with the decode-side work structured for fair comparison with `mlx-lm`. Use `--require-mlx-lm-reference` for publishable parity claims so a missing Python reference cannot silently become a local-only run.
- `bun run bench:generation:context` runs long-context retrieval ladders. Use `--needle-placements all` for early/middle/late marker placement and `--report-json <path>` when recording publishable context-window evidence.
- `bun run bench:serve` runs endpoint-level serving benchmarks. The default `--protocol completions` path uses exact token-array prompts; `--protocol chat` and `--protocol responses` exercise the wire adapters and chat-template path with deterministic text prompts. Prefer explicit capability ladders with `--rungs 128x128@1,1024x512@1,...` and `--report-json <path>` for overnight serving evidence; add `--request-stagger-ms <n>` when testing waiting-row scheduler fairness instead of simultaneous request admission.
- `bun run regression:qwen-gemma -- --profile quick` composes the package-owned Qwen/Gemma transformer and serving regression matrices without loading real checkpoints. Use `--profile real` before high-risk model/serving commits when cached Qwen/Gemma checkpoints are available; use `--profile substantial` for heavier local proof work that adds endpoint capability rungs and a 32k Qwen retrieval assertion.
- The benchmark implementations live with the transformer generation surface in `packages/transformers/scripts/`; shared benchmark data and traces live under repo-root `benchmarks/`.
- Baselines in `benchmarks/baselines.json` now carry both surfaces. Parity targets also record the paired MLX-LM reference numbers captured on the same machine and date.
- The benchmarks warn on >2x regression. They do not hard-fail `bun run validate` (tok/s varies with system load), but the numbers must be reviewed for any hot-path diff.
- `--metal-trace` wraps either benchmark surface in `startMetalCapture()` / `stopMetalCapture()` for Instruments analysis. Zero overhead when not used.

### Performance review requirement

When a diff touches generation hot-path files, the `docs/reviews/` artifact must include **Memory / Performance Evidence** with:
- Before/after numbers from `bun run bench:generation`
- Parity numbers from `bun run bench:generation:parity` when the change affects generation behavior rather than only low-level throughput
- Explanation of any regression (even if intentional)
- `bun run check:runtime-review` enforces the review artifact requirement

### Profiling without code changes

Available tools that do not require modifying the hot path:
- **Metal System Trace**: `startMetalCapture()` / `stopMetalCapture()` (bound in `@mlxts/core`) → open `.gputrace` in Instruments
- **DTrace**: trace mlx-c dylib calls with timing (e.g., `dtrace -n 'pid$target::mlx_eval:entry { self->ts=timestamp; } pid$target::mlx_eval:return { printf("%d ns", timestamp-self->ts); }'`)
- **MLX memory telemetry**: `getActiveMemoryBytes()`, `getPeakMemoryBytes()`, `getCacheMemoryBytes()`, `resetPeakMemory()` — all bound in core, sample between operations for memory profiles

### Key performance invariants

- **One eval per token in steady state.** The decode loop should have exactly one `mxEval` (or `mxAsyncEval` + deferred read) per generated token. Additional synchronization points indicate a bug.
- **Sampling stays on GPU.** Logit tensors should not cross to CPU for sampling. The token ID crosses as a 4-byte scalar via `.item()`, not the full vocab-sized logit vector.
- **Prefill is chunked.** Prompts longer than `prefillStepSize` (default 2048 tokens) are processed in chunks with cache-only eval and `clearMemoryCache()` between chunks.
- **GPU never idles between tokens.** Async eval double-buffering ensures the next forward pass is dispatched before the current token's result is read.

### Forward pass performance invariants

The invariants above protect the macro loop (one eval per token, no idle gaps). These invariants protect what happens *inside* a single forward pass. MLX's lazy evaluation means that the **graph structure is the performance** — two graphs that produce the same output tensor can have dramatically different throughput depending on node count, kernel dispatch paths, and intermediate allocations.

- **SDPA must receive `null` mask during single-token decode when all positions are visible.** MLX's fused scaled-dot-product attention has a fast maskless kernel and a slower masked kernel. Passing a dense all-true boolean mask is functionally equivalent to `null` but routes to the slow path. During single-token decode, if the query can attend to every key in the cache (no sliding window, or the window covers the entire visible range), the mask must be `null`. This is a qualitative GPU utilization change, not a minor optimization.
- **KV cache updates must be O(1) amortized.** Pre-allocate cache buffers in chunks and write via `sliceUpdateDynamic` (or equivalent in-place scatter). Concatenating the full cache history per token is O(n) and creates a new allocation every step — forbidden in steady-state decode. This applies to all cache types: standard, rotating, and mixed-pattern caches.
- **Composite activation functions must collapse to one hot-path primitive.** Multi-op activations like tanh-based GELU create multiple graph nodes and intermediate tensor allocations per call if expressed naively. The preferred order is: use an existing native/core primitive when one exists; otherwise `compile({ shapeless: true })` the full composite helper until a native primitive is warranted by measurements. The hot path should not carry a long chain of elementwise ops per layer when one fused activation is what the model semantically needs.
- **Repeated pure decode motifs should try compile before native helper work.** If the hot path repeatedly rebuilds the same pure subgraph with explicit tensor inputs, a bounded compile spike should be attempted before introducing custom native bindings. Compile is the first lever for reducing Bun/FFI boundary churn; native helpers are for the remaining hot state that compile cannot express cleanly.
- **Choose deeper seams by semantic-stage ownership, not by op trivia.** When compile-first and obvious waste removal stop moving the paired parity metric, the next candidate should be a whole hot semantic stage such as decode attention or cache update plus visible fetch. Group together the intermediates that are hot, adjacent, and private to that stage. If a tensor only exists to feed the next hot step, it usually should not cross back through Bun in between.
- **Compile is an internal strategy, not a semantic API name.** Call sites and public helpers should read in terms of the math they perform. Keep compile reuse behind semantic function names such as `swiglu`, `crossEntropy`, or `repetition penalty`; do not turn runtime strategy into the dominant vocabulary of model code.
- **Readable reference surfaces are a performance constraint too.** If a hot-path optimization makes the main inference or training flow hard to read, move that strategy behind a backend/helper seam instead of normalizing the complexity into the teaching surface of the repo.
- **Weight-derived invariants must be computed once, not per forward call.** If a value depends only on model weights and does not change between tokens (e.g., `add(weight, 1.0)` for Gemma-style offset norms), compute it once after weight loading and store the result. Recomputing invariants per call creates unnecessary graph nodes and FFI round-trips that compound across layers.
- **Per-token graph node count must be comparable to mlx-lm.** When implementing a new model family, count the MLX ops and FFI round-trips per decode token and compare against the equivalent mlx-lm model. If yours is 5x higher, something is structurally wrong — investigate before benchmarking. Common sources of excess nodes: uncompiled composite ops, materialized masks that should be `null`, O(n) cache ops, and per-call recomputation of invariants.
- **Parity research is a paired, revertable loop.** Measure against the same-machine `mlx-lm` reference, prefer multi-trial averages over single lucky runs, and judge experiments by the paired gap or ratio rather than by raw local tok/s alone. Keep the winners, revert the losers, and record the result so the next deeper seam is chosen from evidence instead of folklore.
- **Keeper candidates must survive longer-context or boundary-sensitive reruns.** Tiny runs are for fast hypothesis screening, not for promotion. If a change looks promising, rerun it where cache growth, sliding-window visibility, or long decode behavior can matter — for example around prompt-length boundaries or with materially longer prompt/decode lengths. Do not keep a winner on the strength of a short micro-benchmark alone.

## Bench and Soak Surfaces

- `cd examples/nanogpt && bun run bench:memory` measures active-memory drift for the leak-prone scenarios we care about.
- `bun run bench:generation` measures generation throughput against recorded baselines.
- `cd examples/nanogpt && bun run soak:gpt-tiny` runs the canonical supervised soak for the tiny preset.
- `cd examples/nanogpt && bun run soak:gpt-small` runs the canonical supervised soak for the small preset.
- `cd examples/nanogpt && bun run acceptance:gpt-tiny` and `cd examples/nanogpt && bun run acceptance:gpt-small` are the loss-targeted acceptance runs.
- These commands are intentionally lock-guarded. If one is already running, the next heavy MLX command must fail fast rather than contending for GPU/runtime state.

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

mlx-c (Apple's C API for MLX) is pre-1.0 (currently v0.6.0). There is no published changelog, semver policy, or stability contract.

- **ABI audit required on upgrade.** When mlx-c is upgraded, `src/core/ffi/symbols.ts` must be re-verified against the new headers. This is documented in AGENTS.md as an ABI integrity rule.
- **No guarantee of backward compatibility.** The 0.1.x → 0.2.0 jump shows willingness to break.
- **Mitigation:** We vendor mlx-c source and pin to a known-good version. Upgrades are deliberate, tested, and reviewed.

If Apple deprioritizes mlx-c, we are blocked on new MLX features reaching TypeScript. This is the most significant external dependency risk.
