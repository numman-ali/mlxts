# Runtime Optimization Matrix

This document is the canonical map for runtime optimization work in `mlxts`.

Use it to decide:

- which hot paths should try compile first
- which surfaces should stay eager for now
- which surfaces need deeper core work such as multi-output compile
- which surfaces justify native helpers only after compile has been exhausted

The guiding order is:

1. existing fused/native primitive
2. selective single-output compile
3. selective multi-output compile
4. native helper for hot mutable state or primitives compile cannot express,
   but only behind a validated backend seam

This matrix is a backend planning document. It is not permission for semantic
model-family or training files to become runtime-strategy soup. The readable
reference surfaces should stay semantic while the implementation choices tracked
here live underneath them.

## Matrix

| Surface | Unit | Hot Path | Purity Class | Owner Layer | Status | Planned Phase | Validation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `@mlxts/nn` | `gelu` | GPT/nanoGPT MLP, generic dense FFN | single-output | `@mlxts/nn` | implemented | phase 1 | eager parity + cross-shape reuse + package tests |
| `@mlxts/nn` | `silu` | SwiGLU and Phi/LLaMA-family helper math | single-output | `@mlxts/nn` | implemented | phase 1 | eager parity + cross-shape reuse + package tests |
| `@mlxts/nn` | `swiglu` | LLaMA/Mistral/Phi decoder MLP | single-output | `@mlxts/nn` | implemented | phase 1 | eager parity + cross-shape reuse + transformer/nn tests |
| `@mlxts/nn` | `crossEntropy` | training loss across packages | single-output | `@mlxts/nn` | implemented | phase 1 | eager parity + cross-shape reuse + train/example tests |
| `@mlxts/transformers` | GEGLU approx helper | Gemma 3, Gemma 4, llama-like GELU MLP paths | single-output | `@mlxts/transformers` | implemented | phase 1 | eager parity + cross-shape reuse + family tests |
| `@mlxts/transformers` | Gemma 4 per-layer gate helper | Gemma 4 block gating path | single-output | `@mlxts/transformers` | implemented via shared GEGLU helper | phase 1 | Gemma 4 model tests + benchmark evidence |
| `@mlxts/transformers` | repetition penalty | generation sampling preprocess | single-output | `@mlxts/transformers` | implemented | phase 1 | sampling behavior tests |
| `@mlxts/transformers` | min-p filter | generation sampling preprocess | single-output | `@mlxts/transformers` | implemented | phase 1 | sampling behavior tests |
| `@mlxts/transformers` | top-k filter | generation sampling preprocess | single-output but shape-sensitive | `@mlxts/transformers` | implemented via local shape/config-memoized transform variants | phase 1b | eager parity + local shape/config-memoized transform reuse tests |
| `@mlxts/transformers` | top-p filter | generation sampling preprocess | single-output but shape-sensitive | `@mlxts/transformers` | implemented via local shape/config-memoized transform variants | phase 1b | eager parity + local shape/config-memoized transform reuse tests |
| `@mlxts/transformers` + `@mlxts/serve` | continuous sampled decode | per-row sampler state over one batched forward | scheduler strategy | `@mlxts/transformers` scheduler + `@mlxts/serve` routing | implemented for eligible LLaMA-like, Qwen 3.6 text, and Gemma 3/4 layer-pattern requests | phase 9 | scheduler tests + real Qwen/Gemma regression matrix |
| `@mlxts/serve` | serving strategy config | existing knobs to typed runtime strategy | strategy seam | `@mlxts/serve` | implemented for current supported strategies; future operator flags still gated by backend proof | phase 9 | `/info` selected-strategy output + route telemetry + focused serve tests |
| `@mlxts/serve` | production metrics | bounded request/generation/scheduler/batch/stream/memory counters and histograms | observability seam | `@mlxts/serve` | implemented first pass; cache-specific metrics deepen with future cache backends | phase 9 | `/metrics` tests + bench report correlation |
| `@mlxts/transformers` + `@mlxts/serve` | prefix cache | prompt-prefix reuse and cache-hit lifecycle | cache backend | family-owned cache snapshots, serve telemetry | landed single-request message path plus continuous-scheduler prefix-hit seeding with live Qwen/Gemma protocol-health cache-hit budgets; paged/block-deduplicated LCP reuse pending | phase 9 cache pass | snapshot/fork and seeded-restore tests across full KV, layer-pattern, and Qwen hybrid caches + repeated-chat cache-hit tests + real Qwen/Gemma protocol-cache regression |
| `@mlxts/transformers` + `@mlxts/serve` | paged KV cache | block allocator, CoW, prefix dedup | cache backend | shared cache contracts | not yet | phase 9 cache pass | cache oracle tests + endpoint concurrency ladder |
| `@mlxts/transformers` + `@mlxts/core` | TurboQuant-style KV | attention over compressed KV states | native/backend strategy | cache backend + attention backend | research only | later phase 9 | quality eval + long-context evidence + native review |
| `@mlxts/serve` | Responses/Anthropic parity | protocol adapters over one internal request | protocol seam | `@mlxts/serve` | Responses text implemented; fuller Responses and Anthropic not yet | phase 9 protocol pass | adapter tests + protocol-health `bench:serve` |
| `@mlxts/transformers` + `@mlxts/core` | Qwen gated-delta update | Qwen 3.5 / 3.6 linear-attention recurrence | hot mutable state | narrow native helper behind semantic Qwen helper | implemented | phase 7/9 performance tranche | TS recurrence oracle tests + staged Qwen parity ladder + long-output stress |
| `@mlxts/transformers` | Qwen full-attention mask reuse | cached prefill full-attention layers | single-output marker / ownership cleanup | `@mlxts/transformers` | implemented | phase 7/9 performance tranche | causal SDPA offset test + Qwen staged prompt ladder |
| `@mlxts/transformers` + `@mlxts/nn` | Qwen quantized `b/a` projection fusion | Qwen linear-attention gate projections | packed-weight helper | private eval-only model helper + package-owned quantized utility | implemented | phase 7/9 performance tranche | quantized fusion equivalence test + runtime profile call-count drop |
| `@mlxts/transformers` | mask builders | prefill and mask-shape changes | single-output but shape-derived JS sizes | `@mlxts/transformers` | not yet | phase 1b | eager parity + per-shape compile safety review |
| `@mlxts/transformers` | cache update/fetch | steady-state decode state machine | hot mutable state | `@mlxts/transformers` + `@mlxts/core` | private seam plus borrowed full-buffer internal views; deeper native work parked as research | parked research | parity bench + cache counters + correctness tests |
| `@mlxts/optimizers` | `AdamW.applySingle` | per-parameter optimizer math | multi-output | `@mlxts/optimizers` + `@mlxts/core` | not yet | phase 3 | eager parity + optimizer state tests |
| `@mlxts/train` | `gradientNorm` | training step gradient clipping | eager reduction with sync bottleneck | `@mlxts/train` | not yet | phase 3 | algorithmic rewrite first, then benchmark |
| `@mlxts/train` | `scaleGradientTree` | training step gradient clipping/accumulation | leaf-wise single-output | `@mlxts/train` | not yet | phase 3 | eager parity + tree-structure tests |
| `@mlxts/train` | `accumulateGradients` | training step gradient accumulation | leaf-wise single-output | `@mlxts/train` | not yet | phase 3 | eager parity + tree-structure tests |
| `@mlxts/core` | `compile()` | single-output pure tensor closures | single-output | `@mlxts/core` | implemented | current | transforms tests |
| `@mlxts/core` | `compileMany()` | multi-output pure tensor closures | multi-output | `@mlxts/core` | implemented; active mainline usage is currently parked during the readability-first pass | later adoption after readability-first cleanup | core transform tests + future runtime/helper coverage |
| `@mlxts/core` | `mlx_detail_compile` binding | stateful compiled execution | stateful | `@mlxts/core` | not yet | phase 4 proposal | dedicated design + ABI review |
| `@mlxts/core` + `@mlxts/transformers` | first-party MLX/MLX-C capability audit | cross-package backend planning | research | shared runtime layer | active recurring audit | cross-phase | `.reference/mlx` / `.reference/mlx-c` diff against `ffi/symbols.ts` + matrix updates + benchmark-backed decisions |
| `examples/nanogpt` | GPT MLP via `nn.gelu` | example training/inference | inherited from `@mlxts/nn` | example surface | implemented indirectly | phase 1 | nanogpt coverage + integration tests |

## Interpretation Rules

- `single-output` means the current `compile()` primitive can express it cleanly.
- `multi-output` means the math is pure but the current TypeScript compile wrapper is too narrow.
- `stateful` means the execution model itself, not just the math expression, needs deeper core support.
- `hot mutable state` means compile is not the main tool; this is where native cache work becomes appropriate after the compile phases are exhausted.

## Current Program Order

1. Preserve readable reference surfaces first.
2. Keep compile behind semantic helpers and use it where it improves repeated pure motifs.
3. Keep deeper native cache or decode work parked on the research track while the repo-wide readability and alignment pass is still active.
4. Resume native work only behind a clearer benchmark-stable seam, not by pushing more strategy into semantic family files.
5. Keep `mlx_detail_compile` as a later dedicated proposal.

## Notes

- Runtime strategy is not model identity. Do not create duplicate model configs for these rows.
- Keep backend/runtime selection private until there is a validated winner worth carrying publicly.
- Keep semantic names semantic. The matrix tracks runtime strategy, but function names and call sites should still read in terms of math and model behavior rather than `compiled*` terminology.
- Keep the readable reference surfaces readable. If a row here starts making the main inference or training flow hard to follow, the strategy belongs in a lower helper/backend layer instead of the teaching surface.
- If a semantic helper needs multiple compiled variants, keep that memoization local until the same pattern earns a shared abstraction.
- Native cache or deeper native decode work is currently a parked research track, not the active mainline thread. The earlier shallow native seam experiments were useful but not benchmark-stable.
- Heavy MLX commands are sequential and lock-guarded. Benchmark, soak, acceptance, and proof runs are not allowed to contend with each other.
- The matrix should be updated any time a runtime-sensitive hot path changes phase or ownership layer.
