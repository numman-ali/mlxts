# Architectural Posture Audit — 2026-04-28

**Audit type**: Repo-wide structural / doctrinal review. Read-only diagnosis.
**Reviewer**: Synthesized from six parallel Opus sub-agent slices over `core/nn/optimizers`, `train/data/tokenizers/lora/align/quantize`, `transformers`, `protocols/serve/agent`, `examples/*`, and `docs/scripts/root` files.
**Mechanical baseline**: `docs/reviews/2026-04-28-audit-metrics.md`.
**No code changes proposed**. No PRs opened.

---

## 1. Executive Read

**Posture rating: solid foundations, accumulated structural pressure in two packages, governance lever underused.**

The repo's doctrine is unusually well-articulated and the code largely respects it. All mechanical gates are green: typecheck, Biome lint (610 files), file-lines (300 prod files all ≤500), assertions, tensor-lifetimes, runtime-review. Cross-package dependency edges match the declared layer graph. Strategy-vs-identity discipline holds — no duplicate model configs for runtime variants. CausalLM contract is tight; multimodal composes correctly; OpenResponses naming is consistent.

The drift is at the file-organization and doc-layering level, not at code logic. Two packages — `serve` and `transformers` — have absorbed most of the recent feature pressure (15k and 21k LOC respectively), and both show structural symptoms: prefix-as-folder substitutes for real subfolders, files clustered near the 500-line cap, vestigial Phase-7 directories, and one family (`qwen3_5/`) with 33 files at one level. Per-package AGENTS.md coverage is 2/13, leaving the auto-injection guardrail mostly idle. CLAUDE.md duplicates AGENTS.md content and MEMORY.md Tier 1 violates its own "don't repeat AGENTS.md" rule.

Top concerns, severity-ordered. Structural items are tracked separately from cheap hygiene so the two do not read as equivalent blockers.

**Structural (🔴):**

1. **`packages/serve/src/` is shallow — 48 non-test files at top level (64 prod files total) plus one `protocols/` subdirectory, with prefix-grouping (`server-*`, `transformers-engine-*`, `serve-*`, `model-*`) doing the work of subfolders.** Doctrinal violation of "small role-based subfolders" rule in code-standards.md. The 500-line cap is concealing structural pressure rather than indicating tight design (10 serve files within 40 lines of cap).
2. **Cross-example coupling** — `examples/lora-finetune/data.ts` (production import of `createTrainingProofCorpus` + `parseUltrachatMessagesRow`) and `examples/chat-canary/dataset.test.ts` (test import of `parseUltrachatMessagesRow`) both reach into `examples/train-proof/`. Violates "examples are independently runnable" doctrine. Extract shared fixture to `@mlxts/data` or an align test-fixture surface.

**Governance (🟡):**

3. **11 of 13 packages lack AGENTS.md** at audit time. Auto-injection on path-touch is the right next governance lever — per-package charters codify boundaries before remediation work begins. (Resolved 2026-04-28; see §9 Status.)

**Cheap hygiene (🟢):** Bundle alongside the structural fixes; none of these are blockers.

4. Five empty Phase-7 legacy directories under `packages/transformers/src/` (`base/`, `gemma/`, `llama/`, `mistral/`, `phi/`).
5. CLAUDE.md / MEMORY.md / continuity.md doctrinal duplication.
6. Stale `@mlxts/vlm` / audio / multimodal references in `docs/python-equivalence-map.md` and `docs/gates-and-milestones.md`.

Mechanical gates and code-logic findings are mostly 🟢✓. The audit's center of gravity is structural cleanup + doc-layer discipline, not bug-hunt or refactor-for-correctness.

---

## 2. Mechanical Snapshot Summary

**Gate status — all green.**
- typecheck: 14/14 packages
- Biome lint: 610 files, no issues
- check:file-lines: 300 prod files, all ≤500
- check:assertions: clean (no `as`/`!` outside FFI)
- check:tensor-lifetimes: clean
- check:runtime-review: no runtime-sensitive prod changes pending

**Package weight (LOC, src files):**
- transformers: 20,994 LOC (112 files) — biggest
- serve: 15,110 LOC (64 files) — second biggest
- core: 6,894 LOC (37 files)
- nn: 2,497 LOC (19 files)
- tokenizers: 2,385 LOC (14 files)
- agent, train, align, quantize, data, lora, optimizers, protocols: ≤1,800 LOC each

**Examples weight:**
- nanogpt: 6,654 LOC (62 files) — heavy, has own dist/, tests, supervisor flow
- train-proof: 1,592 LOC; lora-finetune: 705; qwen3_5-image: 474; chat: 351; serve-completions: 209; chat-canary: data only

**Cross-package dependency edges** match declared layer graph. align (7 internal deps) is the heaviest consumer; serve does not import `@mlxts/nn` directly (uses transformers' generation surface — correct). nn does not import optimizers (the apparent edge was a test-file import — false alarm).

**File-cap pressure**: 10 serve files and 9 transformers files within 50 lines of the 500-line cap. The cap is doing its job but pressure is elevated.

**Per-package AGENTS.md status**: 2/13 (`serve`, `transformers`). All 13 have READMEs.

---

## 3. Critical Cross-Cutting Findings

### 3.1 🔴 `serve/src/` flat structure

Forty-eight non-test source files at the top level (64 prod files total including the `protocols/` subdirectory), with prefix groups acting as informal folders:

- `server-*` (20 files): HTTP routes, streaming writers, lifecycle, abort, events, info, json, sse-heartbeat, stop-filter, stream-runtime
- `transformers-engine-*` (13 files): the model-backed generation engine split into 13 pieces
- `serve-*` (~7 files): metrics, registry, runtime-strategy, scheduler-metrics, stream-metrics
- `model-*` (8 files): context, execution-lane, router, server, server-options, sources

The naming literally describes what folders should exist. Code-standards.md says "when a concern grows beyond a couple of files, prefer a small role-based subfolder over a crowded flat directory" — serve is the textbook violation. The one existing subfolder, `protocols/`, demonstrates the right shape: 20 protocol files cluster cleanly, average 230 LOC, much easier to scan.

Detailed proposed restructuring is in §8 below.

### 3.2 🔴 Vestigial directories in `transformers/src/`

Five empty top-level directories: `base/`, `gemma/`, `llama/`, `mistral/`, `phi/`. Zero files, zero imports anywhere in the codebase. Phase 7 family-extraction residue (the live family code now lives in `families/<name>/`). Delete in cleanup pass.

### 3.3 🔴 Cross-example coupling

Two examples reach into `examples/train-proof/`:

- `examples/lora-finetune/data.ts:10` imports `createTrainingProofCorpus` and `parseUltrachatMessagesRow` from `../train-proof/datasets` — production code path.
- `examples/chat-canary/dataset.test.ts:4` imports `parseUltrachatMessagesRow` from `../train-proof/datasets` — test code path.

Doctrine: examples are independently runnable workbooks. Both reaches break the model — production import most clearly, but a test that depends on a sibling example's source is the same coupling expressed in the test boundary. Two fixes possible:

- Move the shared corpus + UltraChat row parser into `@mlxts/data` (reusable, lives where data lives)
- Or into a small align test-fixture surface

qwen3_5-image and chat have parallel `createProgressReporter` copies (~55 LOC each), which is mild duplication but not coupling.

### 3.4 🔴 CLAUDE.md / MEMORY.md / continuity.md doctrinal duplication

- **CLAUDE.md** does `@AGENTS.md` and then restates "Current Phase" (lives in PLAN.md), "Quick Reference" (duplicates AGENTS.md § Build Commands), and a doc table (already in AGENTS.md § Documentation). Slim to ~22 lines.
- **MEMORY.md Tier 1** (10 bullets) overlaps AGENTS.md on heavy-MLX commands, runtime-review, typecheck/coverage gates, and mechanical gates. The file's own rule says "Keep Tier 1 focused on durable repo facts and recurring sharp edges, not doctrine already stated in AGENTS.md" — and then violates it. Trim to ~6 bullets covering only the irreplaceable items.
- **continuity.md** has grown to 239 lines, ~⅓ of which is per-rung Qwen evidence already captured in `docs/reviews/2026-04-24-qwen-serve-benchmark-ladder.md`. Slim to "Current Focus" + "Current State" + "Next Work", target ~80 lines.

Proposed slimmed CLAUDE.md content is in §6 below.

### 3.5 🔴 `serve/AGENTS.md` is tactical, not architectural

The existing file covers thin protocol adapters, OpenResponses spec discipline, admission control vocabulary, memory preflight honesty, and the admission-vs-continuous-batching distinction. All correct, all worth keeping. But it does **not** address: folder structure, family-cache vs serve-scheduling line, protocol-stream-writer sharing, image transport vs preprocessing line, protocols-package scope, or paged-cache / speculative-decode forward seams. The structural doctrine that prevents the next round of organic growth is missing. Recommended additions in §7.4.

### 3.6 🔴 Cross-family cache layer-type taxonomy

`families/qwen3_5/types.ts` declares `Qwen3_5LayerType = "linear_attention" | "full_attention"` while gemma3/4 use `"sliding_attention" | "full_attention"`. Each is correct for its checkpoint, but the cross-family vocabulary is not unified. When paged KV / TurboQuant land, the cache contract will need to know "is this layer's state trimmable?" without family-specific dispatch. Worth a shared infrastructure-level layer-classification before backend work begins.

### 3.7 🔴 Image preprocessing seam location

The doctrinal split is correct (`serve/media-image.ts` owns host I/O + macOS sips decode; `transformers/families/qwen3_5/preprocessing.ts` owns model preprocessing/expansion). But `media-image.ts` (477 LOC) sits next to `server-streaming.ts` and `transformers-engine-content.ts` (382 LOC) at serve top level — the file tree doesn't reveal the seam. Restructuring (§8) puts media transport under `serve/src/media/` and the model-handoff adapter under `serve/src/engine/content.ts`, surfacing the seam.

---

## 4. Per-Slice Findings

### 4.1 Foundation: core, nn, optimizers

🟢✓ **FFI / runtime safety is tight.** Per-call `OutSlot` reentrant by construction. Vector-array temporaries use `try/finally` (`withArrayVector`, `applyValueAndGrad`, `applyClosureTransform`). Native handle pairs carry explicit `try { ... } catch { output.free(); throw }` rescue. ABI shape in `symbols.ts` matches conventions: creation by value, ops with output pointer + status, getters returning directly. FinalizationRegistry correctly used as safety net.

🟢✓ **Type-system honesty.** Only `as const` / `as const satisfies` outside `ffi/` (constant narrowing, not escape hatches). `unwrapPointer`/`sizeToNumber` teach the type system real invariants.

🟢✓ **`nn/activations/{index,runtime}.ts` and `nn/losses/{index,runtime}.ts`** are the canonical semantic-vs-runtime pattern. Public file is math-named, adjacent `runtime.ts` owns compile/shape strategy. This pattern should be quoted as the standard.

🟢✓ **Module pattern compliance uniform.** Linear, Embedding, LayerNorm, RMSNorm, RoPE, GroupedQueryAttention all follow public-MxArray + `#private` config split. `#transposedWeight` cache is correctly held in `#` field as derived value.

🟢✓ **Optimizers**: clean, dispose contract honest, no schedule duplication with train. `Optimizer.applySingle` is the right per-parameter contract.

🟡 **`core/src/` has 47 top-level files.** Prefix groups already look like folders: `array-*` (4), `io-*` (4), `transforms-*` (3), `fast-*` (4), `runtime-*` (2). Subfolders only for `ffi/` and `ops/`. Naming is consistent so cost today is mostly cognitive load on the file pane. Suggested groupings: `array/`, `io/`, `transforms/`, `fast/`, small `runtime/` for the two profile files.

🟡 **`nn/` should add `layers/` and `quantized/` subfolders.** 19 files with three loose groups: layers (linear, embedding, layer-norm, dropout, conv1d, rope, rms-norm, lora-linear, grouped-query-attention), quantized variants (quantized-linear, quantized-embedding), and module/transform infrastructure (module, value-and-grad, checkpoint). Apply the same `losses/`/`activations/` pattern.

🟢 **Smell: `enableCompile`/`disableCompile`/`setCompileMode`/`clearCompileCache`** exported from `@mlxts/core` top-level barrel. These are runtime-strategy debug knobs leaking into the public API. Either gate behind `core.runtime.*` namespace or keep package-private. `compile`/`compileMany`/`checkpoint` should remain public (transform constructors).

🟢 **Minor: `array.ts` has runtime-profiling instrumentation (`coreRuntimeProfileTimestamp`, `recordWrapperConstructDuration`)** woven into hot constructors. Visible noise on the most-read class. Consider hiding behind a build/dev flag.

🟢 **Minor: `Linear`/`Embedding` `#transposedWeight` cache** is correctly invalidated by identity but lacks a JSDoc explaining the optimization. One-line note would help.

🟢 **Minor: `nn/module.ts` at 498 lines** is one line under cap. Splitting `moduleArrayState` discriminated union and freeze/update validation into `module-tree.ts` helper would relieve pressure without losing the readable single-class invariant.

### 4.2 Training stack: train, data, tokenizers, lora, align, quantize

🟢✓ **`@mlxts/train` doctrine respected.** `trainLoop` is a thin orchestrator with caller-supplied callbacks (`runStep`, `evaluate`, `onStep`, `onEval`, `onDone`, `shouldStop`, `schedule`). No Trainer base class, no override surface, no lifecycle protocol. Index barrel re-exports plain functions.

🟢✓ **`@mlxts/lora` vs `transformers/src/lora-adapters.ts` split correct in principle.** lora owns generic Module-level primitives (apply, merge, remove, native `mlxts-lora` format). transformers owns CausalLM-specific PEFT-format I/O, family-prefixed key translation (e.g., `language_model.` for mistral3). Each piece in the right place.

🟢✓ **`@mlxts/quantize` boundary clean.** Live module quantization, pre-allocated quantized layers from checkpoint plan, mode/bits resolution, GGUF I/O, pretrained-config quantization metadata parsing. No serving scheduler, no engine pool, no KV-cache concerns.

🟢✓ **`@mlxts/data` coherent and tight.** 8 files, all about row → MxArray batches. No premature unification.

🟢✓ **`@mlxts/align` 7-deps justified.** Each maps to one observable surface. Recipe layer legitimately composes everything below.

🟡 **`align/recipes.ts` (454 LOC, near cap) mixes three concerns**: dataset evaluation (`evaluateSupervisionDatasetLoss`, `evaluatePreferenceDatasetLoss`, `evaluatePreferenceMetrics` ~140 LOC), batch-picker plumbing, and step runners (`runSupervisionTrainingSteps`, `runPreferenceTrainingSteps`). Split into `recipes.ts` (training step runners) + `evaluation.ts` (dataset eval) before the next preference algorithm lands. Removes cap pressure, aligns with future `@mlxts/eval` extraction.

🟡 **`tokenizers/` has 5 `bpe-*` files at top level** (`bpe.ts`, `bpe-base.ts`, `bpe-load.ts`, `bpe-merges.ts`, `bpe-added-tokens.ts`) plus `byte-level.ts` which is BPE-only. Should cluster into `tokenizers/src/bpe/`. The 7-line `bpe.ts` barrel already proves the boundary.

🟡 **`transformers/src/lora-module-traversal.ts` partially duplicates `@mlxts/lora/src/traversal.ts`.** Transformers uses a thinner slot type (no parent/key) because `loadCausalLMAdapters` doesn't replace children. One canonical traversal in `@mlxts/lora` re-exported (or a thin path-only listing helper) would remove the duplication.

🟡 **`align/sft.ts`'s `@mlxts/optimizers` dependency** appears to be type-borrow only (via `OptimizerLike` shape, not runtime import). Verify; if so, drop the dep and let the type ride from `@mlxts/nn`.

🟢 **`train/checkpoint*.ts`** has 5 files (`checkpoint`, `checkpoint-io`, `checkpoint-manifest`, `checkpoint-serialization`, `checkpoint-types`). At borderline cluster size; could become `checkpoint/` subfolder if a 6th file lands.

🟢 **`quantize/providers/`** is empty (forward-looking placeholder for AWQ/GPTQ providers). Either land providers with parsing logic moved out of `checkpoint-plan.ts`, or remove until needed.

🟡 **`align/dpo.ts` and `sft.ts` per-step trainers compose correctly**, but `recipes.ts` mixed concerns (above) is the cap-pressure signal.

### 4.3 Transformers (deepest slice)

🔴 **Vestigial residue**: 5 empty top-level dirs (`base/`, `gemma/`, `llama/`, `mistral/`, `phi/`) — confirmed dead. Delete.

🟡 **Family seam asymmetry**: two clear tiers but the boundary isn't named.
- **Lean families** (`llama`, `mistral`, `mistral3`, `phi`, `gemma`): config + weights only, share `llama-like/` backbone. 5 families.
- **Full families** (`gemma3/`, `gemma4/`, `qwen3_5/`): own block/attention/mlp/model/norm/types/weights. 3 families.
- **Shared backbone**: `llama-like/` (8 files: attention, block, mlp, model, norm, types).

Within full families, naming is mostly consistent (config, model, attention, block, mlp, norm, weights, types) but with drift: `gemma3/` lacks `block.test.ts`/`attention.test.ts`/`weights.test.ts`; `qwen3_5/` adds many one-off files (`config-feedforward.ts`, `config-helpers.ts`, `conditional-support.ts`, `vision-support.ts`); `gemma4/` is the only family with `runtime/` subfolder; `gemma3/` and `qwen3_5/` lack `rope.ts` (gemma4 has one).

🟡 **`families/qwen3_5/` decomposition is the most acute density problem in transformers.** 33+ files at one level. Cleanly clusters into:
- `multimodal/` (~9 files): conditional, conditional-support, vision, vision-support, preprocessing
- `cache/` (~3 files): cache, batch-cache
- `linear-attention/` (~5 files): gated-delta, gated-delta-recurrence, rotary
- text-core stays at family root: model, attention, block, mlp, norm, config*, weights, types, load

Restructuring relieves cap pressure on 6 files and restores the 30-second readability test.

🟡 **`families/gemma4/runtime/` pattern is correct but only gemma4 has it.** qwen3_5 has the same compile-and-shape-keyed pattern inline. Inconsistency at pattern level. Codify the `families/<family>/runtime/` pattern in transformers AGENTS.md.

🔴 **`families/qwen3_5/types.ts:Qwen3_5LayerType` ("linear_attention" | "full_attention") vs gemma3/4 ("sliding_attention" | "full_attention").** Cross-family layer taxonomy will hurt for paged KV and TurboQuant. Family-owned cache snapshot/fork must declare trimmable vs non-trimmable per layer; today the contract dispatches family-specifically.

🟡 **3 LoRA-related files at top level** (`lora-adapters.ts` 497 LOC, `lora-module-traversal.ts`, `lora-targets.ts`). Should sit in `transformers/src/lora/`. `lora-adapters.ts` is at cap.

🟡 **`infrastructure/runtime-profile.ts` is small (1.8KB) at top level beside heavy folders.** Could move into `cache/`, `generation/`, or `infrastructure/profiling/` if more counters land.

🟢 **`infrastructure/generation/` has 18 files with `continuous-batch-*` prefix** doing folder-by-prefix. Smaller scale of the serve issue. Decomposition is fine — splits a 453-line orchestrator into typed slices — but worth keeping under review.

🟢✓ **No subtle differences hidden in lean families.** Gemma's `gelu_pytorch_tanh`, normWeightOffset, embeddingScale all surface in `gemma/config.ts` → `LlamaLikeConfig`.

🟢✓ **`infrastructure/` boundaries genuinely model-agnostic.** No family imports from another family.

🟢✓ **Strategy-vs-identity discipline clean.** No `useNative*`, `useCompile*`, `useManaged*` flags. No model-config forks for cache backend. `gemma4Family`/`gemma4TextFamily` (and Qwen equivalents) are different checkpoint truths, not runtime forks.

🟢✓ **Attention call sites semantic** (boolean/causal/null masks via `AttentionMask`).

🟡 **Forward-readiness gap: attention backend seam is per-family inline.** SDPA call site lives inside `families/<family>/attention.ts`. A future quantized-KV attention backend has nowhere to plug in without touching every family. Today's per-family inline is fine — but the next backend pass will need a semantic attention call surface in `infrastructure/`.

### 4.4 Serving stack: protocols, serve, agent

🔴 **serve/src/ flat structure** — see §3.1. Detailed restructuring in §8.

🔴 **serve/AGENTS.md is tactical, not architectural** — see §3.5. Recommended additions in §7.4.

🟢✓ **Endpoint convergence is genuinely clean.** All four protocols (`openai-completions`, `openai-chat-completions`, `openai-responses`, `anthropic-messages`) call `normalize…` and produce a `NormalizedGenerationRequest` directly. None routes through another. `transformers-engine.ts:140-244` is one entry point switching on `request.input.kind`. Doctrine respected.

🟢✓ **Family-owned-cache vs serve-owned-scheduling line clean.** `transformers-engine-prefix-cache.ts` only manipulates `TransformerCacheSnapshot`/`TransformerCache` opaque handles via public `canFork`/`fork`/`store`/`Symbol.dispose`. Serve owns lookup, eviction, accounting, identity gating, event emission. No model-family KV manipulation leaks into serve.

🟢 **`transformers-engine-*` 13-file split is logically modular, not fragmented.** Concerns mapped: routing (174 LOC), prefix-cache (316), shared (396), generation (431), streaming (226), static (246), continuous (459), batch (280), content (382), engine entry (244). They group naturally as `engine/` subfolder.

🟡 **Reasoning-tag normalization has bureaucratic drift but correct implementation.** `@mlxts/protocols` is the source of truth. Both `serve/src/protocols/reasoning-tags.ts` (12 lines) and `agent/src/reasoning-tags.ts` (6 lines) are pure re-exports. Two re-export shims for one symbol. Consumers should import from `@mlxts/protocols` directly.

🟡 **Three SSE writers** (`server-streaming.ts`, `server-responses-streaming.ts`, `server-anthropic-messages-streaming.ts`) share scaffolding implicitly: `toAsyncIterator` → `withSseHeartbeat` → state machine → terminal chunk. Each writer is the protocol-specific state machine; the loop scaffold is duplicated. Extract a shared `streaming/writer-base.ts` that owns the loop and exposes typed hooks for protocol delta handling.

🟡 **`createOpenAIChatCompletionReasoningStream` is consumed by all 4 writers** (chat, responses, anthropic, completions), not just chat. Rename to `createReasoningContentStream` and locate in `streaming/`. The OpenAI-chat name is an accident of where it was first written.

🟡 **Single-request and continuous paths replicate progress/lifecycle plumbing.** `generateSinglePreparedRequest`, `streamSinglePreparedRequest`, and `createContinuousTransformersGeneration.{generate,stream}Continuous` all set up: `emitGenerationProgress`, `createPrefillProgressReporter`, `createProgressReporter`, streaming decode state, cache session, terminal dispose. Different paths, same template. Could be one inner `runGeneration({prepared, mode, stream}): result` helper.

🟡 **`@mlxts/agent` boundary correct but underspecified.** No model execution. No protocol-spec generation. Tool-call parsing in `agent/tool-calls.ts` is client-side (parses server-emitted text); serve's `protocols/openai-chat-tool-calls.ts` is server-side (parses generated text into wire format). No code duplication, but the doctrinal split is implicit. Needs an AGENTS.md.

🟡 **`@mlxts/protocols` scope underused.** 261 LOC, one capability (reasoning tags). Naturally adjacent zero-dep wire material lives inside serve: `protocols/openai-stop.ts` (30 LOC), `protocols/openai-usage.ts` (38 LOC), `protocols/openai-models.ts` (65 LOC). Promote on need (when `@mlxts/agent` consumes them too). Don't pre-emptively widen.

🔴 **Image preprocessing seam location** — see §3.7.

🟢 **Forward-readiness landing zones reasonable.** Paged KV → `engine/cache/`. Quantized KV / TurboQuant → cache precision dimension + attention backend. Speculative decode → `engine/decoding/`. Embedding endpoints → new `http/route-embeddings.ts` + non-generation `engine/embedding.ts` (don't model embeddings as `maxTokens: 0` generation).

### 4.5 Examples

🟡 **`examples/nanogpt/src/run/` is partially generic supervised-train scaffolding hiding under nanoGPT.** `supervisor.ts`, `manager.ts`, `manager-args.ts`, `manager-status.ts`, `manager-run.ts`, `files-types.ts` describe a generic supervised-run lifecycle (start/resume/status/watch/stop/cancel, run-dir layout, control-file polling, heartbeat, structured events). Only nanoGPT-specific bindings are: (a) `RunStatus.config?: GPTConfig` and `preset?: string`, (b) `acceptance.ts` importing GPT-specific helpers, (c) `acceptance-options.ts` referencing `GPT_TINY`/`GPT_SMALL`. Strip those and the rest is reusable. When SFT/DPO/QLoRA long-runs need supervised flows, this should move to `@mlxts/train/supervised-run/` or a future `@mlxts/run`. nanoGPT keeps GPT-typed `RunStatus` extension + acceptance loss/sample logic.

🟡 **nanogpt has `package.json` with workspace deps and `dist/` build output.** Marked `private: true`. But shaped like a publishable workspace package. Doctrine says it's "intentional example, not publishable" — the build artifact is a vestigial half-step toward publishability. Either keep justified as editor convenience (and document) or stop building.

🟡 **nanogpt missing top-level README.md.** Given it's the canonical regression surface and run-manager flow is documented across the repo, this is a gap.

🔴 **Cross-example coupling** — see §3.3.

🟡 **`train-proof/stages.ts` uses `Reflect.get(model, "model")` → `layers` → `selfAttention` → `qProjection`** to verify QLoRA quantized base preservation. Structural inspection in example code. If QLoRA preservation is a real product invariant, this belongs as `assertQuantizedBasePreserved` helper in `@mlxts/lora`.

🟡 **`expectTrainableModule(model: CausalLM): Module` `instanceof Module` runtime narrowing** appears in both `train-proof/runtime.ts` and `lora-finetune/runtime.ts`. The fix is a typed helper exposed by `@mlxts/transformers`, not widening the `CausalLM` contract — `CausalLM` is a behavior contract, and pulling `Module` into it would leak implementation shape into a behavior boundary.

🟡 **`createProgressReporter` (~55 LOC) duplicated** between `examples/qwen3_5-image/index.ts` and `examples/chat/index.ts`. Reusable load-progress rendering. Should be a small `@mlxts/transformers` helper.

🟢✓ **Documentation discipline holds.** No `acceptance:gpt-*`, `soak:gpt-*`, or `manager:*` script in root package.json — those live only in `examples/nanogpt/package.json`. Root scripts are appropriately repo-wide.

🟢✓ **`train-proof` is correctly proof-shaped** (pinned UltraChat/UltraFeedback subsets, deterministic slicing, real-data parity assertions). Honest README.

🟢✓ **`chat-canary` clean.** Data only; no logic creep; review-judge sub-example correctly labelled experimental.

🟢✓ **`qwen3_5-image`, `chat`, `serve-completions`** all thin and good.

### 4.6 Doctrine + tooling

🔴 **CLAUDE.md duplication** — see §3.4. Slimmed proposal in §6.1.

🟡 **MEMORY.md Tier 1 overlap with AGENTS.md** — see §3.4. Trim to ~6 bullets.

🟡 **continuity.md growing into permanent ledger** — see §3.4. Slim to ~80 lines.

🟢 **`docs/python-equivalence-map.md` and `docs/gates-and-milestones.md` mention `@mlxts/vlm` / audio / multimodal.** Update to match doctrine ("there is no such package; multimodal lives in @mlxts/transformers + @mlxts/diffusion"). Two stale matches.

🟢 **`scripts/` inventory coherent.** Gates: file-lines, type-assertions, tensor-lifetimes, runtime-review, coverage. All running. **One real gap**: no `check:cross-package-imports` to enforce the dependency graph in `docs/ecosystem-structure.md`. align (7 deps) is the easiest place to creep.

🟢✓ **Generation-paradigm package boundary respected.** No `@mlxts/vlm`, `@mlxts/audio`, `@mlxts/multimodal` package. Multimodal lives correctly in `@mlxts/transformers`.

🟢✓ **OpenResponses naming consistent** in AGENTS.md, serve/AGENTS.md, serve/README.md, MEMORY Tier 2. Two stale matches in legacy docs (above).

🟢✓ **Per-package READMEs all present** (13/13 packages, 6/7 examples — `nanogpt` is the missing example README; `chat-canary` has one despite being data-only). Quality is reasonable; not all are equally tight.

---

## 5. Cross-Cutting Recommendations

### 5.1 Boundary integrity is mostly clean

The cross-package dependency graph matches the declared layer graph. Family-owned-cache vs serve-owned-scheduling is honored. CausalLM is the universal contract. MoE is a block-level swap. Image transport (serve) vs preprocessing (transformers) is at the right line.

The two real boundary smells are: (a) `transformers/src/lora-module-traversal.ts` partially duplicates `@mlxts/lora/src/traversal.ts`, and (b) the `expectTrainableModule(model: CausalLM)` instanceof escape that appears in two examples. The fix is a typed `@mlxts/transformers` helper, not a wider `CausalLM` contract.

### 5.2 Endpoint convergence is real, not aspirational

All four protocols normalize directly into `NormalizedGenerationRequest`. The engine entry point switches on `request.input.kind`. The remaining duplication is in **stream writers** (3 sibling SSE state machines) and **lifecycle plumbing** (single vs continuous paths). Both are factor-out-able without changing the protocol-thin contract.

### 5.3 DRY at the right altitude

The repo applies DRY correctly at most layers: shared protocol normalization (one request model), shared cache contracts (family-owned snapshot, serve-owned lifecycle), shared runtime helpers under semantic call sites. The remaining duplication is local and at the right altitude to extract:

- Stream writer scaffolding (3 copies → 1)
- Reasoning-tag re-export shims (2 trivial copies → import directly from `@mlxts/protocols`)
- Generic supervised-run primitives (currently in nanogpt, should be in train)
- Lora module traversal (transformers fork → consume from `@mlxts/lora`)
- Progress-reporter formatter (2 example copies → `@mlxts/transformers` helper)

### 5.4 Forward-readiness landing zones

| Capability | Lands at | Seam clarity |
|---|---|---|
| Phi-3.5 / Mistral 7B v3 / Llama 3.1 quirks | New lean family or extend llama-like | Clear |
| Paged KV cache | `infrastructure/cache/` + family snapshot/fork | Clear contract; cross-family layer-type taxonomy needs work first |
| TurboQuant KV | `infrastructure/cache/` precision + attention backend | Doctrine ready, attention-backend seam is the gap |
| Speculative decode / MTP | `infrastructure/generation/` + cache trim/restore | Cache-restore exists; no draft-verify split yet |
| Embedding endpoints | New `http/route-embeddings.ts` + `engine/embedding.ts` | Clear (do not model as maxTokens:0 generation) |
| QLoRA proper / DoRA | `@mlxts/lora` (apply/merge), `@mlxts/quantize` (NF4 provider), `transformers/lora-adapters` (PEFT recognition) | Clear |
| ORPO / KTO | `@mlxts/align` siblings to dpo.ts, sharing loss-utils | Clear |
| GRPO | `@mlxts/align` + sampling-during-training extension | Stresses today's seam; design before implementation |
| New attention backend | New `infrastructure/attention/` semantic call surface | Today's per-family inline is the gap to design first |

The single weakest seam is **attention backend selection**: it's per-family inline today, with no infrastructure-layer dispatch. When the next backend pass begins (TurboQuant, FlashAttention variant, quantized-KV), the first design discussion should be where backend selection lives.

---

## 6. Doc-Structure Recommendations

### 6.1 Proposed slimmed CLAUDE.md (~22 lines)

```md
@AGENTS.md

## Project: mlxts

TypeScript-native ML stack for Apple Silicon. MLX bindings, neural networks,
training, pretrained model loading, generation, and serving.

Per-package agent notes (auto-injected when those paths are touched):
- packages/<name>/AGENTS.md for any package you edit

Always read first:
- AGENTS.md — doctrine
- MEMORY.md Tier 1 — durable cross-session sharp edges
- continuity.md — current-phase handoff state

Then as needed:
- PLAN.md for phase status and exit criteria
- docs/agentic-loop.md for the engineering workflow
- docs/design-reasoning.md for the reasoning behind structural choices
- docs/ecosystem-structure.md for the @mlxts/* package map
- docs/runtime-safety.md and docs/runtime-optimization-matrix.md for hot paths

Build, test, gate, and bench commands live in AGENTS.md § Build Commands.
Do not duplicate them here.
```

Deletions: "Current Phase" block (lives in PLAN.md), "Quick Reference" block (duplicates AGENTS.md § Build Commands), the doc table (already in AGENTS.md § Documentation).

### 6.2 MEMORY.md Tier 1 cleanup

Remove bullets 3–6 (heavy MLX commands, runtime-review, typecheck/coverage gates, mechanical gates) — these live in AGENTS.md. Keep the irreplaceable ones:
- examples/nanogpt as committed example surface
- sub-agent posture
- mlx-c-first reflex
- reference-first model truth
- runtime-strategy-not-identity
- benchmark-truth-first

Promote the 2026-04-28 protocols/agent reasoning-tag split entry from Tier 2 to Tier 1 — recurring boundary that's easy to violate when adding new wire protocols.

### 6.3 continuity.md scope discipline

Slim to: Current Focus + Current State + Next Work. Move per-rung Qwen evidence sentences into existing `docs/reviews/2026-04-24-qwen-serve-benchmark-ladder.md` and link out. Target ~80 lines.

### 6.4 Stale doc updates

- `docs/python-equivalence-map.md` and `docs/gates-and-milestones.md`: remove `@mlxts/vlm` / `@mlxts/audio` / `@mlxts/multimodal` references.
- Any reference to legacy top-level transformers family directories (deleted as part of remediation).

---

## 7. Per-Package AGENTS.md Proposal

Adopt per-package AGENTS.md files for auto-injection on path-touch by Claude / Codex. Charter format (boundary doctrine + sharp edges + conventions) — distinct from README (developer/user-facing surface description). Audience split: README answers "how do I use this package?", AGENTS answers "if I'm editing this package, what must not break?".

### 7.1 Recommendation table

| Package | Recommend | Rationale |
|---|---|---|
| **core** | YES | FFI/ABI rules, OutSlot pattern, `using`/`Pointer` boundary, mlx-c-first reflex — easiest to violate inside this package. JIT guardrails high-value. |
| **nn** | YES | Module parameter scanning rules; `#private` for non-parameter state; weight tying via `Embedding.asLinear()`. JIT-injection prevents real bugs. |
| **transformers** | KEEP + extend | Has one. Add boundary discipline, family layout convention, vestigial-folder cleanup, lora-adapters seam, forward-readiness checklist. See §7.3. |
| **serve** | KEEP + extend | Has one. Add folder discipline, engine-vs-protocol-vs-HTTP layer split, family-cache vs serve-scheduling rule, image transport-vs-preprocessing line, protocol-stream-writer sharing rule, forward-seam discipline. See §7.4. |
| **agent** | YES | "Tool-loop primitive only; no model semantics; reasoning-tag normalization in @mlxts/protocols, not duplicated here." |
| **protocols** | YES | "Zero internal deps. Reasoning-tag, finish-reason, usage normalization shared by serve/agent live here. Don't import from @mlxts/transformers." Charter-style. |
| **tokenizers** | YES | "BPE longest-match must bound by max vocab token length, not prompt length." Charter + sharp-edge. |
| **train** | YES | Snapshot vs resume checkpoint distinction; "no Trainer base class, user owns the loop." |
| **lora** | YES | "Adapter mechanics only. CausalLM-specific PEFT I/O lives in @mlxts/transformers/src/lora/. Don't couple to a model family here." |
| **align** | YES | 7 internal deps → most likely to creep. "Recipe helpers and data prep, not a trainer." |
| **optimizers** | MAYBE | 4 src files, 515 LOC. 15-line charter could fit in README. |
| **data** | MAYBE | Small package. Could consolidate. |
| **quantize** | MAYBE | Small. One-paragraph charter probably enough. |

### 7.2 Per-package AGENTS.md sketches (foundation + training stack)

**`packages/core/AGENTS.md`** (~40 lines): @mlxts/core is the FFI boundary and the only place native pointers, `as`/`!` casts, and `any` are allowed (and only inside `src/ffi/`). All higher layers consume `MxArray` through helper functions that hide the `Pointer` brand. Hot rules: per-call `OutSlot` (no shared output buffers), `try/finally` for every native temporary, creation symbols return `Pointer` by value, ops take an output pointer and return `i32` status (use `checkStatus`), getters return values directly. New ops land in `ops/` (or `fast.ts` for fused MLX kernels). New compile-strategy plumbing stays under `transforms-*.ts`; do not promote it to the public barrel — semantic names (`mxEval`, `compile`, `valueAndGrad`) win. ABI audit `ffi/symbols.ts` against `.reference/mlx-c/mlx/c/*.h` whenever mlx-c is upgraded. Always check `mlx/c/ops.h` before writing a JS workaround.

**`packages/nn/AGENTS.md`** (~40 lines): @mlxts/nn is pure TypeScript above @mlxts/core. Every learnable component extends `Module`. Public `MxArray` and `Module` (and `Module[]`) fields are scanned as parameters; non-parameter state — config scalars, derived caches like `#transposedWeight`, dropout probabilities — must use JS `#` private fields. The own-key set is captured on first scan, so assign all public fields in the constructor. Weight tying stays functional via `Embedding.asLinear()`. Semantic-vs-runtime pattern: see `activations/{index,runtime}.ts` and `losses/{index,runtime}.ts` — math-named entry points with compile/strategy plumbing in adjacent `runtime.ts`. New layers go in `layers/` (planned regroup) and follow the `Linear`/`LayerNorm` shape: validate shapes in `forward`, throw with full context. Use `using` for disposable intermediates.

**`packages/optimizers/AGENTS.md`** (~30 lines): Per-parameter optimizer state above @mlxts/nn. Optimizers extend `Optimizer` and implement `applySingle(key, param, grad, prevState)`. `Optimizer.update(model, gradients)` orchestrates path-keyed gradient lookup, atomic staging, and old-array cleanup. Schedules live in @mlxts/train, not here — optimizers expose `setLearningRate(lr)`. State is `Map<string, Record<string, MxArray>>` keyed by dot-joined parameter path. Future fused/compiled optimizer steps replace `update()`, not `applySingle()`.

**`packages/train/AGENTS.md`** (~40 lines): Composable training primitives, not a framework. `trainLoop` is a thin function, never a base class; `runStep`, `evaluate`, and event callbacks belong to the caller. Forbid lifecycle hook surface growth ("before_optimizer_step", etc.). `applyGradientStep`/`materializeTrainingState` are gradient orchestration helpers — keep them honest about ownership and let users see `mxEval`/`synchronize` calls. Checkpoints split into `snapshot` (lightweight saves) vs `resume` (exact continuation with optimizer state); manifest parsing is typed and validated. New schedules are pure `(step) => lr` functions; never store training state on the schedule object.

**`packages/data/AGENTS.md`** (~25 lines): Row-to-MxArray batching only. No model imports, no tokenizer dependency from this package. Keep `Dataset<T>` interface tight; only add streaming/iterable surface when the second consumer needs it. `huggingface.ts` is the HF datasets-server transport — not a general HF Hub client. Pure functions for collation; no MLX state ownership beyond returning the batch.

**`packages/tokenizers/AGENTS.md`** (~35 lines): Pure local tokenizer implementations. Never depend on `@mlxts/core`, `@mlxts/nn`, or any model package — the canonical `Tokenizer` interface is the contract everyone else codes to. BPE-family files cluster under `bpe/` (multiple `bpe-*` surfaces is a real subpackage). Each tokenizer class is self-contained; `load.ts` is dispatch. New formats arrive as siblings, never widen `Tokenizer`. SHARP EDGE: BPE longest-match scanning must be bounded by maximum vocab token length, not by remaining prompt length (Gemma 4/Pi exposed an O(n²) path on a 33k-character tool prompt — see MEMORY.md 2026-04-27 [TOKENIZERS]).

**`packages/lora/AGENTS.md`** (~30 lines): Generic LoRA primitives over any `Module`. Public surface: `applyLoRA`, `mergeLoRA`, `removeLoRA`, plus the native `mlxts-lora` adapter format. Keep all CausalLM/PEFT/HF naming convention knowledge in `@mlxts/transformers/lora-adapters` — never grow a registry, never import from transformers. Native I/O is one config + one safetensors, no model-shape knowledge. Module traversal is canonical here (`traversal.ts`); transformers should not maintain a fork of slot iteration.

**`packages/align/AGENTS.md`** (~40 lines): Recipe layer above `@mlxts/train`. SFT/DPO/future ORPO/KTO are per-step trainer functions composed of `valueAndGrad` + `applyGradientStep`; never grow into framework lifecycle. `loss-utils.ts` is shared math, kept pure. Dataset evaluation (`evaluate*Loss`, `evaluatePreferenceMetrics`) lives separately from training step runners — the file split should follow that line so `recipes.ts` doesn't drift past cap. Chat-template-aware example construction belongs here (it composes tokenizer + transformers `ChatTemplate`); raw-row → trainable shape is align's responsibility, not data's. Seven internal deps is by design.

**`packages/quantize/AGENTS.md`** (~30 lines): Checkpoint and tensor quantization utilities. Owns: live module quantization, pre-allocated quantized layers from a checkpoint plan, mode/group_size/bits resolution, GGUF I/O (delegated to core), and pretrained-config quantization metadata parsing (`mxfp4`, `compressed-tensors`, `awq`, `gptq`). Does NOT own: KV cache representation, runtime serving scheduling, engine memory policy. New checkpoint providers go through `registerQuantizedCheckpointProvider`. Future runtime KV quantization (TurboQuant, FP8 cache) belongs in transformers/core, not here.

### 7.3 Updated AGENTS.md content for `transformers` (additions)

Keep existing Qwen-perf workflow content. Add four sections:

**Package boundary discipline.** @mlxts/transformers owns autoregressive architecture truth: configs, weights, family models, generation, caches, MoE blocks, vision encoders, VLM wrappers, chat templates. Diffusion is not in scope; use @mlxts/diffusion when that phase begins. CausalLM is the universal autoregressive contract. MoE is a block-level swap inside the decoder. VLMs compose vision encoder + projector with `CausalLM.forward({ inputEmbeddings })`. Do not widen `CausalLM` for anticipated future consumers. Strategy is not identity — no model config forks for cache backend, attention backend, compile choice, or KV precision.

**Family seam patterns.** Lean families (`llama`, `mistral`, `mistral3`, `phi`, `gemma`) are config + weights only and dispatch to `llama-like/`. Subtle differences (norm offset, activation, embedding scale) belong in config flags consumed by the shared backbone. Full families (`gemma3`, `gemma4`, `qwen3_5`) own their own `model.ts`, `attention.ts`, `block.ts`, `mlp.ts`, `norm.ts`, `config.ts`, `weights.ts`, `types.ts`. Add a `runtime/` subfolder when compile-keyed transforms or shape-memoized helpers cross 100 LOC. When a family grows past ~12 files at one level, split into intent-named subfolders (`multimodal/`, `cache/`, `linear-attention/`).

**Infrastructure ownership.** `infrastructure/cache/` owns cache contracts and snapshot/fork primitives. Families implement their own caches by composing `KVCache` and family-specific state. `infrastructure/generation/` owns model-agnostic generation. Family-specific generation logic does not belong here. `infrastructure/masks.ts`, `moe.ts`, `input-embeddings.ts`, `sampling/`, `gated-activations/` are model-agnostic primitives. No family imports from another family.

**Forward-readiness checklist.** New lean family: add `families/<name>/{config.ts, weights.ts}`, register in `registry.ts`, dispatch to `llama-like/`. New cache backend: extend `infrastructure/cache/` with a new variant — family-owned snapshot/fork must declare trimmable vs non-trimmable per layer. New attention backend: introduce a semantic attention call surface in `infrastructure/` before scattering backend choice across family `attention.ts` files. New decoding strategy: extend `infrastructure/generation/` and the cache trim/restore contract; do not add decoding flags to family configs. Vestigial cleanup: top-level `base/`, `gemma/`, `llama/`, `mistral/`, `phi/` (distinct from `families/<name>/`) are empty Phase-7 residue and must be deleted.

### 7.4 Updated AGENTS.md content for `serve` (additions)

Keep existing thin-protocols / OpenResponses-discipline / admission-vocabulary / memory-preflight-honesty / chunked-prefill content. Add five sections:

**Folder discipline.** `serve/src/` is organized into `http/`, `streaming/`, `engine/`, `protocols/`, `admission/`, `runtime/`, `observability/`, `model-loading/`, `media/`. Each folder names one engineering role. Do not add new top-level files. New protocol adapters land in `protocols/`. New stream writers land in `streaming/writer-*.ts`. New cache backends, scheduler variants, attention dispatch, and decoding strategies land in `engine/` (subfolders only when one role grows past five files).

**Engine vs protocol vs HTTP.** `engine/` executes generation against a `CausalLM`. It does not parse wire bodies, format wire responses, or talk to `Request`/`Response`. `protocols/` parses wire bodies into `NormalizedGenerationRequest` and formats `NormalizedGenerationResult` back into wire shapes. It does not touch streaming controllers, model execution, or admission budgets. `http/` is the only layer that sees `Request`/`Response`. `streaming/` is the only layer that touches `ReadableStreamDefaultController`. Do not collapse these layers.

**Family-owned cache, serve-owned scheduling.** `engine/prefix-cache.ts` only manipulates `TransformerCacheSnapshot` and `TransformerCache` through public `canFork`/`fork`/`store`/`Symbol.dispose`. KV layout, layer pattern handling, recurrent state, and quantized storage stay in `@mlxts/transformers`. Serve owns matching, identity gating, eviction, accounting, metrics, and protocol usage fields. New cache backends start as a transformers-side snapshot capability widening; serve adapts only after the family contract supports the operation.

**Image and modality transport vs preprocessing.** Serve owns host-side I/O and platform-native decode in `media/`. Model-family preprocessing (smart-resize, patch tokens, grid-thw, vision tower wiring) stays in `@mlxts/transformers`. The seam is `engine/content.ts` — protocol-neutral `GenerationContentPart` enters, model-prepared `PreparedPrompt` exits. Audio, video, and PDF parts will follow the same shape.

**Protocol-stream writer sharing.** SSE writers (openai-completions, openai-chat, openai-responses, anthropic-messages) consume `GenerationStreamEvent` and run reasoning-tag and stop-sequence streams. Shared scaffolding lives in `streaming/runtime.ts`, `streaming/lifecycle.ts`, `streaming/heartbeat.ts`, `streaming/observability.ts`, `streaming/stop-filter.ts`. Each writer is the protocol-specific state machine on top. Reasoning-tag normalization is in `@mlxts/protocols` — serve consumes from there, does not fork.

### 7.5 New AGENTS.md content for `protocols` and `agent`

**`packages/protocols/AGENTS.md`** (~25 lines): Zero-dep wire helpers shared across `@mlxts/serve` and `@mlxts/agent`. Today: reasoning-tag normalization (`splitReasoningTags`, `createReasoningTagStream`, `cleanReasoningFromText`). The package's reason for existence is to keep wire-format normalization out of any package that owns side effects (no fetch, no streams, no model state, no FFI). Promote a helper from serve into protocols only when both serve and agent need it; do not pre-emptively widen the surface. Future candidates that fit: shared SSE event helpers, finish-reason normalization, OpenAI-style usage formatting, cross-protocol model-id parsing. Anything that depends on `Request`, `Response`, MLX, transformers, or fetch does not belong here.

**`packages/agent/AGENTS.md`** (~30 lines): Local tool-loop and CLI primitives on top of OpenAI-compatible chat surfaces. Owns: agent message types, tool registration, tool-call parsing from generated assistant text (the client side of the same wire format that serve emits), tool execution scheduling, max-iteration discipline, conversation state, CLI presentation. Does not execute models. Does not parse server-side wire bodies. Does not duplicate reasoning-tag logic — re-export from `@mlxts/protocols`. The boundary with serve is one fetch call to `/v1/chat/completions`; the boundary with protocols is one shared symbol. Future approval/sandbox policy, multi-step planning, and approval prompts all live here, not in serve. Keep the agent package free of MLX and transformers imports so it stays usable against any OpenAI-compatible server.

### 7.6 Per-example AGENTS.md (selective)

- **`examples/nanogpt/AGENTS.md`** (~20 lines): YES. Scope: `run/` is "production code" for the supervised run path. AGENTS.md should record (a) `run/` ownership and runtime-review expectations, (b) GPT-only fields that should stay (and which fields should not grow), (c) checkpoint metadata format guarantees, (d) the rule that reusable abstractions move out, not in. **Mention** the future extraction of generic supervised-run primitives.
- **`examples/train-proof/AGENTS.md`** (~20 lines): YES. Scope: pinned dataset slices, the four canonical stages, the `Reflect.get` introspection that is example-only and should not multiply, the rule that proof-stage code stays workflow-shaped (no growing into a recipe framework that competes with `@mlxts/align`).

Other examples (`lora-finetune`, `qwen3_5-image`, `chat`, `serve-completions`, `chat-canary`) do not yet warrant per-example AGENTS.md — their READMEs and the global AGENTS.md cover them. Revisit if any grow past ~10 files or pick up a second consumer.

---

## 8. Concrete Folder Restructuring Proposals

### 8.1 `serve/src/` (the biggest win)

```
packages/serve/src/
├── index.ts, errors.ts, types.ts                (root: barrels, types, errors)
├── cli.ts, cli-options.ts, cli.test.ts          (root: CLI entry)
│
├── http/                                        HTTP transport + lifecycle
│   ├── server.ts                                <- server.ts
│   ├── server.test.ts
│   ├── routes-openai.ts                         (extracted from server.ts)
│   ├── routes-anthropic.ts                      <- server-anthropic-messages.ts
│   ├── routes-responses.ts                      <- server-responses.ts
│   ├── route-info.ts                            <- server-info.ts
│   ├── route-generation.ts                      <- server-generation.ts
│   ├── json.ts                                  <- server-json.ts (+test)
│   ├── abort.ts                                 <- server-abort.ts
│   └── events.ts                                <- server-events.ts
│
├── streaming/                                   SSE writers + lifecycle
│   ├── runtime.ts                               <- server-stream-runtime.ts (+test)
│   ├── lifecycle.ts                             <- server-stream-lifecycle.ts
│   ├── observability.ts                         <- server-stream-observability.ts
│   ├── heartbeat.ts                             <- server-sse-heartbeat.ts
│   ├── stop-filter.ts                           <- server-stop-filter.ts
│   ├── writer-base.ts                           NEW: shared SSE scaffolding
│   ├── writer-openai-completions.ts             <- server-streaming.ts (split: completion path)
│   ├── writer-openai-chat.ts                    <- server-streaming.ts (split: chat path)
│   ├── writer-openai-responses.ts               <- server-responses-streaming.ts
│   └── writer-anthropic-messages.ts             <- server-anthropic-messages-streaming.ts
│
├── protocols/                                   UNCHANGED — already clean
│
├── engine/                                      Transformer-backed generation engine
│   ├── index.ts                                 <- transformers-engine.ts (entry only)
│   ├── shared.ts                                <- transformers-engine-shared.ts
│   ├── routing.ts                               <- transformers-engine-routing.ts
│   ├── generation.ts                            <- transformers-engine-generation.ts
│   ├── streaming.ts                             <- transformers-engine-streaming.ts
│   ├── batch.ts                                 <- transformers-engine-batch.ts
│   ├── static.ts                                <- transformers-engine-static.ts
│   ├── continuous.ts                            <- transformers-engine-continuous.ts
│   ├── content.ts                               <- transformers-engine-content.ts (+test)
│   ├── prefix-cache.ts                          <- transformers-engine-prefix-cache.ts (+test)
│   ├── execution-lane.ts                        <- model-execution-lane.ts (+test)
│   └── engine.test.ts                           <- transformers-engine.test.ts
│
├── admission/                                   Request budgets + concurrency
│   ├── concurrency.ts                           <- concurrency-engine.ts (+test)
│   ├── batching.ts                              <- batching-engine.ts (+test)
│   ├── continuous-budget.ts                     <- continuous-scheduler-budget.ts (+test)
│   └── request-limits.ts                        <- request-limits.ts (+test)
│
├── runtime/                                     Strategy + memory + model context
│   ├── strategy.ts                              <- serve-runtime-strategy.ts (+test)
│   ├── memory.ts                                <- memory-telemetry.ts
│   └── model-context.ts                         <- model-context.ts (+test)
│
├── observability/                               Metrics families
│   ├── metrics.ts                               <- serve-metrics.ts (+test)
│   ├── metrics-registry.ts                      <- serve-metrics-registry.ts
│   ├── scheduler-metrics.ts                     <- serve-scheduler-metrics.ts
│   └── stream-metrics.ts                        <- serve-stream-metrics.ts
│
├── model-loading/                               Model-id routing + bootstrap
│   ├── server.ts                                <- model-server.ts (+test)
│   ├── server-options.ts                        <- model-server-options.ts
│   ├── router.ts                                <- model-router.ts (+test)
│   └── sources.ts                               <- model-sources.ts (+test)
│
└── media/                                       Host-side image/file transport
    └── image.ts                                 <- media-image.ts (+test)
```

Each folder names one engineering role. New cache backends and attention backends land under `engine/`. Embedding endpoints land under `http/` + new `engine/embedding.ts`. Speculative decode gets `engine/decoding/`. Anthropic tool-call streaming gets one writer file under `streaming/`. The 500-line cap stops being the only thing keeping files honest.

### 8.2 `transformers/src/` adjustments

```
packages/transformers/src/
├── (root: index, registry, types, generation, chat-template, etc. — keep)
├── lora/                                        NEW: cluster the 3 lora-* files
│   ├── adapters.ts                              <- lora-adapters.ts
│   ├── module-traversal.ts                      <- lora-module-traversal.ts (or remove if @mlxts/lora absorbs)
│   └── targets.ts                               <- lora-targets.ts
├── pretrained-loading/                          DEFERRED: keep flat unless 4th file
├── pretrained/                                  UNCHANGED
├── infrastructure/                              UNCHANGED (boundaries clean)
└── families/
    ├── llama-like/, llama/, mistral/, mistral3/, phi/, gemma/, gemma3/, gemma4/   UNCHANGED
    └── qwen3_5/                                 RESTRUCTURE
        ├── multimodal/                          (conditional, conditional-support, vision, vision-support, preprocessing)
        ├── cache/                               (cache, batch-cache)
        ├── linear-attention/                    (gated-delta, gated-delta-recurrence, rotary)
        └── (text-core stays at family root: model, attention, block, mlp, norm, config*, weights, types, load)

DELETE:
├── base/, gemma/, llama/, mistral/, phi/        (vestigial empty top-level dirs)
```

### 8.3 `core/`, `nn/`, `tokenizers/` adjustments

- **core**: subfolders for `array/`, `io/`, `transforms/`, `fast/`. Keep singletons (`dtype`, `error`, `device`, `memory`, `metal`, `quantization`, `random`, `tree`, `format-shape`) at root. Stop re-exporting `enableCompile`/`disableCompile`/`setCompileMode`/`clearCompileCache` from the public barrel.
- **nn**: add `layers/` (linear, embedding, layer-norm, dropout, conv1d, rope, rms-norm, lora-linear, grouped-query-attention) and `quantized/` (quantized-linear, quantized-embedding). Keep `module.ts`, `value-and-grad.ts`, `checkpoint.ts` at root. `activations/` and `losses/` already correct.
- **tokenizers**: add `bpe/` (bpe, bpe-base, bpe-load, bpe-merges, bpe-added-tokens, byte-level). Keep SentencePiece, Tekken, char at root.
- **align**: split `recipes.ts` into `recipes.ts` (training step runners) + `evaluation.ts` (dataset eval).

---

## 9. Remediation Backlog (Tiered)

**Status (2026-04-28)**: Phase A (per-package AGENTS.md set for the 11 packages without one + slim CLAUDE.md) merged in commit `ef7e0d8`. The serve and transformers structural AGENTS guidance was tightened immediately after; remediation tranches below assume that governance baseline.

### Tier 1 — Structural must-fix before next features

1. **Fix cross-example coupling**: extract `createTrainingProofCorpus` + `parseUltrachatMessagesRow` from `examples/train-proof/datasets.ts` into `@mlxts/data` (or an align test-fixture surface). Update `examples/lora-finetune/data.ts` and `examples/chat-canary/dataset.test.ts` to import from the new home. This is the only true structural blocker — examples are doctrinally required to be independently runnable.

### Tier 1.5 — Cheap hygiene (parallel; not blockers)

Low-risk doc and folder cleanup. Bundle alongside Tier 1, but do not gate next-features work on these.

1. **Delete `transformers/src/` empty legacy directories** (`base/`, `gemma/`, `llama/`, `mistral/`, `phi/`). Free move, no risk.
2. **Trim MEMORY.md Tier 1** to ~7 bullets (remove items already in AGENTS.md).
3. **Slim continuity.md** to ~80 lines; move per-rung Qwen evidence into review docs.
4. **Update `docs/python-equivalence-map.md` and `docs/gates-and-milestones.md`** to remove `@mlxts/vlm` / audio / multimodal references.

### Tier 2 — Worth doing next sprint

1. **`serve/src/` folder restructure** (§8.1). Pure file-move with import updates; no behavior change. Single tranche, paired review artifact.
2. **`families/qwen3_5/` decomposition** into `multimodal/`, `cache/`, `linear-attention/` subfolders (§8.2). Relieves cap pressure on 6 files.
3. **Move `transformers/src/lora-*.ts` into `transformers/src/lora/`** subfolder.
4. **Split `align/recipes.ts`** into `recipes.ts` + `evaluation.ts`.
5. **Add `bpe/` subfolder** in tokenizers; cluster the 5 `bpe-*` files + `byte-level.ts`.
6. **Add `layers/` and `quantized/` subfolders** in nn.
7. **Stop re-exporting `enableCompile`/`disableCompile`/`setCompileMode`/`clearCompileCache`** from `@mlxts/core` top-level barrel.
8. **Extract `streaming/writer-base.ts`** in serve to share SSE scaffolding across the 3-4 writers.
9. **Rename `createOpenAIChatCompletionReasoningStream` → `createReasoningContentStream`** and relocate to streaming/.
10. **Extract supervised-run primitives** from `examples/nanogpt/src/run/` into `@mlxts/train/supervised-run/` (or future `@mlxts/run`). nanoGPT keeps GPT-typed extension.
11. **Extract `assertQuantizedBasePreserved`** helper to `@mlxts/lora` (replaces `train-proof/stages.ts:Reflect.get` introspection).
12. **Extract `expectTrainableModule(model: CausalLM): Module`** as a `@mlxts/transformers` helper (eliminates instanceof escapes in 2 examples). Widening the `CausalLM` contract is rejected — `CausalLM` is a behavior contract and should not pull `Module` into it.
13. **Extract `createProgressReporter` formatter** to a small `@mlxts/transformers` helper (eliminates duplication across qwen3_5-image and chat).
14. **Cross-family layer-type taxonomy**: define `infrastructure/cache/` shared `CacheLayerKind = "full" | "sliding" | "linear-recurrent"` so paged/quantized backends can dispatch without family-specific knowledge.

### Tier 3 — Fence-only via per-package AGENTS.md

The remaining drift is preventable rather than acute:
- Keep per-package AGENTS.md files current as package boundaries move.
- Keep family-seam patterns (`runtime/` subfolder, full-vs-lean tier) current in transformers AGENTS.md.
- Keep serve folder discipline + engine-vs-protocol-vs-HTTP split current in serve AGENTS.md.
- Keep reasoning-tag and protocols-package boundary current in protocols/agent AGENTS.md.
- Codify supervised-run "production code" status in nanogpt AGENTS.md.
- The `nanogpt/dist/` build artifact: either delete or document as editor convenience.
- The `quantize/providers/` empty folder: either populate or remove until needed.
- nanogpt missing top-level README.md: add one.
- `array.ts` runtime-profiling instrumentation: hide behind dev flag (minor).
- `Linear`/`Embedding` `#transposedWeight` cache: add JSDoc explaining the optimization.
- Verify and possibly drop `align`'s `@mlxts/optimizers` dep (likely type-only).

---

## 10. Governance Recommendations

1. **Add `check:per-package-agents` gate**: every package directory under `packages/` meeting a size threshold (e.g., >5 src files or >300 LOC) must have an AGENTS.md. Trivial packages exempted by allowlist. Cheap to write; permanent.

2. **Add `check:cross-package-imports` gate** (currently missing). Statically enforce the dependency graph in `docs/ecosystem-structure.md`. Catches stack inversions immediately. Reference: align (7 deps) is the easiest place to creep.

3. **Add a one-line "Sync target" to every AGENTS.md** (`Last reviewed against AGENTS.md: <date>`). Sub-agents touching a package check this; if older than ~60 days, do a quick reread.

4. **Add a review-checklist line to `docs/agentic-loop.md`**: "If you change `AGENTS.md` doctrine, search per-package `AGENTS.md` files for now-stale guidance."

5. **No script needed for CLAUDE.md slim-down** — once trimmed, the `@AGENTS.md` import is the single source of truth and any drift is structurally visible.

6. **Promote one MEMORY.md Tier 2 entry to Tier 1**: the 2026-04-28 protocols/agent reasoning-tag split — recurring boundary easy to violate when adding new wire protocols.

7. **Consider a `docs/per-package-charter.md` short policy** if per-package AGENTS.md adoption needs a charter document, but a 6-bullet section in AGENTS.md ("Per-package AGENTS.md charter") is probably enough.

---

## 11. Files Reviewed

This audit reviewed the structure, contents, or summary metrics of the following files and directories. No files were modified.

**Doctrine and root files**:
- `AGENTS.md`, `CLAUDE.md`, `MEMORY.md`, `continuity.md`, `README.md`, `PLAN.md`, `CONTRIBUTING.md`
- `package.json` (root + per-package)
- `biome.json`, `tsconfig.json`, `tsconfig.build.json`, `tsconfig.docs.json`, `typedoc.json`

**docs/**:
- `agentic-loop.md`, `architecture.md`, `code-standards.md`, `design-reasoning.md`, `ecosystem-structure.md`, `future-backends.md`, `gates-and-milestones.md`, `inference-optimizations.md`, `mlx-bindings.md`, `product-surfaces.md`, `python-equivalence-map.md`, `release-checklist.md`, `runtime-optimization-matrix.md`, `runtime-safety.md`, `serving-runtime-strategy.md`, `setup.md`
- `proposals/` (listing only)
- `reviews/` (listing only)

**Packages — all 13** (`packages/<name>/`):
- `core/` (full structure, src/, src/ffi/, src/ops/, package.json, README.md)
- `nn/` (full structure, src/activations/, src/losses/, package.json, README.md)
- `optimizers/`, `train/`, `data/`, `tokenizers/`, `lora/`, `align/`, `quantize/`, `protocols/` (full structure)
- `transformers/` (full src/, src/families/<all>, src/infrastructure/, src/pretrained/, AGENTS.md, README.md)
- `serve/` (full src/, src/protocols/, AGENTS.md, README.md)
- `agent/` (full src/, README.md)

**Examples** (`examples/<name>/`):
- `nanogpt/` (src/, src/run/, src/cli/, src/bench/, package.json — no top-level README)
- `train-proof/`, `lora-finetune/`, `qwen3_5-image/`, `chat/`, `serve-completions/`, `chat-canary/` (full structure, READMEs)

**Scripts** (`scripts/`):
- `check-coverage.ts`, `check-file-lines.ts`, `check-runtime-review.ts`, `check-type-assertions.ts`, `check-visible-tensor-lifetimes.ts`, `runtime-sensitive-ops.ts`, `runtime-command-lock.ts`, `regression-qwen-gemma.ts`, `build-package.ts`, `build-workspaces.ts`, `pack-public-packages.ts`

**Mechanical metrics**: `docs/reviews/2026-04-28-audit-metrics.md` (companion to this audit; gitignored `.tmp/` baseline relocated for durability).

**Sub-agent reports**: 6 parallel Opus sub-agent slices. No sub-agent wrote files; output returned in tool-result message bodies and synthesized into this document, which is the canonical record. The transcripts themselves are not preserved as durable evidence — this audit is the artifact.
