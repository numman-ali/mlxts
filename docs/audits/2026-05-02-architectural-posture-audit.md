# Architectural Posture Audit — 2026-05-02

**Audit type**: Repo-wide holistic posture review covering repo governance, architecture, product surface coherence, quality bar, and direction.
**Reviewer**: Synthesized from six parallel Opus sub-agent slices over `core/nn/optimizers/quantize`, `train/data/tokenizers/lora/align`, `transformers`, `diffusion`, `serve/agent/protocols`, and `examples/docs/governance` files. Mechanical baseline by a separate metrics agent.
**Predecessor**: `docs/audits/2026-04-28-architectural-posture-audit.md`. This doc anchors on its findings and reports drift, resolution, and new pressure four days later.
**Companion**: `docs/audits/2026-05-02-audit-metrics.md` (mechanical baseline).
**No code changes proposed**. No PRs opened. Read-only diagnosis.

---

## 1. Executive Read

**Posture rating: foundations are stronger than four days ago, product-surface coherence is the next horizon, and one structural pressure is forming that should be split before it ossifies.**

The four days since the last audit have been transformative. Every 🔴 from 2026-04-28 is resolved. `serve/src/` went from 48 flat files to 9 named subfolders. Per-package `AGENTS.md` coverage went from 2/13 to 14/14 with an enforcement gate. Cross-example coupling is fixed. `CLAUDE.md` slimmed from doctrine-duplication to a 23-line pointer. Cache layer-type taxonomy unified at the infrastructure layer. All mechanical gates green. The audit cadence is working — naming the drift made it actionable.

What changed beyond cleanup: `@mlxts/diffusion` went from skeleton to multi-family in five days, hosting Stable Diffusion / SDXL, FLUX.1, FLUX.2 Klein, Z-Image-Turbo, Qwen-Image, Qwen-Image-2512, Stable Diffusion 3, LTX-Video, and LTX-2 (video + audio + vocoder). 220 files, 37,627 prod LOC. The package is now plausibly the third-largest in the repo. Five families have documented real-checkpoint proofs; three do not.

The audit's center of gravity has moved. **Architectural cleanup is mostly done. Product-surface coherence is the new question.** Seven diffusion families each ship their own example CLI with ~4,000 lines of duplicated boilerplate. There is no `mlxts generate` unified surface, no diffusion serving route, no SDK shape, and no shared CLI primitives. The fragmentation is not architectural drift — it is the expected state during the per-family build-out — but it is the right time to design the consolidation rather than let it harden.

Top concerns, severity-ordered:

**Structural (🔴):**

1. **`@mlxts/diffusion`'s `families/ltx/` hosts two architecturally distinct families with prefix-as-folder pressure.** 53 files, 14,803 LOC, of which 32 files (8,788 LOC) are LTX-2 and 21 files (6,015 LOC) are classic LTX-Video. The shared code is ~350 LOC (~2.4% of folder). Same prefix-as-folder pattern that drove the 2026-04-28 serve restructure. Should split into `families/ltx-video/`, `families/ltx2/`, and `families/ltx-shared/` before any LTX-2.3 tranche begins.
2. **`continuity.md` regrew to 593 lines (target: ~80; cap-equivalent: ~150).** The prior audit recommended slimming; the file grew from 239 to 593 lines instead. The `## Latest Evidence` section (249 lines) duplicates content already in `docs/reviews/` artifacts. Doctrinal-duplication finding was not addressed — every new tranche appended evidence bullets instead of pointing to the review artifact.
3. **AXI Principle 7 (session hooks) is absent repo-wide.** No CLI in the repo self-installs into Claude Code or Codex `SessionStart` hooks. This is the one AXI principle no Phase 9.5 tranche has addressed. Largest remaining AXI gap.

**Product-surface (🔴):**

4. **No `mlxts generate {image|video|audio}` unified CLI exists.** Seven diffusion family CLIs each invent their own flag conventions, with at least one inversion (`--allow-download` vs `--local-files-only`). ~4,000 lines of duplicated CLI boilerplate. AI SDK's `generateText`/`generateImage`/`generateObject` shape teaches the right answer: per-paradigm CLIs/SDK exports, sharing primitives but with paradigm-specific surfaces. The unification needs to be designed deliberately as a phase artifact, not improvised per family.
5. **No design decision recorded for diffusion serving.** `@mlxts/serve` handles text generation (chat, completions, responses, anthropic). There is no `/v1/images/generations` route, no `DiffusionEngine` interface, no serve→diffusion package coupling. The text-generation `GenerationEngine` shape does not fit image/video/audio output. The architectural decision (widen serve vs sibling package vs explicit hole) is not in any AGENTS.md or design doc. Will become a 🔴 the moment diffusion serving work begins without it.

**Quality bar (🔴):**

6. **Classic LTX-Video and LTX-2 lack documented real-checkpoint proofs.** Together they are 39% of `@mlxts/diffusion` LOC. Five other families have `*-real-checkpoint-proof.md` review docs; LTX has only "Add" and finite-proof evidence. Largest open `Prove` debt in the repo.
7. **Training stack has completeness gaps that the "full end-to-end ML library" framing exposes.** `@mlxts/align` has SFT + DPO but no PPO/ORPO/KTO/GRPO. `@mlxts/lora` has standard LoRA but no QLoRA / DoRA. No dedicated evaluation harness package. These are not bugs — they are absent surfaces. If the repo's product framing is "complete and thorough, industry-standard, better than everything else," they are gaps.

**Governance (🟡):**

8. **5 of 16 examples lack `AGENTS.md`** and the `check:per-package-agents` gate does not scan `examples/`. Missing: `chat`, `chat-canary`, `lora-finetune`, `serve-completions`, `train-proof` — the most complex non-diffusion examples.
9. **`MEMORY.md` Tier 2 has entries spanning 60–70 lines.** Defeats the "lookup index" intent.
10. **`serve-completions` example is a test fixture posing as a product surface.** Should move to `packages/serve/tests/integration/` or be marked test-only.

**Cheap hygiene (🟢):** 19 prod files within 50 LOC of the 500-line cap (up from 10 at last audit). 27 of these have hidden multi-concern structure that should split. ~13 represent single logical units that could legitimately breathe to 600–700. The cap question is one section below.

**Resolved since 2026-04-28 (the reassuring news):** Serve flat structure, AGENTS.md tactical-not-architectural, image preprocessing seam location, cache layer-type taxonomy, vestigial Phase-7 dirs in transformers, qwen3_5/ flat 33 files, LoRA at top level of transformers, cross-example coupling, CLAUDE.md duplication. Every 🔴 from last audit fixed.

---

## 2. Mechanical Snapshot Summary

Full numbers in the companion `2026-05-02-audit-metrics.md`. Headline:

**Gate status — all green.**
- typecheck: 15/15 packages (was 14)
- Biome lint: 1,076 files, 0 issues (was 610)
- check:file-lines: 542 prod files, all ≤500 (was 300)
- check:assertions: clean
- check:tensor-lifetimes: clean
- check:runtime-review: clean
- check:per-package-agents: 14/14 (was 2/13 at last audit, was 14/14 by 2026-04-28 close — held)

**Package weight (LOC, src files, delta vs last audit):**
- `transformers`: 26,744 LOC / 151 files (was 20,994 / 112) — +27% LOC, +35% files in 4 days
- `serve`: 21,679 LOC / 96 files (was 15,110 / 64) — +43% LOC, +50% files in 4 days
- `diffusion`: 37,627 LOC / 141 files (was ~0) — net-new third-largest package
- All others: stable

**Cross-package edges:** clean. Layering matches declared graph. `diffusion` does not import `transformers` (correct — design constraint). `serve` does not import `nn` directly. `agent` is experimentally fenced (no escape into serve).

**File-cap pressure:** 19 serve, 13 diffusion, 9 transformers, 5 core, 1 nn, 1 tokenizers prod files within 50 lines of cap. Total: 47 files in 450–499 band (up from ~30).

**Per-package AGENTS.md:** 14/14 packages. 11/16 examples (governance gap — gate doesn't scan examples).

**New review docs since 2026-04-28:** 135 docs in 4 days. Daily distribution: 04-29 (9), 04-30 (61), 05-01 (60), 05-02 (5 in progress).

**CLI surface inventory:** 2 published binaries (`mlxts-serve`, `mlxts-agent`), 23 example CLIs (one per family + tooling).

---

## 3. Repo Posture

This section answers: *is the discipline holding? where is governance working, where is it slipping?*

### 3.1 Discipline holding

🟢 **Sub-agent workflow is the de facto operating mode.** Six parallel Opus slice agents produced this audit's input. The 2026-04-28 audit used the same shape. The pattern is now embedded in the agentic-loop docs and respected.

🟢 **Reference-first model-truth discipline is observable in commit cadence.** Every diffusion family added in the past four days has a paired `2026-05-XX-<family>-snapshot-skeleton.md` or `-runtime-foundation.md` review doc establishing reference parity before code lands. The "Add" → "Prove" linguistic distinction in commit messages (used carefully: `Add LTX media proof verifier` vs `Prove FLUX.2 Klein real checkpoint generation`) is consistent.

🟢 **"Strategy is not identity" doctrine holds.** No duplicate model configs for runtime variants. Native gated-delta is exposed as a backend flag, not a `qwen3_6-native` family. Compile-vs-eager decisions live in runtime helpers, not config schemas.

🟢 **MLX-C-first habit is visible.** `Conv3d` landed via mlx-c binding (`2026-05-01-core-nn-conv3d.md`), not a JS shim. Safetensors int64 interop extended through proper FFI (`2026-05-01-safetensors-int64-interop.md`).

🟢 **Mechanical gates have grown without weakening.** `check:training-proofs`, `check:phase10-proofs`, `check:tensor-lifetimes`, `check:runtime-review`, `check:per-package-agents`, `check:cross-package-imports` all present. None disabled. None bypassed.

### 3.2 Discipline slipping

🔴 **`continuity.md` doctrinal duplication** (see §1, finding 2). The pattern: every new tranche appends evidence bullets to `## Latest Evidence` rather than producing a concise pointer. The file is the most-touched governance doc and the weakest enforcement signal. There is no gate on its size or shape.

🟡 **`MEMORY.md` Tier 2 is becoming an archive, not an index.** Single entries 60–70 lines long defeat the "search-when-you-touch-the-area" intent. The LTX/LTX-2 entry alone is 69 lines.

🟡 **Example-level `AGENTS.md` is unguarded.** 5/16 examples missing. The most-complex non-diffusion examples (`chat`, `lora-finetune`, `train-proof`, `chat-canary`, `serve-completions`) are exactly the ones that need governance most.

🟡 **`toon()` formatting helper is independently implemented in 4+ files** under `packages/serve/src/`. Mild DRY drift.

### 3.3 Governance assessment

The repo's governance machinery (gates, AGENTS.md, review docs, audit cadence) is largely working. The two slipping signals are both narrative-doc related — `continuity.md` and `MEMORY.md` Tier 2. They share a root cause: there is no soft gate on length or recency of pointer-vs-content in narrative docs. A `check:continuity-shape` script could enforce: ≤200 lines, no `## Latest Evidence` block longer than 30 lines, all evidence sections must be a table of `(date, review-doc-link, one-line)`. Without this gate, the file will continue regrowing.

---

## 4. Architectural Posture

Per-slice findings, drift since 2026-04-28, severity-marked.

### 4.1 Foundation primitives (`core`, `nn`, `optimizers`, `quantize`)

**Resolved since 2026-04-28:**
- 🟢 `nn/src/layers/` and `nn/src/quantized/` subfolders exist and are populated
- 🟢 `quantize/src/providers/` placeholder concern void (never created, doctrine confirmed)

**Persisting from prior audit:**
- 🟡 `core/src/array.ts` (498 LOC) still has runtime-profiling instrumentation woven into the hot `MxArray` constructor. Six profiling imports inflate the file and add noise to the most-read class.
- 🟡 `core/src/` root has 27 non-test files with prefix-grouping (`array-*`, `io-*`, `transforms-*`) doing folder work manually.
- 🟡 `nn/src/module.ts` (498 LOC) — `moduleArrayState` and freeze/update validation could split into `module-tree.ts`.

**New since 2026-04-28:**
- 🟡 `core/src/fast.ts` (474 LOC) now holds Qwen-specific `qwenGatedDeltaUpdate` (~157 lines + types) alongside generic fused ops. Model-family vocabulary (`QwenGatedDeltaUpdateResult`) leaking into the `mx.fast` namespace.
- 🟢 Conv3d, safetensors int64 interop landed cleanly via mlx-c.

**Cross-package layer discipline:** clean. FFI types (`Pointer`, `_ctx`) confined to `core/ffi/`. No assertions outside the FFI boundary. Compile flags (`enableCompile`/`disableCompile`) leak to top-level barrel — flagged at last audit, still present.

### 4.2 Training stack (`train`, `data`, `tokenizers`, `lora`, `align`)

**Resolved since 2026-04-28:**
- 🟢 Cross-example coupling FIXED: `lora-finetune/data.ts` and `chat-canary/dataset.test.ts` no longer reach into `train-proof/`. Shared corpus moved to `@mlxts/data` (`createTrainingProofCorpus`, `parseUltrachatMessagesRow`).
- 🟢 `align/recipes.ts` cap pressure FIXED: `evaluation.ts` extracted at 290 LOC, `recipes.ts` now 200 LOC.
- 🟢 `tokenizers/src/bpe/` subfolder exists with 10 files clustered cleanly.
- 🟢 All 5 in-scope packages have `AGENTS.md` (was 2/13 across whole repo).

**New since 2026-04-28:**
- 🟡 `tokenizers/src/bpe/bpe-base.ts` (468 LOC) and `clip.ts` (432 LOC) approaching cap.
- 🟡 `supervised-run/supervised-run.test.ts` at 1,005 LOC. Test files exempt from cap, but readability debt.
- 🟡 `transformers/src/lora/module-traversal.ts` still duplicates `@mlxts/lora/src/traversal.ts`. Lora AGENTS.md now forbids forking but the existing fork persists.

**AXI hardening landed across four CLIs** (training-proof, training-proof-matrix, lora-finetune, supervised-run-manager). Three are AXI-strong; supervised-run manager has acknowledged human-prose default status output (opt-in TOON via `--json`).

### 4.3 Transformers (text + multimodal encoders)

**Resolved since 2026-04-28:**
- 🟢 Five empty Phase-7 dirs (`base/`, `gemma/`, `llama/`, `mistral/`, `phi/`) all deleted.
- 🟢 `families/qwen3_5/` decomposed: 33 flat files → 17 root + `cache/` (3) + `linear-attention/` (5) + `multimodal/` (9). Subfolder structure is the audit's proposed shape.
- 🟢 LoRA files moved to `transformers/src/lora/` subfolder.
- 🟢 Cross-family layer-kind taxonomy unified: `infrastructure/cache/layer-kind.ts` provides canonical `CacheLayerKind = "full" | "sliding" | "linear-recurrent"` with `cacheLayerKindFromAttentionType()` bridge.

**New since 2026-04-28:**
- 🔴 `qwen3_5/cache/batch-cache.ts` (499 LOC) and `index.ts` (480 LOC) — cap pressure migrated *into* the new subfolder. The split moved code without relieving density.
- 🟡 Encoder family pattern (T5, CLIP, Whisper, Qwen2-VL) emerged with no doctrinal home. AGENTS.md describes lean families and full families but not "conditioning encoders." T5/CLIP/Whisper correctly kept *out* of the `CausalLM` registry, but the pattern is undocumented.
- 🟡 `runWithHiddenStates()` exposed on `LlamaLikeModel` (used by Qwen2 for Qwen-Image, Qwen3 for Z-Image). Not part of `CausalLM` contract — callers must reach below the interface. Defining `EncoderHiddenStates` interface before a third caller arrives would prevent silent contract widening.
- 🟡 `pretrained/progress.ts` defaults `writeLine` to `console.log`. Any caller using the default contaminates agent stdout. AXI hazard.
- 🟡 Cross-family RMSNorm and RoPE duplication across 3-4 families.

### 4.4 Diffusion

This package didn't exist meaningfully at the last audit. Five days of breadth-first family build-out has accumulated genuine pressure. **Slice D's findings are the densest in this audit.**

**Folder shape:** `families/<family>/` mirrors transformers. Shared infrastructure in `schedulers/` and `pretrained/`. AGENTS.md promises a `src/sampling/` directory that does not exist.

**🔴 `families/ltx/` hosts two architecturally distinct families.** 53 files, 14,803 LOC. 21 classic LTX files + 32 LTX-2 files + ~350 LOC shared. Shared fraction is 2.4% — does not justify co-location. Same prefix-as-folder pattern that drove the 2026-04-28 serve restructure. **Recommendation: split into `families/ltx-video/`, `families/ltx2/`, `families/ltx-shared/` before LTX-2.3 lands.**

**🔴 `families/flux/config.ts` (477 LOC) duplicates utilities already extracted into `families/flux2/config-parsing.ts`.** Flux was built first; later families import the extraction; flux was never refactored. ~270 LOC of duplication. After refactor, flux/config.ts drops to ~280 LOC and `flux2/config-parsing.ts` should move to package-level `src/config-parsing.ts`.

**🔴 `families/stable-diffusion/pipeline.ts` (498 LOC, 2 lines from cap)** mixes 30+ symbols across latent shapes, CFG, SDXL conditioning, denoising loop, public API. Should split into `pipeline-core.ts` (denoising loop) + `pipeline.ts` (public surface).

**🟡 Cross-family duplication audit** identified four patterns:
- Config parsing helpers (~270 LOC duplicated in flux)
- VAE decode wiring (~150 LOC near-duplicate across 5 families)
- Latent initialization + scheduler sigma scaling (5 families re-implement)
- CLI flag parsers (~400 LOC across 8 example CLIs)

**🟡 Package-vs-example boundary leaks:**
- `examples/ltx-video/audio-output.ts` (217 LOC) is general-purpose PCM16/WAV encoding living in an example. Belongs in `@mlxts/diffusion`.
- 7 example CLIs duplicate ~4,000 lines of CLI boilerplate. Shared CLI primitives (flag-readers, TOON output, error format) should extract to `packages/diffusion/src/cli/`.

**Add vs Prove status by family:**

| Family | Add | Tests | Finite-proof | Real-checkpoint |
|---|---|---|---|---|
| stable-diffusion (SD/SDXL) | ✅ | ✅ | ✅ | ✅ (SDXL) |
| flux (FLUX.1) | ✅ | ✅ | ✅ | ✅ |
| flux2 (FLUX.2 Klein) | ✅ | ✅ | ✅ | ✅ |
| z-image (Z-Image-Turbo) | ✅ | ✅ | ✅ | ✅ |
| qwen-image (incl. 2512) | ✅ | ✅ | ✅ | ✅ (2512) |
| stable-diffusion-3 | ✅ | ✅ | ✅ | ❌ blocked on gated Hub access |
| **ltx (classic LTX-Video)** | ✅ | ✅ | ✅ | **❌ no doc** |
| **ltx2 (LTX-2)** | ✅ | ✅ | ✅ | **❌ no doc** |

Five of eight families have real-checkpoint evidence. SD3 is blocker-bound (access). Classic LTX and LTX-2 — together 39% of package LOC — have only finite-proof evidence. **Largest `Prove` debt in the repo.**

### 4.5 Serve, agent, protocols

**Resolved since 2026-04-28 (this is the largest cleanup in the audit):**
- 🟢 Serve flat structure 🔴 → 🟢: 48 top-level files → 11 (and 9 named subfolders). Section 8 of the prior audit's restructuring proposal *fully executed and exceeded*. New subfolders: `engine/`, `http/`, `streaming/`, `media/`, `model-loading/`, `observability/`, `admission/`, `runtime/`, plus existing `protocols/`.
- 🟢 `serve/AGENTS.md` tactical-not-architectural 🔴 → 🟢: now 80 lines covering folder structure, role separation, family-cache-vs-serve-scheduling line, media transport ownership, SSE writer sharing, continuous batching evidence discipline, lazy pool pressure-relief contract. All six 2026-04-28 gaps closed.
- 🟢 Image preprocessing seam: `media-image.ts` moved to `media/image.ts` alongside `local-image.ts`, `remote-image.ts`, `decoded-image-cache.ts`. Seam visible in tree.
- 🟢 Cache layer-type taxonomy unified at infrastructure layer (see §4.3).

**New since 2026-04-28:**
- 🟡 File-cap density doubled: 19 prod files within 50 LOC of cap (was 10). Three engine files at 497–498 LOC.
- 🟡 `toon()` helper duplicated across 4+ CLI files. Extract to shared formatting utility.
- 🟡 `--base-url` (serve status) vs `--endpoint` (agent) flag inconsistency for the same concept.
- 🟡 `prompt-cache-observability.ts` still at top level (55 LOC). Defer until grows.
- 🟡 **No diffusion serving design recorded.** This is the §1 finding 5.

**Cross-package coupling:** clean. Serve does not import `nn`. Agent fenced experimentally — no escape into serve. Protocols package stays type-only.

### 4.6 Examples, docs, governance, AXI uniformity

**Resolved since 2026-04-28:**
- 🟢 Cross-example coupling FIXED.
- 🟢 Per-package AGENTS.md: 14/14 (was 2/13). Enforced by gate.
- 🟢 CLAUDE.md slimmed to 23 lines (was duplicating AGENTS.md).
- 🟢 Stale `@mlxts/vlm`/audio/multimodal references mostly removed.

**New since 2026-04-28:**
- 🔴 `continuity.md` regrew to 593 lines (target was ~80; was 239 at last audit).
- 🔴 AXI Principle 7 (session hooks) absent across every CLI in the repo.
- 🟡 5/16 examples lack `AGENTS.md`; gate doesn't scan examples.
- 🟡 `MEMORY.md` Tier 2 entries 60–70 lines long.
- 🟡 7 diffusion CLIs duplicate ~4,000 lines of identical boilerplate.
- 🟡 `examples/chat/index.ts` not AXI-compliant (plain `console.error`, no `exit_codes`, no non-TTY guard).
- 🟡 `mlxts-serve` help lacks `exit_codes` table.

**AXI compliance scorecard** (Slice F):
- 14 example CLIs score 26/30 each — diffusion family CLIs are the repo's AXI template
- 5 published binaries score 24–25/30 — minor gaps (`exit_codes` block, `bin:` self-ID)
- 3 outliers (`acceptance.ts`, `soak.ts`, `chat/index.ts`) score 11/30

---

## 5. Product Surface Coherence (Central Section)

This section answers the audit's core question: *what is the user-facing story for `mlxts`, and does the code support it coherently?*

### 5.1 The framing

The repo's product is a TypeScript-native ML stack for Apple Silicon that should expose three peer surfaces:

- **CLI**: agent-driven, AXI-shaped (TOON output, content-first, structured errors, session hooks)
- **API**: HTTP-served, OpenAI/Anthropic/OpenResponses-compatible
- **SDK**: programmatic, importable from TypeScript code

These are not stacked. The SDK is not a wrapper over the API; the CLI is not a wrapper over the SDK. They are three coherent surfaces backed by the same core packages.

### 5.2 What the AI SDK teaches

Direct read of `.reference/ai-sdk/packages/ai/src/`:

- **Per-paradigm functions, not modality-generic.** `generateText`, `generateImage`, `generateObject`, `streamText` are separate exports. There is no `generate({modality: "image"})` function. This validates the repo's "packages by paradigm not modality" doctrine and tells us the SDK shape should be per-paradigm too.
- **60+ provider packages.** `openai`, `anthropic`, `fal`, `black-forest-labs`, `luma`, `replicate`, `prodia` for media generation. Each provider is a separate npm package; the core SDK abstracts over them. Translation for `mlxts`: providers are MLX-backed model families, not external services. The shape transfers.
- **`mcp` package included.** Model Context Protocol is a peer concern, not a separate SDK. Translation: MCP support is part of the agent surface, not a sibling.
- **`open-responses` package included.** Confirms OpenResponses as a first-class API surface (not just OpenAI).
- **No mega-CLI.** The AI SDK is SDK-first; it does not ship a unified `ai generate` CLI. Translation: our CLI design is original work, not a copy. The AI SDK shape constrains the SDK and API surfaces, not the CLI.

### 5.3 The current state of `mlxts` product surfaces

**Text generation surface:** Coherent.
- CLI: `mlxts-serve` start/discover/status; `mlxts-agent run/REPL`; `examples/chat/`
- API: `/v1/completions`, `/v1/chat/completions`, `/v1/responses`, `/v1/messages` — four protocols, all normalize into `NormalizedGenerationRequest` → `GenerationEngine`
- SDK: not formally exposed; consumers import `@mlxts/transformers` `loadCausalLM` + `generate()`. Functional but not productized.

**Image generation surface:** Fragmented.
- CLI: 7 separate example CLIs (`examples/{flux,flux2,stable-diffusion,stable-diffusion-3,z-image,qwen-image,ltx-video}/`) with ~4,000 lines of duplicated boilerplate. No `mlxts generate image`.
- API: **none**. No `/v1/images/generations` route. No `DiffusionEngine`. Serve does not depend on `@mlxts/diffusion`.
- SDK: not formally exposed. Examples assemble pipeline directly from `@mlxts/diffusion` exports.

**Video generation surface:** Same as image but only LTX (classic + LTX-2).

**Audio generation surface:** Same as image but only LTX-2 audio + vocoder. Whisper is *transcription*, not generation; lives correctly in `@mlxts/transformers` not `@mlxts/diffusion`.

**Multimodal understanding surface:** Coherent.
- Image *input* through serve (`media/` transport + `engine/content.ts` preprocessing dispatch)
- Qwen3.5 VL works end-to-end with vision encoder + projector → `inputEmbeddings` → `CausalLM.forward()`

**Training/fine-tuning surface:** Partially coherent.
- CLI: `examples/train-proof/cli.ts`, `examples/lora-finetune/cli.ts`, `supervised-run/manager.ts` — three separate AXI-shaped CLIs
- API: not exposed. No serving-side route for training or fine-tuning.
- SDK: implicit. Consumers import `@mlxts/train` + `@mlxts/align` + `@mlxts/lora`.
- **Completeness gaps**: SFT + DPO done; PPO/ORPO/KTO/GRPO absent. LoRA done; QLoRA/DoRA absent. No dedicated evals harness.

### 5.4 The recommended product surface

This is a Phase 11+ proposal, not a current task. Document it now so the work compounds rather than fragments.

**Surface 1 — Unified `mlxts` CLI:**

```
mlxts serve                  # start the OpenAI/Anthropic-compatible server
mlxts agent run              # one-shot agent invocation
mlxts agent repl             # interactive REPL
mlxts generate text          # one-shot text generation
mlxts generate image         # one-shot image generation
mlxts generate video         # one-shot video generation
mlxts generate audio         # one-shot audio generation
mlxts train sft              # SFT training run
mlxts train dpo              # DPO training run
mlxts train lora             # LoRA finetune
mlxts evals run              # evaluation harness
mlxts model discover         # list available models
mlxts model status           # model pool state
```

One binary, structured subcommands, AXI-shaped throughout. Each subcommand is content-first (no-arg shows live state or usage). Session hooks self-install on first run. TOON output by default, `--json` for raw JSON, `--full` for non-truncated payloads. Replaces all 7 diffusion example CLIs as thin family-specific wrappers becomes one unified entry with family as a flag (`--family flux2`, `--model black-forest-labs/FLUX.1-Klein`). Per-family flag sets remain available because the differences are real (`--frames` for video, `--audio-output` for LTX-2).

**Surface 2 — Programmatic SDK:**

```typescript
import { generateText, streamText } from '@mlxts/sdk';
import { generateImage, generateVideo, generateAudio } from '@mlxts/sdk';
import { generateObject } from '@mlxts/sdk';
import { sftTrain, dpoTrain, loraFinetune } from '@mlxts/sdk';
import { createAgent } from '@mlxts/sdk';
```

Mirrors AI SDK's per-paradigm shape. `@mlxts/sdk` is a thin facade package re-exporting from `@mlxts/transformers`, `@mlxts/diffusion`, `@mlxts/align`, `@mlxts/agent`. The core packages remain importable directly for advanced use. The SDK is the "happy path."

**Surface 3 — HTTP API:**

```
POST /v1/chat/completions     # existing
POST /v1/completions          # existing
POST /v1/responses            # existing (OpenResponses)
POST /v1/messages             # existing (Anthropic)
POST /v1/images/generations   # NEW (text→image, OpenAI Images-compatible shape)
POST /v1/videos/generations   # NEW (custom; no industry standard yet — design from first principles)
POST /v1/audio/generations    # NEW (custom; OpenAI TTS shape as reference but not blind copy)
POST /v1/audio/transcriptions # existing intent (Whisper); not yet routed
GET  /v1/models               # existing
```

`@mlxts/serve` widens to include image/video/audio routes. New `DiffusionEngine` interface alongside existing `GenerationEngine`. New admission model for media generation (no token budgets, no SSE streaming for image, byte/duration budgets instead). Serve depends on `@mlxts/diffusion` (this coupling is the architectural trigger for the change).

**The architectural decision the repo needs to make explicit:**

The `DiffusionEngine` interface and `media-generation` admission model live in `@mlxts/serve`, not in a sibling `@mlxts/diffusion-serve` package. Justification: keeping serving unified means agents and operators learn one product, not two. The serve surface widens; `@mlxts/diffusion` does not import or depend on serve. This decision should land in `serve/AGENTS.md` and `docs/serving-runtime-strategy.md` *before* the first diffusion route is implemented.

### 5.5 Quality bar: industry-standard, not paltry-proof

The user's framing: "complete and thorough, full end-to-end, working in the serve family, using an industry-standard approach, and we want to do better than everything else."

Translation into gates:

- **No family ships in serve without real-checkpoint proof.** Currently SD3 / LTX-Video / LTX-2 lack real-checkpoint docs. The serve route for image generation should *not* expose any family that hasn't passed `Prove`-grade validation.
- **`Add` vs `Prove` distinction must be machine-readable.** Today it's linguistic (commit message "Add" vs "Prove"). A `families.json` manifest in `@mlxts/diffusion` that classifies each family by evidence stage would let the CLI and API report capability honestly: `mlxts model status --include-evidence-stage` returns `family: ltx-2, stage: finite-proof, real-checkpoint: missing`.
- **Industry-standard means OpenAI Images API shape for image generation.** `POST /v1/images/generations` should match OpenAI's request/response shape (prompt, n, size, response_format) so AI SDK clients work unmodified. Custom shapes for video and audio are unavoidable since OpenAI has no video API; design them deliberately, document them as `mlxts`-spec, not `OpenAI`-spec.
- **Better than everything else means a measurable axis.** Serve already tracks `mean_server_prefill_tps` and `mean_server_decode_tps`. Diffusion serve should track `mean_server_image_seconds_per_megapixel`, `mean_server_video_seconds_per_frame_per_megapixel`, `mean_server_audio_seconds_per_second`. Comparable ladder to mlx-lm's diffusion examples and Comfy.

### 5.6 The training stack as part of the same product

The user's framing extends to training: *"a full end-to-end ML library."* This means the train/fine-tune/align stack gets the same treatment as generation:

**Current state:**
- SFT training (`@mlxts/align/sft.ts`) — production-quality
- DPO training (`@mlxts/align/dpo.ts`) — production-quality
- LoRA fine-tuning (`@mlxts/lora` + recipe in `align/`) — production-quality
- Supervised run manager — operator-friendly long-run control
- Loss tracing, checkpoint resume — landed
- Evaluation hook (`@mlxts/align/evaluation.ts`) — minimal

**Completeness gaps:**
- PPO / GRPO (preference-RL post-DPO) — absent
- ORPO / KTO (alternatives to DPO) — absent
- QLoRA (quantized LoRA, very common production target) — absent
- DoRA (decomposed LoRA) — absent
- Dedicated evals harness (HumanEval, MMLU, MT-Bench-style) — absent
- Reward modeling — absent
- Distillation — absent
- Continual pretraining / mid-training — absent

These are not bugs. They are absent surfaces. If the product framing is "complete and thorough, better than everything else," the audit should call them out so the next phase plan integrates them.

**Recommended Phase 8 completion checklist** (timeline TBD by user):
1. PPO/GRPO support in `@mlxts/align` — closes the post-DPO RLHF loop
2. ORPO/KTO support — alternative preference recipes
3. QLoRA in `@mlxts/lora` — quantization-aware LoRA, the mainstream production fine-tune
4. `@mlxts/evals` package — dedicated harness (HumanEval, lm-evaluation-harness compatibility, MT-Bench-style)
5. Reward modeling primitive in `@mlxts/align`

---

## 6. Direction (Proposed Phase 11+ Roadmap)

Synthesized from audit findings. Not a commitment — a deliberate plan for user review.

### 6.1 Phase 10b — Diffusion completeness

**Goal:** every diffusion family at `Prove` stage. Architectural cleanup before the family count grows.

1. **Split `families/ltx/` into `ltx-video/`, `ltx2/`, `ltx-shared/`.** Before LTX-2.3 lands. ~2-hour mechanical refactor.
2. **Real-checkpoint proofs for classic LTX-Video and LTX-2.** Operator runs with `Lightricks/LTX-Video` and `Lightricks/LTX-2`. Already has working proof CLI; just needs the run + review doc.
3. **Refactor `flux/config.ts` to import from `flux2/config-parsing.ts`.** Move shared helpers to `src/config-parsing.ts`. ~1-hour refactor.
4. **Split `stable-diffusion/pipeline.ts` (498 LOC) into `pipeline-core.ts` + `pipeline.ts`.** Proactive split before cap.
5. **Move `examples/ltx-video/audio-output.ts` to `packages/diffusion/src/media/wav-encoder.ts`.** Eliminate package-vs-example boundary leak.
6. **Create `packages/diffusion/src/sampling/`** — extract `createInitialLatents`, `applyCfg`, denoising-loop helpers shared across 5 families.

### 6.2 Phase 11 — Unified product surface

**Goal:** `mlxts generate {image|video|audio}` exists, AXI-shaped, AI SDK-compatible API surface.

1. **Design doc** for `mlxts` unified CLI (subcommands as §5.4). One artifact. User-approved before implementation.
2. **Extract `packages/diffusion/src/cli/`** — shared flag readers, TOON output, error formatters. 7 example CLIs become thin wrappers.
3. **`packages/sdk/`** — facade re-exporting `generateText`, `generateImage`, `generateVideo`, `generateAudio`, `generateObject`, `streamText`. Mirrors AI SDK shape.
4. **`@mlxts/serve` widens to include `DiffusionEngine`.** New `/v1/images/generations` route (OpenAI Images-spec compatible). Custom `/v1/videos/generations` and `/v1/audio/generations` (mlxts-spec, documented as such).
5. **`mlxts` binary consolidates** all subcommands. `mlxts-serve`/`mlxts-agent` remain as deprecated aliases.
6. **AXI session hooks land** for `mlxts` and (deprecated) `mlxts-serve`/`mlxts-agent`. Self-install into Claude Code/Codex on first run.

### 6.3 Phase 12 — Training stack completeness

See §5.6 for the checklist. Sequenced after Phase 11 to avoid splitting attention.

1. PPO/GRPO in `@mlxts/align`
2. ORPO/KTO in `@mlxts/align`
3. QLoRA in `@mlxts/lora`
4. `@mlxts/evals` package
5. Reward modeling primitive

### 6.4 Continuous

**Audit cadence:** every 3–5 days while Phase 10b–12 in active build-out. Each audit anchors on the prior doc, reports drift/resolution/new pressure. After Phase 12 closes, slow to weekly.

**Doctrine evolution:** the audit-as-practice format itself is an artifact. This doc is the second iteration; the format should converge to a stable shape (sections 1–11) and any change to the format should be noted in the next doc's §1.

---

## 7. The 500-Line Cap Question

The user explicitly asked for this question to be evaluated against data, not opinion.

### 7.1 Data

47 prod files in the 450–499 band. Two distinct populations:

**Population 1 — single logical unit, inherently dense (~13 files).**
Files where the code is one tightly-coupled invariant cluster. Splitting fragments one concept across two files. Examples:
- `serve/engine/continuous.ts` (498) — continuous-batch scheduler state machine
- `serve/engine/content.ts` (498) — media dispatch + token-sequence prompt construction
- `serve/engine/prefix-cache.ts` (497) — prefix cache invariant cluster
- `transformers/infrastructure/cache/tensor-block-snapshot.ts` (499) — cache snapshot logic
- `transformers/families/qwen3_5/cache/batch-cache.ts` (499) — batch hybrid cache
- `core/array.ts` (498) — the MxArray class itself
- `core/ffi/symbols.ts` (498) — FFI symbol declarations
- `nn/module.ts` (498) — Module base class
- `diffusion/families/ltx/autoencoder-ltx2-blocks.ts` (491) — single architecture, single file

For these, splitting would create artificial seams the reviewer can't name cleanly.

**Population 2 — structural pressure pretending to be density (~27 files).**
Files near cap because they secretly contain multiple concerns. Examples:
- `diffusion/families/stable-diffusion/pipeline.ts` (498) — 30+ symbols across 5 concerns
- `diffusion/families/flux/config.ts` (477) — duplicates flux2 utilities
- `transformers/lora/adapters.ts` (497) — mixes I/O with apply
- `core/fast.ts` (474) — generic fused ops + Qwen-specific helpers
- `nn/module.ts` (498) — base class + helpers + validation (could split)

For these, splitting is a genuine improvement.

### 7.2 The recommendation: structured cap with explicit exception

```
Default cap: 500 lines
Hard ceiling: 700 lines
Exception band (500–700): allowed when file represents a single logical unit,
                          marked with a header comment naming the unit
```

Implementation:

```typescript
// @file-cap: 600 — continuous-batch scheduler state machine
```

`scripts/check-file-lines.ts` parses the marker. Without marker: cap 500. With marker: cap = marker value, max 700. The marker is grep-able, code-reviewable, and a contract the author commits to.

This:
- Keeps Population 1 working (continuous.ts adds marker, breathes to 600)
- Keeps Population 2 honest (pipeline.ts stays gated at 500, must split)
- Makes the exception explicit and discoverable
- Preserves the forcing function on the 27 files of structural pressure
- Doesn't require human judgment on every check

### 7.3 Alternative: simpler global raise

If the marker mechanism feels heavy, the simpler answer is "raise hard cap to 600 globally, document 500 as aspirational." Cost: ~5 of Population 2's files use the headroom to grow worse before notice. Benefit: zero new mechanism.

**Recommended: marker-based exception, hard cap at 700.** Population 2's debt is large enough today (27 files) that the forcing function is worth keeping.

### 7.4 Doctrine update

`docs/code-standards.md` and `AGENTS.md` references to "500-line cap" should become "500-line guidance / 700-line hard ceiling, with named-unit exception via `@file-cap:` header marker." Code review for any file with a marker should ask: *can the reviewer name the single logical unit the marker claims, in one sentence?* If not, the marker is not justified.

---

## 8. Audit-as-Practice

The audit format itself is becoming a stable artifact. This doc is iteration 2. Iteration 3 should anchor on this one and report drift over the next 3–5 days.

### 8.1 Stable structure

Every audit doc has:
1. Executive Read (2–3 paragraphs, severity-marked top concerns)
2. Mechanical Snapshot (one-paragraph summary; full numbers in companion `*-audit-metrics.md`)
3. Repo Posture (governance, gates, sub-agent workflow)
4. Architectural Posture (per-slice findings, drift since prior audit)
5. Product Surface Coherence (the central section)
6. Direction (proposed phase plan, not committed work)
7. Cap question / live design questions
8. Audit-as-Practice (this section — process meta)
9. Specific Recommendations (severity-ordered, integrated)
10. Files Reviewed

### 8.2 Cadence

Every 3–5 days during active phase work. Each iteration:
- Anchors on prior doc by date
- Reports per-finding: resolved / persisting / new
- Highlights what was *not addressed* from prior audit (drift signal)
- Updates the "stable structure" only deliberately

### 8.3 Sub-agent workflow

Six parallel slice agents. One metrics baseline pre-pass. One synthesis (lead agent). The slices are stable:
- A: core / nn / optimizers / quantize
- B: train / data / tokenizers / lora / align
- C: transformers
- D: diffusion
- E: serve / agent / protocols
- F: examples / docs / governance / AXI cross-cutting

When new packages emerge (e.g., `@mlxts/sdk`, `@mlxts/evals`), slice ownership decisions live in this section of the audit doc, not in chat.

### 8.4 What the audit *is not*

- Not a code review for individual changes
- Not a remediation PR (read-only diagnosis)
- Not a feature roadmap (direction proposals, not commitments)
- Not a substitute for `docs/reviews/<feature>.md` per-tranche reviews

It is a posture audit. State of the repo at a moment in time. Drift detector. Compounding mechanism for cleanup.

---

## 9. Specific Recommendations (Severity-Ordered)

### 🔴 P1 — Address before next family lands

**R1. Split `families/ltx/`** into `ltx-video/`, `ltx2/`, `ltx-shared/`. Path: `packages/diffusion/src/families/ltx/`. Effort: ~2 hours. **Trigger: do this before any LTX-2.3 work begins** — that tranche will compound the prefix-as-folder pressure.

**R2. Slim `continuity.md` to ≤200 lines.** Replace `## Latest Evidence` (lines 302–550) with a pointer table: `(date, review-doc-link, one-line)`. Add `check:continuity-shape` script to enforce the shape.

**R3. AXI session hooks for `mlxts-serve` and `mlxts-agent`.** Self-install into `~/.claude/settings.json` and `~/.codex/hooks.json` `SessionStart`. The compact dashboard shows configured models, model pool state, prompt cache health. Closes the largest AXI gap.

**R4. Real-checkpoint proofs for classic LTX-Video and LTX-2.** Operator runs with `Lightricks/LTX-Video` and `Lightricks/LTX-2`. Documented in `docs/reviews/`. **No diffusion serve route should expose either family until this lands.**

**R5. Document the diffusion serving design decision.** `serve/AGENTS.md` or `docs/serving-runtime-strategy.md`. The decision: `DiffusionEngine` interface lives in `@mlxts/serve`; serve depends on `@mlxts/diffusion`; new admission model for media generation. Lands *before* the first diffusion route implementation.

### 🟡 P2 — Address during Phase 10b

**R6. Refactor `families/flux/config.ts`** to import from `families/flux2/config-parsing.ts`; move that file to package-level `src/config-parsing.ts`. ~1 hour.

**R7. Split `families/stable-diffusion/pipeline.ts`** into `pipeline-core.ts` (denoising) + `pipeline.ts` (public surface). Proactive — currently 2 lines from cap.

**R8. Move `examples/ltx-video/audio-output.ts`** to `packages/diffusion/src/media/wav-encoder.ts`. Resolve package-vs-example boundary leak.

**R9. Create `packages/diffusion/src/sampling/`** — extract shared `createInitialLatents`, scheduler-sigma-scaling, CFG application helpers. Closes 5-family duplication.

**R10. Extract `packages/diffusion/src/cli/`** — shared flag readers, TOON output, error formatters for the 7 example CLIs. Eliminates ~4,000 LOC of duplication.

**R11. Implement marker-based file cap** (§7). Update `scripts/check-file-lines.ts`. Update `docs/code-standards.md` and `AGENTS.md` doctrine references.

**R12. Normalize `--allow-download` / `--local-files-only`** flag conventions across all diffusion CLIs. Standardize on `--allow-download` (opt-in, agent-safer default).

**R13. Add `check:per-example-agents`** gate. Scan `examples/` with same threshold as packages. 5 examples will need AGENTS.md.

**R14. Define `EncoderHiddenStates` interface** in `packages/transformers/src/types.ts`. `LlamaLikeCausalLM` implements both `CausalLM` and `EncoderHiddenStates`. Prevents silent contract widening before a third caller.

**R15. Extract `pretrained/progress.ts` default writeLine** to no-op default. Rename current default-stdout function to `createConsoleProgressReporter`. Closes AXI hazard (R from Slice C).

**R16. Add `exit_codes[3]{code,meaning}:` table** to `mlxts-serve` help output (`packages/serve/src/cli-usage.ts`).

**R17. Extract shared `toon()` helper** to `packages/serve/src/cli-formatting.ts`. Replace 4+ independent implementations.

**R18. Split `qwen3_5/cache/batch-cache.ts`** (499 LOC) along snapshot/restore vs batch-layer-ops boundary. Before next cache feature.

**R19. Split `transformers/lora/adapters.ts`** (497 LOC) into `adapters-io.ts` + `adapters-apply.ts`.

**R20. `core/src/fast.ts`**: split into `fast.ts` (generic fused ops) + `fast-qwen.ts` (Qwen-specific gated-delta). Doctrinal — `mx.fast` namespace shouldn't carry model-family vocabulary.

### 🟢 P3 — Hygiene / forward seams

**R21. Trim `MEMORY.md` Tier 2 entries** spanning >30 lines. Compress to 4–6 line summaries pointing to review artifacts.

**R22. Move `examples/serve-completions/`** to `packages/serve/tests/integration/` or mark as test-only with `AGENTS.md`.

**R23. Codify encoder-family doctrine** in `packages/transformers/AGENTS.md` — the third family category alongside lean and full.

**R24. Add `.reference/anthropic-typescript-sdk`** and `.reference/modelcontextprotocol/typescript-sdk` to `.reference/`. AI SDK is in. Anthropic SDK and MCP TypeScript SDK are the missing canonical references.

**R25. Add three skills:** `add-diffusion-family`, `phase10-proof-qa`, `memory-compaction`. The diffusion family pattern is now repeated 7 times; capture it.

**R26. Advisory warning** in `check-file-lines.ts` for files in 400–499 band. Surface pressure before violation.

**R27. Document `Add` vs `Prove` evidence stages** in `docs/gates-and-milestones.md`. Make the linguistic distinction machine-readable via a `families.json` evidence-stage manifest in `@mlxts/diffusion`.

**R28. Stale `mlx-vlm` column header** in `docs/python-equivalence-map.md` line 51.

### 🔵 P4 — Phase 11+ direction (proposal, not commitment)

**R29. Phase 11 design doc**: unified `mlxts generate {image|video|audio}` CLI shape, `@mlxts/sdk` facade, serve diffusion routes. User-approved before implementation.

**R30. Phase 12 design doc**: training stack completeness — PPO/GRPO/ORPO/KTO, QLoRA/DoRA, `@mlxts/evals` harness.

---

## 10. Open Questions for the User

These are decisions the audit cannot make.

1. **Cap mechanism**: marker-based (§7.2) or simple global raise to 600 (§7.3)?
2. **`mlxts` binary consolidation**: replace `mlxts-serve`/`mlxts-agent` immediately or keep them as long-term aliases?
3. **`@mlxts/sdk` package**: facade-only re-export, or carry its own minimal types? (Recommended: facade-only.)
4. **OpenAI Images-spec at `/v1/images/generations`**: exact wire-compatible (so AI SDK clients work unmodified) or `mlxts`-spec extension? (Recommended: exact for image, custom for video/audio.)
5. **Phase ordering**: Phase 10b → Phase 11 → Phase 12, or interleave Phase 12 (training completeness) earlier?
6. **`@mlxts/evals` package**: separate package or part of `@mlxts/align`? (Recommended: separate, named for the role.)
7. **Audit cadence formalization**: schedule via `/loop` or remain user-triggered?

---

## 11. Files Reviewed

This audit synthesized inputs from six parallel sub-agent slices. Each slice's review listed ≤30 most-informative files; the full union exceeds the practical limit for this section. Canonical inputs:

**Audit framework anchors:**
- `docs/audits/2026-04-28-architectural-posture-audit.md` (predecessor)
- `docs/audits/2026-04-28-audit-metrics.md` (predecessor metrics)
- `docs/audits/2026-05-02-audit-metrics.md` (this audit's metrics; companion doc)
- `AGENTS.md`, `CLAUDE.md`, `MEMORY.md`, `continuity.md`, `PLAN.md`
- `.agents/skills/axi/SKILL.md`
- `docs/design-reasoning.md`, `docs/code-standards.md`, `docs/runtime-safety.md`, `docs/runtime-optimization-matrix.md`, `docs/serving-runtime-strategy.md`

**Reference inputs (newly added):**
- `.reference/ai-sdk/packages/ai/src/{generate-text,generate-image,generate-object}/index.ts` (AI SDK shape)
- `.reference/ai-sdk/packages/` directory listing (provider-package taxonomy)

**Per-slice review inputs:** see slice-A through slice-F output sections (synthesized; ~180 unique file paths across the six slices).

---

**End of audit.** Next iteration anchors on this doc. Next-iteration date: when phase 10b first 🔴 lands or +5 days, whichever sooner.
