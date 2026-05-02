# Audit Phase 0 — Mechanical Snapshot — 2026-05-02

Companion to `docs/audits/2026-05-02-architectural-posture-audit.md`. Captures the mechanical baseline so future readers can verify the numbers. Generated with repo gates plus `find` / `wc -l` / `grep`.

**Predecessor**: `docs/audits/2026-04-28-audit-metrics.md`. Deltas reported throughout.

---

## 1. Gate Status

| Gate | Result | Key numbers |
|---|---|---|
| `bun run typecheck` | PASS | 15/15 packages clean (was 14/14) |
| `bun run check:file-lines` | PASS | 542 prod files, all ≤500 (was 300) |
| `bun run check:assertions` | PASS | no `as`/`!` outside FFI |
| `bun run check:tensor-lifetimes` | PASS | no suspicious nested calls |
| `bun run check:runtime-review` | PASS | no runtime-sensitive prod changes pending |
| `bun run check:per-package-agents` | PASS | 14/14 packages (was 2/13 at 2026-04-28 open) |
| `bun run check:coverage` | EXISTS | not run in audit (heavy gate) |
| Biome lint | PASS | 1,076 files, 0 issues, 229ms (was 610 files) |

**Delta:** +1 package (diffusion typecheck-registered), +242 prod files, biome scope +76% (file count).

---

## 2. Package Weight

| Package | Src files | Test files | Src LOC | Test LOC | Test/src ratio |
|---|---:|---:|---:|---:|---:|
| core | 41 | 22 | 7,526 | 5,066 | 0.67 |
| nn | 23 | 21 | 3,181 | 3,040 | 0.96 |
| optimizers | 4 | 3 | 515 | 506 | 0.98 |
| train | 23 | 8 | 3,512 | 2,414 | 0.69 |
| data | 9 | 6 | 690 | 510 | 0.74 |
| tokenizers | 17 | 7 | 3,355 | 1,630 | 0.49 |
| transformers | 151 | 67 | 26,744 | 16,082 | 0.60 |
| lora | 6 | 2 | 592 | 335 | 0.57 |
| align | 10 | 5 | 1,188 | 775 | 0.65 |
| quantize | 8 | 4 | 780 | 579 | 0.74 |
| protocols | 1 | 1 | 261 | 79 | 0.30 |
| serve | 96 | 38 | 21,679 | 19,318 | 0.89 |
| agent | 12 | 6 | 2,009 | 1,639 | 0.82 |
| diffusion | 141 | 79 | 37,627 | 20,242 | 0.54 |

**Totals:** 542 src, 269 test, 109,459 src LOC, 71,215 test LOC.

**Notable deltas (4 days):**
- transformers: 21k → 26.7k LOC (+27%); 112 → 151 files (+35%)
- serve: 15k → 21.7k LOC (+43%); 64 → 96 files (+50%)
- diffusion: 0 → 37.6k LOC (net-new); 0 → 141 files

---

## 3. Examples

| Example | Src files | Src LOC |
|---|---:|---:|
| nanogpt | 62 | 8,809 |
| train-proof | 18 | 3,969 |
| ltx-video | 18 | 4,134 |
| stable-diffusion-3 | 9 | 2,148 |
| lora-finetune | 12 | 1,963 |
| stable-diffusion | 9 | 1,882 |
| image-proof | 5 | 1,640 |
| flux2 | 9 | 1,580 |
| flux | 9 | 1,567 |
| qwen-image | 9 | 1,439 |
| z-image | 9 | 1,318 |
| qwen3_5-image | 4 | 910 |
| whisper | 4 | 859 |
| chat | 1 | 281 |
| serve-completions | 1 | 209 |
| chat-canary | 2 | 218 (data+fixture) |

**New since 2026-04-28:** flux, flux2, image-proof, ltx-video, qwen-image, stable-diffusion, stable-diffusion-3, whisper, z-image (9 new examples).

---

## 4. Cross-Package Import Edges

Counts are line-occurrences of `from "@mlxts/<dep>"` in each package's `src/`.

| importer ↓ / dep → | core | nn | tokenizers | lora | quantize | transformers | protocols | optimizers | data | train |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transformers | 150 | 50 | 13 | 4 | 2 | — | — | — | — | — |
| serve | 9 | — | 10 | — | — | 46 | 1 | — | — | — |
| nn | 66 | — | — | — | — | — | — | 1 | — | — |
| optimizers | 9 | 5 | — | — | — | — | — | — | — | — |
| train | 12 | 4 | — | — | — | — | — | 6 | — | — |
| lora | 6 | 7 | — | — | — | — | — | — | — | — |
| quantize | 6 | 6 | — | — | — | — | — | — | — | — |
| data | 4 | — | — | — | — | — | — | — | — | — |
| align | 6 | 3 | 4 | — | — | 13 | — | 3 | 11 | 2 |
| agent | — | — | — | — | — | — | 1 | — | — | — |
| diffusion | 222 | 52 | — | — | — | — | — | — | — | — |

**Notable:** `diffusion → core` (222) is now the heaviest single edge in the repo, surpassing `transformers → core` (150). `align` remains the most fan-out importer (7 distinct deps).

---

## 5. File-Cap Pressure

**Total prod files:** 542 (was 300).

**Top 25 prod files closest to 500-line cap:**

| LOC | Path |
|---:|---|
| 499 | transformers/infrastructure/cache/tensor-block-snapshot.ts |
| 499 | transformers/families/qwen3_5/cache/batch-cache.ts |
| 498 | serve/engine/continuous.ts |
| 498 | serve/engine/content.ts |
| 498 | nn/module.ts |
| 498 | diffusion/families/stable-diffusion/pipeline.ts |
| 498 | core/ffi/symbols.ts |
| 498 | core/array.ts |
| 497 | transformers/lora/adapters.ts |
| 497 | transformers/families/qwen3_5/weights.ts |
| 497 | serve/engine/prefix-cache.ts |
| 496 | serve/types.ts |
| 495 | serve/observability/metrics.ts |
| 492 | serve/protocols/openai-responses.ts |
| 491 | transformers/infrastructure/generation/continuous-batch.ts |
| 491 | diffusion/families/ltx/autoencoder-ltx2-blocks.ts |
| 489 | serve/model-loading/sources.ts |
| 484 | serve/protocols/openai-completions.ts |
| 483 | serve/media/image.ts |
| 483 | diffusion/families/flux2/pipeline.ts |
| 481 | core/ops/shape.ts |
| 480 | transformers/families/qwen3_5/cache/index.ts |
| 480 | diffusion/families/ltx/latent-upsampler-ltx2.ts |
| 479 | serve/http/server.ts |
| 478 | serve/protocols/anthropic-messages-input.ts |

**Per-package files >450 lines:**

| Package | Count |
|---|---:|
| serve | 19 |
| diffusion | 13 |
| transformers | 9 |
| core | 5 |
| nn | 1 |
| tokenizers | 1 |

**Total in 450–499 band:** 47 (was ~30 at 2026-04-28).

---

## 6. Per-Package AGENTS.md

| Package | AGENTS.md | Status |
|---|---|---|
| core | YES | Added since 04-28 |
| nn | YES | Added since 04-28 |
| optimizers | YES | Added since 04-28 |
| train | YES | Added since 04-28 |
| data | YES | Added since 04-28 |
| tokenizers | YES | Added since 04-28 |
| transformers | YES | Pre-existing |
| lora | YES | Added since 04-28 |
| align | YES | Added since 04-28 |
| quantize | YES | Added since 04-28 |
| protocols | YES | Added since 04-28 |
| serve | YES | Pre-existing (substantially extended) |
| agent | YES | Added since 04-28 |
| diffusion | YES | Added since 04-28 |

**Coverage: 14/14 (100%).** Was 2/13 at 2026-04-28 open. Enforced by `check:per-package-agents`.

**Per-example AGENTS.md:** 11/16 (69%). Missing: `chat`, `chat-canary`, `lora-finetune`, `serve-completions`, `train-proof`. Gate does not scan `examples/`.

---

## 7. `@mlxts/diffusion` Deep Dive

**Subdirectory tree:**

```
diffusion/src/
├── families/
│   ├── flux/                    11 src, 6 test, 2,782 LOC
│   ├── flux2/                   13 src, 7 test, 3,520 LOC
│   ├── ltx/                     53 src, 26 test, 14,803 LOC  ← dual-variant hosting
│   ├── qwen-image/              15 src, 9 test, 3,857 LOC
│   ├── stable-diffusion/        12 src, 6 test, 3,693 LOC
│   ├── stable-diffusion-3/      12 src, 6 test, 2,761 LOC
│   └── z-image/                 10 src, 6 test, 2,380 LOC
├── pretrained/                  8 src, 4 test, ~1,500 LOC
└── schedulers/                  4 src, 4 test, ~500 LOC
```

**LTX dual-variant prefix split:**
- Classic LTX (no `ltx2` in name): 21 src files, 6,015 LOC
- LTX-2 (`*ltx2*` prefix): 32 src files, 8,788 LOC
- Shared (`config-common.ts`, `embeddings-rope.ts`): ~350 LOC (~2.4% of folder)

**Top 10 longest diffusion files:**

| LOC | Path |
|---:|---|
| 498 | families/stable-diffusion/pipeline.ts |
| 491 | families/ltx/autoencoder-ltx2-blocks.ts |
| 483 | families/flux2/pipeline.ts |
| 480 | families/ltx/latent-upsampler-ltx2.ts |
| 477 | families/flux/config.ts |
| 474 | families/stable-diffusion/weights.ts |
| 468 | pretrained/pipeline-specs.ts |
| 468 | families/ltx/pipeline.ts |
| 465 | families/ltx/config.ts |
| 464 | pretrained/snapshot-source.ts |

---

## 8. `@mlxts/serve` Follow-Up

**2026-04-28 finding:** 48 top-level files, prefix groups doing folder work manually.

**Current state:** **11 top-level files** (10 prod + 1 test). Prefix-as-folder problem **fully resolved**.

**Subfolders with prod file counts:**

| Subfolder | Prod files |
|---|---:|
| protocols/ | 20 |
| model-loading/ | 12 |
| streaming/ | 10 |
| engine/ | 15 (was 13 transformers-engine-* prefix) |
| observability/ | 9 |
| http/ | 8 |
| admission/ | 5 |
| media/ | 4 |
| runtime/ | 3 |

Zero `server-*`, `transformers-engine-*`, `serve-*`, or `model-*` files at the top level.

---

## 9. `@mlxts/transformers` Follow-Up

**2026-04-28 findings:** 5 empty Phase-7 dirs, qwen3_5/ flat 33 files.

**Empty dirs:** ALL GONE (`base/`, `gemma/`, `llama/`, `mistral/`, `phi/` deleted at top level).

**qwen3_5/ decomposition:** 33 flat files → 17 root + `cache/` (3) + `linear-attention/` (5) + `multimodal/` (9). Subfolder structure matches 2026-04-28 proposal.

**Top 10 longest transformers files:**

| LOC | Path |
|---:|---|
| 499 | infrastructure/cache/tensor-block-snapshot.ts |
| 499 | families/qwen3_5/cache/batch-cache.ts |
| 497 | lora/adapters.ts |
| 497 | families/qwen3_5/weights.ts |
| 491 | infrastructure/generation/continuous-batch.ts |
| 480 | families/qwen3_5/cache/index.ts |
| 461 | quantize.ts |
| 461 | load.ts |
| 459 | families/qwen3_5/multimodal/vision.ts |
| 448 | families/qwen3_5/config.ts |

---

## 10. New Review Docs Since 2026-04-28

| Date | Count |
|---|---:|
| 2026-04-29 | 9 |
| 2026-04-30 | 61 |
| 2026-05-01 | 60 |
| 2026-05-02 | 5 (in progress) |

**Total new in 4 days: 135 docs.**

Dominant themes:
- 2026-04-29: prefix cache block architecture
- 2026-04-30: image generation stack (flux, stable-diffusion), agent cache, serve CLI hardening, diffusion foundation
- 2026-05-01: full diffusion family coverage (ltx2, flux2-klein, qwen-image, z-image, sd3, whisper), CLIP/T5/Qwen2-VL/Qwen3 encoders
- 2026-05-02: LTX media proof verifier, LTX-2 audio-video proof assembly, Qwen-Image-Edit (in flight)

---

## 11. CLI Surface Inventory

**Published `bin` entrypoints:**

| Package | Binary | Source |
|---|---|---|
| agent | `mlxts-agent` | packages/agent/src/cli.ts |
| serve | `mlxts-serve` | packages/serve/src/cli.ts |

**Shebang entrypoints (`#!/usr/bin/env bun`) in src:**

- 3 in `packages/`: agent/cli.ts, serve/cli.ts, train/supervised-run/supervisor.ts
- 23 in `examples/`: nanogpt (5 entries), train-proof (3), ltx-video (2), lora-finetune (2), image-proof (1), and 10 single-entry families/utilities

**Total:** 2 published binaries, 26 total CLI entrypoints across the repo.

---

## 12. Cross-Example Coupling

**2026-04-28 finding:** `lora-finetune/data.ts` and `chat-canary/dataset.test.ts` reaching into `train-proof/`.

**Current grep `from "../[a-z-]*-proof`:**

- 7 examples import from `../image-proof/`: flux, flux2, qwen-image, stable-diffusion, stable-diffusion-3, ltx-video, z-image
- 0 examples import from `../train-proof/` ✅ (resolved)

`image-proof/` has become an intentional shared utility example for artifact verification (BMP writer, SHA256 helpers, validity checks). Same structural pattern as the previous `train-proof` coupling but **deliberate and broader**. The audit synthesis flags this as a `🟢` boundary question for future review: should `image-proof/` become `packages/diffusion/src/media/` rather than remain as a shared example?

---

## 13. Drift Summary vs 2026-04-28

**Resolved (🔴 → 🟢):**
- serve flat structure (48 → 11 top-level files, 9 named subfolders)
- serve/AGENTS.md tactical-not-architectural (now 80 lines architectural)
- image preprocessing seam location (now in `serve/src/media/`)
- cross-family cache layer-type taxonomy (unified at `infrastructure/cache/layer-kind.ts`)
- 5 empty Phase-7 dirs in transformers (deleted)
- qwen3_5/ flat 33 files (decomposed into 4 subdirs)
- LoRA files at top of transformers (moved to `lora/` subfolder)
- cross-example coupling (`train-proof` reach-ins eliminated)
- CLAUDE.md doctrinal duplication (slimmed to 23 lines)
- per-package AGENTS.md coverage (2/13 → 14/14, gate enforced)

**Persisting (🟡):**
- `core/src/array.ts` runtime-profiling instrumentation in hot constructor
- `core/src/` prefix-grouping without subfolders
- `nn/src/module.ts` inline `moduleArrayState` helper
- compile flags exported from top-level `core` barrel
- `transformers/lora/module-traversal.ts` duplicates `@mlxts/lora/traversal.ts`
- stale `mlx-vlm` column header in `docs/python-equivalence-map.md`

**New pressure (🔴/🟡):**
- `diffusion/families/ltx/` prefix-as-folder pressure (53 files, dual-family)
- `continuity.md` regrew to 593 lines (target: ≤200)
- AXI Principle 7 (session hooks) absent repo-wide
- Real-checkpoint proofs missing for classic LTX-Video and LTX-2
- No diffusion serving design recorded
- 7 diffusion CLIs duplicate ~4,000 LOC of boilerplate
- `--allow-download` vs `--local-files-only` flag inversion
- `examples/ltx-video/audio-output.ts` (217 LOC) trapped in example
- `qwen3_5/cache/batch-cache.ts` cap pressure (subfolder split moved code without relieving density)
- file-cap density doubled (10 → 19 serve files within 50 of cap)
- 5/16 examples lack AGENTS.md (gate doesn't scan examples)
- `MEMORY.md` Tier 2 entries 60–70 lines long
