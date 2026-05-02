# Audit Phase 0 — Mechanical Snapshot
Date: 2026-04-28

Companion to `docs/audits/2026-04-28-architectural-posture-audit.md`.
Captures the mechanical baseline used during the audit so future readers can
verify the numbers. Generated with the existing repo gates plus shell tooling
(`find`, `wc -l`, `grep`).

## Gate Status (all green)
- typecheck: 14/14 packages clean (root + 13 packages + nanogpt example)
- biome lint: 610 files, no issues
- check:file-lines: 300 prod files, all ≤500
- check:assertions: clean (no `as` / `!` outside FFI)
- check:tensor-lifetimes: clean
- check:runtime-review: no runtime-sensitive prod changes

## Per-Package LOC and Test Density
| package | src files | test files | src LOC | test LOC | test/src ratio |
|---|---:|---:|---:|---:|---:|
| core | 37 | 22 | 6,894 | 4,744 | 0.59 |
| nn | 19 | 17 | 2,497 | 2,663 | 0.89 |
| optimizers | 4 | 3 | 515 | 506 | 0.75 |
| train | 10 | 7 | 1,404 | 1,409 | 0.70 |
| data | 8 | 5 | 586 | 473 | 0.62 |
| tokenizers | 14 | 6 | 2,385 | 1,147 | 0.43 |
| transformers | 112 | 49 | 20,994 | 12,662 | 0.44 |
| lora | 5 | 2 | 572 | 320 | 0.40 |
| align | 9 | 5 | 1,151 | 757 | 0.56 |
| quantize | 8 | 4 | 780 | 579 | 0.50 |
| protocols | 1 | 1 | 261 | 79 | 1.00 |
| serve | 64 | 24 | 15,110 | 11,400 | 0.38 |
| agent | 9 | 5 | 1,776 | 1,511 | 0.56 |

## Examples
| example | src files | src LOC |
|---|---:|---:|
| nanogpt | 62 | 6,654 |
| train-proof | 9 | 1,592 |
| lora-finetune | 6 | 705 |
| qwen3_5-image | 2 | 474 |
| chat | 1 | 351 |
| serve-completions | 1 | 209 |
| chat-canary | 0 | 0 (data only) |

## Cross-Package Dependency Edges (declared in package.json)
- core → (none, leaf)
- protocols → (none, leaf)
- tokenizers → (none, leaf)
- nn → core
- optimizers → core, nn
- data → core
- lora → core, nn
- quantize → core, nn
- train → core, nn, optimizers
- transformers → core, nn, lora, quantize, tokenizers + @huggingface/hub, @huggingface/jinja
- agent → protocols
- align → core, data, lora, nn, tokenizers, train, transformers (7 internal deps — most of any package)
- serve → core, protocols, tokenizers, transformers (notably no @mlxts/nn)

## Folder-Hygiene Findings
1. `packages/serve/src/` is shallow: 48 non-test source files at the top level plus a single `protocols/` subdirectory (64 prod files total). Naming-prefix groups exist: `server-*` (20), `transformers-engine-*` (13), `serve-*` (~7), `model-*` (8) — folders are doing the job manually via prefix.
2. `packages/transformers/src/` has empty legacy directories: `gemma/`, `llama/`, `mistral/`, `phi/`, `base/` — 0 files each. Vestigial residue from Phase 7 family extraction.
3. `packages/transformers/src/families/` shapes inconsistent: `gemma/`, `llama/`, `mistral/`, `mistral3/`, `phi/` only have config + weights (share llama-like backbone — fine). `gemma3/`, `gemma4/`, `qwen3_5/` have full file sets. qwen3_5/ has 35+ files at one level (no internal subfolders).
4. `transformers/src/lora-adapters.ts` (497 LOC) + `lora-module-traversal.ts` + `lora-targets.ts` at top level — not a duplicate of `@mlxts/lora` (it adds CausalLM-specific PEFT-format I/O on top), but could live in `transformers/src/lora/`.
5. `core/src/` has 47 top-level files. Patterns: array-* (4), io-* (4), transforms-* (3), fast-* (4), runtime-* (2). Subfolders only for `ffi/` and `ops/`.

## Top Files Near 500-Line Cap
serve:
- 486 server.ts
- 485 serve-metrics.ts
- 484 protocols/openai-completions.ts
- 480 protocols/openai-responses.ts
- 477 media-image.ts
- 470 protocols/anthropic-messages.ts
- 464 types.ts
- 464 server-streaming.ts
- 462 protocols/openai-chat-completions.ts
- 459 transformers-engine-continuous.ts

transformers:
- 497 lora-adapters.ts
- 497 families/qwen3_5/weights.ts
- 489 infrastructure/cache/batch.ts
- 461 quantize.ts
- 459 families/qwen3_5/vision.ts
- 456 families/qwen3_5/batch-cache.ts
- 453 infrastructure/generation/continuous-batch.ts
- 453 families/qwen3_5/config.ts
- 450 load.ts

The 500-line cap is doing its job, but pressure is high — many files within ~40 lines of the cap. Cap is concealing structure pressure rather than indicating naturally tight design.

## Per-Package AGENTS.md Status (at audit time)
- HAVE: serve, transformers (2/13)
- MISSING: core, nn, optimizers, train, data, tokenizers, lora, align, quantize, protocols, agent (11/13)
- ALL 13 packages have README.md
- ALL examples except `nanogpt` have README.md; `chat-canary` has a README despite being data-only

## Top Files by Export Density
- 32: transformers/scripts/benchmark-common.ts (script)
- 31: transformers/src/types.ts (canonical contracts hub)
- 29: serve/src/model-server-options.ts
- 27: core/src/ops/shape.ts
- 27: core/src/index.ts (barrel)
- 23: core/src/ffi/symbols.ts
- 21: core/src/ops/arithmetic.ts
- 20: serve/src/transformers-engine-shared.ts
- 18: serve/src/serve-runtime-strategy.ts

## Deep Relative Imports (../../../)
16 total occurrences:
- 6 are scripts importing `runtime-command-lock` (acceptable)
- 3 are `transformers/src/families/gemma4/runtime/{model,attention,mlp}.ts` → `../../../infrastructure/{masks,gated-activations}` — 3-deep cross-folder reach inside the package
- 4 are dist/ output (ignore)
- 3 are example bench scripts importing root scripts (acceptable)
