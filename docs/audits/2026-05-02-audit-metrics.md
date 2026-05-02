# Audit Metrics - 2026-05-02 Fresh Pass

Companion to `docs/audits/2026-05-02-architectural-posture-audit.md`.

This fresh pass supersedes the archived draft metrics at
`docs/audits/archive/2026-05-02-audit-metrics-superseded.md`. It reuses the
same mechanical baseline where the numbers are still valid and adds product
surface checks that the first May 2 draft underweighted.

## 1. Gate Baseline

This fresh pass reran the lightweight gates after the same-day audit was
superseded. Coverage was not rerun because this pass updates audit
documentation. A targeted native smoke and the Phase 10 proof gate were then
run after synthesis because the audit identified proof-path risk; both exposed
a current native runtime blocker.

| Gate | Result | Key numbers |
| --- | --- | --- |
| `bun run typecheck` | PASS | 15/15 workspace typechecks including `examples/nanogpt` |
| `bun run check:file-lines` | PASS | 542 prod files, all <=500 |
| `bun run check:assertions` | PASS | no `as` / `!` outside FFI |
| `bun run check:tensor-lifetimes` | PASS | no suspicious nested tensor calls |
| `bun run check:runtime-review` | PASS | no runtime-sensitive prod changes pending |
| `bun run check:per-package-agents` | PASS | 13 non-trivial packages checked; all have `AGENTS.md` |
| `bun run check:cross-package-imports` | PASS | 474 package dependency edges, no stack inversions |
| `bun run check:skills` | PASS | 2 repo-local skills |
| `bun run lint` | PASS | 1,076 files, 0 issues |
| `bun test packages/core/src/ffi.test.ts` | FAIL | dylib load and empty array pass; scalar/data arrays and GPU stream creation fail |
| `bun run check:phase10-proofs` | FAIL | 82 pass, 58 fail; failures collapse to null MLX array/stream pointers |

All 14 package directories have `AGENTS.md`; the gate enforces the 13 packages
above its non-triviality threshold. The heavy full validation suite and
coverage gate were not rerun because this pass updates audit documentation.

Native failure signatures observed:

- `mlx_array_new_float(42.0)` returned null in `packages/core/src/ffi.test.ts`.
- `mlx_array_new_data(...)` returned null in `packages/core/src/ffi.test.ts`
  and across Phase 10 image/audio proof tests.
- `mlx_default_gpu_stream_new` returned null in FFI and media output shape
  tests.

This is a P0 blocker for any new runtime/proof claim until isolated.

## 2. Package Weight

From the May 2 mechanical baseline:

| Package | Src files | Src LOC | Test files | Test LOC |
| --- | ---: | ---: | ---: | ---: |
| core | 41 | 7,526 | 22 | 5,066 |
| nn | 23 | 3,181 | 21 | 3,040 |
| optimizers | 4 | 515 | 3 | 506 |
| train | 23 | 3,512 | 8 | 2,414 |
| data | 9 | 690 | 6 | 510 |
| tokenizers | 17 | 3,355 | 7 | 1,630 |
| transformers | 151 | 26,744 | 67 | 16,082 |
| lora | 6 | 592 | 2 | 335 |
| align | 10 | 1,188 | 5 | 775 |
| quantize | 8 | 780 | 4 | 579 |
| protocols | 1 | 261 | 1 | 79 |
| serve | 96 | 21,679 | 38 | 19,318 |
| agent | 12 | 2,009 | 6 | 1,639 |
| diffusion | 141 | 37,627 | 79 | 20,242 |

Total production LOC: 109,459. The important delta since April 28 is that
`@mlxts/diffusion` is now the largest package by LOC.

## 3. Examples

From the May 2 baseline:

| Example | Src files | Src LOC |
| --- | ---: | ---: |
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
| chat-canary | 2 | 218 |

Product-surface signal: `examples/image-proof` and `examples/ltx-video` contain
reusable artifact/report behavior that other examples now depend on.

## 4. CLI Surface Inventory

Published binaries:

| Package | Binary | Source |
| --- | --- | --- |
| `@mlxts/serve` | `mlxts-serve` | `packages/serve/src/cli.ts` |
| `@mlxts/agent` | `mlxts-agent` | `packages/agent/src/cli.ts` |

No top-level `mlxts` binary exists. There is no `mlxts generate`, `mlxts
serve`, `mlxts train`, or `mlxts model` umbrella CLI.

Shebang entrypoints recorded by the archived metrics:

- 3 in packages: `agent/cli.ts`, `serve/cli.ts`,
  `train/supervised-run/supervisor.ts`
- 23 in examples: proof commands, workbooks, verifier commands, nanoGPT
  manager/bench/acceptance flows

## 5. Cross-Example Coupling

The April 28 coupling through `examples/train-proof` is resolved.

Fresh grep shows the same structural pattern has migrated to
`examples/image-proof`:

- `examples/flux/image-output.ts`
- `examples/flux/index.test.ts`
- `examples/flux2/image-output.ts`
- `examples/flux2/index.test.ts`
- `examples/qwen-image/image-output.ts`
- `examples/qwen-image/index.test.ts`
- `examples/stable-diffusion/image-output.ts`
- `examples/stable-diffusion/index.test.ts`
- `examples/stable-diffusion-3/image-output.ts`
- `examples/stable-diffusion-3/index.test.ts`
- `examples/z-image/image-output.ts`
- `examples/z-image/index.test.ts`
- `examples/ltx-video/video-output.ts`
- `examples/ltx-video/verify-report.ts`
- `examples/ltx-video/index.test.ts`
- `examples/ltx-video/verify-report.test.ts`

The archived metrics correctly noticed this but marked it as a low-severity
boundary question. This fresh audit treats it as P1 because the behavior is no
longer example-specific: BMP writing, artifact hashing, report verification,
and media proof schemas are product infrastructure.

## 6. AXI Skill Verification

Local file:

- `.agents/skills/axi/SKILL.md`

Verified against upstream:

- Repository: `https://github.com/kunchenguid/axi`
- Commit: `a7193a0d4143696a969e999bbc1e9463f848005d`
- SHA-256: `a196c3b3eca191e87147a1b619f40361df7230bd20b1afc25f22e0065f42a23e`

The local skill now matches upstream exactly. The previous local file was an
adapted version.

## 7. LTX Proof Status

Evidence reviewed:

- `examples/ltx-video/README.md`
- `docs/reviews/2026-05-01-ltx-video-proof-cli.md`
- `docs/reviews/2026-05-02-ltx2-proof-assembly.md`
- `docs/reviews/2026-05-02-ltx-video-proof-verifier.md`
- Recent LTX commits from `git log -- packages/diffusion/src/families/ltx examples/ltx-video docs/reviews`

Finding:

- Classic LTX-Video has a finite proof command and verifier path, but the
  review doc explicitly says the official checkpoint proof still needs a local
  authenticated/cached snapshot run.
- LTX-2 has proof assembly and BMP/WAV verifier support, but the review doc
  explicitly says real-checkpoint execution still needs an operator run with a
  local LTX-2 snapshot and runtime lock.

The gap is real, but the precise wording should be "no documented
real-checkpoint proof yet," not "unsupported."

## 8. AGENTS Coverage

Packages: 14/14 have `AGENTS.md`.

Examples: 11/16 have `AGENTS.md`.

Missing example charters:

- `examples/chat`
- `examples/chat-canary`
- `examples/lora-finetune`
- `examples/serve-completions`
- `examples/train-proof`

This remains useful, but it is secondary to the product-ownership issue.

## 9. Continuity Size

`continuity.md` is now 105 lines.

The superseded same-day audit found it had regrown to 593 lines with large
evidence blocks better stored as review links. The follow-up documentation pass
slimmed it back to a compact active-state record. The remaining governance risk
is keeping that shape durable, ideally with a lightweight check.

## 10. Product Gaps Confirmed By Search

Searches found no implemented surfaces for:

- `@mlxts/sdk`
- `@mlxts/generate`
- top-level `mlxts generate`
- `DiffusionEngine`
- `/v1/images/generations`
- `/v1/videos/generations`
- `/v1/audio/generations`

`packages/diffusion/src/families/stable-diffusion/pipeline-loading.ts` exposes
a family-specific `generateImage()` bundle method, but there is no product
layer that owns image/video/audio generation across families.
