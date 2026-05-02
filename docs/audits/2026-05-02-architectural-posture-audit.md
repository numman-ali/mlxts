# Architectural Posture Audit - 2026-05-02 Fresh Pass

**Audit type**: repo-wide posture audit, replacing the same-day draft archived at
`docs/audits/archive/2026-05-02-architectural-posture-audit-superseded.md`.
**Predecessor**: `docs/audits/2026-04-28-architectural-posture-audit.md`.
**Companion metrics**: `docs/audits/2026-05-02-audit-metrics.md`.

This audit is read-only diagnosis. It proposes direction and remediation; it
does not itself implement product behavior.

## 1. Executive Read

The April 28 cleanup worked at the file-tree level, but the May 2 draft was too
quick to declare the red items "all resolved." Several literal paths were fixed,
while the same pressure migrated into new product-surface gaps. The honest read:
the repo is stronger, but the next risk is not another model family; it is
ownership of the media-generation product surface.

The strongest current shape is the package architecture. `serve/src/` is no
longer flat, per-package `AGENTS.md` coverage is effectively complete, Qwen
cache taxonomy is shared, and diffusion families have made very fast progress.
The weakest current shape is the user journey. Media generation is proven
through family workbooks and shared example utilities, not through a coherent
`mlxts` CLI, SDK facade, serving contract, or package-owned artifact/proof
infrastructure.

Important validation correction: this fresh pass also found a current native
runtime failure. `@mlxts/core` can load the dylib and create an empty array, but
non-empty array creation and default GPU stream creation return null pointers in
local tests. That makes the current `check:phase10-proofs` failure a P0 gate
blocker until the native runtime or packaging issue is isolated.

Top concerns:

- **P0 - Native runtime gate failure**: a simple `@mlxts/core` array smoke,
  `packages/core/src/ffi.test.ts`, and `bun run check:phase10-proofs` fail with
  `mlx_array_new_data returned a null pointer` or
  `mlx_default_gpu_stream_new returned a null pointer`.
- **P1 - Product ownership**: reusable media artifact/report behavior is trapped
  in `examples/image-proof` and `examples/ltx-video`, while seven media family
  CLIs duplicate flag parsing, output shaping, and proof/report behavior.
- **P1 - Diffusion serving decision**: there is no recorded decision for whether
  `@mlxts/serve` owns media-generation routes, a `DiffusionEngine`, and media
  admission semantics.
- **P1 - LTX structure and proof**: `packages/diffusion/src/families/ltx/`
  hosts classic LTX-Video and LTX-2 in one prefix-organized folder, and neither
  has documented real-checkpoint proof yet.
- **P2 - Narrative governance**: `continuity.md` had regrown into a large
  evidence archive before the follow-up cleanup, and `MEMORY.md` Tier 2 is
  drifting toward archival entries instead of searchable index notes.
- **P2 - Training completeness wording**: QLoRA is not absent; it exists in the
  proof path and `@mlxts/lora` exposes quantized-base preservation helpers. The
  real gaps are DoRA, PPO/GRPO/ORPO/KTO, reward modeling, distillation, and a
  dedicated `@mlxts/eval` harness.

## 2. Intent Check

Nomi's apparent intent is not "add more demos." It is a complete, end-to-end
TypeScript ML stack on Apple Silicon: train, fine-tune, serve, generate media,
inspect models, and let agents operate those surfaces without brittle terminal
guesswork.

The repo serves that intent well at the package and proof-tranche level. It is
not yet serving it as a product surface. A user can run family examples, but
cannot discover a coherent top-level `mlxts generate image`, `mlxts model`,
`mlxts train`, or `@mlxts/sdk` happy path. That is fine for Phase 10 build-out,
but it becomes harmful if another wave of families lands before shared product
contracts are designed.

The next phase should therefore convert repeated proof behavior into package
and product primitives before widening the family matrix further.

## 3. Drift Ledger

**Resolved from April 28:**

- `packages/serve/src/` flatness resolved: the prefix groups became role-named
  subfolders.
- `serve/AGENTS.md` tactical-only posture resolved: it now names architectural
  boundaries.
- Empty transformer Phase-7 directories resolved: the vestigial directories are
  gone.
- Qwen 3.5 folder flatness resolved at the root level: cache, linear-attention,
  and multimodal subfolders exist.
- Transformer LoRA top-level files resolved: the files moved under
  `transformers/src/lora/`.
- Cache layer-type taxonomy resolved: shared cache-layer kind exists.
- `CLAUDE.md` duplication resolved: the file is now a short pointer.

**Migrated rather than fully resolved:**

- Cross-example coupling moved from `examples/train-proof` to
  `examples/image-proof`. The old training reach-in is gone, but six image
  examples and LTX now import image-proof artifact/test utilities. This is the
  same structural smell with a stronger product signal.
- File-cap pressure moved inward. Qwen 3.5 root decomposition helped, but
  `families/qwen3_5/cache/batch-cache.ts` and related cache files now sit close
  to the cap.
- Documentation duplication moved from `CLAUDE.md` into `continuity.md` and
  long `MEMORY.md` Tier 2 entries.

**Persisting from April 28:**

- `core/src/array.ts`, `core/src/` root prefix grouping, and `nn/src/module.ts`
  remain near-cap / mixed-concern candidates.
- `transformers/src/lora/module-traversal.ts` still duplicates traversal shape
  from `@mlxts/lora`.
- Some public/runtime knobs and progress defaults still risk leaking runtime or
  terminal strategy into semantic surfaces.

**New since April 28:**

- `@mlxts/diffusion` is now the largest package by LOC and has a real product
  surface problem, not just a package-structure problem.
- Classic LTX-Video and LTX-2 have finite proof paths but no documented
  real-checkpoint proof.
- No `@mlxts/sdk`, top-level `mlxts` binary, media serving route, or
  `DiffusionEngine` exists.
- AXI session-hook behavior exists in the upstream standard, but mlxts should
  not treat hooks as a CLI-v1 requirement. The simpler first product is a
  compact no-argument `mlxts` dashboard.

## 4. Mechanical Snapshot Summary

The companion metrics doc carries the detailed counts. Headline:

- Typecheck passes across 15 workspace typechecks, including `examples/nanogpt`.
- `check:file-lines` passes across 542 production files.
- Biome lint, assertions, tensor-lifetime, runtime-review, per-package AGENTS,
  cross-package imports, and skill checks pass in the focused rerun.
- All 14 package directories have `AGENTS.md`; `check:per-package-agents`
  enforces the 13 packages above its non-triviality threshold.
- Examples are 11/16 on `AGENTS.md`; missing examples are `chat`,
  `chat-canary`, `lora-finetune`, `serve-completions`, and `train-proof`.
- `@mlxts/diffusion` is 37,627 production LOC, larger than `transformers`
  (26,744) and `serve` (21,679).

Coverage was not rerun because this pass updates audit documentation.

## 5. Product Surface Coherence

Text serving is coherent. `mlxts-serve` owns OpenAI Completions, Chat
Completions, OpenResponses, and Anthropic Messages adapters through a normalized
generation request. `@mlxts/agent` remains experimental and separate. This is
the repo's best product surface today.

Media generation is not yet coherent. The package owns impressive runtime
pieces, but the user-facing surface is a family-by-family example matrix:
`examples/stable-diffusion`, `examples/flux`, `examples/flux2`,
`examples/qwen-image`, `examples/z-image`, `examples/stable-diffusion-3`, and
`examples/ltx-video`. These examples are now carrying shared responsibilities:
artifact writing, report verification, flag policy, TOON/JSON choices, snapshot
source policy, and error format.

The top-level CLI is missing. There is no `mlxts generate image`, `mlxts
generate video`, `mlxts generate audio`, `mlxts model discover`, or
`mlxts model status`. The future wrapper is mentioned in
`docs/ecosystem-structure.md`, but no design or implementation owns it.

The SDK is missing. There is no `@mlxts/sdk` facade. Existing package APIs are
usable by advanced callers, but there is no happy-path programmatic surface
that says "generate an image" or "run SFT" across packages.

The API is missing for media generation. There is no `/v1/images/generations`,
`/v1/videos/generations`, `/v1/audio/generations`, or `DiffusionEngine`.
Do not start by copying the text `GenerationEngine`; image/video/audio
admission, output shape, streaming, and artifact lifetime are different
concerns.

AXI session hooks should not be a near-term mlxts product goal. The upstream
AXI skill includes them, and this repo now mirrors that skill exactly, but
mlxts can adopt the output ergonomics without adopting automatic hook installs.
A future top-level `mlxts` binary should first make `mlxts` with no arguments
show a compact live dashboard: local machine, MLX/native status, known models,
available generation/training/serving commands, and the next useful checks.

## 6. Architectural Posture

`@mlxts/diffusion` needs the next structural pass. `families/ltx/` combines two
architecturally distinct families with only a small shared core. Split it into
`ltx-video`, `ltx2`, and a small shared helper folder before LTX-2.3 or more
sidecar work lands.

`examples/image-proof` has crossed from "example helper" into product
infrastructure. The importing examples need BMP writing, hashing, report
schemas, and saved-report verification. Those should move under
`packages/diffusion/src/media/`, `packages/diffusion/src/proofs/`, or a similarly
package-owned surface. LTX WAV/PCM16 output belongs in the same package-owned
media layer.

`@mlxts/serve` is structurally much healthier than it was on April 28. The
remaining architectural gap is forward-looking: it has no doctrine for media
generation serving. The decision should land in `docs/serving-runtime-strategy.md`
or `packages/serve/AGENTS.md` before a route is added.

`@mlxts/transformers` needs small follow-through, not a new big split.
Qwen cache files near the cap, the emerging encoder-hidden-states pattern, and
progress-output defaults are the main concerns. The CausalLM contract should
stay tight; hidden-state consumers need a separate interface rather than
quietly widening CausalLM.

The training packages are in better shape than the superseded draft implied.
SFT, DPO, LoRA, and QLoRA proof paths exist. The missing architecture is a
broader alignment/evals layer, not basic QLoRA existence.

## 7. Quality And Proof Bar

The repo's `Add` vs `Prove` language is useful, but it should become
machine-readable before product surfaces expose media generation broadly.
Family examples should be able to report evidence stage without a human reading
review docs.

Current media proof posture:

| Family | Current evidence |
| --- | --- |
| Stable Diffusion / SDXL | finite proof plus bounded SDXL real-checkpoint proof |
| FLUX.1 | finite proof plus bounded real-checkpoint proof |
| FLUX.2 Klein | finite proof plus bounded real-checkpoint proof |
| Z-Image-Turbo | finite proof plus bounded real-checkpoint proof |
| Qwen-Image / 2512 | finite proof plus bounded 2512 real-checkpoint proof |
| Stable Diffusion 3 / 3.5 | finite path; real checkpoint blocked by gated access |
| Classic LTX-Video | finite proof and verifier; no documented real-checkpoint proof yet |
| LTX-2 | proof assembly and BMP/WAV verifier; no documented real-checkpoint proof yet |

Training proof posture:

- SFT and DPO are implemented and verified through the Phase 8 proof surfaces.
- LoRA and QLoRA proof paths exist; QLoRA preservation is package-owned through
  `@mlxts/lora` helpers.
- Remaining capability gaps are DoRA, PPO/GRPO/ORPO/KTO, reward modeling,
  distillation, and a dedicated evaluation harness.

The quality bar for product-ready media serving should be: real-checkpoint
proof, package-owned artifact/report utilities, documented model license/access
expectations, structured errors, and operator-visible resource metrics.

The macro posture issue is not that the repo fails to make proofs. It makes
many proofs. The issue is that a proof can become psychologically satisfying
before the capability is finished. For media generation, "finished" means the
full intended capability is covered end to end: image, video, and audio modes
where claimed; package-owned artifact and verification utilities; coherent
CLI/API/SDK entrypoints; real-checkpoint evidence where access permits; and
clear unsupported-mode errors where it does not.

## 8. Risk Lanes

Security: serving image input already has useful boundaries around local image
roots, traversal rejection, byte limits, and remote URL validation. Media
generation serving must reuse that caution. Do not add arbitrary file, URL, or
artifact serving paths without explicit roots, byte budgets, and allowlists.

Dependency hygiene: runtime dependencies remain relatively small and mostly
official Hugging Face JS packages plus Bun/MLX bindings. Diffusion increases
the weight of Hugging Face snapshot behavior and model-card/license reliance,
so dependency and checkpoint provenance should become part of proof metadata.

Native distribution: the repo is still Apple Silicon / MLX-first. Before a
public release story, native binary build reproducibility, CMake/MLX-C version
pinning, and package install behavior need a dedicated release-readiness pass.

Licensing: media checkpoints have heterogeneous licenses and gated access.
Real-checkpoint proof docs should record the checkpoint id, access mode, and
license posture at least at the operator-evidence level. A working local proof
does not imply the product can advertise the model freely.

Supply chain: Hub downloads, local cache discovery, symlinked safetensors, and
generated artifacts are central paths now. Keep local-only modes explicit and
standardize the flags before those examples become a top-level CLI.

## 9. Direction

Recommended next sequence:

1. **Native runtime blocker**: isolate why non-empty array creation and default
   GPU stream creation return null pointers locally, then restore
   `packages/core/src/ffi.test.ts` and `check:phase10-proofs`.
2. **Phase 10b product/proof infrastructure**: extract media artifact writers,
   report schemas, saved-report verification, and WAV/BMP helpers into
   `@mlxts/diffusion`.
3. **LTX cleanup and Prove pass**: split `families/ltx/`, then run and document
   real-checkpoint proofs for classic LTX-Video and LTX-2.
4. **Diffusion serving design**: decide whether `@mlxts/serve` owns
   `DiffusionEngine`, media-generation admission, and OpenAI-compatible image
   routes.
5. **Unified CLI/SDK design**: design `mlxts generate {text|image|video|audio}`,
   `mlxts model`, and a thin `@mlxts/sdk` facade before implementing them.
   Keep session hooks out of v1; the no-argument dashboard is the agent-native
   entrypoint.
6. **Narrative-doc guardrail**: keep `continuity.md` slim and add a shape check
   so it stays a compact continuity record instead of a duplicate review
   archive.

Avoid starting another broad media family until at least the first two items are
done. The repo has enough family breadth to reveal the shared product seams.

## 10. Specific Recommendations

**P0. Restore native runtime health before new proof claims.** Start from
`packages/core/src/ffi.test.ts`, because the same null pointer signatures
explain the current Phase 10 proof gate failures.

**P1. Extract media proof infrastructure from examples.** Move BMP writing,
artifact hashing, report schemas, saved-report verification, and LTX WAV/PCM16
encoding into `@mlxts/diffusion`. Keep examples as thin workbooks.

**P1. Split `packages/diffusion/src/families/ltx/`.** Create distinct
`ltx-video` and `ltx2` family folders with a small shared helper home.

**P1. Run real-checkpoint LTX proofs.** Document classic LTX-Video and LTX-2
proofs under `docs/reviews/` before exposing either through future serve/SDK
surfaces.

**P1. Record the diffusion serving decision.** Add the decision to
`docs/serving-runtime-strategy.md` or `packages/serve/AGENTS.md` before the
first route implementation.

**P1. Design the unified `mlxts` CLI and SDK facade.** Treat this as a design
artifact first, not a quick wrapper around examples.

**P2. Gate `continuity.md` shape.** Keep evidence blocks out of the continuity
record, prefer dated review links, and enforce a compact limit.

**P2. Add audit-outcome routing to repo doctrine.** Accepted audit outcomes
should update the right home: `PLAN.md` for phase order and mission changes,
`continuity.md` for active continuity state only, `MEMORY.md` for durable sharp
edges, `docs/reviews/` for evidence, and package/product docs for stable
contracts.

**P2. Add an example AGENTS gate.** The missing examples are exactly where
cross-example coupling and product-surface confusion can regrow.

**P2. Normalize download/cache flags.** Choose one safer convention before
family examples become top-level CLI subcommands. Prefer cached/local-only by
default with an explicit opt-in for downloads.

**P2. Correct training roadmap wording.** Do not list QLoRA as absent. Track
DoRA, PPO/GRPO/ORPO/KTO, reward modeling, distillation, and `@mlxts/evals`.

**P2. Define the hidden-state interface.** Add a narrow
`EncoderHiddenStates`-style contract for text encoders used by diffusion
conditioning instead of widening CausalLM.

**P2. Fix progress default stdout hazards.** Package pretrained progress helpers
should not default to `console.log` in agent-facing paths.

**P3. Revisit near-cap files after the P1 work.** Do not solve the file-cap
question abstractly while product seams are still misplaced.

## 11. Open Questions For Nomi

These are the recommended answers, not blank questions.

1. **Media generation serving should wait for the generation product contract.**
   `@mlxts/serve` should likely own HTTP routes, admission, SSE/streaming where
   relevant, and server resource policy. It should not own family artifact
   writers, proof schemas, or checkpoint behavior. Define the package-owned
   generation contract first, then adapt it into serving.
2. **Package-owned proof infrastructure comes before the top-level CLI
   implementation.** The top-level CLI should be designed now, but not built by
   wrapping example workbooks. Move artifact writers, verifiers, proof reports,
   and media output helpers into packages first.
3. **Stable generation types should live below the SDK.** A package such as
   `@mlxts/generate` should own request/response types, artifact descriptors,
   model capability records, and proof metadata. `@mlxts/sdk` can then be a thin
   friendly facade over `generate`, `serve`, `train`, and model discovery.
4. **Example `AGENTS.md` should be threshold-based, not ceremonial.** Require it
   for any example with a CLI, multiple files, cross-example imports, proof
   behavior, or product-adjacent workflows. Tiny one-file examples can stay
   exempt.
5. **Media checkpoint license/access metadata should be recorded now.** Keep it
   lightweight: checkpoint id, revision when known, local/cache/download mode,
   gated-access status, and observed license/source link. This is product truth,
   not release bureaucracy.
6. **Session hooks should stay out of mlxts CLI v1.** Keep the local AXI skill
   exact upstream, but treat hook installation as optional upstream guidance,
   not mlxts doctrine. A no-argument `mlxts` dashboard is the first agent-native
   surface.
7. **Documentation routing should be explicit.** `PLAN.md` is the mission and
   phase roadmap. `continuity.md` is the compact active state for long-running
   work. `MEMORY.md` is durable doctrine and sharp edges. `docs/reviews/`
   holds evidence for completed tranches. `docs/audits/` holds posture
   diagnosis and should not become the task tracker.

## 12. Files Reviewed

Primary audit artifacts:

- `docs/audits/README.md`
- `docs/audits/2026-04-28-architectural-posture-audit.md`
- `docs/audits/2026-04-28-audit-metrics.md`
- `docs/audits/archive/2026-05-02-architectural-posture-audit-superseded.md`
- `docs/audits/archive/2026-05-02-audit-metrics-superseded.md`
- `docs/audits/2026-05-02-audit-metrics.md`

Doctrine and planning:

- `AGENTS.md`
- `MEMORY.md`
- `PLAN.md`
- `continuity.md`
- `docs/product-surfaces.md`
- `docs/serving-runtime-strategy.md`
- `docs/ecosystem-structure.md`
- `.agents/skills/axi/SKILL.md`

Representative source and examples:

- `packages/diffusion/src/families/ltx/`
- `packages/diffusion/src/families/stable-diffusion/pipeline.ts`
- `packages/diffusion/src/families/flux/config.ts`
- `packages/diffusion/src/pretrained/snapshot-source.ts`
- `packages/serve/src/media/local-image.ts`
- `packages/serve/src/media/remote-image.ts`
- `packages/transformers/src/families/qwen3_5/cache/`
- `packages/lora/src/quantized-base.ts`
- `examples/image-proof/`
- `examples/ltx-video/`
- media family example `index.ts`, `image-output.ts`, and tests

Representative review evidence:

- `docs/reviews/2026-05-01-diffusion-sdxl-real-checkpoint-proof.md`
- `docs/reviews/2026-05-01-flux-real-checkpoint-proof.md`
- `docs/reviews/2026-05-01-flux2-klein-real-checkpoint-proof.md`
- `docs/reviews/2026-05-01-z-image-turbo-real-checkpoint-proof.md`
- `docs/reviews/2026-05-01-qwen-image-2512-real-checkpoint-proof.md`
- `docs/reviews/2026-05-01-ltx-video-proof-cli.md`
- `docs/reviews/2026-05-02-ltx2-proof-assembly.md`
- `docs/reviews/2026-05-02-ltx-video-proof-verifier.md`
- `docs/reviews/2026-04-30-training-proof-live-hardening.md`
- `docs/reviews/2026-04-28-trainable-module-lora-helpers.md`
