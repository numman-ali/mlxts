## Qwen3.6 Hybrid VLM Platform Tranche

### Why this tranche exists

`Qwen/Qwen3.6-27B` is not a simple dense decoder addition. Its current Hugging Face
config is a top-level `qwen3_5` multimodal wrapper with:

- a hybrid text backbone (`qwen3_5_text`) that alternates linear-attention and full-attention layers
- a vision encoder with projector output in the decoder hidden space
- image/video sentinel tokens and nested `text_config` / `vision_config`

If mlxts supports this checkpoint truthfully, the repo must grow reusable support
for:

- multimodal decoder prefill via `inputEmbeddings`
- hybrid cache state that is not just KV tensors
- grouped depthwise convolution for recurrent linear-attention blocks
- a top-level multimodal wrapper that composes a vision encoder with `CausalLM`

This tranche treats Qwen3.6 as the forcing function for shared platform work,
not as an excuse to ship a one-off family fork.

### Design constraints

- `CausalLM` remains the universal autoregressive decoder contract.
- Multimodal composition happens around `CausalLM`, not through a separate
  modality contract.
- Shared changes must stay reusable for future Gemma 4 multimodal work and
  future hybrid/MoE families.
- Example-owned runnable surfaces live under `examples/`, not root scripts.
- Runtime-sensitive changes must leave tensor ownership and cache behavior
  explicit in code and review artifacts.

### Execution slices

1. Shared contract tranche
   - Add `inputEmbeddings?: MxArray` to decoder forward options.
   - Add generation-prefill support for an initial embedding-aligned prompt.
   - Keep existing dense families behaviorally identical.

2. Shared runtime primitives
   - Add `Conv1d` support in `@mlxts/core` / `@mlxts/nn`.
   - Add reusable cache support for hybrid state arrays without widening model
     identity.
   - Keep public contracts as narrow as the current implementation allows.

3. `qwen3_5_text`
   - Add config parsing, registry wiring, weight sanitation, hybrid decoder
     blocks, recurrent linear-attention state, and text-only generation.
   - Validate on synthetic fixtures first, then on a real Qwen-family snapshot.

4. `qwen3_5` multimodal wrapper
   - Add vision config parsing, vision encoder, projector merge, and top-level
     wrapper loading from nested config.
   - Support image-conditioned prefill and then ordinary text decode.
   - Scope the first example to images, not video.

5. Example and docs
   - Add an example-local image path runner under `examples/`.
   - Keep image decoding/resizing outside the reusable package surface.
   - Document the current limits plainly.

### Commit posture

- Commit 1: this execution doc only
- Commit 2: shared decoder/generation contract changes
- Commit 3: shared runtime primitives
- Commit 4: `qwen3_5_text`
- Commit 5: `qwen3_5` multimodal wrapper plus example
- Commit 6: validation artifacts and docs

Each commit should keep concerns isolated and avoid mixing unrelated repo work.

### Validation bar

- Focused unit tests for every new primitive and config parser
- Transformer load round-trips for synthetic `qwen3_5_text` and top-level
  `qwen3_5` snapshots
- One real text smoke on the practical MLX snapshot
- One real image-conditioned smoke once the multimodal path exists
- Required gates before handoff:
  - `bun test`
  - `bun run typecheck`
  - `bun run check:coverage`
  - `bun run check:runtime-review`
  - `bun run check:tensor-lifetimes` when runtime-sensitive tensor code changes

### Explicit non-scope for this tranche

- `qwen3_5_moe`
- full video support
- Gemma 4 multimodal implementation
- prompt-cache rollback for non-trimmable hybrid state

Those remain follow-on work, but the shared seams in this tranche should not
block them.
