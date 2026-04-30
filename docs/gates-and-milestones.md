# Gates and Milestones

Every phase has exit criteria — concrete, testable conditions that prove the work is done. This document defines them.

---

## Quality Gates (Apply to Every Phase)

These gates are non-negotiable at every phase boundary. Code does not advance until all pass.

### Code Quality Gates

| Gate | Command | What it checks |
|------|---------|---------------|
| Type safety | `bun run typecheck` | Zero TypeScript errors in strict mode |
| Lint | `bun run lint` | Biome passes clean with `--error-on-warnings` |
| Type assertions | `bun run check:assertions` | No `as` or `!` outside FFI boundary (AST-checked) |
| Tensor lifetimes | `bun run check:tensor-lifetimes` | No anonymous disposable intermediates in nested calls |
| Runtime review | `bun run check:runtime-review` | Runtime-sensitive diffs have review artifacts |
| Coverage | `bun run check:coverage` | Package-specific line/function/branch thresholds |
| Training proof surfaces | `bun run check:training-proofs` | Phase 8 example proof code typechecks and report-verifier tests pass |
| Agent-facing CLI contract | Focused CLI formatter/parser tests plus manual AXI review | Finite commands emit compact structured stdout, actionable stdout errors, stable exit codes, and no non-TTY prompts |
| Full validation | `bun run validate` | All of the above in sequence |

### Architectural Gates

| Gate | How to verify | What it ensures |
|------|--------------|----------------|
| Human readability | Manual review by a different agent or human | A TS developer unfamiliar with the code can understand any function within 30 seconds |
| Error quality | Manual review | Every user-facing error includes: context, expected vs actual, and an actionable hint |
| JSDoc coverage | Manual review | Every public API has at least a one-line JSDoc |
| No dead code | Manual review | No stale compatibility layers, unused exports, or commented-out code |
| Runnable reference surfaces | `bun run` each committed example surface | Every committed end-to-end reference surface executes successfully |

### Operational Gates (for phases with training/inference)

| Gate | Command | What it checks |
|------|---------|---------------|
| Memory stability | `cd examples/nanogpt && bun run bench:memory` | No unbounded memory growth over repeated operations |
| Soak test | `cd examples/nanogpt && bun run soak:<preset>` | Throughput stable, memory stable over sustained runs |
| Acceptance | `cd examples/nanogpt && bun run acceptance:<preset>` | Loss reaches target, generation produces coherent output |

### Change-Specific Gates

Use the narrowest gate that proves the change, then run the full required gates
before commit.

| Change area | Minimum focused proof | Required broader gate before handoff |
|-------------|----------------------|--------------------------------------|
| Package API or pure TypeScript logic | Package-local tests for the touched package | `bun run typecheck`, `bun run check:coverage` |
| Runtime-sensitive tensor, generation, cache, or serving code | Focused tests plus `bun run check:tensor-lifetimes` and a `docs/reviews/` artifact | `bun run check:runtime-review`, `bun run check:coverage` |
| Model-family generation or performance | Family tests plus `bun run bench:generation:parity --require-mlx-lm-reference` when making parity claims | Qwen/Gemma real regression when cached checkpoints are available |
| Serving behavior | `bun run --filter '@mlxts/serve' regression:serve` | `bun run regression:qwen-gemma -- --profile real` for high-risk serving/model commits |
| Serving capability claim | `bench:serve --report-json` ladder with route, scheduler, stream, and memory evidence | `bun run regression:qwen-gemma -- --profile substantial` when cached models fit |
| Agent loop behavior | `bun test packages/agent/src` plus a served-model smoke when practical | Serve regression if protocol or streaming semantics changed |
| Agent-facing CLI work | Parser/formatter tests plus stdout/stderr/exit-code assertions | Package typecheck and coverage; served-model or example smoke when the command executes model work |
| Training or alignment proof | Example/package-focused tests and the relevant proof command | Promote to self-hosted Apple Silicon gate only after the proof is stable |
| Example/workbook docs or scripts | The example's documented smoke command | No root example script; reusable behavior belongs in packages |

---

## Phase 4: nanoGPT

**Status:** Complete.

### Milestone: "GPT trains on Shakespeare and generates text"

| Criterion | How to verify |
|-----------|--------------|
| `bun run validate` passes | Run it |
| gpt-tiny trains to <1.8 val loss | `cd examples/nanogpt && bun run acceptance:gpt-tiny` |
| gpt-small has a loss-targeted acceptance run | `cd examples/nanogpt && bun run acceptance:gpt-small` |
| Generated text is recognizably English and vaguely Shakespearean | Manual inspection of sample output |
| Supervised long runs use `cd examples/nanogpt && bun run manager ...` | Verify soak ladder: 50 → 250 → 1000 → 5000 steps |
| Runtime review artifact exists for hot-path changes | `bun run check:runtime-review` |
| Checkpoint save is atomic | Code review: write-to-temp, rename |
| NaN loss detection stops training | Test coverage |

---

## Phase 5: Ecosystem Restructure

**Goal:** Extract the canonical `@mlxts/*` packages, stabilize the package-first repo contract, and clearly separate publishable packages from committed example surfaces such as `examples/nanogpt`.

### Milestone: "Canonical packages extracted, validated, and documented"

| Criterion | How to verify |
|-----------|--------------|
| Root workspace and docs use the `mlxts` package identity | Manual review |
| `@mlxts/core` exists with MxArray, ops, transforms, FFI, and native build ownership | `cd packages/core && bun test` and `cd packages/core && bun run build:native` |
| `@mlxts/nn` exists with Module, layers, losses, activations | All existing nn tests pass |
| `@mlxts/optimizers` exists with Adam, AdamW, SGD | All existing optimizer tests pass |
| `@mlxts/train` exists with training loop, checkpointing | All existing train tests pass |
| `@mlxts/data` exists with text data loading | All existing data tests pass |
| `@mlxts/tokenizers` exists with char tokenizer | All existing tokenizer tests pass |
| `bun run validate` passes across entire monorepo | Run it |
| Each package's tests pass independently, not just monorepo-level | `cd packages/<pkg> && bun test` for each |
| Active production source honors the 500-line cap | `bun run check:file-lines` |
| Coverage thresholds match the current package-first gate | `bun run check:coverage` |
| `examples/nanogpt` is documented as a committed in-repo example surface rather than a package publish target | Manual review |
| Runtime-sensitive extraction work has a review artifact | `bun run check:runtime-review` |
| All top-level docs describe the package-first state consistently | Manual review |

### What "done" looks like
A developer runs `bun install`, opens `packages/core/src/index.ts` and `packages/train/src/index.ts`, sees the canonical `@mlxts/*` package surfaces, and `bun run validate` passes. `examples/nanogpt` is present as a clearly documented in-repo example and regression surface, not a package.

---

## Phase 6: Publish Core Packages

**Goal:** The repo is ready for first npm publish. TypeDoc, CI, dist output, and package manifests are in place even if the actual publish step is still manual.

### Milestone: "External developers can install and use mlxts"

| Criterion | How to verify |
|-----------|--------------|
| Public package manifests, `dist/`, and exports are publish-ready | Manual review + `bun run build` |
| Semver versioning with changesets | `changeset status` |
| TypeDoc API docs generate cleanly | `bun run docs:api` |
| GitHub Actions CI definitions exist for fast checks, Apple Silicon validation, and pack dry-runs | Manual review |
| README with quick-start example | Manual review |
| `bun pm pack --dry-run` succeeds for every public package | `bun run pack:dry-run` |
| Educational walkthrough: "Building GPT from scratch in TypeScript" | Published (blog or repo doc) |
| Benchmarks: mlxts vs Python MLX for core ops | Results documented |
| Contributing guide | CONTRIBUTING.md exists |
| Consider publishing unscoped `mlxts` package that re-exports core + nn + optimizers for beginners | Evaluate ergonomics vs namespace clarity |

---

## Phase 6.5: Modern Transformer Primitives

**Goal:** The nn and ops layers have everything modern architectures need.

**Current state:** Complete in-repo. The canonical `@mlxts/core` and
`@mlxts/nn` packages now provide the primitives this milestone requires.

### Milestone: "Ready for LLaMA"

| Criterion | How to verify |
|-----------|--------------|
| RMSNorm module implemented and tested | Test: matches manual computation |
| RoPE module implemented and tested | Test: rotary embeddings produce correct positional encoding |
| Grouped-Query Attention (GQA) module works | Test: forward pass with different num_kv_heads < num_heads |
| SwiGLU activation works | Test: matches manual gate * silu(x) computation |
| sort and topk ops bound from mlx-c | Test: sort returns correct order, topk returns correct k elements |
| float16/bfloat16 safetensors I/O | Test: save float16 model, reload, verify values |
| mlx_quantize/mlx_dequantize bound | Test: quantize to 4-bit, dequantize, verify approximate values |
| `bun run validate` passes | Run it |

**Must complete before Phase 7 begins.**

---

## Phase 7: Model Architectures

**Goal:** Load and run pretrained LLaMA, Mistral, etc. The `@mlxts/transformers` package.

### Milestone: "Load a HuggingFace model and generate text"

| Criterion | How to verify |
|-----------|--------------|
| Pretrained loading downloads from HuggingFace Hub via official `@huggingface/hub` | Test: download a small model |
| Pretrained loading parses `config.json` and tokenizer sidecars | Test: parse known configs |
| `@mlxts/transformers` streams safetensors weights into mlxts arrays | Test: load and verify shapes/dtypes |
| `@mlxts/tokenizers` implements special-token-aware `tokenizer.json` encoding for supported families | Test: encode/decode and special-token IDs match Hugging Face tokenizer output |
| Chat prompt compilation matches Hugging Face for supported chat-capable checkpoints | Test: rendered prompt text and token IDs match `apply_chat_template` + tokenizer output |
| `@mlxts/transformers` implements LLaMA architecture | Test: forward pass matches MLX Python output for same weights |
| KV cache works for efficient generation | Test: generate 100 tokens, verify speed improvement vs no cache |
| Checkpoint generation defaults and end-of-turn stop behavior are honored | Test: generation stops on the checkpoint's full EOS / turn-boundary token set |
| `AutoModel.fromPretrained("meta-llama/Llama-3.2-1B-Instruct")` works | End-to-end test |
| At least 3 model architectures supported | LLaMA, Mistral, Phi or Gemma |
| Generation quality: coherent multi-sentence output | Manual inspection |
| `examples/chat/` runs interactively | Demo it |

### What "done" looks like
A developer writes:
```typescript
import { AutoModel, AutoTokenizer, generateText } from '@mlxts/transformers';
const model = await AutoModel.fromPretrained('meta-llama/Llama-3.2-1B-Instruct');
const tokenizer = await AutoTokenizer.fromPretrained('meta-llama/Llama-3.2-1B-Instruct');
const output = generateText(model, tokenizer, 'Hello, world!', { maxTokens: 100 });
console.log(output);
```
And it works. On their Mac. From TypeScript.

---

## Phase 8: Fine-Tuning

**Goal:** LoRA fine-tuning, DPO alignment, dataset loading.

### Milestone: "Fine-tune a model on custom data"

| Criterion | How to verify |
|-----------|--------------|
| `@mlxts/lora` implements LoRA adapter injection | Test: inject LoRA into LLaMA, verify parameter count reduction |
| LoRA training converges on a small task | Test: fine-tune on a 1000-example dataset, verify loss decreases |
| LoRA merge produces a standalone model | Test: merge adapters, verify generation quality matches |
| `@mlxts/align` implements DPO trainer | Test: DPO loss decreases on preference pairs |
| `@mlxts/data` supports HuggingFace datasets format | Test: load a dataset from Hub |
| Chat template support for instruction tuning | Test: format conversations correctly |
| Canonical proof uses pinned real-data subsets | Test: short runs on `HuggingFaceH4/ultrachat_200k` and `HuggingFaceH4/ultrafeedback_binarized` produce held-out metrics |
| Canonical proof reports are machine-checkable | `bun run check:training-proofs` plus `examples/train-proof/verify-report.ts <report.json>` |
| Canonical training proof is runnable on self-hosted Apple Silicon | Manual `Training Proof` workflow or local `bun run examples/train-proof/index.ts` succeeds on the official anchor |
| `examples/lora-finetune/` runs end-to-end | Fine-tune, merge, generate |
| Memory fits within 64GB for 1B-3B parameter models with LoRA | Measure peak memory |
| QLoRA works with 4-bit base model | Test: load quantized model, apply LoRA, train |
| Training orchestration stays explicit | Review: no black-box pipeline framework or reactive dependency lands inside `@mlxts/train` |

### What "done" looks like
A developer fine-tunes `meta-llama/Llama-3.2-1B-Instruct` on a pinned real-data
subset using 4 lines of config and `mlxts train --lora`, then merges the
adapter and serves the result. The same short proof run is cheap enough to stay
in CI as a regression gate.

---

## Phase 9a: Quantization

**Goal:** Quantize and dequantize models. GGUF tensor support.

### Milestone: "Quantize a model to 4-bit and run it"

| Criterion | How to verify |
|-----------|--------------|
| `@mlxts/quantize` quantizes a model to 4-bit | Test: quantized model produces reasonable output |
| Dequantization works for inference | Test: dequantized weights are approximately correct |
| GGUF tensor dequantization works | Test: load Q4_K_M tensors, dequantize, verify values |
| Quantized inference performance | Benchmark: tokens/sec for quantized vs full-precision |

---

## Phase 9b: KV Cache and Generation Optimization

**Goal:** Efficient inference with KV caching.

> **Note:** PagedAttention-style cache management is a major engineering effort. Start with simple sequential KV cache.

### Milestone: "Fast generation with KV cache"

| Criterion | How to verify |
|-----------|--------------|
| Simple sequential KV cache works | Test: generate 100 tokens, verify speed improvement vs no cache |
| KV cache memory is bounded | Test: generate 2048 tokens without unbounded memory growth |
| Batch generation works | Test: generate for multiple sequences in parallel |

---

## Phase 9c: Serving

**Goal:** Production-quality inference server with OpenAI-compatible API and
bounded Anthropic-compatible Messages support.

### Milestone: "Serve a model with an OpenAI-compatible API"

| Criterion | How to verify |
|-----------|--------------|
| `@mlxts/serve` starts a server on `Bun.serve()` | Test: server responds to health check |
| `/v1/chat/completions` endpoint works | Test: curl request returns valid response |
| `/v1/completions` endpoint works | Test: completion request returns text |
| Text-only `/v1/responses` endpoint works | Test: responses request returns valid text response and semantic SSE stream |
| Text-only `/v1/messages` endpoint works | Test: Anthropic Messages request returns valid text/thinking blocks and Anthropic SSE events |
| Streaming responses (SSE) | Test: completions/chat/responses/messages stream token or semantic deltas with usage where supported |
| Protocol adapters share one internal request path | Test: chat, completions, text Responses, and bounded Anthropic Messages normalize to the protocol-neutral request model |
| `mlxts-serve meta-llama/Llama-3.2-1B-Instruct` works | End-to-end demo through the package-owned binary |
| Qwen/Gemma continuous scheduler routes are honest | Real regression asserts route decisions, scheduler phases, stream health, and memory budgets |
| Future broader Anthropic content/tool support works | Test: Anthropic clients can use images/tools once those adapters exist |
| Future `/v1/embeddings` endpoint works | Test: embedding request returns vector once embedding engines exist |
| Future dynamic model loading/unloading works | Test: switch models without restart once the engine pool exists |

### What "done" looks like
`mlxts serve --model meta-llama/Llama-3.2-1B-Instruct --quantize 4bit` starts a server. OpenAI-compatible clients (Cursor, Continue, LangChain.js, etc.) and Anthropic Messages-compatible text clients can connect and get fast, streaming responses.

---

## Phase 9.5: Product-Agent Experience and AXI Hardening

**Goal:** Agent-operated CLI surfaces are predictable, token-efficient, and
safe to drive through shell tools.

### Milestone: "Agents can inspect, run, and recover without reading logs"

| Criterion | How to verify |
|-----------|--------------|
| Local AXI skill is the canonical CLI contract | `.agents/skills/axi/SKILL.md` exists and is linked from root/product docs |
| Finite commands emit compact structured stdout | Formatter/parser tests assert TOON-shaped defaults and explicit empty states |
| Errors are structured and actionable | Tests assert stdout error bodies plus exit `1` or `2` as appropriate |
| Progress and diagnostics stay off stdout | Tests or manual CLI review checks stdout/stderr separation |
| Non-TTY paths never prompt | Tests run missing-required-value paths without hanging |
| `mlxts-serve` finite inspection commands are AXI-shaped | `mlxts-serve discover --model-root <dir>` and `mlxts-serve status --base-url <url>` have snapshot-style tests |
| Experimental `mlxts-agent` one-shot and non-TTY error paths are AXI-shaped | `bun test packages/agent/src` covers one-shot help/error/status output |
| Training proof, run manager, benchmark, and Phase 10 proof commands adopt AXI before becoming canonical | Focused tests or review artifacts for each command tranche |
| Long-running servers, REPLs, and managers expose status/report/transcript surfaces instead of one final data blob | Manual review plus command-specific smokes |

---

## Phase 10a: Multimodal Understanding

**Goal:** Image, audio, and encoder-decoder understanding through
`@mlxts/transformers` without widening `CausalLM`.

### Milestone: "Describe and answer questions about media from TypeScript"

| Criterion | How to verify |
|-----------|--------------|
| Qwen image-conditioned generation remains green | `bun run regression:qwen-image` |
| VLM support extends beyond the first Qwen path | Test: at least one additional VLM family describes a known image |
| Vision/audio encoder preprocessing is family-owned | Code review: file decode/transport stays in serve/examples; checkpoint preprocessing stays in transformers |
| Image+text prompting works through serving protocols | Test: OpenAI Chat/OpenResponses/Anthropic media requests produce coherent output where supported |
| Encoder-decoder or audio path works | Test: Whisper or another encoder-decoder model produces text from local media |
| `examples/vlm-chat/` or equivalent runs | Describe a known image and verify coherent output |
| Finite proof commands are AXI-shaped | CLI tests assert compact stdout and structured errors |

---

## Phase 10b: Diffusion and Flow Generation

**Goal:** Image generation on Apple Silicon.

### Milestone: "Generate an image from a text prompt"

| Criterion | How to verify |
|-----------|--------------|
| `@mlxts/diffusion` implements Stable Diffusion pipeline | Test: generate a 512x512 image |
| DDIM and Euler schedulers work | Test: different schedulers produce valid images |
| Image preprocessing pipeline | Test: load, resize, normalize images |
| `examples/stable-diffusion/` runs | Generate an image from "a cat sitting on a laptop" |
| Finite proof commands are AXI-shaped | CLI tests assert compact stdout and structured errors |

---

## Phase 10 Completion Fence

**Goal:** Phase 10 is represented by both multimodal understanding and
diffusion/flow generation as package-owned product surfaces.

### Milestone: "On-device media understanding and generation are real product paths"

| Criterion | How to verify |
|-----------|--------------|
| `@mlxts/transformers` owns at least one real VLM path beyond toy fixtures | Real checkpoint smoke plus package tests |
| `@mlxts/diffusion` owns at least one real text-to-image pipeline | Real checkpoint smoke plus package tests |
| Serving advertises only implemented media semantics | Protocol tests reject unsupported media/tool/file shapes clearly |
| Examples are workbooks, not hidden product surfaces | Manual review: reusable logic lives in packages |
| Runtime-sensitive media paths have review artifacts | `bun run check:runtime-review` |
| Full validation is green | `bun run validate` |

---

## Phase 11: Multi-Backend (Future)

Phase 11 (multi-backend) is future work. See docs/future-backends.md. No gates defined until a go/no-go decision is made.

---

## Phase 12: Evaluation and Benchmarks

**Goal:** Standardized model evaluation. Credibility through reproducible benchmarks.

### Milestone: "Evaluate any model on standard benchmarks"

| Criterion | How to verify |
|-----------|--------------|
| `@mlxts/eval` implements 6 core tasks | MMLU, HellaSwag, ARC, WinoGrande, TruthfulQA, GSM8K |
| Evaluation results match Python lm-eval-harness within 5% or directionally consistent (exact match unlikely due to floating-point and tokenizer differences) | Cross-validate on same model |
| `mlxts eval --model Llama-3.2-1B --tasks mmlu,hellaswag` works | End-to-end |
| JSON result output for comparison | Structured results file |
| Benchmark suite for mlxts vs Python MLX core ops | Published results |

---

## Ultimate Milestone: "Implement Any Paper"

This is the north star. The ecosystem is complete when:

| Criterion | What it proves |
|-----------|---------------|
| A developer can implement a custom model architecture using `@mlxts/nn` building blocks | The nn layer is general enough |
| A developer can train it using `@mlxts/train` | The training infra is model-agnostic |
| A developer can fine-tune a pretrained model using `@mlxts/lora` | Fine-tuning works across architectures |
| A developer can serve it using `@mlxts/serve` | The serving layer is model-agnostic |
| A developer can evaluate it using `@mlxts/eval` | Evaluation is standardized |
| A developer can add custom Metal kernels via FFI and compose them with existing ops | The extension story works |
| A developer can do all of the above without leaving TypeScript | No Python required |
| A developer can read the code and understand how it works | Human readability maintained |

### How to Test This Milestone

Create `examples/custom-model/`:
1. Pick a recent paper with a novel architecture element (e.g., a new attention mechanism, a new normalization technique, a novel positional encoding)
2. Implement it using only `@mlxts/*` packages
3. Train it on a standard dataset
4. Evaluate it on benchmarks
5. Serve it via API

If a developer can follow the example and do the same for a different paper, the ecosystem is working.

---

## Gate Enforcement

### Automated (CI)
- `bun run validate` runs on every push
- Coverage thresholds enforced per package
- Type assertion check prevents `as` leaking out of FFI
- Tensor lifetime check prevents anonymous intermediate leaks
- `check:training-proofs` keeps Phase 8 proof/example surfaces statically checked without running heavy model training
- The heavier `bun run examples/train-proof/index.ts` proof currently lives on a manual self-hosted Apple Silicon workflow; it can be promoted to a stricter regression gate later

### Semi-Automated (Agent Review)
- Runtime review artifacts required for hot-path changes
- No agent's output ships without review by a different agent or human
- Error message quality reviewed manually

### Manual (Human Decision)
- Phase transitions require Nomi's explicit approval
- Example quality assessed by actually running them
- "Can a TS developer understand this?" assessed by fresh-eyes review

### Skill-Based (Future)
- Create a Claude Code skill that, given a user intent ("I want to fine-tune LLaMA on my data"), produces the correct mlxts code using the right packages
- The skill references canonical examples and API docs
- If the skill can't produce working code, the API or docs need improvement
