# mlxts

TypeScript-native MLX packages for Apple Silicon, built around Bun and Apple's
official `mlx-c` C API.

The repo is package-first now. The canonical product surface is the extracted
`@mlxts/*` package family:

- `@mlxts/core`
- `@mlxts/nn`
- `@mlxts/optimizers`
- `@mlxts/train`
- `@mlxts/data`
- `@mlxts/tokenizers`
- `@mlxts/transformers`
- `@mlxts/lora`
- `@mlxts/align`
- `@mlxts/quantize`
- `@mlxts/protocols`
- `@mlxts/serve`
- `@mlxts/agent`
- `@mlxts/diffusion`

`examples/nanogpt` is the committed in-repo nanoGPT example and regression
surface. It proves the package stack works end to end, but it is an example,
not a publishable package.

## Why this repo exists

Most JavaScript ML tooling stops at inference. mlxts is trying to make training,
fine-tuning, and native Apple Silicon MLX workflows feel natural from
TypeScript without smuggling Python in as the real runtime.

The priorities are:

- correctness first
- readable APIs and readable code
- explicit ownership and runtime safety
- package surfaces that can grow into a real ecosystem
- first-class local serving and agent loops, not example-only demos
- examples that act like ML workbooks: real flows, real evidence, thin over
  reusable packages

## Repo shape

```text
packages/
  core/         Native MLX runtime, tensors, ops, transforms, I/O
  nn/           Modules, layers, activations, losses
  optimizers/   Adam, AdamW, SGD
  train/        Schedules, gradient utilities, checkpoints, loop helpers
  data/         Text loading and batching
  tokenizers/   Tokenizer implementations
  transformers/ Hugging Face checkpoint loading, model families, generation
  quantize/     Quantization primitives and checkpoint conversion support
  protocols/    Shared protocol helpers for serve/agent wire semantics
  lora/         Adapter layers, injection, merge helpers
  align/        SFT/DPO data prep and recipe helpers
  serve/        Local OpenAI-compatible serving and scheduling surfaces
  agent/        Local tool-using agent loops over served models
  diffusion/    Diffusion/flow generation primitives and image proof paths
examples/
  nanogpt/      Committed nanoGPT example and regression surface
  chat/         Interactive local chat over transformer checkpoints
  lora-finetune/ Fine-tuning workbook over @mlxts/lora and @mlxts/align
  train-proof/  Training/alignment proof workflow
  qwen3_5-image/ Dedicated Qwen 3.5 / Qwen 3.6 multimodal image example
  stable-diffusion/ Stable Diffusion / SDXL image-generation proof workbook
  flux/         FLUX.1 image-generation proof workbook
  z-image/      Z-Image-Turbo image-generation proof workbook
  qwen-image/   Qwen-Image / Qwen-Image-2512 generation proof workbook
  image-proof/  Saved BMP/report verification helper for image proof outputs
  serve-completions/ Serving concurrency/protocol example
docs/           Architecture, setup, standards, roadmap
scripts/        Validation, packaging, and repo tooling
```

## Requirements

- macOS on Apple Silicon
- Xcode 16+ with the Metal Toolchain
- CMake 3.24+
- Bun 1.3+

See [docs/setup.md](./docs/setup.md) for the full environment setup.

## Quick start

```bash
bun install
bun run build:native
bun run validate
```

The native build step only applies to `@mlxts/core`; the other packages build
on top of that runtime surface.

If you want to sanity-check the nanoGPT example after validation:

```bash
cd examples/nanogpt && bun run manager --help
cd examples/nanogpt && bun run acceptance:gpt-tiny
```

If you want to sanity-check the multimodal Qwen path on Apple Silicon, start
with the dedicated image example and an MLX-converted checkpoint:

```bash
bun run examples/qwen3_5-image/index.ts mlx-community/Qwen3.6-27B-4bit \
  --image ./.reference/transformers/tests/fixtures/tests_samples/COCO/000000039769.png \
  --prompt "Describe this image." \
  --greedy
```

If you want to sanity-check the current Phase 10 image-generation proof
surface, use one of the finite proof workbooks. Stable Diffusion / SDXL,
FLUX.1, Z-Image-Turbo, and Qwen-Image / Qwen-Image-2512 have bounded real
checkpoint evidence; FLUX.2 Klein and Stable Diffusion 3 / 3.5 are later
separate family tranches.

```bash
bun run examples/stable-diffusion/index.ts /models/stable-diffusion \
  --prompt "a cat sitting on a laptop" \
  --output .tmp/stable-diffusion/sample.bmp \
  --steps 2 \
  --height 256 \
  --width 256 \
  --json

bun run examples/flux/index.ts black-forest-labs/FLUX.1-schnell \
  --local-files-only \
  --prompt "a small red apple on a white table, product photo" \
  --output .tmp/flux/sample.bmp \
  --steps 2 \
  --height 256 \
  --width 256 \
  --json

bun run examples/z-image/index.ts Tongyi-MAI/Z-Image-Turbo \
  --local-files-only \
  --prompt "a small red apple on a white table, product photo" \
  --output .tmp/z-image/sample.bmp \
  --steps 2 \
  --height 256 \
  --width 256 \
  --guidance-scale 0 \
  --json

bun run examples/qwen-image/index.ts Qwen/Qwen-Image-2512 \
  --local-files-only \
  --prompt "a small red apple on a white table, product photo" \
  --negative-prompt " " \
  --output .tmp/qwen-image/sample.bmp \
  --steps 2 \
  --height 256 \
  --width 256 \
  --true-cfg-scale 4 \
  --json
```

If you want to serve a local model and talk to it through the package-owned
agent loop:

```bash
mlxts-serve mlx-community/Qwen3.6-27B-4bit --model-id mlx-community/Qwen3.6-27B-4bit --port 8000
mlxts-agent --model mlx-community/Qwen3.6-27B-4bit --endpoint http://127.0.0.1:8000 --cwd .
```

## Package examples

```ts
import { mxEval, ones, matmul } from "@mlxts/core";

using a = ones([3, 3]);
using b = matmul(a, a);
mxEval(b);
console.log(b.toList());
```

```ts
import { Linear } from "@mlxts/nn";
import { AdamW } from "@mlxts/optimizers";

const layer = new Linear(4, 8);
const optimizer = new AdamW(3e-4, 0.1);
```

## Repo ergonomics

The repo now has explicit local release-prep tooling even though the actual npm
publish step is manual:

```bash
bun run build
bun run docs:api
bun run pack:dry-run
bun run release:check
```

That gives us dist builds, declaration output, TypeDoc generation, and tarball
packing checks for the public packages before any real release happens.

## What is intentionally deferred

- a broader dedicated examples repo beyond the committed in-repo examples
- real npm publishing
- hosted API docs
- production deployment ergonomics beyond the current local serving/runtime
  surfaces
- larger/default-step quality and performance characterization for SDXL,
  FLUX.1, Z-Image-Turbo, and Qwen-Image-2512 beyond bounded capability proofs
- FLUX.2 Klein, Stable Diffusion 3 / 3.5, image-to-image, inpainting,
  ControlNet, video, and audio generation
- advanced serving backends such as paged KV, TurboQuant-style KV compression,
  speculative decoding, and full multimodal serving until the cache/scheduler
  seams are proven

Those come later in the roadmap once the current package ergonomics are fully settled.
