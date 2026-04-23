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

## Repo shape

```text
packages/
  core/         Native MLX runtime, tensors, ops, transforms, I/O
  nn/           Modules, layers, activations, losses
  optimizers/   Adam, AdamW, SGD
  train/        Schedules, gradient utilities, checkpoints, loop helpers
  data/         Text loading and batching
  tokenizers/   Tokenizer implementations
examples/
  nanogpt/      Committed nanoGPT example and regression surface
  qwen3_5-image/ Dedicated Qwen 3.5 / Qwen 3.6 multimodal image example
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
- serving and deployment ergonomics beyond the current local/runtime surfaces
- diffusion and broader generative-media families beyond the first multimodal transformer tranche

Those come later in the roadmap once the current package ergonomics are fully settled.
