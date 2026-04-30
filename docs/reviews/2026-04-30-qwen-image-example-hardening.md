# Qwen Image Example Hardening

## Scope

Hardened the direct `examples/qwen3_5-image` workbook after the serving regression harness landed. The example now has import-safe CLI parsing, a shared runtime lock, cached/local source resolution by default, explicit Qwen thinking controls, and `--json` structured output for finite-run proof capture.

## Files Reviewed

- `examples/qwen3_5-image/AGENTS.md`
- `examples/qwen3_5-image/README.md`
- `examples/qwen3_5-image/index.ts`
- `examples/qwen3_5-image/index.test.ts`
- `examples/qwen3_5-image/image-io.ts`
- `packages/transformers/src/interaction-profile.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/conditional.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/preprocessing.ts`

## Boundary Check

The example stays a direct `@mlxts/transformers` workbook. It owns local file reads, macOS `sips` decode, and CLI policy. Checkpoint loading, chat-template rendering, Qwen multimodal prompt expansion, smart resize, and image patchification remain package-owned. Serving protocol routes, media transport policy, and prompt-prefix cache assertions remain in `@mlxts/serve` and `bun run regression:qwen-image`.

## Validation

- `bun test examples/qwen3_5-image` passed: `6 pass`.
- `bun run typecheck` passed.
- `bun run lint` passed.
- `bun run check:tensor-lifetimes` passed.
- Real cached-model proof:

```bash
bun run examples/qwen3_5-image/index.ts mlx-community/Qwen3.6-27B-4bit \
  --image .tmp/qwen-image-example/quadrants.bmp \
  --prompt "Describe the colored quadrants in one sentence." \
  --max-tokens 64 \
  --greedy \
  --json
```

The proof used cached/local-only source resolution for snapshot `c000ac2c2057`, resized the generated `96x96` BMP to `256x256`, finished with `eos`, generated `34` tokens, and returned: `The image displays four colored quadrants: red in the top-left, green in the top-right, blue in the bottom-left, and yellow in the bottom-right.`

## Out-of-scope Drift Noticed

- The example still resolves the already-local snapshot several times through sidecar loaders. This is noisy in progress output but not a correctness or download issue because all loaders receive the resolved local directory and cached/local-only option.
- `examples/qwen3_5-image` remains a one-shot local-image workbook, not a VLM chat loop, multi-image runner, remote-image transport surface, or serving compatibility harness.
