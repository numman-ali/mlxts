# LTX-Video Proof CLI Review

## Files Reviewed

- `examples/ltx-video/index.ts`
- `examples/ltx-video/conditioning.ts`
- `examples/ltx-video/conditioning-runtime.ts`
- `examples/ltx-video/conditioning-result.ts`
- `examples/ltx-video/conditioning-types.ts`
- `examples/ltx-video/video-output.ts`
- `examples/ltx-video/index.test.ts`
- `examples/ltx-video/conditioning.test.ts`
- `examples/ltx-video/video-output.test.ts`
- `examples/image-proof/verify-report.ts`
- `examples/image-proof/verify-report.test.ts`
- `package.json`
- `tsconfig.phase10-examples.json`

## Reference Audit

Diffusers `LTXPipeline` uses T5 prompt embeddings and attention masks, FlowMatch
Euler timesteps, packed BCFHW video latents, classic LTX transformer denoising,
and `AutoencoderKLLTXVideo` decode into BFHWC video frames. The proof command
stays on that classic text-to-video boundary and rejects other pipeline kinds
through snapshot manifest checks.

The command deliberately does not claim `LTXConditionPipeline`, image/video
conditioning, guidance rescale, decode timestep/noise interpolation, sidecar
latent upsampling, or LTX-2 audio-video support. Those remain separate runtime
tranches because their references add distinct semantics.

## Tensor Lifetime Audit

Prompt-conditioning results own T5 hidden states and attention masks through a
disposable result object. The command keeps `using` ownership around scheduler
latents, denoised packed latents, decoded video, transformer, VAE, conditioner,
and RNG key. Preview-sheet creation slices and reshapes sampled BFHWC frames,
then frees temporary frame tensors after concatenation.

## Memory / Performance Evidence

The proof defaults are bounded for local operator use: 128x128, 9 frames, 4
steps, float16 latents, and a single BMP preview artifact. Progress remains on
stderr, structured stdout remains finite, and the runtime lock prevents
concurrent heavy MLX proof runs from contending with other model workloads.

Validated:

```bash
bunx tsc -p tsconfig.phase10-examples.json --pretty false
bun test examples/ltx-video examples/image-proof
bun run check:phase10-proofs
```

## Independent Review

Darwin reviewed the Phase 10 example patterns and recommended a classic-only
LTX proof command with AXI stdout, stderr progress, runtime-lock discipline,
T5/SentencePiece prompt conditioning, `@mlxts/diffusion` LTX snapshot/model
APIs, and a BMP preview artifact rather than a production video container.

## Remaining Risks / Follow-ups

- The official LTX checkpoint proof still needs a local authenticated/cached
  snapshot run on the target Apple Silicon machine.
- The current artifact is a sampled preview sheet, not MP4/GIF/video container
  output.
- LTX sidecar latent upsampling is implemented in `@mlxts/diffusion` but not
  composed into this first finite proof command.
- LTX-2 audio-video denoising, latent upsampling, vocoder output, and connector
  semantics remain future Phase 10 runtime tranches.
