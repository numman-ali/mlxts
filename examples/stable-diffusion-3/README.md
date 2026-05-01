# Stable Diffusion 3 Example

This workbook composes the package-owned SD3 runtime with tokenizer and text
encoder packages. It is the application layer for SD3 / SD3.5 prompt
conditioning and finite proof commands.

Current support covers Diffusers-style prompt conditioning:

- `tokenizer` + `text_encoder`
- `tokenizer_2` + `text_encoder_2`
- `tokenizer_3` + `text_encoder_3`
- positive and negative classifier-free guidance conditioning
- CLIP hidden-state concatenation, T5 padding alignment, and pooled CLIP
  projection concatenation

Official Stability checkpoints are gated, so real image proof remains a
separate authenticated run.
