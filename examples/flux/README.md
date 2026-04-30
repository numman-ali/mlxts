# FLUX Example

This workbook composes `@mlxts/diffusion`, `@mlxts/tokenizers`, and
`@mlxts/transformers` for FLUX prompt conditioning.

The first supported FLUX path is `FLUX.1-schnell`: CLIP produces pooled prompt
projections, T5 produces prompt sequence embeddings, and `@mlxts/diffusion`
consumes the resulting tensors through its FLUX sampling contract.

Backbone construction, transformer weight loading, and a finite image proof
command remain follow-on Phase 10 tranches.
