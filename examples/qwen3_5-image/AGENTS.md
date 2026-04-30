# examples/qwen3_5-image

This example is a workbook for one-shot Qwen image-conditioned generation.

Reusable checkpoint loading, chat-template compilation, Qwen prompt expansion, smart resize policy, and tensor patchification live in `@mlxts/transformers`.

Local image file reads, macOS image decode, and CLI argument parsing live in this example.

Serving protocol behavior, admission policy, prefix-cache routing, and product endpoint compatibility live in `@mlxts/serve`.

Heavy model runs acquire the shared runtime command lock.

Example changes prove behavior with fast parser/image tests and a real cached Qwen image run when checkpoint files are available.
