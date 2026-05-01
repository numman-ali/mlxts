# examples/whisper

This example is a workbook for local Whisper speech-to-text proofs.

`@mlxts/transformers` owns Whisper config parsing, audio feature extraction,
encoder-decoder execution, special-token prompting, greedy transcription, and
checkpoint loading.

`@mlxts/tokenizers` owns tokenizer loading and text decoding.

This example owns local WAV decoding, cross-package composition, and finite
AXI-shaped proof commands.

The proof path accepts 16 kHz PCM or float WAV files. Broad audio container
decoding and resampling live outside this example until a package-owned media
transport exists.

Heavy model runs acquire the shared runtime command lock.

Example changes prove behavior with command tests and fixture-backed WAV tests.
Real checkpoint proof runs use a cached local Whisper snapshot when checkpoint
files are available.
