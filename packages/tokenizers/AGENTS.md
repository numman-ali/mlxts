# @mlxts/tokenizers

Zero internal dependencies. `@mlxts/core`, `@mlxts/nn`, and any model package are off-limits. The `Tokenizer` interface is the contract every consumer codes against.

Each tokenizer class is self-contained — `BPETokenizer`, `SentencePieceTokenizer`, `CharTokenizer`. `loadTokenizer` is pure dispatch over `TokenizerFormat`. New formats arrive as siblings, never as additions to the `Tokenizer` interface.

BPE longest-match scanning is bounded by the maximum vocab token length, not by the remaining prompt length. Bounding by remaining prompt length is O(n²) on long prompts.

Hugging Face `tokenizer.json` added tokens use sparse IDs above the base vocab size. BPE decode tables register `added_tokens` by ID, not just base vocab entries.

Tekken is its own format with its own loader (`loadTekken`, `loadTekkenJson`). Collapsing Tekken into the BPE loader is forbidden.

`UnsupportedTokenizerError` is the boundary error for loaders. Hitting an unsupported configuration throws with full `format` and `reason` context. Silent degradation is forbidden.
