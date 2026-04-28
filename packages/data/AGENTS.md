# @mlxts/data

Row-to-`MxArray` batching only. The package depends on `@mlxts/core`. Imports of `@mlxts/nn`, model packages, and tokenizer packages are forbidden.

`Dataset<T>` is the row-iteration contract. Streaming and iterable surfaces widen only when a second consumer needs them.

`huggingface.ts` is the HF datasets-server transport. Building a general HF Hub client here is forbidden — `@huggingface/hub` is consumed inside `@mlxts/transformers`.

Collation functions (`collateTokenSupervisionBatch`, `collatePreferenceBatch`) are pure. They produce batches of `MxArray` and return ownership to the caller. Datasets do not retain batch tensors.

Chat-template-aware example construction is out of scope. Raw rows in, ready-to-collate rows out. Tokenizer plus `ChatTemplate` composition belongs to `@mlxts/align`.

`loadJsonlDataset` and `loadHuggingFaceRowsDataset` return rows untyped (`unknown`). Downstream packages narrow rows themselves; the data package does not embed schema knowledge.
