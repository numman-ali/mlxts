# @mlxts/serve Agent Notes

Serving is a first-class package surface, not an example. Keep protocol adapters
thin: OpenAI completions, chat completions, Responses/OpenResponses, and
Anthropic Messages should normalize into the shared `GenerationEngine` contract
without copying generation logic between protocols.

Admission controls must be explicit and operator-facing. Keep generated-token,
prompt-token, total-token, concurrency, and batching limits separate in code,
errors, CLI output, and `/info` so long-context failures explain which budget was
hit. Treat `/info` context metadata as configured admission truth, not a promise
that every advertised model window fits local memory.

Memory preflight should stay best-effort and honest. Estimate cache and prefill
memory from family config geometry, compare it with current MLX active memory
and the configured utilization budget, and skip rather than fake certainty when
the model config is not understood. A preflight pass is not a throughput
scheduler guarantee.

Do not call admission micro-batching continuous batching. True continuous
batching needs a scheduler-owned decode loop plus batch-aware cache semantics in
`@mlxts/transformers`, especially for Qwen hybrid full-attention plus recurrent
linear-attention cache state.

For serving reliability work, prefer small audited tranches: admission,
observability, cancellation, memory preflight, scheduler/cache architecture, then
wire-protocol expansion. Runtime-sensitive changes need a review artifact under
`docs/reviews/` and the usual repo gates.
