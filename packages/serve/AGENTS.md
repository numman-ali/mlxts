# @mlxts/serve Agent Notes

Serving is a first-class package surface, not an example. Keep protocol adapters
thin: OpenAI completions, chat completions, OpenResponses, and
Anthropic Messages should normalize into the shared `GenerationEngine` contract
without copying generation logic between protocols.

`/v1/responses` is an OpenResponses-spec surface, not an OpenAI Responses clone.
Keep OpenAI wording only where a compatibility client or wire shape requires
it, and check the OpenResponses OpenAPI/source guidance before widening response
items, tools, files, state, or streaming events.

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
`@mlxts/transformers`. That route is now real for eligible LLaMA-like, Qwen
3.6 text, and Gemma 3/4 layer-pattern requests; keep claims tied to the exact
route evidence and do not imply prefix/paged cache or multimodal batching.

For full-KV continuous batching, long waiting prompts must be chunk-prefilled
between active decode steps rather than forwarded as one admission wall. For
Qwen/Gemma routes, preserve the real regression guardrails that assert
per-request TTFT, scheduler queue time, SSE lifecycle, and mixed long-prefill /
short-arrival fairness separately from aggregate throughput.

For serving reliability work, prefer small audited tranches: admission,
observability, cancellation, memory preflight, scheduler/cache architecture, then
wire-protocol expansion. Runtime-sensitive changes need a review artifact under
`docs/reviews/` and the usual repo gates.
