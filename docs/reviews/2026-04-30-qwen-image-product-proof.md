# Runtime Proof: Qwen Image Product Path

## Summary

This proof exercises the real Qwen image-serving path through `@mlxts/serve`
using the cached `mlx-community/Qwen3.6-27B-4bit` checkpoint. It uses a local
generated 96x96 BMP with four colored quadrants and sends it through the Chat
Completions, OpenResponses, and Anthropic Messages image routes.

This tranche adds a repeatable regression harness and does not change model or
runtime production files. The goal is product evidence: protocol media
transport, host image decode, Qwen image prompt preparation, single-request
media routing, prompt-prefix media cache reuse, and visible model output all
work together on the real checkpoint.

## Files Reviewed

- `packages/serve/scripts/regression-qwen-image.ts`
- `packages/serve/scripts/regression-qwen-image.test.ts`
- `packages/serve/src/engine/content.ts`
- `packages/serve/src/engine/index.ts`
- `packages/serve/src/media/image.ts`
- `packages/serve/src/protocols/openai-chat-messages.ts`
- `packages/serve/src/protocols/openai-responses-input.ts`
- `packages/serve/src/protocols/openai-chat-completions.ts`
- `packages/serve/src/protocols/openai-responses.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/conditional.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/preprocessing.ts`

## Harness

The repeatable proof command is:

```bash
bun run regression:qwen-image -- \
  --qwen-model mlx-community/Qwen3.6-27B-4bit \
  --report-dir .tmp/qwen-image-regression
```

The harness starts a local Qwen image-capable server with cached files by
default, sends cold and repeated image requests for all three advertised image
protocols, and writes `.tmp/qwen-image-regression/qwen-image-regression.json`.
It fails if any request has empty visible output, misses prompt preparation,
routes anywhere other than `single:media_input`, touches continuous scheduling,
or fails to read from the prompt-prefix cache on exact repeats.

Latest harness run:

- `openai-chat-cold`: HTTP 200, `2117.9ms`, route `single:media_input`,
  `cache_write_tokens=92`, `output_chars=166`
- `openai-chat-repeat`: HTTP 200, `1564.3ms`, route `single:media_input`,
  `cache_read_tokens=92`, `output_chars=166`
- `openai-responses-repeat`: HTTP 200, `1559.1ms`, route
  `single:media_input`, `cache_read_tokens=92`, `output_chars=166`
- `anthropic-messages-repeat`: HTTP 200, `1565.6ms`, route
  `single:media_input`, `cache_read_tokens=92`, `output_chars=166`

Every harness request emitted one prompt-preparation completion event and zero
continuous scheduler phases. The visible answer for each protocol was a concise
description of the four quadrants: red top-left, green top-right, blue
bottom-left, yellow bottom-right, separated by black lines.

## Evidence

Server launch:

```bash
bun run packages/serve/src/cli.ts \
  --model mlx-community/Qwen3.6-27B-4bit \
  --port 8000 \
  --api-key mlxts \
  --local-files-only \
  --max-generated-tokens 64 \
  --max-prompt-tokens 262144 \
  --max-total-tokens 262144 \
  --gpu-memory-utilization 0.85 \
  --max-batch-size 2 \
  --max-concurrent-requests 1 \
  --prompt-prefix-cache-max-entries 2 \
  --verbose
```

The server loaded the cached 16.1 GB snapshot and advertised:

- `chat_completions: "text_and_image_when_supported"`
- `responses: "text_and_image_when_supported"`
- `anthropic_messages: "text_and_image_when_supported"`
- `context_window: 262144`

The first Chat Completions probe used the default thinking mode and showed why
short visual-description calls need either more output budget or explicit
thinking-off controls. It returned HTTP 200 and spent the 48-token cap inside
`reasoning_content`, with no visible `message.content`.

The product probe used:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  },
  "max_tokens": 48,
  "temperature": 0
}
```

Cold Chat Completions request:

- HTTP 200
- total time `1.836070s`
- route `single`, reason `media_input`
- prompt prep `33.1ms`
- prompt tokens `93`
- prompt cache read/write: `cached_tokens=90`, `cache_write_tokens=2`
- finish reason `stop`
- visible answer: "The image displays a simple 2x2 grid layout with four
  solid-colored squares: red (top-left), green (top-right), blue
  (bottom-left), and yellow (bottom-right), separated by thin black lines."

Exact repeated Chat Completions request:

- HTTP 200
- total time `1.681337s`
- prompt tokens `93`
- prompt cache read/write: `cached_tokens=92`, `cache_write_tokens=0`
- finish reason `stop`
- visible answer matched the cold request.

OpenResponses image request:

- HTTP 200
- total time `1.933174s`
- status `completed`
- input tokens `93`
- cached input tokens `92`
- output tokens `42`
- visible `output_text`: "A square divided into four equal quadrants by black
  lines, each filled with a solid primary color: red (top-left), green
  (top-right), blue (bottom-left), and yellow (bottom-right)."

Anthropic Messages uses the same media content path. The harness includes the
Anthropic route with `thinking: { "type": "disabled" }` and asserts visible text
plus the same `single:media_input` route decision and repeat-cache read.

## Boundaries

Media requests correctly stay off continuous batching:

```text
route=single eligible=no reason=media_input model_type=qwen3_5
```

Prompt-prefix reuse remains media-aware. The repeated request only hit after
the same image bytes and text prompt were used, and the repeated Chat request
reported `cached_tokens=92`.

This is not a multimodal batching proof, remote image transport proof,
high-throughput image-decoder proof, or Pi image-client proof. It proves that
the real local server product path can answer image questions through the
protocol surfaces already advertised by `/info`.

## Remaining Risks / Follow-ups

Short Qwen visual-description requests with thinking enabled can finish inside
reasoning content before emitting visible content. Product clients should set
`enable_thinking: false` for short descriptive image calls, or request a larger
output budget when reasoning traces are desired.

The current image decode path still uses local host decoding and request-local
prepared tensors. Future multimodal batching needs a transformer-owned prepared
prompt batching contract rather than forcing media inputs through text-only
continuous scheduling.
