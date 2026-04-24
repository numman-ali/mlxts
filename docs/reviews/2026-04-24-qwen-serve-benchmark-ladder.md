# Qwen 3.6 Serving Benchmark Ladder

Date: 2026-04-24

This records the first post-harness serving ladder using explicit `bench:serve --rungs`
and JSON reports. The goal was to separate endpoint health, decode parity,
long-output stability, long-context capability, and real batching evidence.

## Commands

```bash
bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local \
  --rungs 128x128@1,1024x512@1,5000x128@1,10000x128@1 \
  --trials 1 --greedy --ignore-eos --stream \
  --report-json .tmp/qwen36-serve-ladder-c1-stream.json \
  --max-batch-size 8 --batch-window-ms 2 --max-concurrent-requests 1 \
  --gpu-memory-utilization 0.85

bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local \
  --rungs 1024x1024@1,1024x2048@1 \
  --trials 1 --greedy --ignore-eos --stream \
  --report-json .tmp/qwen36-serve-output-c1-stream.json \
  --max-batch-size 8 --batch-window-ms 2 --max-concurrent-requests 1 \
  --max-prompt-tokens 1024 --max-total-tokens 3072 \
  --gpu-memory-utilization 0.85 --request-timeout-ms 3600000

bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local \
  --rungs 32768x128@1,65536x128@1 \
  --trials 1 --greedy --ignore-eos --stream \
  --report-json .tmp/qwen36-serve-long-context-stream.json \
  --max-batch-size 8 --batch-window-ms 2 --max-concurrent-requests 1 \
  --max-prompt-tokens 65536 --max-total-tokens 65664 \
  --gpu-memory-utilization 0.85 --request-timeout-ms 3600000

bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local \
  --rungs 131072x128@1 \
  --trials 1 --greedy --ignore-eos --stream \
  --report-json .tmp/qwen36-serve-128k-stream.json \
  --max-batch-size 8 --batch-window-ms 2 --max-concurrent-requests 1 \
  --max-prompt-tokens 131072 --max-total-tokens 131200 \
  --gpu-memory-utilization 0.85 --request-timeout-ms 3600000

bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit \
  --prompt-tokens 1024 --generation-tokens 128 --trials 1 \
  --memory-sample-interval 16 --require-mlx-lm-reference \
  --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python

bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit \
  --prompt-tokens 10000 --generation-tokens 128 --trials 1 \
  --memory-sample-interval 16 --require-mlx-lm-reference \
  --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python

bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit \
  --prompt-tokens 1024 --generation-tokens 1024 --trials 1 \
  --memory-sample-interval 64 --require-mlx-lm-reference \
  --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python

bun run bench:generation:context --model mlx-community/Qwen3.6-27B-4bit \
  --rungs 32768 --needle-placements all --generation-tokens 24 \
  --prefill-step-size 2048 \
  --report-json .tmp/qwen36-context-32k-all-needles.json

bun run bench:serve --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --model-id llama-local --rungs 32x32@2,128x32@2 \
  --trials 1 --no-warmup --greedy --ignore-eos \
  --request-stagger-ms 25 \
  --report-json .tmp/llama-serve-stagger-smoke.json \
  --max-concurrent-requests 1 --max-batch-size 4 --batch-window-ms 1 \
  --max-prompt-tokens 128 --max-total-tokens 160 \
  --gpu-memory-utilization 0.85

bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit \
  --model-id qwen-local --rungs 128x128@2,1024x128@2 \
  --trials 1 --no-warmup --greedy --ignore-eos --stream \
  --request-stagger-ms 100 \
  --report-json .tmp/qwen36-serve-stagger-c2-stream.json \
  --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 \
  --max-prompt-tokens 1024 --max-total-tokens 1152 \
  --gpu-memory-utilization 0.85 --request-timeout-ms 3600000

bun run bench:serve --model google/gemma-4-E2B-it \
  --model-id gemma-local --rungs 128x128@2,1024x128@2 \
  --trials 1 --no-warmup --greedy --ignore-eos --stream \
  --request-stagger-ms 100 \
  --report-json .tmp/gemma4-e2b-serve-stagger-c2-stream.json \
  --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 \
  --max-prompt-tokens 1024 --max-total-tokens 1152 \
  --gpu-memory-utilization 0.85 --request-timeout-ms 3600000
```

## Qwen Endpoint Results

| Rung | Wall | TTFT | Post-TTFT Decode | Peak | Active Delta | Finish |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128x128@1 | 5.0s | 1.1s | 33.068 tok/s | 18.481 GB | 0.000 GB | length |
| 1024x512@1 | 22.0s | 4.5s | 29.183 tok/s | 19.934 GB | 0.000 GB | length |
| 5000x128@1 | 25.4s | 21.1s | 29.542 tok/s | 21.298 GB | 0.000 GB | length |
| 10000x128@1 | 47.5s | 43.3s | 30.845 tok/s | 21.969 GB | 0.000 GB | length |
| 1024x1024@1 | 40.1s | 4.5s | 28.709 tok/s | 19.934 GB | 0.000 GB | length |
| 1024x2048@1 | 76.3s | 4.9s | 28.661 tok/s | 19.934 GB | 0.000 GB | length |
| 32768x128@1 | 158.2s | 153.7s | 27.664 tok/s | 25.993 GB | 0.000 GB | length |
| 65536x128@1 | 351.4s | 346.1s | 24.105 tok/s | 31.406 GB | 0.000 GB | length |
| 131072x128@1 | 865.0s | 858.5s | 19.624 tok/s | 42.543 GB | 0.000 GB | length |

The serving path is stable through 128k context on this 64 GB machine. The
honest gap is long-prefill usability: 128k completes, but the TTFT is around
14.3 minutes, and post-TTFT decode falls below the shorter-context range.

## Long-Context Retrieval

| Rung | Needle | Prompt Tokens | Needle Center | Prefill | Decode | Peak | Exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32768 | early | 32774 | 0.101 | 158.5s | 22.444 tok/s | 25.995 GB | yes |
| 32768 | middle | 32774 | 0.500 | 156.9s | 22.608 tok/s | 25.995 GB | yes |
| 32768 | late | 32774 | 0.999 | 156.8s | 22.850 tok/s | 25.995 GB | yes |

The 32k all-needle retrieval run exact-matched early, middle, and late markers
with flat active decode memory slope. This proves the new ladder can catch
position-sensitive failures, but it does not replace the older 64k/128k
late-needle capability evidence; those rungs still need all-needle runs before
claiming broad full-window recall.

## Paired MLX-LM Parity

| Rung | mlx-lm Decode | mlxts Decode | mlx-lm Peak | mlxts Peak | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024/128 | 29.135 tok/s | 29.236 tok/s | 17.022 GB | 19.934 GB | mlxts slightly faster, higher peak |
| 10000/128 | 27.332 tok/s | 27.241 tok/s | 19.219 GB | 21.969 GB | practical parity, memory just under warning ratio |
| 1024/1024 | 28.965 tok/s | 28.537 tok/s | 17.022 GB | 19.934 GB | small decode gap, higher peak |

Decode parity is strong enough that the next Qwen work should not chase random
micro-optimizations. The clearer remaining target is peak-memory parity versus
`mlx-lm`, then long-prefill scheduling/chunking.

## Controls

Gemma 4 E2B serving control:

| Rung | Wall | TTFT | Post-TTFT Decode | Peak | Active Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128x128@1 | 1.575s | 0.133s | 88.041 tok/s | 9.446 GB | 0.000 GB |
| 1024x512@1 | 6.459s | 0.215s | 81.845 tok/s | 9.892 GB | 0.000 GB |
| 5000x128@1 | 2.211s | 0.687s | 83.321 tok/s | 10.091 GB | 0.000 GB |

LLaMA 3.2 1B continuous-batching control:

| Rung | Completion TPS | Continuous Rows | Max Generation Batch | Peak |
| --- | ---: | ---: | ---: | ---: |
| 16x16@1 | 118.729 | 1 | 1 | 1.102 GB |
| 16x16@2 | 259.032 | 2 | 2 | 1.127 GB |
| 16x16@4 | 212.584 | 4 | 4 | 1.181 GB |

Staggered arrival controls:

| Model | Rung | Stagger | Wall | Completion TPS | Peak | Continuous Rows | Max Generation Batch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaMA 3.2 1B 4bit | 32x32@2 | 25 ms | 0.359s | 178.126 | 1.123 GB | 3 | 2 |
| LLaMA 3.2 1B 4bit | 128x32@2 | 25 ms | 0.283s | 226.097 | 1.230 GB | 3 | 2 |
| Qwen 3.6 27B 4bit | 128x128@2 | 100 ms | 10.056s | 25.457 | 18.481 GB | 0 | 0 |
| Qwen 3.6 27B 4bit | 1024x128@2 | 100 ms | 17.226s | 14.862 | 19.934 GB | 0 | 0 |
| Gemma 4 E2B | 128x128@2 | 100 ms | 3.234s | 79.158 | 9.446 GB | 0 | 0 |
| Gemma 4 E2B | 1024x128@2 | 100 ms | 3.373s | 75.907 | 9.892 GB | 0 | 0 |

The staggered LLaMA run proves waiting-row continuous scheduling is visible when
the model/cache contract is eligible: delayed request arrivals still merged into
a generation batch with `max_generation_batch=2`. Qwen and Gemma correctly
reported no generation batch rows, which is the desired honest result until
their hybrid/sliding cache semantics are represented in the scheduler.

During the Gemma stagger run, the first attempt stalled before the second rung
printed. The cause was benchmark-side prompt preparation: completions rungs were
building a text prompt even though the request body uses exact token-array
prompts. The harness now skips text tokenization for completions and keeps text
prompt synthesis only for chat/Responses protocol-health runs; the rerun
completed both Gemma rungs immediately.

Qwen queued concurrency control:

| Rung | Wall | Mean Request | P95 Request | Batch Rows | Max Generation Batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024x128@1 | 8.543s | 8.543s | 8.543s | 0 | 0 |
| 1024x128@2 | 17.254s | 15.164s | 17.254s | 0 | 0 |
| 1024x128@4 | 35.159s | 25.339s | 35.159s | 0 | 0 |

This is the expected distinction: Qwen can serve parallel callers safely through
serialization, but it does not yet have true hybrid-cache batched decode. The
LLaMA control proves the full-KV scheduler metrics light up when the model is
eligible.

## Next Gaps

- Qwen long-prefill serving needs better chunking/scheduling before 128k feels
  product-grade. It is capable, but not ergonomic enough.
- Qwen peak memory is still above `mlx-lm` at 1k contexts and should be profiled
  around cache representation and temporary peak accounting.
- Qwen hybrid-cache continuous batching remains separate work. Do not claim Qwen
  batching from queued concurrency.
- Protocol benchmarks now exist through `bench:serve --protocol chat|responses`;
  tool-quality benchmarks are still separate work.
- Long-context retrieval now supports early/middle/late needle placement and has
  one 32k all-needle Qwen proof. Run 64k/128k all-needle ladders before making
  broad recall claims beyond the prior late-needle Qwen evidence.
