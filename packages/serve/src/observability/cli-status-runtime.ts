/**
 * Runtime strategy and limit formatting for the finite status CLI.
 * @module
 */

import type { ServeInfoResponse } from "../http/route-info";

function toon(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

/** Format full serve admission limits for the status CLI. */
export function formatLimits(info: ServeInfoResponse): string[] {
  const limits = info.limits;
  return [
    "limits:",
    `  max_generated_tokens: ${toon(limits.max_generated_tokens)}`,
    `  max_prompt_tokens: ${toon(limits.max_prompt_tokens)}`,
    `  max_total_tokens: ${toon(limits.max_total_tokens)}`,
    `  max_client_batch_size: ${toon(limits.max_client_batch_size)}`,
    `  batch_window_ms: ${toon(limits.batch_window_ms)}`,
    `  prefill_step_size: ${toon(limits.prefill_step_size)}`,
    `  active_prefill_step_size: ${toon(limits.active_prefill_step_size)}`,
    `  active_decode_steps_per_prefill_chunk: ${toon(
      limits.active_decode_steps_per_prefill_chunk,
    )}`,
    `  stream_decode_interval: ${toon(limits.stream_decode_interval)}`,
    `  max_concurrent_requests: ${toon(limits.max_concurrent_requests)}`,
    `  prompt_prefix_cache_max_entries: ${toon(limits.prompt_prefix_cache_max_entries)}`,
    `  prompt_prefix_cache_max_bytes: ${toon(limits.prompt_prefix_cache_max_bytes)}`,
    `  gpu_memory_utilization: ${toon(limits.gpu_memory_utilization)}`,
  ];
}

/** Format compact serve admission limits for the default status view. */
export function formatDefaultLimits(info: ServeInfoResponse): string[] {
  const limits = info.limits;
  return [
    "limits:",
    `  max_generated_tokens: ${toon(limits.max_generated_tokens)}`,
    `  max_prompt_tokens: ${toon(limits.max_prompt_tokens)}`,
    `  max_total_tokens: ${toon(limits.max_total_tokens)}`,
    `  gpu_memory_utilization: ${toon(limits.gpu_memory_utilization)}`,
  ];
}

/** Format runtime strategy details for the full status view. */
export function formatRuntimeStrategy(info: ServeInfoResponse): string[] {
  const strategy = info.runtime_strategy;
  return [
    "runtime_strategy:",
    `  scheduler: ${toon(strategy.scheduler.mode)}`,
    `  max_batch_size: ${toon(strategy.scheduler.max_batch_size)}`,
    `  cache: ${toon(`${strategy.cache.backend}/${strategy.cache.precision}`)}`,
    `  attention: ${toon(strategy.attention.backend)}`,
    `  decoding: ${toon(strategy.decoding.backend)}`,
    `  stream_decode_interval: ${toon(strategy.streaming.stream_decode_interval)}`,
    `  memory_policy: ${toon(strategy.memory.policy)}`,
    `  gpu_memory_utilization: ${toon(strategy.memory.gpu_memory_utilization)}`,
  ];
}
