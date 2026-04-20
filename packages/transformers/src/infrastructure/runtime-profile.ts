const ENABLED_FLAG = "MLXTS_RUNTIME_PROFILE";

const COUNTER_NAMES = [
  "cache.append_full_state",
  "cache.sliding_single_token",
  "cache.sliding_existing_capacity",
  "cache.sliding_growth",
  "cache.sliding_merge",
  "cache.write_range",
  "cache.return_prefix_view",
  "cache.return_tail_view",
  "cache.return_full_buffer",
  "cache.return_ordered_concat_view",
  "cache.buffer_replaced",
  "attention.mask_created",
  "attention.mask_borrowed",
] as const;

export type TransformerRuntimeCounter = (typeof COUNTER_NAMES)[number];

export type TransformerRuntimeProfileSnapshot = {
  enabled: boolean;
  counters: Record<TransformerRuntimeCounter, number>;
};

const counters: Record<TransformerRuntimeCounter, number> = {
  "cache.append_full_state": 0,
  "cache.sliding_single_token": 0,
  "cache.sliding_existing_capacity": 0,
  "cache.sliding_growth": 0,
  "cache.sliding_merge": 0,
  "cache.write_range": 0,
  "cache.return_prefix_view": 0,
  "cache.return_tail_view": 0,
  "cache.return_full_buffer": 0,
  "cache.return_ordered_concat_view": 0,
  "cache.buffer_replaced": 0,
  "attention.mask_created": 0,
  "attention.mask_borrowed": 0,
};

export function isTransformerRuntimeProfilingEnabled(): boolean {
  return process.env[ENABLED_FLAG] === "1";
}

export function resetTransformerRuntimeProfile(): void {
  for (const name of COUNTER_NAMES) {
    counters[name] = 0;
  }
}

export function recordTransformerRuntimeCounter(name: TransformerRuntimeCounter, delta = 1): void {
  if (!isTransformerRuntimeProfilingEnabled()) {
    return;
  }
  counters[name] += delta;
}

export function snapshotTransformerRuntimeProfile(): TransformerRuntimeProfileSnapshot {
  return {
    enabled: isTransformerRuntimeProfilingEnabled(),
    counters: { ...counters },
  };
}
