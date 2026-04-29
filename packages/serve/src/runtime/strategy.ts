/**
 * Runtime strategy description for model serving.
 * @module
 */

export type ServeRuntimeKnobs = {
  maxBatchSize?: number;
  batchWindowMs?: number;
  prefillStepSize?: number;
  activePrefillStepSize?: number;
  activeDecodeStepsPerPrefillChunk?: number;
  streamDecodeInterval?: number;
  maxConcurrentRequests?: number;
  promptPrefixCacheMaxEntries?: number;
  gpuMemoryUtilization?: number;
};

export type ServeRuntimeDefaults = {
  maxBatchSize: number;
  batchWindowMs: number;
  prefillStepSize: number;
  activePrefillStepSize: number;
  activeDecodeStepsPerPrefillChunk: number;
  streamDecodeInterval: number;
  maxConcurrentRequests: number;
  promptPrefixCacheMaxEntries: number;
  gpuMemoryUtilization?: number;
};

export type ServeSchedulerStrategy = {
  mode: "auto";
  maxBatchSize: number;
  batchWindowMs: number;
  prefillStepSize: number;
  activePrefillStepSize: number;
  activeDecodeStepsPerPrefillChunk: number;
  maxConcurrentRequests: number;
};

export type ServeCacheStrategy = {
  backend: "managed";
  precision: "model";
  promptPrefixMaxEntries: number;
};

export type ServeAttentionStrategy = {
  backend: "auto";
};

export type ServeDecodingStrategy = {
  backend: "model";
};

export type ServeStreamingStrategy = {
  decodeInterval: number;
};

export type ServeMemoryStrategy =
  | {
      policy: "none";
    }
  | {
      policy: "admit_only";
      gpuMemoryUtilization: number;
    };

export type ServeRuntimeStrategy = {
  scheduler: ServeSchedulerStrategy;
  cache: ServeCacheStrategy;
  attention: ServeAttentionStrategy;
  decoding: ServeDecodingStrategy;
  streaming: ServeStreamingStrategy;
  memory: ServeMemoryStrategy;
};

export type ServeRuntimeStrategyInfo = {
  scheduler: {
    mode: "auto";
    max_batch_size: number;
    batch_window_ms: number;
    prefill_step_size: number;
    active_prefill_step_size: number;
    active_decode_steps_per_prefill_chunk: number;
    max_concurrent_requests: number;
  };
  cache: {
    backend: "managed";
    precision: "model";
    prompt_prefix_max_entries: number;
  };
  attention: {
    backend: "auto";
  };
  decoding: {
    backend: "model";
  };
  streaming: {
    stream_decode_interval: number;
  };
  memory:
    | {
        policy: "none";
        gpu_memory_utilization: null;
      }
    | {
        policy: "admit_only";
        gpu_memory_utilization: number;
      };
};

export const DEFAULT_SERVE_PREFILL_STEP_SIZE = 512;
export const DEFAULT_SERVE_PROMPT_PREFIX_CACHE_MAX_ENTRIES = 1;

export const TRANSFORMERS_ENGINE_RUNTIME_DEFAULTS: ServeRuntimeDefaults = {
  maxBatchSize: 1,
  batchWindowMs: 0,
  prefillStepSize: DEFAULT_SERVE_PREFILL_STEP_SIZE,
  activePrefillStepSize: 128,
  activeDecodeStepsPerPrefillChunk: 16,
  streamDecodeInterval: 1,
  maxConcurrentRequests: 1,
  promptPrefixCacheMaxEntries: DEFAULT_SERVE_PROMPT_PREFIX_CACHE_MAX_ENTRIES,
};

export function requirePositiveInteger(name: string, value: number): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return value;
}

export function requireNonNegativeInteger(name: string, value: number): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer.`);
  }
  return value;
}

export function requirePositiveFraction(name: string, value: number): number {
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    throw new Error(`${name} must be a number greater than 0 and less than or equal to 1.`);
  }
  return value;
}

function runtimeValue(
  value: number | undefined,
  defaultValue: number,
  validate: (name: string, value: number) => number,
  name: string,
): number {
  return validate(name, value ?? defaultValue);
}

function memoryStrategy(gpuMemoryUtilization: number | undefined): ServeMemoryStrategy {
  if (gpuMemoryUtilization === undefined) {
    return { policy: "none" };
  }
  return {
    policy: "admit_only",
    gpuMemoryUtilization: requirePositiveFraction("gpuMemoryUtilization", gpuMemoryUtilization),
  };
}

/** Resolve loose serving knobs into the selected runtime strategy. */
export function resolveServeRuntimeStrategy(
  options: ServeRuntimeKnobs,
  defaults: ServeRuntimeDefaults,
): ServeRuntimeStrategy {
  const scheduler: ServeSchedulerStrategy = {
    mode: "auto",
    maxBatchSize: runtimeValue(
      options.maxBatchSize,
      defaults.maxBatchSize,
      requirePositiveInteger,
      "maxBatchSize",
    ),
    batchWindowMs: runtimeValue(
      options.batchWindowMs,
      defaults.batchWindowMs,
      requireNonNegativeInteger,
      "batchWindowMs",
    ),
    prefillStepSize: runtimeValue(
      options.prefillStepSize,
      defaults.prefillStepSize,
      requirePositiveInteger,
      "prefillStepSize",
    ),
    activePrefillStepSize: runtimeValue(
      options.activePrefillStepSize,
      defaults.activePrefillStepSize,
      requirePositiveInteger,
      "activePrefillStepSize",
    ),
    activeDecodeStepsPerPrefillChunk: runtimeValue(
      options.activeDecodeStepsPerPrefillChunk,
      defaults.activeDecodeStepsPerPrefillChunk,
      requirePositiveInteger,
      "activeDecodeStepsPerPrefillChunk",
    ),
    maxConcurrentRequests: runtimeValue(
      options.maxConcurrentRequests,
      defaults.maxConcurrentRequests,
      requirePositiveInteger,
      "maxConcurrentRequests",
    ),
  };
  const streaming: ServeStreamingStrategy = {
    decodeInterval: runtimeValue(
      options.streamDecodeInterval,
      defaults.streamDecodeInterval,
      requirePositiveInteger,
      "streamDecodeInterval",
    ),
  };
  return {
    scheduler,
    cache: {
      backend: "managed",
      precision: "model",
      promptPrefixMaxEntries: runtimeValue(
        options.promptPrefixCacheMaxEntries,
        defaults.promptPrefixCacheMaxEntries,
        requirePositiveInteger,
        "promptPrefixCacheMaxEntries",
      ),
    },
    attention: {
      backend: "auto",
    },
    decoding: {
      backend: "model",
    },
    streaming,
    memory: memoryStrategy(options.gpuMemoryUtilization ?? defaults.gpuMemoryUtilization),
  };
}

/** Resolve transformer-engine defaults for direct engine construction. */
export function transformersRuntimeStrategy(options: ServeRuntimeKnobs): ServeRuntimeStrategy {
  return resolveServeRuntimeStrategy(options, TRANSFORMERS_ENGINE_RUNTIME_DEFAULTS);
}

/** Format the selected strategy for `/info`. */
export function formatServeRuntimeStrategyInfo(
  strategy: ServeRuntimeStrategy,
): ServeRuntimeStrategyInfo {
  return {
    scheduler: {
      mode: strategy.scheduler.mode,
      max_batch_size: strategy.scheduler.maxBatchSize,
      batch_window_ms: strategy.scheduler.batchWindowMs,
      prefill_step_size: strategy.scheduler.prefillStepSize,
      active_prefill_step_size: strategy.scheduler.activePrefillStepSize,
      active_decode_steps_per_prefill_chunk: strategy.scheduler.activeDecodeStepsPerPrefillChunk,
      max_concurrent_requests: strategy.scheduler.maxConcurrentRequests,
    },
    cache: {
      backend: strategy.cache.backend,
      precision: strategy.cache.precision,
      prompt_prefix_max_entries: strategy.cache.promptPrefixMaxEntries,
    },
    attention: {
      backend: strategy.attention.backend,
    },
    decoding: {
      backend: strategy.decoding.backend,
    },
    streaming: {
      stream_decode_interval: strategy.streaming.decodeInterval,
    },
    memory:
      strategy.memory.policy === "none"
        ? {
            policy: "none",
            gpu_memory_utilization: null,
          }
        : {
            policy: "admit_only",
            gpu_memory_utilization: strategy.memory.gpuMemoryUtilization,
          },
  };
}
