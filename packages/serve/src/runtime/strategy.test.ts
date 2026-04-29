import { describe, expect, test } from "bun:test";

import {
  DEFAULT_MODEL_SERVER_PREFILL_STEP_SIZE,
  DEFAULT_MODEL_SERVER_PROMPT_PREFIX_CACHE_MAX_ENTRIES,
} from "../model-loading/server";
import {
  DEFAULT_SERVE_PREFILL_STEP_SIZE,
  DEFAULT_SERVE_PROMPT_PREFIX_CACHE_MAX_ENTRIES,
  resolveServeRuntimeStrategy,
  TRANSFORMERS_ENGINE_RUNTIME_DEFAULTS,
} from "./strategy";

describe("serve runtime strategy defaults", () => {
  test("uses one cold prefill chunk-size default across model serving and engine strategy", () => {
    expect(DEFAULT_MODEL_SERVER_PREFILL_STEP_SIZE).toBe(DEFAULT_SERVE_PREFILL_STEP_SIZE);
    expect(TRANSFORMERS_ENGINE_RUNTIME_DEFAULTS.prefillStepSize).toBe(
      DEFAULT_SERVE_PREFILL_STEP_SIZE,
    );
    expect(
      resolveServeRuntimeStrategy({}, TRANSFORMERS_ENGINE_RUNTIME_DEFAULTS).scheduler
        .prefillStepSize,
    ).toBe(DEFAULT_SERVE_PREFILL_STEP_SIZE);
  });

  test("uses one prompt-prefix cache retention default across serving and engine strategy", () => {
    expect(DEFAULT_MODEL_SERVER_PROMPT_PREFIX_CACHE_MAX_ENTRIES).toBe(
      DEFAULT_SERVE_PROMPT_PREFIX_CACHE_MAX_ENTRIES,
    );
    expect(TRANSFORMERS_ENGINE_RUNTIME_DEFAULTS.promptPrefixCacheMaxEntries).toBe(
      DEFAULT_SERVE_PROMPT_PREFIX_CACHE_MAX_ENTRIES,
    );
    expect(
      resolveServeRuntimeStrategy({}, TRANSFORMERS_ENGINE_RUNTIME_DEFAULTS).cache
        .promptPrefixMaxEntries,
    ).toBe(DEFAULT_SERVE_PROMPT_PREFIX_CACHE_MAX_ENTRIES);
  });
});
