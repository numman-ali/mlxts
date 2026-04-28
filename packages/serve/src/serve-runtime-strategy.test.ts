import { describe, expect, test } from "bun:test";

import { DEFAULT_MODEL_SERVER_PREFILL_STEP_SIZE } from "./model-server";
import {
  DEFAULT_SERVE_PREFILL_STEP_SIZE,
  resolveServeRuntimeStrategy,
  TRANSFORMERS_ENGINE_RUNTIME_DEFAULTS,
} from "./serve-runtime-strategy";

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
});
