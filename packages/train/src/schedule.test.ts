import { describe, expect, test } from "bun:test";

import { getLearningRate, validateLearningRateConfig, warmupCosineSchedule } from "./schedule";

function scheduleConfig() {
  return {
    learningRate: 1e-3,
    warmupSteps: 2,
    minLearningRate: 1e-4,
    maxSteps: 10,
  };
}

describe("schedule", () => {
  test("getLearningRate starts at zero during warmup", () => {
    expect(getLearningRate(0, scheduleConfig())).toBe(0);
  });

  test("getLearningRate reaches the peak at the end of warmup", () => {
    expect(getLearningRate(2, scheduleConfig())).toBeCloseTo(1e-3);
  });

  test("warmupCosineSchedule reaches minLearningRate at maxSteps", () => {
    const schedule = warmupCosineSchedule(scheduleConfig());
    expect(schedule(10)).toBeCloseTo(1e-4);
  });

  test("validateLearningRateConfig rejects invalid warmup ranges", () => {
    expect(() =>
      validateLearningRateConfig({
        learningRate: 1e-3,
        warmupSteps: 10,
        minLearningRate: 1e-4,
        maxSteps: 10,
      }),
    ).toThrow("warmupSteps must be < maxSteps");
  });
});
