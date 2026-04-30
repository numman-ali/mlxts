import { describe, expect, test } from "bun:test";
import { array } from "@mlxts/core";

import { DDIMScheduler } from "./ddim";

function expectTensorClose(
  actual: ArrayLike<number>,
  expected: readonly number[],
  digits = 5,
): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const actualValue = actual[index];
    const expectedValue = expected[index];
    if (actualValue === undefined || expectedValue === undefined) {
      throw new Error("expectTensorClose: missing comparison value.");
    }
    expect(actualValue).toBeCloseTo(expectedValue, digits);
  }
}

describe("DDIMScheduler", () => {
  test("creates leading timesteps and paired steps", () => {
    const scheduler = new DDIMScheduler({ numTrainTimesteps: 10 });

    expect(scheduler.numTrainTimesteps).toBe(10);
    expect(scheduler.timesteps(5)).toEqual([8, 6, 4, 2, 0]);
    expect(scheduler.steps(5)).toEqual([
      { timestep: 8, previousTimestep: 6 },
      { timestep: 6, previousTimestep: 4 },
      { timestep: 4, previousTimestep: 2 },
      { timestep: 2, previousTimestep: 0 },
      { timestep: 0, previousTimestep: -2 },
    ]);
  });

  test("variance follows the DDIM formula", () => {
    const scheduler = new DDIMScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 4,
    });

    const alphaT = scheduler.requireAlphaCumprod(3);
    const alphaPrev = scheduler.requireAlphaCumprod(1);
    const expected = ((1 - alphaPrev) / (1 - alphaT)) * (1 - alphaT / alphaPrev);
    expect(scheduler.variance(3, 1)).toBeCloseTo(expected);
  });

  test("adds forward-process noise", () => {
    const scheduler = new DDIMScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
      clipSample: false,
    });
    using original = array([1, -2], "float32");
    using noise = array([0.5, 0.25], "float32");
    using noisy = scheduler.addNoise(original, noise, 1);

    noisy.eval();
    const alpha = scheduler.requireAlphaCumprod(1);
    expect(scheduler.scaleModelInput(noisy)).toBe(noisy);
    expectTensorClose(noisy.toTypedArray(), [
      Math.sqrt(alpha) * 1 + Math.sqrt(1 - alpha) * 0.5,
      Math.sqrt(alpha) * -2 + Math.sqrt(1 - alpha) * 0.25,
    ]);
  });

  test("rejects invalid clip range and timestep reads", () => {
    expect(() => new DDIMScheduler({ clipSampleRange: 0 })).toThrow("clipSampleRange");
    const scheduler = new DDIMScheduler({ numTrainTimesteps: 2 });
    expect(() => scheduler.requireAlphaCumprod(2)).toThrow("outside the alpha schedule");
  });

  test("moves one deterministic DDIM step", () => {
    const scheduler = new DDIMScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 4,
      clipSample: false,
    });
    using sample = array([0.25, -0.5], "float32");
    using modelOutput = array([0.1, -0.2], "float32");
    const output = scheduler.step(modelOutput, sample, { timestep: 3, previousTimestep: 1 });
    try {
      output.prevSample.eval();
      output.predOriginalSample.eval();

      const alphaT = scheduler.requireAlphaCumprod(3);
      const alphaPrev = scheduler.requireAlphaCumprod(1);
      const betaT = 1 - alphaT;
      const firstPredOriginal = (0.25 - Math.sqrt(betaT) * 0.1) / Math.sqrt(alphaT);
      const secondPredOriginal = (-0.5 - Math.sqrt(betaT) * -0.2) / Math.sqrt(alphaT);
      const firstPrevious =
        Math.sqrt(alphaPrev) * firstPredOriginal + Math.sqrt(1 - alphaPrev) * 0.1;
      const secondPrevious =
        Math.sqrt(alphaPrev) * secondPredOriginal + Math.sqrt(1 - alphaPrev) * -0.2;

      expectTensorClose(output.predOriginalSample.toTypedArray(), [
        firstPredOriginal,
        secondPredOriginal,
      ]);
      expectTensorClose(output.prevSample.toTypedArray(), [firstPrevious, secondPrevious]);
    } finally {
      output.prevSample.free();
      output.predOriginalSample.free();
    }
  });

  test("clips predicted original samples when requested", () => {
    const scheduler = new DDIMScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
      clipSample: true,
      clipSampleRange: 0.5,
    });
    using sample = array([4, -4], "float32");
    using modelOutput = array([0, 0], "float32");
    const output = scheduler.step(modelOutput, sample, { timestep: 1, previousTimestep: 0 });
    try {
      output.predOriginalSample.eval();
      expectTensorClose(output.predOriginalSample.toTypedArray(), [0.5, -0.5]);
    } finally {
      output.prevSample.free();
      output.predOriginalSample.free();
    }
  });
});
