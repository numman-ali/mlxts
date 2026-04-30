import { describe, expect, test } from "bun:test";
import { array } from "@mlxts/core";

import { EulerScheduler } from "./euler";

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

describe("EulerScheduler", () => {
  test("creates high-to-low timestep pairs", () => {
    const scheduler = new EulerScheduler({ numTrainTimesteps: 10 });

    const steps = scheduler.timesteps(3);
    expect(steps.length).toBe(3);
    expect(steps[0]?.timestep).toBeCloseTo(10);
    expect(steps[0]?.previousTimestep).toBeCloseTo(6.666666666666667);
    expect(steps[1]?.timestep).toBeCloseTo(6.666666666666667);
    expect(steps[1]?.previousTimestep).toBeCloseTo(3.333333333333333);
    expect(steps[2]?.timestep).toBeCloseTo(3.333333333333333);
    expect(steps[2]?.previousTimestep).toBeCloseTo(0);
  });

  test("scales model input by sigma energy", () => {
    const scheduler = new EulerScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });
    using sample = array([2, -4], "float32");
    using scaled = scheduler.scaleModelInput(sample, 2);

    scaled.eval();
    const sigma = scheduler.sigmaAt(2);
    const factor = 1 / Math.sqrt(sigma * sigma + 1);
    expectTensorClose(scaled.toTypedArray(), [2 * factor, -4 * factor]);
  });

  test("adds forward-process noise with normalized sigma scaling", () => {
    const scheduler = new EulerScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });
    using sample = array([1, 2], "float32");
    using noise = array([0.5, -0.25], "float32");
    using noisy = scheduler.addNoise(sample, noise, 1);

    noisy.eval();
    const sigma = scheduler.sigmaAt(1);
    const factor = 1 / Math.sqrt(sigma * sigma + 1);
    expectTensorClose(noisy.toTypedArray(), [
      (1 + 0.5 * sigma) * factor,
      (2 - 0.25 * sigma) * factor,
    ]);
  });

  test("moves one Euler step", () => {
    const scheduler = new EulerScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });
    using sample = array([0.25, -0.5], "float32");
    using modelOutput = array([0.1, -0.2], "float32");
    using previous = scheduler.step(modelOutput, sample, {
      timestep: 2,
      previousTimestep: 1,
    });

    previous.eval();
    const sigma = scheduler.sigmaAt(2);
    const previousSigma = scheduler.sigmaAt(1);
    const outputScale = 1 / Math.sqrt(previousSigma * previousSigma + 1);
    expectTensorClose(previous.toTypedArray(), [
      (Math.sqrt(sigma * sigma + 1) * 0.25 + 0.1 * (previousSigma - sigma)) * outputScale,
      (Math.sqrt(sigma * sigma + 1) * -0.5 + -0.2 * (previousSigma - sigma)) * outputScale,
    ]);
  });

  test("scales caller-provided prior noise", () => {
    const scheduler = new EulerScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });
    using noise = array([1, -1], "float32");
    using prior = scheduler.scaleInitialNoise(noise);

    prior.eval();
    expectTensorClose(prior.toTypedArray(), [scheduler.initNoiseSigma, -scheduler.initNoiseSigma]);
  });

  test("samples prior noise with caller-selected shape", () => {
    const scheduler = new EulerScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });
    using prior = scheduler.samplePrior([2, 3], "float32");

    prior.eval();
    expect(prior.shape).toEqual([2, 3]);
    expect(prior.dtype).toBe("float32");
  });
});
