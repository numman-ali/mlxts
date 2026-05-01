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
    expect(steps[0]?.timestep).toBeCloseTo(9);
    expect(steps[0]?.previousTimestep).toBeCloseTo(4.5);
    expect(steps[1]?.timestep).toBeCloseTo(4.5);
    expect(steps[1]?.previousTimestep).toBeCloseTo(0);
    expect(steps[2]?.timestep).toBeCloseTo(0);
    expect(steps[2]?.previousTimestep).toBeCloseTo(0);
    expect(steps[2]?.previousSigma).toBe(0);
  });

  test("creates leading and trailing Diffusers timestep spacing", () => {
    const leading = new EulerScheduler({
      numTrainTimesteps: 10,
      timestepSpacing: "leading",
      stepsOffset: 1,
    });
    const trailing = new EulerScheduler({
      numTrainTimesteps: 10,
      timestepSpacing: "trailing",
    });

    expect(leading.timesteps(3).map((step) => step.timestep)).toEqual([7, 4, 1]);
    expect(trailing.timesteps(3).map((step) => step.timestep)).toEqual([9, 6, 2]);
  });

  test("supports Diffusers final sigma policies", () => {
    const baseConfig = {
      betaSchedule: "linear" as const,
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 10,
    };
    const zero = new EulerScheduler(baseConfig);
    const sigmaMin = new EulerScheduler({ ...baseConfig, finalSigmasType: "sigma_min" });

    expect(zero.timesteps(3)[2]?.previousSigma).toBe(0);
    expect(sigmaMin.timesteps(3)[2]?.previousSigma).toBeCloseTo(0.33333333333333326);
  });

  test("scales leading-spaced prior noise from the active inference schedule", () => {
    const scheduler = new EulerScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 10,
      timestepSpacing: "leading",
      stepsOffset: 1,
    });

    expect(scheduler.timesteps(3)[0]?.sigma).toBeCloseTo(1.5229237364484844);
    expect(scheduler.initialNoiseSigma(3)).toBeCloseTo(1.8218937145284335);
  });

  test("scales model input by sigma energy", () => {
    const scheduler = new EulerScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });
    using sample = array([2, -4], "float32");
    using scaled = scheduler.scaleModelInput(sample, 1);

    scaled.eval();
    const sigma = scheduler.sigmaAt(1);
    const factor = 1 / Math.sqrt(sigma * sigma + 1);
    expectTensorClose(scaled.toTypedArray(), [2 * factor, -4 * factor]);
  });

  test("adds forward-process noise in sigma space", () => {
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
    expectTensorClose(noisy.toTypedArray(), [1 + 0.5 * sigma, 2 - 0.25 * sigma]);
  });

  test("moves one Euler step", () => {
    const scheduler = new EulerScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 10,
      timestepSpacing: "leading",
      stepsOffset: 1,
    });
    using sample = array([0.25, -0.5], "float32");
    using modelOutput = array([0.1, -0.2], "float32");
    const step = scheduler.timesteps(3)[0];
    if (step === undefined) {
      throw new Error("test expected an Euler step.");
    }
    using previous = scheduler.step(modelOutput, sample, step);

    previous.eval();
    expectTensorClose(previous.toTypedArray(), [0.19365280896604434, -0.3873056179320887]);
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
