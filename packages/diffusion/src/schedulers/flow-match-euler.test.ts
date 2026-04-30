import { describe, expect, test } from "bun:test";
import { array } from "@mlxts/core";

import { calculateFlowMatchShift, FlowMatchEulerScheduler } from "./flow-match-euler";

function expectNumbersClose(
  actual: readonly number[],
  expected: readonly number[],
  digits = 6,
): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const actualValue = actual[index];
    const expectedValue = expected[index];
    if (actualValue === undefined || expectedValue === undefined) {
      throw new Error("expectNumbersClose: missing comparison value.");
    }
    expect(actualValue).toBeCloseTo(expectedValue, digits);
  }
}

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

function exponentialShift(mu: number, sigma: number): number {
  const expMu = Math.exp(mu);
  return expMu / (expMu + (1 / sigma - 1));
}

describe("FlowMatchEulerScheduler", () => {
  test("creates FLUX schnell-style sigma steps", () => {
    const scheduler = new FlowMatchEulerScheduler({
      numTrainTimesteps: 1000,
      shift: 1,
      useDynamicShifting: false,
    });

    const steps = scheduler.timesteps(4);

    expectNumbersClose(
      steps.map((step) => step.sigma),
      [1, 0.75, 0.5, 0.25],
    );
    expectNumbersClose(
      steps.map((step) => step.nextSigma),
      [0.75, 0.5, 0.25, 0],
    );
    expectNumbersClose(
      steps.map((step) => step.timestep),
      [1000, 750, 500, 250],
    );
  });

  test("applies static FlowMatch sigma shifting", () => {
    const scheduler = new FlowMatchEulerScheduler({
      numTrainTimesteps: 1000,
      shift: 3,
      useDynamicShifting: false,
    });

    const steps = scheduler.timesteps(4);

    expectNumbersClose(
      steps.map((step) => step.sigma),
      [1, 0.9, 0.75, 0.5],
    );
    expectNumbersClose(
      steps.map((step) => step.nextSigma),
      [0.9, 0.75, 0.5, 0],
    );
  });

  test("matches Flux dynamic exponential time shifting", () => {
    const mu = calculateFlowMatchShift(1024, {
      baseImageSeqLen: 256,
      maxImageSeqLen: 4096,
      baseShift: 0.5,
      maxShift: 1.15,
    });
    const scheduler = new FlowMatchEulerScheduler({
      numTrainTimesteps: 1000,
      useDynamicShifting: true,
      baseImageSeqLen: 256,
      maxImageSeqLen: 4096,
      baseShift: 0.5,
      maxShift: 1.15,
      timeShiftType: "exponential",
    });

    const steps = scheduler.timesteps(4, { imageSequenceLength: 1024 });
    const expectedSigmas = [1, 0.75, 0.5, 0.25].map((sigma) => exponentialShift(mu, sigma));

    expect(mu).toBeCloseTo(0.63, 6);
    expectNumbersClose(
      steps.map((step) => step.sigma),
      expectedSigmas,
    );
  });

  test("requires shift context when dynamic shifting is enabled", () => {
    const scheduler = new FlowMatchEulerScheduler({ useDynamicShifting: true });

    expect(() => scheduler.timesteps(4)).toThrow("mu or imageSequenceLength");
  });

  test("adds forward-process flow noise", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using sample = array([1, 2], "float32");
    using noise = array([3, -1], "float32");
    using noisy = scheduler.addNoise(sample, noise, 0.75);

    noisy.eval();
    expectTensorClose(noisy.toTypedArray(), [2.5, -0.25]);
  });

  test("moves one deterministic Euler flow step", () => {
    const scheduler = new FlowMatchEulerScheduler();
    const step = scheduler.timesteps(4)[1];
    if (step === undefined) {
      throw new Error("missing scheduler step");
    }
    using sample = array([1, 2], "float32");
    using modelOutput = array([0.5, -1], "float32");
    using previous = scheduler.step(modelOutput, sample, step);

    previous.eval();
    expectTensorClose(previous.toTypedArray(), [0.875, 2.25]);
  });

  test("retains unscaled prior and model inputs", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using noise = array([1, -1], "float32");
    using sample = array([2, -2], "float32");
    using prior = scheduler.scaleInitialNoise(noise);
    using modelInput = scheduler.scaleModelInput(sample);

    prior.eval();
    modelInput.eval();
    expectTensorClose(prior.toTypedArray(), [1, -1]);
    expectTensorClose(modelInput.toTypedArray(), [2, -2]);
  });

  test("samples plain normal prior noise with caller-selected shape", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using prior = scheduler.samplePrior([2, 3], "float32");

    prior.eval();
    expect(prior.shape).toEqual([2, 3]);
    expect(prior.dtype).toBe("float32");
  });
});
