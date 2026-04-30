import { describe, expect, test } from "bun:test";

import {
  interpolateSchedule,
  linspace,
  makeAlphaCumprodSchedule,
  makeBetaSchedule,
  makeDiscreteTimesteps,
  makeSigmaSchedule,
} from "./schedule";

describe("diffusion schedules", () => {
  test("linear beta schedule uses evenly spaced endpoints", () => {
    const betas = makeBetaSchedule({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.4,
      numTrainTimesteps: 4,
    });

    expect(Array.from(betas)).toEqual([
      expect.closeTo(0.1),
      expect.closeTo(0.2),
      expect.closeTo(0.3),
      expect.closeTo(0.4),
    ]);
  });

  test("scaled linear beta schedule squares sqrt-spaced endpoints", () => {
    const betas = makeBetaSchedule({
      betaSchedule: "scaled_linear",
      betaStart: 0.01,
      betaEnd: 0.04,
      numTrainTimesteps: 3,
    });

    const values = Array.from(betas);
    expect(values[0]).toBeCloseTo(0.01);
    expect(values[1]).toBeCloseTo(0.0225);
    expect(values[2]).toBeCloseTo(0.04);
  });

  test("alpha cumulative product and sigmas follow the diffusion schedule", () => {
    const alphas = makeAlphaCumprodSchedule({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });
    const sigmas = makeSigmaSchedule({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });

    expect(Array.from(alphas)).toEqual([0.9, 0.7200000000000001]);
    expect(sigmas[0]).toBe(0);
    expect(sigmas[1]).toBeCloseTo(Math.sqrt(0.1 / 0.9));
    expect(sigmas[2]).toBeCloseTo(Math.sqrt(0.28 / 0.72));
  });

  test("interpolation reads fractional schedule positions", () => {
    const schedule = Float64Array.of(0, 2, 4);

    expect(interpolateSchedule(schedule, 0)).toBe(0);
    expect(interpolateSchedule(schedule, 1.5)).toBe(3);
    expect(() => interpolateSchedule(schedule, 3)).toThrow("outside the schedule");
  });

  test("timestep spacings match diffusers-compatible recipes", () => {
    expect(makeDiscreteTimesteps(5, 10, "leading")).toEqual([8, 6, 4, 2, 0]);
    expect(makeDiscreteTimesteps(5, 10, "trailing")).toEqual([9, 7, 5, 3, 1]);
    expect(makeDiscreteTimesteps(3, 10, "linspace")).toEqual([9, 5, 0]);
  });

  test("linspace validates count", () => {
    expect(Array.from(linspace(1, 3, 3))).toEqual([1, 2, 3]);
    expect(Array.from(linspace(5, 7, 1))).toEqual([5]);
    expect(() => linspace(0, 1, 0)).toThrow("count");
  });

  test("schedule config rejects invalid ranges", () => {
    expect(() => makeBetaSchedule({ numTrainTimesteps: 0 })).toThrow("numTrainTimesteps");
    expect(() => makeBetaSchedule({ betaStart: 0 })).toThrow("betaStart");
    expect(() => makeBetaSchedule({ betaEnd: 0 })).toThrow("betaEnd");
    expect(() => makeBetaSchedule({ betaStart: 0.3, betaEnd: 0.2 })).toThrow("betaEnd");
    expect(() => makeDiscreteTimesteps(11, 10, "leading")).toThrow("cannot exceed");
    expect(() => interpolateSchedule(Float64Array.of(1), Number.NaN)).toThrow("finite");
  });
});
