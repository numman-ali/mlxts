import { describe, expect, test } from "bun:test";

import { trainLoop, validateTrainLoopConfig } from "./loop";

function loopConfig() {
  return {
    maxSteps: 5,
    learningRate: 1e-3,
    warmupSteps: 1,
    minLearningRate: 1e-4,
    evalInterval: 2,
    logInterval: 2,
  };
}

describe("trainLoop", () => {
  test("runs step, eval, and done callbacks with the configured cadence", () => {
    const stepped: number[] = [];
    const evaluations: number[] = [];
    const completed: number[] = [];

    const totalSteps = trainLoop({
      config: loopConfig(),
      runStep(step, learningRate) {
        return { step, learningRate };
      },
      evaluate(step) {
        return step * 10;
      },
      onStep(step) {
        stepped.push(step);
      },
      onEval(step, result) {
        evaluations.push(step + result);
      },
      onDone(total) {
        completed.push(total);
      },
    });

    expect(totalSteps).toBe(5);
    expect(stepped).toEqual([2, 4]);
    expect(evaluations).toEqual([22, 44]);
    expect(completed).toEqual([5]);
  });

  test("stops early when shouldStop becomes true", () => {
    let seenSteps = 0;

    const totalSteps = trainLoop({
      config: loopConfig(),
      runStep() {
        seenSteps += 1;
        return seenSteps;
      },
      shouldStop() {
        return seenSteps >= 3;
      },
    });

    expect(totalSteps).toBe(3);
  });

  test("validateTrainLoopConfig rejects invalid startStep", () => {
    expect(() =>
      validateTrainLoopConfig({
        ...loopConfig(),
        startStep: 5,
      }),
    ).toThrow("startStep must be < maxSteps");
  });
});
