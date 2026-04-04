import { describe, expect, test } from "bun:test";
import { full, type MxArray, random, treeFlatten } from "@mlxts/core";
import { Module } from "@mlxts/nn";
import { AdamW } from "@mlxts/optimizers";

import { GPT_TINY, resolveConfig } from "./config";
import { prepareData } from "./data";
import { GPT } from "./model/gpt";
import { initializeGPT } from "./model/init";
import { CharTokenizer } from "./tokenizer";
import { getLearningRate, type TrainConfig, type TrainEvent, train } from "./train";

function baseTrainConfig(): TrainConfig {
  return {
    maxSteps: 10,
    batchSize: 2,
    learningRate: 1e-3,
    weightDecay: 0,
    warmupSteps: 2,
    minLearningRate: 1e-4,
    gradAccumSteps: 1,
    evalInterval: 5,
    evalSteps: 1,
    logInterval: 1,
    maxGradNorm: 1,
    seed: 42,
  };
}

function trainingFixture() {
  const text = "abcdefghijklmnopqrstuvwxyz ".repeat(120);
  const tokenizer = CharTokenizer.fromText(text);
  const tokens = tokenizer.encode(text);
  const config = resolveConfig(
    { ...GPT_TINY, nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 16, dropout: 0 },
    tokenizer.vocabSize,
  );
  const model = new GPT(config);
  random.seed(42);
  initializeGPT(model, config);
  return {
    config,
    model,
    ...prepareData(tokens, 0.9),
  };
}

function checkpointedTrainingFixture() {
  const text = "abcdefghijklmnopqrstuvwxyz ".repeat(120);
  const tokenizer = CharTokenizer.fromText(text);
  const tokens = tokenizer.encode(text);
  const config = resolveConfig(
    {
      ...GPT_TINY,
      nLayer: 2,
      nHead: 2,
      nEmbd: 32,
      blockSize: 16,
      dropout: 0,
      gradientCheckpointing: true,
    },
    tokenizer.vocabSize,
  );
  const model = new GPT(config);
  random.seed(42);
  initializeGPT(model, config);
  return {
    config,
    model,
    ...prepareData(tokens, 0.9),
  };
}

class NonFiniteLossModel extends Module {
  weight: MxArray;

  constructor() {
    super();
    this.weight = full([1], 1);
  }

  forward(input: MxArray): MxArray {
    const [batch, time] = input.shape;
    if (batch === undefined || time === undefined) {
      throw new Error("NonFiniteLossModel: expected rank-2 token input");
    }
    return full([batch, time, 2], Number.NaN);
  }
}

function snapshotParameterScalars(model: GPT): Array<{ path: string; values: Float64Array }> {
  return treeFlatten(model.parameters()).map(([path, value]) => ({
    path: path.join("."),
    values: Float64Array.from(value.toTypedArray(), Number),
  }));
}

function totalParameterDelta(
  before: Array<{ path: string; values: Float64Array }>,
  model: GPT,
): number {
  const after = treeFlatten(model.parameters());
  let total = 0;

  for (let index = 0; index < before.length; index++) {
    const baseline = before[index];
    const currentEntry = after[index];
    if (baseline === undefined || currentEntry === undefined) {
      throw new Error(`missing parameter entry at index ${index}`);
    }
    const [path, value] = currentEntry;
    const joinedPath = path.join(".");
    if (baseline.path !== joinedPath) {
      throw new Error(
        `parameter path mismatch at index ${index} (${baseline.path} vs ${joinedPath})`,
      );
    }
    const currentValues = value.toTypedArray();
    for (let valueIndex = 0; valueIndex < currentValues.length; valueIndex++) {
      const current = currentValues[valueIndex];
      const previous = baseline.values[valueIndex];
      if (current === undefined || previous === undefined) {
        throw new Error(`missing scalar at ${joinedPath}:${valueIndex}`);
      }
      total += Math.abs(current - previous);
    }
  }

  return total;
}

describe("getLearningRate", () => {
  test("LR starts at 0 during warmup", () => {
    expect(getLearningRate(0, baseTrainConfig())).toBe(0);
  });

  test("LR reaches peak at end of warmup", () => {
    expect(getLearningRate(2, baseTrainConfig())).toBeCloseTo(1e-3);
  });

  test("LR at end of training is minLR", () => {
    expect(getLearningRate(10, baseTrainConfig())).toBeCloseTo(1e-4);
  });

  test("LR is monotonically decreasing after warmup", () => {
    const config = baseTrainConfig();
    let previous = getLearningRate(config.warmupSteps, config);
    for (let step = config.warmupSteps + 1; step <= config.maxSteps; step++) {
      const current = getLearningRate(step, config);
      expect(current).toBeLessThanOrEqual(previous);
      previous = current;
    }
  });
});

describe("train", () => {
  test("returns a summary and loss decreases over a short run", () => {
    const { config, model, trainTokens, valTokens } = trainingFixture();

    try {
      const losses: number[] = [];
      const events: TrainEvent[] = [];
      const summary = train({
        model,
        config,
        trainTokens,
        valTokens,
        trainConfig: baseTrainConfig(),
        onEvent(event) {
          events.push(event);
          if (event.type === "step") {
            losses.push(event.loss);
          }
        },
      });

      expect(losses).toHaveLength(10);
      expect(losses.at(-1) ?? Infinity).toBeLessThan(losses[0] ?? 0);
      expect(summary.totalSteps).toBe(10);
      expect(summary.lastStepLoss).toBe(losses.at(-1) ?? null);
      expect(summary.lastTrainLoss).not.toBeNull();
      expect(summary.lastValLoss).not.toBeNull();
      expect(events.some((event) => event.type === "done")).toBe(true);
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("emits eval progress events during longer evaluation windows", () => {
    const { config, model, trainTokens, valTokens } = trainingFixture();

    try {
      const progressEvents: Extract<TrainEvent, { type: "progress" }>[] = [];
      train({
        model,
        config,
        trainTokens,
        valTokens,
        trainConfig: {
          ...baseTrainConfig(),
          maxSteps: 2,
          warmupSteps: 1,
          evalInterval: 1,
          evalSteps: 6,
        },
        onEvent(event) {
          if (event.type === "progress") {
            progressEvents.push(event);
          }
        },
      });

      expect(progressEvents.length).toBeGreaterThan(0);
      expect(progressEvents.some((event) => event.split === "train")).toBe(true);
      expect(progressEvents.some((event) => event.split === "val")).toBe(true);
      expect(progressEvents.some((event) => event.completed === 5)).toBe(true);
      expect(progressEvents.some((event) => event.completed === 6)).toBe(true);
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("restores the caller's training mode after training", () => {
    const { config, model, trainTokens, valTokens } = trainingFixture();
    model.eval();

    try {
      train({
        model,
        config,
        trainTokens,
        valTokens,
        trainConfig: { ...baseTrainConfig(), maxSteps: 2, warmupSteps: 1, evalInterval: 2 },
      });
      expect(model.isTraining).toBe(false);
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("training with the same seed is repeatable", () => {
    const first = trainingFixture();
    const second = trainingFixture();

    try {
      const runConfig = { ...baseTrainConfig(), maxSteps: 2, warmupSteps: 1, evalInterval: 2 };
      const summaryA = train({
        model: first.model,
        config: first.config,
        trainTokens: first.trainTokens,
        valTokens: first.valTokens,
        trainConfig: runConfig,
      });
      const summaryB = train({
        model: second.model,
        config: second.config,
        trainTokens: second.trainTokens,
        valTokens: second.valTokens,
        trainConfig: runConfig,
      });

      expect(summaryA).toEqual(summaryB);
      const firstParams = treeFlatten(first.model.parameters());
      const secondParams = treeFlatten(second.model.parameters());
      expect(firstParams).toHaveLength(secondParams.length);
      for (let index = 0; index < firstParams.length; index++) {
        expect(firstParams[index]?.[0]).toEqual(secondParams[index]?.[0]);
        expect(firstParams[index]?.[1].toList()).toEqual(secondParams[index]?.[1].toList());
      }
    } finally {
      first.model[Symbol.dispose]();
      second.model[Symbol.dispose]();
    }
  });

  test("supports an externally owned optimizer and clean early stop", () => {
    const { config, model, trainTokens, valTokens } = trainingFixture();
    const optimizer = new AdamW({
      learningRate: 1e-3,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0,
    });
    let stopChecks = 0;

    try {
      const summary = train({
        model,
        optimizer,
        config,
        trainTokens,
        valTokens,
        trainConfig: baseTrainConfig(),
        shouldStop() {
          stopChecks += 1;
          return stopChecks >= 3;
        },
      });

      expect(summary.totalSteps).toBe(3);
      expect(optimizer.step).toBe(3);
      expect(summary.lastStepLoss).not.toBeNull();
    } finally {
      optimizer[Symbol.dispose]();
      model[Symbol.dispose]();
    }
  });

  test("supports short training runs with gradient checkpointing enabled", () => {
    const { config, model, trainTokens, valTokens } = checkpointedTrainingFixture();

    try {
      const summary = train({
        model,
        config,
        trainTokens,
        valTokens,
        trainConfig: {
          ...baseTrainConfig(),
          maxSteps: 2,
          warmupSteps: 1,
          evalInterval: 2,
        },
      });

      expect(summary.totalSteps).toBe(2);
      expect(summary.lastStepLoss).not.toBeNull();
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("supports gradient accumulation with more than one micro-step", () => {
    const { config, model, trainTokens, valTokens } = trainingFixture();

    try {
      const summary = train({
        model,
        config,
        trainTokens,
        valTokens,
        trainConfig: {
          ...baseTrainConfig(),
          gradAccumSteps: 2,
          maxSteps: 2,
          warmupSteps: 1,
          evalInterval: 2,
        },
      });

      expect(summary.totalSteps).toBe(2);
      expect(summary.lastStepLoss).not.toBeNull();
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("gradient clipping materially reduces parameter updates", () => {
    const unclipped = trainingFixture();
    const clipped = trainingFixture();

    try {
      const beforeUnclipped = snapshotParameterScalars(unclipped.model);
      const beforeClipped = snapshotParameterScalars(clipped.model);

      train({
        model: unclipped.model,
        config: unclipped.config,
        trainTokens: unclipped.trainTokens,
        valTokens: unclipped.valTokens,
        trainConfig: {
          ...baseTrainConfig(),
          maxSteps: 1,
          warmupSteps: 0,
          evalInterval: 1,
          maxGradNorm: null,
        },
      });
      train({
        model: clipped.model,
        config: clipped.config,
        trainTokens: clipped.trainTokens,
        valTokens: clipped.valTokens,
        trainConfig: {
          ...baseTrainConfig(),
          maxSteps: 1,
          warmupSteps: 0,
          evalInterval: 1,
          maxGradNorm: 1e-6,
        },
      });

      const unclippedDelta = totalParameterDelta(beforeUnclipped, unclipped.model);
      const clippedDelta = totalParameterDelta(beforeClipped, clipped.model);

      expect(unclippedDelta).toBeGreaterThan(clippedDelta);
    } finally {
      unclipped.model[Symbol.dispose]();
      clipped.model[Symbol.dispose]();
    }
  });

  test("rejects non-finite loss before mutating parameters", () => {
    const text = "ab".repeat(40);
    const tokenizer = CharTokenizer.fromText(text);
    const tokens = tokenizer.encode(text);
    const { trainTokens, valTokens } = prepareData(tokens, 0.9);
    const config = resolveConfig(
      { ...GPT_TINY, nLayer: 1, nHead: 1, nEmbd: 8, blockSize: 4, dropout: 0 },
      2,
    );
    const model = new NonFiniteLossModel();
    const before = model.weight.toList();

    try {
      expect(() =>
        train({
          model: model as unknown as GPT,
          config,
          trainTokens,
          valTokens,
          trainConfig: { ...baseTrainConfig(), maxSteps: 1, warmupSteps: 0, evalInterval: 1 },
        }),
      ).toThrow("non-finite");
      expect(model.weight.toList()).toEqual(before);
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("validates train config before starting", () => {
    const scenarios: Array<{ name: string; override: Partial<TrainConfig>; message: string }> = [
      { name: "learningRate", override: { learningRate: 0 }, message: "learningRate must be > 0" },
      {
        name: "minLearningRate",
        override: { minLearningRate: 0 },
        message: "minLearningRate must be > 0",
      },
      { name: "maxGradNorm", override: { maxGradNorm: 0 }, message: "maxGradNorm must be > 0" },
      {
        name: "warmupSteps",
        override: { warmupSteps: 10, maxSteps: 10 },
        message: "warmupSteps must be < maxSteps",
      },
      {
        name: "startStep",
        override: { startStep: 10, maxSteps: 10 },
        message: "startStep must be < maxSteps",
      },
      {
        name: "gradAccumSteps",
        override: { gradAccumSteps: 0 },
        message: "gradAccumSteps must be >= 1",
      },
    ];

    for (const scenario of scenarios) {
      const { config, model, trainTokens, valTokens } = trainingFixture();
      try {
        expect(() =>
          train({
            model,
            config,
            trainTokens,
            valTokens,
            trainConfig: { ...baseTrainConfig(), ...scenario.override },
          }),
        ).toThrow(scenario.message);
      } finally {
        model[Symbol.dispose]();
      }
    }
  });
});
