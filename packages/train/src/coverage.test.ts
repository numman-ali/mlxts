import { afterEach, describe, expect, test } from "bun:test";
import { full, type MxArray, type ParameterTree, treeFlatten } from "@mlxts/core";
import { Module } from "@mlxts/nn";
import { AdamW } from "@mlxts/optimizers";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { applyCheckpoint, loadCheckpoint, saveCheckpoint } from "./checkpoint";
import { writeCheckpointDirectory } from "./checkpoint-io";
import {
  bytesPerElement,
  isSupportedCheckpointDType,
  readManifest,
  shiftOptimizerOffsets,
  shiftTensorMeta,
} from "./checkpoint-manifest";
import type { CheckpointManifest } from "./checkpoint-types";
import {
  accumulateGradients,
  accumulateGradientTrees,
  clipGradientTree,
  evalGradientTree,
  freeGradientTree,
  gradientNorm,
  scaleGradientTree,
} from "./gradients";
import { applyGradientStep, materializeTrainingState } from "./step";

const tempRoots: string[] = [];

class TinyModel extends Module {
  weight: MxArray;
  bias: MxArray;

  constructor() {
    super();
    this.weight = full([2], 1);
    this.bias = full([1], 2);
  }

  override forward(): MxArray {
    return this.weight;
  }
}

function createTempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempRoots.push(directory);
  return directory;
}

afterEach(() => {
  while (tempRoots.length > 0) {
    const directory = tempRoots.pop();
    if (directory !== undefined) {
      rmSync(directory, { recursive: true, force: true });
    }
  }
});

function checkpointPath(name: string): string {
  return join(createTempDir(name), "checkpoint");
}

function freeTree(tree: ParameterTree): void {
  freeGradientTree(tree);
}

function createGradients(value: number): ParameterTree {
  return {
    weight: full([2], value),
    bias: full([1], value),
  };
}

describe("train coverage", () => {
  test("checkpoint manifest helpers parse resume metadata and shift offsets", () => {
    const path = checkpointPath("train-manifest");
    const manifest: CheckpointManifest = {
      version: 2,
      kind: "resume",
      metadata: { run: "coverage" },
      step: 4,
      parameters: {
        weight: {
          shape: [2],
          dtype: "float32",
          offset: 0,
          byteLength: 8,
        },
      },
      optimizer: {
        kind: "adamw",
        step: 4,
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        weightDecay: 0.1,
        state: {
          weight: {
            m: {
              shape: [2],
              dtype: "float32",
              offset: 8,
              byteLength: 8,
            },
          },
        },
      },
    };

    writeCheckpointDirectory(path, manifest, new Uint8Array(16));
    const parsed = readManifest(path);

    expect(parsed.kind).toBe("resume");
    expect(parsed.optimizer?.state.weight?.m?.offset).toBe(8);
    expect(isSupportedCheckpointDType("float32")).toBe(true);
    expect(isSupportedCheckpointDType("bfloat16")).toBe(false);
    expect(bytesPerElement("bool")).toBe(1);
    expect(bytesPerElement("uint8")).toBe(1);
    expect(bytesPerElement("int8")).toBe(1);
    expect(bytesPerElement("uint16")).toBe(2);
    expect(bytesPerElement("int16")).toBe(2);
    expect(bytesPerElement("uint32")).toBe(4);
    expect(bytesPerElement("int32")).toBe(4);
    expect(bytesPerElement("float64")).toBe(8);
    const weightMeta = parsed.parameters.weight;
    expect(weightMeta).toBeDefined();
    if (weightMeta === undefined) {
      throw new Error('expected checkpoint parameter "weight" to be present');
    }
    expect(shiftTensorMeta(weightMeta, 5).offset).toBe(5);
    expect(shiftOptimizerOffsets(parsed.optimizer?.state ?? {}, 12).weight?.m?.offset).toBe(20);
  });

  test("readManifest rejects malformed checkpoint shapes and metadata", () => {
    const path = checkpointPath("train-manifest-errors");
    const manifestPath = join(path, "manifest.json");

    writeCheckpointDirectory(
      path,
      {
        version: 2,
        kind: "snapshot",
        metadata: { ok: true },
        step: 1,
        parameters: {
          weight: {
            shape: [2],
            dtype: "float32",
            offset: 0,
            byteLength: 8,
          },
        },
      },
      new Uint8Array(8),
    );

    writeFileSync(manifestPath, `${JSON.stringify({ version: 9 })}\n`);
    expect(() => readManifest(path)).toThrow("unsupported");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "resume",
        step: 1,
        parameters: {},
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("metadata");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "resume",
        metadata: {},
        step: 1,
        parameters: {},
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("optimizer metadata");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: {},
        optimizer: {
          kind: "adamw",
          step: 1,
          lr: 0.01,
          beta1: 0.9,
          beta2: 0.95,
          eps: 1e-8,
          weightDecay: 0.1,
          state: {},
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("must not include optimizer metadata");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: {
          weight: {
            shape: [2],
            dtype: "float32",
            offset: 0,
            byteLength: 4,
          },
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("byteLength");
  });

  test("readManifest validates numeric fields, parameter records, and optimizer metadata structure", () => {
    const path = checkpointPath("train-manifest-structure-errors");
    const manifestPath = join(path, "manifest.json");
    writeCheckpointDirectory(
      path,
      {
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: {},
      },
      new Uint8Array(0),
    );

    writeFileSync(manifestPath, `${JSON.stringify([])}\n`);
    expect(() => readManifest(path)).toThrow("expected an object");

    writeFileSync(manifestPath, `${JSON.stringify({ version: "two" })}\n`);
    expect(() => readManifest(path)).toThrow("expected a finite number");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "mystery",
        metadata: {},
        step: 1,
        parameters: {},
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow('kind "mystery" is unsupported');

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: [],
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("parameters: expected an object");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: {
          weight: null,
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow('parameter "weight": expected an object');

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: {
          weight: {
            shape: "bad",
            dtype: "float32",
            offset: 0,
            byteLength: 4,
          },
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("expected a shape array");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: {
          weight: {
            shape: [-1],
            dtype: "float32",
            offset: 0,
            byteLength: 4,
          },
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("expected a non-negative integer");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: {
          weight: {
            shape: [1],
            dtype: "float32",
            offset: -1,
            byteLength: 4,
          },
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("offset: expected a non-negative integer");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: {
          weight: {
            shape: [1],
            dtype: "float32",
            offset: 0,
            byteLength: -1,
          },
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("byteLength: expected a non-negative integer");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: {
          weight: {
            shape: [1],
            dtype: 5,
            offset: 0,
            byteLength: 4,
          },
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("dtype: expected a string");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "snapshot",
        metadata: {},
        step: 1,
        parameters: {
          weight: {
            shape: [1],
            dtype: "bfloat16",
            offset: 0,
            byteLength: 4,
          },
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow('unsupported checkpoint dtype "bfloat16"');

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "resume",
        metadata: {},
        step: 1,
        parameters: {},
        optimizer: [],
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("optimizer: expected an object");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "resume",
        metadata: {},
        step: 1,
        parameters: {},
        optimizer: {
          kind: "sgd",
          step: 1,
          lr: 0.01,
          beta1: 0.9,
          beta2: 0.95,
          eps: 1e-8,
          weightDecay: 0.1,
          state: {},
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow('optimizer.kind: expected "adamw"');

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "resume",
        metadata: {},
        step: 1,
        parameters: {},
        optimizer: {
          kind: "adamw",
          step: -1,
          lr: 0.01,
          beta1: 0.9,
          beta2: 0.95,
          eps: 1e-8,
          weightDecay: 0.1,
          state: {},
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("optimizer.step: expected a non-negative integer");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "resume",
        metadata: {},
        step: 1,
        parameters: {},
        optimizer: {
          kind: "adamw",
          step: 1,
          lr: 0.01,
          beta1: 0.9,
          beta2: 0.95,
          eps: 1e-8,
          weightDecay: 0.1,
          state: [],
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("optimizer.state: expected an object");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "resume",
        metadata: {},
        step: 1,
        parameters: {},
        optimizer: {
          kind: "adamw",
          step: 1,
          lr: 0.01,
          beta1: 0.9,
          beta2: 0.95,
          eps: 1e-8,
          weightDecay: 0.1,
          state: {
            weight: [],
          },
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("optimizer.state.weight: expected an object");

    writeFileSync(
      manifestPath,
      `${JSON.stringify({
        version: 2,
        kind: "resume",
        metadata: {},
        step: 2,
        parameters: {},
        optimizer: {
          kind: "adamw",
          step: 1,
          lr: 0.01,
          beta1: 0.9,
          beta2: 0.95,
          eps: 1e-8,
          weightDecay: 0.1,
          state: {},
        },
      })}\n`,
    );
    expect(() => readManifest(path)).toThrow("optimizer.step 1 does not match checkpoint step 2");
  });

  test("writeCheckpointDirectory replaces existing checkpoints atomically", () => {
    const path = checkpointPath("train-atomic");
    writeCheckpointDirectory(
      path,
      {
        version: 2,
        kind: "snapshot",
        metadata: { version: "old" },
        step: 1,
        parameters: {},
      },
      Uint8Array.from([1, 2]),
    );

    writeCheckpointDirectory(
      path,
      {
        version: 2,
        kind: "snapshot",
        metadata: { version: "new" },
        step: 2,
        parameters: {},
      },
      Uint8Array.from([9, 8, 7]),
    );

    expect(readManifest(path).step).toBe(2);
    expect(readFileSync(join(path, "tensors.bin"))).toEqual(Buffer.from([9, 8, 7]));
    expect(existsSync(join(path, "manifest.json"))).toBe(true);
  });

  test("gradient helpers cover identity, mismatch, and non-finite cases", () => {
    const left = createGradients(2);
    const right = createGradients(3);
    const mismatch: ParameterTree = { rogue: full([1], 1) };
    const nanTree: ParameterTree = { weight: full([1], Number.NaN) };

    try {
      const summed = accumulateGradients(left, right);
      try {
        expect(gradientNorm(summed)).toBeCloseTo(Math.sqrt(75));
      } finally {
        freeTree(summed);
      }

      expect(scaleGradientTree(left, 1)).toBe(left);
      expect(clipGradientTree(left, null)).toBe(left);
      expect(() => accumulateGradients(left, mismatch)).toThrow(
        "gradient tree leaf counts do not match",
      );
      expect(() => clipGradientTree(nanTree, 1)).toThrow("non-finite");
      expect(clipGradientTree(left, 10)).toBe(left);

      const renamed: ParameterTree = { other: full([2], 3), bias: full([1], 3) };
      try {
        expect(() => accumulateGradients(left, renamed)).toThrow("gradient path mismatch");
      } finally {
        freeTree(renamed);
      }

      const viaAlias = accumulateGradientTrees(left, right);
      try {
        expect(gradientNorm(viaAlias)).toBeCloseTo(Math.sqrt(75));
      } finally {
        freeTree(viaAlias);
      }

      evalGradientTree({});
      freeGradientTree(mismatch);
      expect(treeFlatten(mismatch)[0]?.[1].isDisposed).toBe(true);
    } finally {
      freeTree(left);
      freeTree(right);
      freeTree(nanTree);
    }
  });

  test("gradient helpers evaluate non-empty trees and surface cleanup on arithmetic failures", () => {
    const left: ParameterTree = { a: full([1], 1), b: full([2], 2) };
    const mismatchedShape: ParameterTree = { a: full([1], 3), b: full([3], 4) };

    try {
      evalGradientTree(left);
      expect(() => accumulateGradients(left, mismatchedShape)).toThrow(
        "train.accumulateGradients:",
      );
    } finally {
      freeTree(left);
      freeTree(mismatchedShape);
    }
  });

  test("applyGradientStep validates inputs and frees staged gradients on failure", () => {
    expect(() =>
      applyGradientStep({
        gradAccumSteps: 0,
        maxGradNorm: null,
        takeMicroStep() {
          return { lossValue: 1, gradients: createGradients(1) };
        },
        applyGradients() {},
      }),
    ).toThrow("positive integer");

    let captured: ParameterTree | null = null;
    expect(() =>
      applyGradientStep({
        gradAccumSteps: 1,
        maxGradNorm: 0.5,
        takeMicroStep() {
          captured = createGradients(2);
          return {
            lossValue: 4,
            gradients: captured,
          };
        },
        applyGradients() {
          throw new Error("boom");
        },
      }),
    ).toThrow("boom");

    expect(treeFlatten(captured ?? {})[0]?.[1].isDisposed).toBe(true);
  });

  test("applyGradientStep frees accumulated and prepared gradients when reduction or clipping fails", () => {
    let firstMicroStep: ParameterTree | null = null;
    let secondMicroStep: ParameterTree | null = null;

    expect(() =>
      applyGradientStep({
        gradAccumSteps: 2,
        maxGradNorm: null,
        takeMicroStep() {
          if (firstMicroStep === null) {
            firstMicroStep = createGradients(1);
            return { lossValue: 1, gradients: firstMicroStep };
          }

          secondMicroStep = { rogue: full([1], 1) };
          return { lossValue: 1, gradients: secondMicroStep };
        },
        applyGradients() {},
      }),
    ).toThrow("gradient tree leaf counts do not match");

    expect(treeFlatten(firstMicroStep ?? {})[0]?.[1].isDisposed).toBe(true);
    expect(treeFlatten(secondMicroStep ?? {})[0]?.[1].isDisposed).toBe(true);

    let nanGradients: ParameterTree | null = null;
    expect(() =>
      applyGradientStep({
        gradAccumSteps: 1,
        maxGradNorm: 1,
        takeMicroStep() {
          nanGradients = { weight: full([1], Number.NaN) };
          return { lossValue: 1, gradients: nanGradients };
        },
        applyGradients() {},
      }),
    ).toThrow("non-finite");

    expect(treeFlatten(nanGradients ?? {})[0]?.[1].isDisposed).toBe(true);
  });

  test("checkpoint helpers enforce optimizer invariants and missing parameters", () => {
    const model = new TinyModel();
    const optimizer = new AdamW({ learningRate: 1e-3 });
    const path = checkpointPath("train-checkpoint-errors");

    try {
      expect(() =>
        saveCheckpoint({
          model,
          kind: "resume",
          metadata: {},
          path,
          step: 1,
        }),
      ).toThrow("require optimizer state");

      expect(() =>
        saveCheckpoint({
          model,
          kind: "snapshot",
          metadata: {},
          optimizer,
          path,
          step: 0,
        }),
      ).toThrow("must not include optimizer state");

      optimizer.restore({
        kind: "adamw",
        step: 3,
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weightDecay: 0,
        state: {},
      });

      expect(() =>
        saveCheckpoint({
          model,
          kind: "resume",
          metadata: {},
          optimizer,
          path,
          step: 2,
        }),
      ).toThrow("does not match checkpoint step");

      saveCheckpoint({
        model,
        kind: "snapshot",
        metadata: {},
        path,
        step: 1,
      });
      const loaded = loadCheckpoint(path);
      delete loaded.parameters.bias;
      expect(() => applyCheckpoint(new TinyModel(), loaded)).toThrow(
        'missing checkpoint parameter "bias"',
      );
    } finally {
      optimizer[Symbol.dispose]();
      model[Symbol.dispose]();
    }
  });

  test("materializeTrainingState tolerates empty optimizer state arrays", () => {
    const model = new TinyModel();

    try {
      materializeTrainingState(model, {
        stateArrays() {
          return [];
        },
      });

      expect(model.weight.toList()).toEqual([1, 1]);
    } finally {
      model[Symbol.dispose]();
    }
  });
});
