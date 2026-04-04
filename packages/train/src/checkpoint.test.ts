import { describe, expect, test } from "bun:test";
import { full, type MxArray, treeFlatten, treeUnflatten } from "@mlxts/core";
import { Module } from "@mlxts/nn";
import { AdamW } from "@mlxts/optimizers";
import { existsSync, mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import {
  applyCheckpoint,
  loadCheckpoint,
  restoreAdamWFromCheckpoint,
  saveCheckpoint,
} from "./checkpoint";

class TinyModel extends Module {
  weight: MxArray;
  bias: MxArray;

  constructor(weightValue: number, biasValue: number) {
    super();
    this.weight = full([2], weightValue);
    this.bias = full([1], biasValue);
  }

  override forward(): MxArray {
    return this.weight;
  }
}

function checkpointPath(name: string): string {
  return join(mkdtempSync(join(tmpdir(), `${name}-`)), "checkpoint");
}

function gradientTree(model: TinyModel) {
  return treeUnflatten(
    treeFlatten(model.trainableParameters()).map(([path, value]) => [
      path,
      full([...value.shape], 1, value.dtype),
    ]),
  );
}

function freeTreeArrays(model: TinyModel): void {
  model[Symbol.dispose]();
}

describe("checkpoint", () => {
  test("saveCheckpoint and loadCheckpoint round-trip generic metadata", () => {
    const model = new TinyModel(2, 3);
    const path = checkpointPath("train-generic");

    try {
      saveCheckpoint({
        model,
        kind: "snapshot",
        metadata: { name: "fixture", version: 1 },
        path,
        step: 5,
      });

      expect(existsSync(join(path, "manifest.json"))).toBe(true);
      expect(existsSync(join(path, "tensors.bin"))).toBe(true);

      const loaded = loadCheckpoint(path);
      expect(loaded.version).toBe(2);
      expect(loaded.kind).toBe("snapshot");
      expect(loaded.step).toBe(5);
      expect(loaded.metadata).toEqual({ name: "fixture", version: 1 });
      expect(Object.keys(loaded.parameters).length).toBeGreaterThan(0);
    } finally {
      freeTreeArrays(model);
    }
  });

  test("applyCheckpoint restores parameter values and disposes replaced arrays", () => {
    const source = new TinyModel(4, 6);
    const target = new TinyModel(1, 1);
    const path = checkpointPath("train-apply");

    try {
      saveCheckpoint({
        model: source,
        kind: "snapshot",
        metadata: { name: "apply" },
        path,
        step: 7,
      });

      const before = treeFlatten(target.parameters());
      const loaded = loadCheckpoint(path);
      applyCheckpoint(target, loaded);
      const after = treeFlatten(target.parameters());
      const expected = treeFlatten(source.parameters());

      expect(before).toHaveLength(after.length);
      for (let index = 0; index < before.length; index++) {
        expect(before[index]?.[1].isDisposed).toBe(true);
        expect(after[index]?.[0]).toEqual(expected[index]?.[0]);
        expect(after[index]?.[1].toList()).toEqual(expected[index]?.[1].toList());
      }
    } finally {
      freeTreeArrays(source);
      freeTreeArrays(target);
    }
  });

  test("restoreAdamWFromCheckpoint recreates optimizer state", () => {
    const model = new TinyModel(2, 3);
    const optimizer = new AdamW({ learningRate: 1e-3, weightDecay: 0.1 });
    const path = checkpointPath("train-resume");

    try {
      const gradients = gradientTree(model);
      try {
        optimizer.update(model, gradients);
      } finally {
        for (const [, value] of treeFlatten(gradients)) {
          value.free();
        }
      }

      saveCheckpoint({
        model,
        kind: "resume",
        metadata: { run: "resume" },
        optimizer,
        path,
        step: optimizer.step,
      });

      const loaded = loadCheckpoint(path);
      if (loaded.optimizer === undefined) {
        throw new Error("expected optimizer payload in resume checkpoint");
      }

      const restored = restoreAdamWFromCheckpoint(loaded.optimizer);
      try {
        const snapshot = restored.checkpoint();
        expect(snapshot.step).toBe(optimizer.step);
        expect(snapshot.lr).toBeCloseTo(1e-3);
        expect(Object.keys(snapshot.state)).toEqual(Object.keys(optimizer.checkpoint().state));
      } finally {
        restored[Symbol.dispose]();
      }
    } finally {
      optimizer[Symbol.dispose]();
      freeTreeArrays(model);
    }
  });
});
