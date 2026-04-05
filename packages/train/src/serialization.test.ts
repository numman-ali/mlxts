import { describe, expect, test } from "bun:test";
import { array, full, type MxArray, type ParameterTree, treeFlatten } from "@mlxts/core";
import { Module } from "@mlxts/nn";
import { AdamW } from "@mlxts/optimizers";

import {
  createCheckpointArray,
  currentParameterEntries,
  loadOptimizerState,
  readTensorSlice,
  serializeOptimizer,
  serializeParameters,
} from "./checkpoint-serialization";
import type {
  CheckpointTensor,
  CheckpointTensorMeta,
  ParameterizedModel,
} from "./checkpoint-types";

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

function freeTree(tree: ParameterTree): void {
  for (const [, value] of treeFlatten(tree)) {
    value.free();
  }
}

function gradientTree(_model: TinyModel): ParameterTree {
  return {
    weight: full([2], 1),
    bias: full([1], 1),
  };
}

describe("checkpoint serialization", () => {
  test("serializeParameters and currentParameterEntries use stable sorted parameter order", () => {
    const model: ParameterizedModel = {
      parameters() {
        return {
          z: full([1], 3),
          nested: {
            a: full([1], 1),
          },
        };
      },
      update() {},
    };

    const liveTree = model.parameters();
    try {
      const entries = currentParameterEntries(model);
      expect(entries.map(([path]) => path.join("."))).toEqual(["nested.a", "z"]);

      const serialized = serializeParameters(model);
      expect(Object.keys(serialized.parameters)).toEqual(["nested.a", "z"]);
      expect(serialized.bytes.byteLength).toBeGreaterThan(0);
    } finally {
      freeTree(liveTree);
    }
  });

  test("createCheckpointArray round-trips every supported checkpoint dtype", () => {
    const tensors: CheckpointTensor[] = [
      { shape: [2], dtype: "bool", data: Uint8Array.from([1, 0]) },
      { shape: [2], dtype: "uint8", data: Uint8Array.from([2, 3]) },
      { shape: [2], dtype: "uint16", data: new Uint8Array(new Uint16Array([4, 5]).buffer) },
      { shape: [2], dtype: "uint32", data: new Uint8Array(new Uint32Array([6, 7]).buffer) },
      { shape: [2], dtype: "int8", data: new Uint8Array(new Int8Array([-1, 2]).buffer) },
      { shape: [2], dtype: "int16", data: new Uint8Array(new Int16Array([-3, 4]).buffer) },
      { shape: [2], dtype: "int32", data: new Uint8Array(new Int32Array([-5, 6]).buffer) },
      { shape: [2], dtype: "float32", data: new Uint8Array(new Float32Array([1.5, 2.5]).buffer) },
      { shape: [2], dtype: "float64", data: new Uint8Array(new Float64Array([3.5, 4.5]).buffer) },
    ];

    for (const tensor of tensors) {
      using restored = createCheckpointArray(tensor);
      expect(restored.dtype).toBe(tensor.dtype);
      expect(restored.shape).toEqual([2]);
    }
  });

  test("serialization rejects unsupported dtypes and enforces tensor byte bounds", () => {
    const unsupportedModel: ParameterizedModel = {
      parameters() {
        return {
          weight: array([1], "bfloat16"),
        };
      },
      update() {},
    };

    const unsupportedTree = unsupportedModel.parameters();
    try {
      expect(() => serializeParameters(unsupportedModel)).toThrow(
        'dtype "bfloat16" is not supported',
      );
    } finally {
      freeTree(unsupportedTree);
    }

    const meta: CheckpointTensorMeta = {
      shape: [2],
      dtype: "float32",
      offset: 4,
      byteLength: 8,
    };
    expect(() => readTensorSlice("weight", meta, new Uint8Array(8))).toThrow(
      "exceeds tensors.bin size",
    );
  });

  test("serializeOptimizer and loadOptimizerState round-trip optimizer slots", () => {
    const model = new TinyModel(1, 2);
    const optimizer = new AdamW({ learningRate: 1e-3, weightDecay: 0.1 });

    try {
      const gradients = gradientTree(model);
      try {
        optimizer.update(model, gradients);
      } finally {
        freeTree(gradients);
      }

      const serialized = serializeOptimizer(optimizer);
      expect(serialized.optimizer.kind).toBe("adamw");
      expect(serialized.optimizer.step).toBe(1);
      expect(Object.keys(serialized.optimizer.state).sort()).toEqual(["bias", "weight"]);

      const loaded = loadOptimizerState(serialized.optimizer.state, serialized.bytes);
      const biasM = loaded.bias?.m;
      const weightV = loaded.weight?.v;
      expect(biasM?.dtype).toBe("float32");
      expect(weightV?.shape).toEqual([2]);
    } finally {
      optimizer[Symbol.dispose]();
      model[Symbol.dispose]();
    }
  });
});
