import { describe, expect, test } from "bun:test";
import { Embedding, Linear, Module, QuantizedEmbedding, QuantizedLinear } from "@mlxts/nn";

import { quantizeModule } from "./quantize-module";

class TinyBlock extends Module {
  up = new Linear(64, 32);
  down = new Linear(32, 16);

  forward(): never {
    throw new Error("unused");
  }
}

class TinyModel extends Module {
  tokens = new Embedding(8, 64);
  input = new Linear(64, 64);
  block = new TinyBlock();

  forward(): never {
    throw new Error("unused");
  }
}

class ArrayModelBody extends Module {
  layers = [new TinyBlock(), new TinyBlock()];

  forward(): never {
    throw new Error("unused");
  }
}

class ArrayModel extends Module {
  model = new ArrayModelBody();

  forward(): never {
    throw new Error("unused");
  }
}

class MixedArrayModel extends Module {
  mixed = [new TinyBlock(), 1];

  forward(): never {
    throw new Error("unused");
  }
}

class ExistingQuantizedModel extends Module {
  projection = QuantizedLinear.fromLinear(new Linear(64, 64), {
    groupSize: 64,
    bits: 4,
    mode: "affine",
  });
  extras = [1, 2, 3];

  forward(): never {
    throw new Error("unused");
  }
}

describe("quantizeModule", () => {
  test("replaces selected dense linear layers with quantized layers", () => {
    using model = new TinyModel();

    const result = quantizeModule(model, {
      bits: 4,
      groupSize: 64,
      select: (path) => path !== "block.down",
    });

    expect(model.input).toBeInstanceOf(QuantizedLinear);
    expect(model.tokens).toBeInstanceOf(QuantizedEmbedding);
    expect(model.block.up).toBeInstanceOf(QuantizedLinear);
    expect(model.block.down).toBeInstanceOf(Linear);
    expect(result.targets.map((target) => target.path)).toEqual(["input", "block.up", "tokens"]);
    expect(result.skipped).toEqual([
      {
        path: "block.down",
        reason: "selection predicate returned false",
      },
    ]);
  });

  test("skips incompatible group sizes without mutating the layer", () => {
    using model = new TinyModel();

    const result = quantizeModule(model, {
      groupSize: 48,
    });

    expect(model.input).toBeInstanceOf(Linear);
    expect(model.tokens).toBeInstanceOf(Embedding);
    expect(result.targets).toEqual([]);
    expect(result.skipped.some((entry) => entry.path === "input")).toBe(true);
    expect(result.skipped.some((entry) => entry.path === "tokens")).toBe(true);
  });

  test("walks module arrays in stable order", () => {
    using model = new ArrayModel();

    const result = quantizeModule(model, {
      groupSize: 64,
      select: (path) => path.endsWith("up"),
    });

    expect(result.targets).toEqual([
      { path: "model.layers.0.up", params: { groupSize: 64, bits: 4, mode: "affine" } },
      { path: "model.layers.1.up", params: { groupSize: 64, bits: 4, mode: "affine" } },
    ]);
    expect(model.model.layers[0]?.up).toBeInstanceOf(QuantizedLinear);
    expect(model.model.layers[1]?.up).toBeInstanceOf(QuantizedLinear);
  });

  test("surfaces mixed module arrays clearly", () => {
    const model = new MixedArrayModel();
    expect(() => quantizeModule(model)).toThrow("mixes Module and non-Module values");
  });

  test("ignores already quantized children and non-module arrays", () => {
    using model = new ExistingQuantizedModel();

    const result = quantizeModule(model);

    expect(result.targets).toEqual([]);
    expect(model.projection).toBeInstanceOf(QuantizedLinear);
  });
});
