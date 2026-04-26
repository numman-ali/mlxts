import { describe, expect, test } from "bun:test";
import { Embedding, Linear, Module, QuantizedEmbedding, QuantizedLinear } from "@mlxts/nn";

import { resolveCheckpointQuantizationPlan } from "./checkpoint-plan";
import { setupQuantizedModule } from "./setup-quantized-module";

class TinyBlock extends Module {
  qProjection = new Linear(64, 32);
  kProjection = new Linear(64, 32);

  forward(): never {
    throw new Error("unused");
  }
}

class TinyModel extends Module {
  tokens = new Embedding(8, 64);
  block = new TinyBlock();
  lmHead = new Linear(64, 128);

  forward(): never {
    throw new Error("unused");
  }
}

describe("setupQuantizedModule", () => {
  test("quantizes every eligible linear layer for a global mlx plan", () => {
    using model = new TinyModel();
    const plan = resolveCheckpointQuantizationPlan({
      quantization: { bits: 4, group_size: 64 },
    });
    if (plan === null) {
      throw new Error("expected plan");
    }

    const result = setupQuantizedModule(model, plan);

    expect(model.block.qProjection).toBeInstanceOf(QuantizedLinear);
    expect(model.block.kProjection).toBeInstanceOf(QuantizedLinear);
    expect(model.tokens).toBeInstanceOf(QuantizedEmbedding);
    expect(model.lmHead).toBeInstanceOf(QuantizedLinear);
    expect(result.targets).toHaveLength(4);
  });

  test("honors explicit-only checkpoint rules", () => {
    using model = new TinyModel();
    const plan = resolveCheckpointQuantizationPlan({
      quantization: {
        bits: 4,
        group_size: 64,
        tokens: true,
        "block.qProjection": true,
      },
    });
    if (plan === null) {
      throw new Error("expected plan");
    }

    const result = setupQuantizedModule(model, plan);

    expect(model.block.qProjection).toBeInstanceOf(QuantizedLinear);
    expect(model.tokens).toBeInstanceOf(QuantizedEmbedding);
    expect(model.block.kProjection).toBeInstanceOf(Linear);
    expect(model.lmHead).toBeInstanceOf(Linear);
    expect(result.targets.map((target) => target.path)).toEqual(["block.qProjection", "tokens"]);
  });

  test("rejects unsupported providers until weight transforms exist", () => {
    using model = new TinyModel();
    const plan = resolveCheckpointQuantizationPlan({
      quantization_config: {
        quant_method: "awq",
        bits: 4,
        group_size: 128,
      },
    });
    if (plan === null) {
      throw new Error("expected plan");
    }

    expect(() => setupQuantizedModule(model, plan)).toThrow(
      'provider "awq" requires a checkpoint transform',
    );
  });

  test("honors explicit disabled rules", () => {
    using model = new TinyModel();
    const plan = resolveCheckpointQuantizationPlan({
      quantization: {
        bits: 4,
        group_size: 64,
        tokens: false,
        "block.qProjection": false,
      },
    });
    if (plan === null) {
      throw new Error("expected plan");
    }

    const result = setupQuantizedModule(model, plan);
    expect(model.block.qProjection).toBeInstanceOf(Linear);
    expect(model.tokens).toBeInstanceOf(Embedding);
    expect(result.skipped).toContainEqual({
      path: "block.qProjection",
      reason: "explicit checkpoint rule disabled quantization",
    });
    expect(result.skipped).toContainEqual({
      path: "tokens",
      reason: "explicit checkpoint rule disabled quantization",
    });
  });

  test("throws when an explicit rule requests an incompatible group size", () => {
    using model = new TinyModel();
    const plan = resolveCheckpointQuantizationPlan({
      quantization: {
        bits: 4,
        group_size: 96,
        "block.qProjection": true,
      },
    });
    if (plan === null) {
      throw new Error("expected plan");
    }

    expect(() => setupQuantizedModule(model, plan)).toThrow("is incompatible with groupSize 96");
  });

  test("skips incompatible layers for implicit global plans", () => {
    using model = new TinyModel();
    const plan = resolveCheckpointQuantizationPlan({
      quantization: {
        bits: 4,
        group_size: 96,
      },
    });
    if (plan === null) {
      throw new Error("expected plan");
    }

    const result = setupQuantizedModule(model, plan);
    expect(result.targets).toEqual([]);
    expect(result.skipped.some((entry) => entry.path === "block.qProjection")).toBe(true);
  });
});
