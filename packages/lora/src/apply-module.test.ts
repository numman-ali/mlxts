import { describe, expect, test } from "bun:test";
import { treeFlatten } from "@mlxts/core";
import { Linear, LoRALinear, Module, QuantizedLinear } from "@mlxts/nn";

import { applyLoRAToModule, mergeLoRAInModule, removeLoRAFromModule } from "./apply-module";

class AttentionBlock extends Module {
  qProjection = new Linear(64, 64);
  kProjection = new Linear(64, 64);

  forward(): never {
    throw new Error("unused");
  }
}

class ModelBody extends Module {
  layers = [new AttentionBlock(), new AttentionBlock()];

  forward(): never {
    throw new Error("unused");
  }
}

class TinyModel extends Module {
  model = new ModelBody();

  forward(): never {
    throw new Error("unused");
  }
}

class HeadModel extends Module {
  lmHead = new Linear(64, 64);

  forward(): never {
    throw new Error("unused");
  }
}

describe("applyLoRAToModule", () => {
  test("targets the selected layer slice and projection keys", () => {
    using model = new TinyModel();

    const result = applyLoRAToModule(model, {
      lastLayers: 1,
      keys: ["q_proj"],
    });

    expect(model.model.layers[0]?.qProjection).toBeInstanceOf(Linear);
    expect(model.model.layers[1]?.qProjection).toBeInstanceOf(LoRALinear);
    expect(model.model.layers[1]?.kProjection).toBeInstanceOf(Linear);
    expect(result.targets).toEqual(["model.layers.1.qProjection"]);
  });

  test("freezes non-adapter weights before LoRA training", () => {
    using model = new TinyModel();

    applyLoRAToModule(model, {
      paths: ["model.layers.1.qProjection"],
    });

    const paths = treeFlatten(model.trainableParameters())
      .map(([path]) => path.join("."))
      .sort();
    expect(paths).toEqual(["model.layers.1.qProjection.loraA", "model.layers.1.qProjection.loraB"]);
  });

  test("can merge and remove LoRA wrappers", () => {
    using model = new TinyModel();
    applyLoRAToModule(model, {
      lastLayers: 2,
      keys: ["q_proj"],
    });

    const removed = removeLoRAFromModule(model);
    expect(removed.targets).toEqual(["model.layers.0.qProjection", "model.layers.1.qProjection"]);
    expect(model.model.layers[0]?.qProjection).toBeInstanceOf(Linear);

    applyLoRAToModule(model, {
      lastLayers: 2,
      keys: ["q_proj"],
    });
    const merged = mergeLoRAInModule(model);
    expect(merged.targets).toEqual(["model.layers.0.qProjection", "model.layers.1.qProjection"]);
    expect(model.model.layers[0]?.qProjection).toBeInstanceOf(Linear);
  });

  test("preserves quantized bases on QLoRA merge by default", () => {
    using model = new TinyModel();
    const quantized = QuantizedLinear.fromLinear(
      model.model.layers[1]?.qProjection ?? new Linear(64, 64),
      {
        groupSize: 64,
        bits: 4,
        mode: "affine",
      },
    );
    const previous = model.model.layers[1]?.replaceChild("qProjection", quantized);
    previous?.[Symbol.dispose]();

    applyLoRAToModule(model, {
      paths: ["model.layers.1.qProjection"],
    });

    const merged = mergeLoRAInModule(model);
    expect(merged.targets).toEqual(["model.layers.1.qProjection"]);
    expect(model.model.layers[1]?.qProjection).toBeInstanceOf(QuantizedLinear);
  });

  test("supports exact path targeting outside the default key selection", () => {
    using model = new TinyModel();

    const result = applyLoRAToModule(model, {
      paths: ["model.layers.0.kProjection"],
    });

    expect(result.targets).toEqual(["model.layers.0.kProjection"]);
    expect(model.model.layers[0]?.kProjection).toBeInstanceOf(LoRALinear);
  });

  test("reports when a custom selector rejects an otherwise valid target", () => {
    using model = new TinyModel();

    const result = applyLoRAToModule(model, {
      lastLayers: 2,
      keys: ["q_proj"],
      select: () => false,
    });

    expect(result.targets).toEqual([]);
    expect(
      result.skipped.some((entry) => entry.reason === "selection predicate returned false"),
    ).toBe(true);
  });

  test("skips paths that are already wrapped", () => {
    using model = new TinyModel();

    applyLoRAToModule(model, {
      paths: ["model.layers.1.qProjection"],
    });
    const result = applyLoRAToModule(model, {
      paths: ["model.layers.1.qProjection"],
    });

    expect(result.targets).toEqual([]);
    expect(result.skipped).toContainEqual({
      path: "model.layers.1.qProjection",
      reason: "already wrapped with LoRA",
    });
  });

  test("can target exact root-level paths when no layer slice exists", () => {
    using model = new HeadModel();

    const result = applyLoRAToModule(model, {
      paths: ["lmHead"],
    });

    expect(result.targets).toEqual(["lmHead"]);
    expect(model.lmHead).toBeInstanceOf(LoRALinear);
  });
});
