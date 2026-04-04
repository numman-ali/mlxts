import { describe, expect, test } from "bun:test";
import type { ParameterTree } from "@mlxts/core";
import { array, MxArray, mxEval, treeFlatten } from "@mlxts/core";
import { Module } from "./module";

// --- Test helpers: concrete Module subclasses ---

class SimpleModule extends Module {
  weight: MxArray;
  bias: MxArray;

  constructor() {
    super();
    this.weight = array([1, 2, 3]);
    this.bias = array([0.1, 0.2, 0.3]);
  }

  forward(x: MxArray): MxArray {
    return x;
  }
}

class NestedModule extends Module {
  layer1: SimpleModule;
  layer2: SimpleModule;

  constructor() {
    super();
    this.layer1 = new SimpleModule();
    this.layer2 = new SimpleModule();
  }

  forward(x: MxArray): MxArray {
    return this.layer1.forward(x);
  }
}

class ModuleWithNull extends Module {
  weight: MxArray;
  bias: MxArray | null;
  // biome-ignore lint/correctness/noUnusedPrivateClassMembers: used to test that #private fields are excluded from parameters()
  #config: number;

  constructor(hasBias: boolean) {
    super();
    this.weight = array([1, 2]);
    this.bias = hasBias ? array([0.5]) : null;
    this.#config = 42;
  }

  forward(x: MxArray): MxArray {
    return x;
  }
}

class ModuleWithConfig extends Module {
  weight: MxArray;
  label: string;

  constructor() {
    super();
    this.weight = array([1, 2]);
    this.label = "config";
  }

  forward(x: MxArray): MxArray {
    return x;
  }
}

// ---------------------------------------------------------------------------
// parameters()
// ---------------------------------------------------------------------------

describe("Module.parameters()", () => {
  test("returns flat tree for simple module", () => {
    const m = new SimpleModule();
    const params = m.parameters();

    expect(Object.keys(params)).toEqual(["weight", "bias"]);
    expect(params.weight).toBe(m.weight);
    expect(params.bias).toBe(m.bias);

    m[Symbol.dispose]();
  });

  test("returns nested tree for nested module", () => {
    const m = new NestedModule();
    const params = m.parameters();

    expect(Object.keys(params)).toEqual(["layer1", "layer2"]);

    const layer1Params = params.layer1 as ParameterTree;
    expect(Object.keys(layer1Params)).toEqual(["weight", "bias"]);
    expect(layer1Params.weight).toBe(m.layer1.weight);

    m[Symbol.dispose]();
  });

  test("null properties are skipped", () => {
    const m = new ModuleWithNull(false);
    const params = m.parameters();

    expect(Object.keys(params)).toEqual(["weight"]);
    expect(params.bias).toBeUndefined();

    m[Symbol.dispose]();
  });

  test("null bias is included when present", () => {
    const m = new ModuleWithNull(true);
    const params = m.parameters();

    expect(Object.keys(params)).toEqual(["weight", "bias"]);

    m[Symbol.dispose]();
  });

  test("#private fields are excluded", () => {
    const m = new ModuleWithNull(true);
    const params = m.parameters();

    // #config should not appear
    expect("config" in params).toBe(false);
    expect("#config" in params).toBe(false);

    m[Symbol.dispose]();
  });

  test("Object.keys includes weight and bias for concrete module", () => {
    const m = new SimpleModule();
    const keys = Object.keys(m);

    expect(keys).toContain("weight");
    expect(keys).toContain("bias");

    m[Symbol.dispose]();
  });
});

// ---------------------------------------------------------------------------
// trainableParameters()
// ---------------------------------------------------------------------------

describe("Module.trainableParameters()", () => {
  test("returns all params when nothing frozen", () => {
    const m = new SimpleModule();
    const params = m.trainableParameters();

    expect(Object.keys(params)).toEqual(["weight", "bias"]);

    m[Symbol.dispose]();
  });

  test("excludes frozen keys", () => {
    const m = new SimpleModule();
    m.freeze(["bias"]);
    const params = m.trainableParameters();

    expect(Object.keys(params)).toEqual(["weight"]);

    m[Symbol.dispose]();
  });

  test("frozen keys still appear in parameters()", () => {
    const m = new SimpleModule();
    m.freeze(["bias"]);

    const allParams = m.parameters();
    const trainableParams = m.trainableParameters();

    expect(Object.keys(allParams)).toEqual(["weight", "bias"]);
    expect(Object.keys(trainableParams)).toEqual(["weight"]);

    m[Symbol.dispose]();
  });
});

// ---------------------------------------------------------------------------
// update()
// ---------------------------------------------------------------------------

describe("Module.update()", () => {
  test("replaces MxArray parameter values", () => {
    const m = new SimpleModule();
    const oldWeight = m.weight;

    const newWeight = array([10, 20, 30]);
    m.update({ weight: newWeight });

    expect(m.weight).toBe(newWeight);
    expect(m.weight).not.toBe(oldWeight);

    // Bias unchanged (partial merge)
    mxEval(m.bias);
    expect(m.bias.toList()).toEqual([
      expect.closeTo(0.1, 4),
      expect.closeTo(0.2, 4),
      expect.closeTo(0.3, 4),
    ]);

    oldWeight.free();
    m[Symbol.dispose]();
  });

  test("is partial merge — untouched keys survive", () => {
    const m = new SimpleModule();
    const originalBias = m.bias;

    m.update({ weight: array([99]) });

    expect(m.bias).toBe(originalBias);

    m[Symbol.dispose]();
  });

  test("recurses into sub-modules", () => {
    const m = new NestedModule();
    const newWeight = array([42]);

    m.update({ layer1: { weight: newWeight } });

    expect(m.layer1.weight).toBe(newWeight);
    // layer2 untouched
    expect(m.layer2.weight).not.toBe(newWeight);

    m[Symbol.dispose]();
  });

  test("throws on unknown keys", () => {
    const m = new SimpleModule();
    const rogue = array([9]);

    expect(() => m.update({ rogue })).toThrow('unknown key "rogue"');
    expect(Object.keys(m)).toEqual(["weight", "bias"]);

    rogue.free();
    m[Symbol.dispose]();
  });

  test("throws when a parameter leaf is updated with a subtree", () => {
    const m = new SimpleModule();
    const replacement = array([9]);

    expect(() => m.update({ weight: { nested: replacement } })).toThrow("expects an MxArray");
    expect(m.weight.toList()).toEqual([1, 2, 3]);

    replacement.free();
    m[Symbol.dispose]();
  });

  test("throws when a sub-module is updated with a leaf", () => {
    const m = new NestedModule();
    const replacement = array([9]);

    expect(() => m.update({ layer1: replacement })).toThrow("expects a nested parameter tree");
    expect(m.layer1.weight.toList()).toEqual([1, 2, 3]);

    replacement.free();
    m[Symbol.dispose]();
  });

  test("throws when updating a non-parameter slot", () => {
    const m = new ModuleWithConfig();
    const replacement = array([9]);

    expect(() => m.update({ label: replacement })).toThrow('key "label" is not a parameter slot');

    replacement.free();
    m[Symbol.dispose]();
  });
});

// ---------------------------------------------------------------------------
// freeze / unfreeze
// ---------------------------------------------------------------------------

describe("freeze / unfreeze", () => {
  test("freeze specific keys", () => {
    const m = new SimpleModule();
    m.freeze(["weight"]);

    const params = m.trainableParameters();
    expect(Object.keys(params)).toEqual(["bias"]);

    m[Symbol.dispose]();
  });

  test("freeze all", () => {
    const m = new SimpleModule();
    m.freeze();

    const params = m.trainableParameters();
    expect(Object.keys(params)).toEqual([]);

    m[Symbol.dispose]();
  });

  test("unfreeze specific keys", () => {
    const m = new SimpleModule();
    m.freeze();
    m.unfreeze(["weight"]);

    const params = m.trainableParameters();
    expect(Object.keys(params)).toEqual(["weight"]);

    m[Symbol.dispose]();
  });

  test("unfreeze all", () => {
    const m = new SimpleModule();
    m.freeze();
    m.unfreeze();

    const params = m.trainableParameters();
    expect(Object.keys(params)).toEqual(["weight", "bias"]);

    m[Symbol.dispose]();
  });

  test("freeze returns this for chaining", () => {
    const m = new SimpleModule();
    const result = m.freeze(["weight"]);
    expect(result).toBe(m);
    m[Symbol.dispose]();
  });

  test("freeze throws on unknown keys", () => {
    const m = new SimpleModule();
    expect(() => m.freeze(["missing"])).toThrow('unknown key "missing"');
    m[Symbol.dispose]();
  });

  test("freeze throws on non-parameter keys", () => {
    const m = new ModuleWithConfig();
    expect(() => m.freeze(["label"])).toThrow('key "label" is not a parameter or sub-module');
    m[Symbol.dispose]();
  });

  test("unfreeze throws on unknown keys", () => {
    const m = new SimpleModule();
    expect(() => m.unfreeze(["missing"])).toThrow('unknown key "missing"');
    m[Symbol.dispose]();
  });
});

// ---------------------------------------------------------------------------
// train / eval
// ---------------------------------------------------------------------------

describe("train / eval", () => {
  test("default is training mode", () => {
    const m = new SimpleModule();
    expect(m.isTraining).toBe(true);
    m[Symbol.dispose]();
  });

  test("eval() sets training to false", () => {
    const m = new SimpleModule();
    m.eval();
    expect(m.isTraining).toBe(false);
    m[Symbol.dispose]();
  });

  test("train() sets training to true", () => {
    const m = new SimpleModule();
    m.eval();
    m.train();
    expect(m.isTraining).toBe(true);
    m[Symbol.dispose]();
  });

  test("train/eval propagates to sub-modules", () => {
    const m = new NestedModule();
    m.eval();

    expect(m.isTraining).toBe(false);
    expect(m.layer1.isTraining).toBe(false);
    expect(m.layer2.isTraining).toBe(false);

    m.train();
    expect(m.layer1.isTraining).toBe(true);
    expect(m.layer2.isTraining).toBe(true);

    m[Symbol.dispose]();
  });

  test("train returns this for chaining", () => {
    const m = new SimpleModule();
    const result = m.train(false);
    expect(result).toBe(m);
    m[Symbol.dispose]();
  });
});

// ---------------------------------------------------------------------------
// treeFlatten integration
// ---------------------------------------------------------------------------

describe("Module + treeFlatten integration", () => {
  test("treeFlatten of nested module produces correct paths", () => {
    const m = new NestedModule();
    const flat = treeFlatten(m.parameters());

    expect(flat).toHaveLength(4);
    expect(flat[0]?.[0]).toEqual(["layer1", "weight"]);
    expect(flat[1]?.[0]).toEqual(["layer1", "bias"]);
    expect(flat[2]?.[0]).toEqual(["layer2", "weight"]);
    expect(flat[3]?.[0]).toEqual(["layer2", "bias"]);

    m[Symbol.dispose]();
  });
});

// ---------------------------------------------------------------------------
// Module[] (array of sub-modules) support
// ---------------------------------------------------------------------------

class ModuleArrayParent extends Module {
  layers: SimpleModule[];

  constructor(count: number) {
    super();
    this.layers = Array.from({ length: count }, () => new SimpleModule());
  }

  forward(x: MxArray): MxArray {
    return x;
  }
}

class MixedModule extends Module {
  weight: MxArray;
  layers: SimpleModule[];

  constructor() {
    super();
    this.weight = array([1, 2]);
    this.layers = [new SimpleModule(), new SimpleModule()];
  }

  forward(x: MxArray): MxArray {
    return x;
  }
}

class InvalidMixedModuleArray extends Module {
  layers: [SimpleModule, string];

  constructor() {
    super();
    this.layers = [new SimpleModule(), "rogue"];
  }

  forward(x: MxArray): MxArray {
    return x;
  }
}

describe("Module[] support", () => {
  test("parameters() produces numeric-keyed subtree for Module[]", () => {
    const m = new ModuleArrayParent(2);
    const params = m.parameters();
    expect(params.layers).toBeDefined();
    const layersTree = params.layers as ParameterTree;
    expect(layersTree["0"]).toBeDefined();
    expect(layersTree["1"]).toBeDefined();
    const layer0 = layersTree["0"] as ParameterTree;
    expect(layer0.weight).toBeInstanceOf(MxArray);
    expect(layer0.bias).toBeInstanceOf(MxArray);
    m[Symbol.dispose]();
  });

  test("treeFlatten produces correct paths with numeric indices", () => {
    const m = new ModuleArrayParent(2);
    const flat = treeFlatten(m.parameters());
    expect(flat).toHaveLength(4);
    expect(flat[0]?.[0]).toEqual(["layers", "0", "weight"]);
    expect(flat[1]?.[0]).toEqual(["layers", "0", "bias"]);
    expect(flat[2]?.[0]).toEqual(["layers", "1", "weight"]);
    expect(flat[3]?.[0]).toEqual(["layers", "1", "bias"]);
    m[Symbol.dispose]();
  });

  test("update() applies to Module[] children", () => {
    const m = new ModuleArrayParent(2);
    const newWeight = array([99, 99, 99]);
    m.update({
      layers: {
        "0": { weight: newWeight },
      },
    });
    mxEval(newWeight);
    const params = m.parameters();
    const layer0 = (params.layers as ParameterTree)["0"] as ParameterTree;
    expect((layer0.weight as MxArray).toList()).toEqual([99, 99, 99]);
    m[Symbol.dispose]();
  });

  test("train(false) propagates into Module[] elements", () => {
    const m = new ModuleArrayParent(2);
    expect(m.layers[0]?.isTraining).toBe(true);
    m.train(false);
    expect(m.layers[0]?.isTraining).toBe(false);
    expect(m.layers[1]?.isTraining).toBe(false);
    m.train(true);
    expect(m.layers[0]?.isTraining).toBe(true);
    m[Symbol.dispose]();
  });

  test("trainableParameters() excludes frozen Module[]", () => {
    const m = new ModuleArrayParent(2);
    m.freeze(["layers"]);
    const trainable = m.trainableParameters();
    expect(Object.keys(trainable)).toHaveLength(0);
    m.unfreeze(["layers"]);
    const unfrozen = m.trainableParameters();
    expect(unfrozen.layers).toBeDefined();
    m[Symbol.dispose]();
  });

  test("dispose recurses into Module[] elements", () => {
    const m = new ModuleArrayParent(2);
    // Just verify it doesn't throw — disposal frees native handles
    m[Symbol.dispose]();
  });

  test("non-Module arrays are silently skipped", () => {
    class HasNumberArray extends Module {
      // biome-ignore lint/correctness/noUnusedPrivateClassMembers: testing that config arrays are skipped
      #config = [1, 2, 3];
      weight: MxArray;
      constructor() {
        super();
        this.weight = array([1]);
      }
      forward(x: MxArray): MxArray {
        return x;
      }
    }
    const m = new HasNumberArray();
    const params = m.parameters();
    expect(Object.keys(params)).toEqual(["weight"]);
    m[Symbol.dispose]();
  });

  test("mixed Module with weight + Module[] works together", () => {
    const m = new MixedModule();
    const flat = treeFlatten(m.parameters());
    expect(flat).toHaveLength(5); // weight + 2 layers × (weight + bias)
    expect(flat[0]?.[0]).toEqual(["weight"]);
    expect(flat[1]?.[0]).toEqual(["layers", "0", "weight"]);
    m[Symbol.dispose]();
  });

  test("mixed arrays of Module and non-Module values throw clearly", () => {
    const m = new InvalidMixedModuleArray();
    try {
      expect(() => m.parameters()).toThrow("mixes Module and non-Module values");
      expect(() => m.freeze()).toThrow("mixes Module and non-Module values");
      expect(() => m.train(false)).toThrow("mixed Module arrays are unsupported");
      expect(() => m[Symbol.dispose]()).toThrow("mixed Module arrays are unsupported");
    } finally {
      m.layers[0][Symbol.dispose]();
    }
  });
});
