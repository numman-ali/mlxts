import { describe, expect, test } from "bun:test";

import { array, type MxArray } from "../core/array";
import { add } from "../core/ops/arithmetic";
import { mxEval } from "../core/transforms";
import type { ParameterTree } from "./tree";
import { treeFlatten, treeLeaves, treeMap, treeUnflatten } from "./tree";

describe("treeFlatten", () => {
  test("single level", () => {
    const w = array([1.0]);
    const b = array([2.0]);
    const tree: ParameterTree = { weight: w, bias: b };

    const flat = treeFlatten(tree);

    expect(flat).toHaveLength(2);
    expect(flat[0]?.[0]).toEqual(["weight"]);
    expect(flat[0]?.[1]).toBe(w);
    expect(flat[1]?.[0]).toEqual(["bias"]);
    expect(flat[1]?.[1]).toBe(b);

    w.free();
    b.free();
  });

  test("nested", () => {
    const w = array([3.0]);
    const tree: ParameterTree = { layer1: { weight: w } };

    const flat = treeFlatten(tree);

    expect(flat).toHaveLength(1);
    expect(flat[0]?.[0]).toEqual(["layer1", "weight"]);
    expect(flat[0]?.[1]).toBe(w);

    w.free();
  });

  test("empty tree", () => {
    const flat = treeFlatten({});
    expect(flat).toEqual([]);
  });
});

describe("treeUnflatten", () => {
  test("single level round-trip", () => {
    const w = array([1.0]);
    const b = array([2.0]);
    const tree: ParameterTree = { weight: w, bias: b };

    const restored = treeUnflatten(treeFlatten(tree));

    expect(restored.weight).toBe(w);
    expect(restored.bias).toBe(b);

    w.free();
    b.free();
  });

  test("nested construction", () => {
    const a = array([5.0]);
    const entries: [string[], MxArray][] = [[["a", "b"], a]];

    const tree = treeUnflatten(entries);

    expect((tree.a as ParameterTree).b).toBe(a);

    a.free();
  });

  test("duplicate paths throw", () => {
    const a = array([1.0]);
    const b = array([2.0]);

    expect(() =>
      treeUnflatten([
        [["weight"], a],
        [["weight"], b],
      ]),
    ).toThrow("duplicate path");

    a.free();
    b.free();
  });

  test("prefix collisions throw when a leaf would become a subtree", () => {
    const a = array([1.0]);
    const b = array([2.0]);

    expect(() =>
      treeUnflatten([
        [["layer"], a],
        [["layer", "weight"], b],
      ]),
    ).toThrow("collides with an existing leaf");

    a.free();
    b.free();
  });

  test("prefix collisions throw when a subtree would become a leaf", () => {
    const a = array([1.0]);
    const b = array([2.0]);

    expect(() =>
      treeUnflatten([
        [["layer", "weight"], a],
        [["layer"], b],
      ]),
    ).toThrow("collides with an existing subtree");

    a.free();
    b.free();
  });
});

describe("round-trip", () => {
  test("treeFlatten then treeUnflatten preserves structure and references", () => {
    const w1 = array([1.0]);
    const b1 = array([2.0]);
    const w2 = array([3.0]);
    const tree: ParameterTree = {
      layer1: { weight: w1, bias: b1 },
      layer2: { weight: w2 },
    };

    const restored = treeUnflatten(treeFlatten(tree));

    // Same MxArray references, not copies.
    expect((restored.layer1 as ParameterTree).weight).toBe(w1);
    expect((restored.layer1 as ParameterTree).bias).toBe(b1);
    expect((restored.layer2 as ParameterTree).weight).toBe(w2);

    w1.free();
    b1.free();
    w2.free();
  });
});

describe("treeLeaves", () => {
  test("returns all arrays in traversal order", () => {
    const w = array([1.0]);
    const b = array([2.0]);
    const tree: ParameterTree = { layer: { weight: w, bias: b } };

    const leaves = treeLeaves(tree);

    expect(leaves).toHaveLength(2);
    expect(leaves[0]).toBe(w);
    expect(leaves[1]).toBe(b);

    w.free();
    b.free();
  });
});

describe("treeMap", () => {
  test("identity preserves structure", () => {
    const w = array([1.0]);
    const b = array([2.0]);
    const tree: ParameterTree = { layer: { weight: w, bias: b } };

    const mapped = treeMap((x) => x, tree);

    expect((mapped.layer as ParameterTree).weight).toBe(w);
    expect((mapped.layer as ParameterTree).bias).toBe(b);

    w.free();
    b.free();
  });

  test("transform applies function to all leaves", () => {
    const w = array([1.0]);
    const b = array([2.0]);
    const ones = array([1.0]);
    const tree: ParameterTree = { weight: w, bias: b };

    // Add 1.0 to every parameter.
    const mapped = treeMap((x) => add(x, ones), tree);

    const newW = mapped.weight as MxArray;
    const newB = mapped.bias as MxArray;
    mxEval(newW, newB);

    expect(newW.toList()).toEqual([2]);
    expect(newB.toList()).toEqual([3]);

    w.free();
    b.free();
    ones.free();
    newW.free();
    newB.free();
  });

  test("structure mismatch throws clear error", () => {
    const a = array([1.0]);
    const b = array([2.0]);
    const tree1: ParameterTree = { weight: a, bias: b };
    const tree2: ParameterTree = { weight: a };

    expect(() => treeMap((x, y) => add(x, y), tree1, tree2)).toThrow("bias");

    a.free();
    b.free();
  });

  test("extra keys in a secondary tree throw clear errors", () => {
    const a = array([1.0]);
    const b = array([2.0]);
    const extra = array([3.0]);
    const tree1: ParameterTree = { weight: a, bias: b };
    const tree2: ParameterTree = { weight: a, bias: b, extra };

    expect(() => treeMap((x, y) => add(x, y), tree1, tree2)).toThrow("extra");

    a.free();
    b.free();
    extra.free();
  });
});
