/**
 * Tree utilities for nested parameter structures.
 *
 * MLX-style modules store parameters as nested objects mirroring module
 * hierarchy. These utilities convert between nested trees and flat
 * path/value arrays for autograd and optimizer state.
 *
 * @module
 */

import { MxArray } from "./array";

/** A nested tree of parameters, mirroring module hierarchy. */
export type ParameterTree = { [key: string]: MxArray | ParameterTree };

/** A flattened entry: path segments and leaf MxArray. */
export type FlatEntry = [path: string[], value: MxArray];

/** Type guard distinguishing subtrees from leaf arrays. */
export function isParameterTree(value: MxArray | ParameterTree): value is ParameterTree {
  return !(value instanceof MxArray);
}

function formatPath(path: readonly string[]): string {
  return path.length === 0 ? "<root>" : path.join(".");
}

function descendToParent(root: ParameterTree, path: readonly string[]): ParameterTree {
  let node = root;
  for (let index = 0; index < path.length - 1; index++) {
    const segment = path[index];
    if (segment === undefined) {
      continue;
    }

    const existing = node[segment];
    if (existing !== undefined && isParameterTree(existing)) {
      node = existing;
      continue;
    }

    if (existing !== undefined) {
      throw new Error(
        `treeUnflatten: path "${formatPath(path.slice(0, index + 1))}" collides with an existing leaf`,
      );
    }

    const child: ParameterTree = {};
    node[segment] = child;
    node = child;
  }
  return node;
}

function assignLeaf(node: ParameterTree, path: readonly string[], value: MxArray): void {
  const lastSegment = path[path.length - 1];
  if (lastSegment === undefined) {
    return;
  }

  const existing = node[lastSegment];
  if (existing !== undefined && isParameterTree(existing)) {
    throw new Error(`treeUnflatten: path "${formatPath(path)}" collides with an existing subtree`);
  }
  if (existing !== undefined) {
    throw new Error(`treeUnflatten: duplicate path "${formatPath(path)}"`);
  }
  node[lastSegment] = value;
}

/**
 * Get a tree value by key with a runtime check.
 * Throws if the key is missing — callers must verify before calling.
 */
function getTreeValue(tree: ParameterTree, key: string, context: string): MxArray | ParameterTree {
  if (!(key in tree)) {
    throw new Error(`${context}: key '${key}' not found`);
  }
  // Object.entries iteration guarantees key is present, but TS index signature
  // adds | undefined. The `in` check above makes this safe.
  const value = tree[key];
  if (value === undefined) {
    throw new Error(`${context}: key '${key}' has undefined value`);
  }
  return value;
}

/**
 * Flatten a nested parameter tree into path/value pairs.
 *
 * Each leaf MxArray is paired with its full path from the tree root.
 * Keys are visited in insertion order at each level.
 *
 * @example
 * ```ts
 * treeFlatten({ layer1: { weight: w, bias: b } })
 * // → [[["layer1", "weight"], w], [["layer1", "bias"], b]]
 * ```
 */
export function treeFlatten(tree: ParameterTree): FlatEntry[] {
  const result: FlatEntry[] = [];

  function walk(node: ParameterTree, prefix: string[]): void {
    for (const [key, value] of Object.entries(node)) {
      const path = [...prefix, key];
      if (isParameterTree(value)) {
        walk(value, path);
      } else {
        result.push([path, value]);
      }
    }
  }

  walk(tree, []);
  return result;
}

/**
 * Reconstruct a nested parameter tree from flattened path/value pairs.
 *
 * Multiple entries sharing a path prefix are merged into the same subtree.
 *
 * @example
 * ```ts
 * treeUnflatten([[["a", "b"], arr]])
 * // → { a: { b: arr } }
 * ```
 */
export function treeUnflatten(entries: FlatEntry[]): ParameterTree {
  const root: ParameterTree = {};

  for (const [path, value] of entries) {
    if (path.length === 0) {
      throw new Error("treeUnflatten: empty paths are not allowed");
    }
    const parent = descendToParent(root, path);
    assignLeaf(parent, path, value);
  }

  return root;
}

function assertExactKeys(
  referenceKeys: readonly string[],
  tree: ParameterTree,
  treeIndex: number,
  path: readonly string[],
): void {
  const currentKeys = Object.keys(tree);
  for (const key of referenceKeys) {
    if (!currentKeys.includes(key)) {
      throw new Error(
        `treeMap: key "${formatPath([...path, key])}" present in tree 0 but missing in tree ${treeIndex}`,
      );
    }
  }

  for (const key of currentKeys) {
    if (!referenceKeys.includes(key)) {
      throw new Error(
        `treeMap: key "${formatPath([...path, key])}" present in tree ${treeIndex} but missing in tree 0`,
      );
    }
  }
}

/**
 * Apply a function to corresponding leaves across one or more trees.
 *
 * The first tree's structure determines the traversal. All trees must
 * share the same key structure at every level.
 *
 * @throws When a key present in the first tree is missing from another tree.
 *
 * @example
 * ```ts
 * // Scale all parameters by 0.1
 * treeMap((x) => multiply(x, 0.1), params)
 * ```
 */
export function treeMap(
  fn: (...values: MxArray[]) => MxArray,
  ...trees: ParameterTree[]
): ParameterTree {
  if (trees.length === 0) {
    throw new Error("treeMap: at least one tree is required");
  }

  const first = trees[0];
  if (first === undefined) {
    throw new Error("treeMap: first tree is undefined");
  }

  function walk(path: string[], nodes: ParameterTree[]): ParameterTree {
    const reference = nodes[0];
    if (reference === undefined) {
      throw new Error(`treeMap: missing reference tree at "${formatPath(path)}"`);
    }

    const referenceKeys = Object.keys(reference);
    for (let index = 1; index < nodes.length; index++) {
      const node = nodes[index];
      if (node === undefined) {
        throw new Error(`treeMap: tree ${index} is undefined at "${formatPath(path)}"`);
      }
      assertExactKeys(referenceKeys, node, index, path);
    }

    const result: ParameterTree = {};
    for (const [key, value] of Object.entries(reference)) {
      const nextPath = [...path, key];
      if (isParameterTree(value)) {
        const subtrees = nodes.map((tree, index) => {
          const subtree = getTreeValue(tree, key, `treeMap: tree ${index}`);
          if (!isParameterTree(subtree)) {
            throw new Error(
              `treeMap: key "${formatPath(nextPath)}" is a subtree in tree 0 but a leaf in tree ${index}`,
            );
          }
          return subtree;
        });
        result[key] = walk(nextPath, subtrees);
      } else {
        const leaves = nodes.map((tree, index) => {
          const leaf = getTreeValue(tree, key, `treeMap: tree ${index}`);
          if (isParameterTree(leaf)) {
            throw new Error(
              `treeMap: key "${formatPath(nextPath)}" is a leaf in tree 0 but a subtree in tree ${index}`,
            );
          }
          return leaf;
        });
        result[key] = fn(...leaves);
      }
    }

    return result;
  }

  return walk([], trees);
}

/**
 * Collect all leaf MxArray values from a parameter tree.
 *
 * Returns leaves in the same order as {@link treeFlatten}, but without
 * the path segments.
 */
export function treeLeaves(tree: ParameterTree): MxArray[] {
  return treeFlatten(tree).map(([, value]) => value);
}
