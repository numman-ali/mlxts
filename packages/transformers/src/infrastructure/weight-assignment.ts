/**
 * Runtime-safe weight assignment helpers for family loaders.
 * @module
 */

import { formatShape, MxArray, type ParameterTree, treeFlatten } from "@mlxts/core";

import { WeightMismatchError } from "../types";

function nextNode(current: unknown, segment: string, path: string): unknown {
  if (Array.isArray(current)) {
    const index = Number(segment);
    if (!Number.isInteger(index) || index < 0 || index >= current.length) {
      throw new Error(
        `assignWeightPath: "${path}" segment "${segment}" is not a valid array index.`,
      );
    }
    return current[index];
  }

  if (typeof current !== "object" || current === null) {
    throw new Error(`assignWeightPath: "${path}" cannot descend through a non-object segment.`);
  }

  return Reflect.get(current, segment);
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((dimension, index) => dimension === right[index]);
}

/** Replace a parameter leaf on a model object with a loaded tensor. */
export function assignWeightPath(root: object, path: string, tensor: MxArray): void {
  if (path.trim() === "") {
    throw new Error("assignWeightPath: path must not be empty.");
  }

  const segments = path.split(".");
  let current: unknown = root;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    if (segment === undefined) {
      throw new Error(`assignWeightPath: "${path}" contains an undefined segment.`);
    }
    current = nextNode(current, segment, path);
  }

  if (typeof current !== "object" || current === null || Array.isArray(current)) {
    throw new Error(`assignWeightPath: "${path}" does not point to an object property.`);
  }

  const leafKey = segments[segments.length - 1];
  if (leafKey === undefined) {
    throw new Error(`assignWeightPath: "${path}" is missing a leaf segment.`);
  }

  const existing = Reflect.get(current, leafKey);
  if (!(existing instanceof MxArray)) {
    throw new Error(`assignWeightPath: "${path}" does not point to an MxArray parameter.`);
  }

  if (!sameShape(existing.shape, tensor.shape)) {
    throw new WeightMismatchError(path, existing.shape, tensor.shape);
  }

  existing.free();
  Reflect.set(current, leafKey, tensor);
}

/** List all leaf parameter paths in stable tree traversal order. */
export function listParameterPaths(tree: ParameterTree): string[] {
  return treeFlatten(tree).map(([path]) => path.join("."));
}

/** Format a parameter path list for logging or error messages. */
export function formatParameterPaths(paths: readonly string[]): string {
  return [...paths].toSorted((left, right) => left.localeCompare(right)).join(", ");
}

/** Format a weight shape mismatch for assertion helpers. */
export function describeWeightShape(tensor: MxArray): string {
  return formatShape(tensor.shape);
}
