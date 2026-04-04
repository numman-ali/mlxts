/**
 * Neural network checkpoint bridge.
 *
 * Wraps a module method in MLX gradient checkpointing while preserving
 * the tree-based parameter interface used by `Module`.
 *
 * @module
 */

import type { FlatEntry, MxArray, ParameterTree } from "@mlxts/core";
import {
  checkpoint as coreCheckpoint,
  type DisposableTransform,
  treeFlatten,
  treeUnflatten,
} from "@mlxts/core";
import type { Module } from "./module";

function samePaths(
  left: readonly (readonly string[])[],
  right: readonly (readonly string[])[],
): boolean {
  if (left.length !== right.length) {
    return false;
  }

  for (let index = 0; index < left.length; index++) {
    const leftPath = left[index];
    const rightPath = right[index];
    if (leftPath === undefined || rightPath === undefined || leftPath.length !== rightPath.length) {
      return false;
    }
    for (let segment = 0; segment < leftPath.length; segment++) {
      if (leftPath[segment] !== rightPath[segment]) {
        return false;
      }
    }
  }

  return true;
}

function formatPaths(paths: readonly (readonly string[])[]): string {
  return paths.map((path) => path.join(".")).join(", ");
}

function createSavedEntries(tree: ParameterTree): FlatEntry[] {
  return treeFlatten(tree).map(([path, value]) => [path, value]);
}

function attachDisposable<T extends (...args: MxArray[]) => unknown>(
  fn: T,
  dispose: () => void,
): DisposableTransform<T> {
  return Object.assign(fn, {
    [Symbol.dispose]: dispose,
  });
}

/**
 * Checkpoint a module-backed forward function.
 *
 * The returned function preserves the module's original parameter handles
 * after each invocation, even if the checkpointed call throws.
 */
export function checkpoint(
  module: Module,
  fn: (...args: MxArray[]) => MxArray = (...args) => module.forward(...args),
): DisposableTransform<(...args: MxArray[]) => MxArray> {
  let cachedPaths: string[][] | null = null;
  let cachedFn: DisposableTransform<(...args: MxArray[]) => MxArray> | null = null;
  let disposed = false;

  const transformed = attachDisposable(
    (...args: MxArray[]): MxArray => {
      if (disposed) {
        throw new Error("nn.checkpoint: transform has already been disposed");
      }
      const flat = treeFlatten(module.trainableParameters());
      const paths = flat.map(([path]) => path);
      const parameterArrays = flat.map(([, value]) => value);

      if (parameterArrays.length === 0) {
        return fn(...args);
      }

      const savedEntries = createSavedEntries(module.parameters());

      if (cachedPaths === null) {
        cachedPaths = paths.map((path) => [...path]);
        const stablePaths = cachedPaths;
        const parameterCount = stablePaths.length;

        const innerFn = (...allArgs: MxArray[]): MxArray => {
          const parameterArgs = allArgs.slice(0, parameterCount);
          const userArgs = allArgs.slice(parameterCount);
          const entries: FlatEntry[] = [];

          for (let index = 0; index < stablePaths.length; index++) {
            const path = stablePaths[index];
            const value = parameterArgs[index];
            if (path === undefined || value === undefined) {
              throw new Error(`nn.checkpoint: missing parameter at index ${index}`);
            }
            entries.push([path, value]);
          }

          module.update(treeUnflatten(entries));
          return fn(...userArgs);
        };

        cachedFn = coreCheckpoint(innerFn);
      } else if (!samePaths(cachedPaths, paths)) {
        throw new Error(
          "nn.checkpoint: trainable parameter structure changed after the transform was created. " +
            `Expected [${formatPaths(cachedPaths)}], got [${formatPaths(paths)}]. ` +
            "Recreate the checkpointed function after freeze/unfreeze or module surgery.",
        );
      }

      if (cachedFn === null) {
        throw new Error("nn.checkpoint: cached transform was not initialized");
      }

      try {
        return cachedFn(...parameterArrays, ...args);
      } finally {
        module.update(treeUnflatten(savedEntries));
      }
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      cachedPaths = null;
      if (cachedFn !== null) {
        cachedFn[Symbol.dispose]();
        cachedFn = null;
      }
    },
  );

  return transformed;
}
