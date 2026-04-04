/**
 * Neural network value-and-grad bridge.
 *
 * Wraps a loss function that uses a Module into Phase 2's flat
 * valueAndGrad transform. Handles tree flatten/unflatten and
 * model parameter save/restore.
 *
 * The restore is in a `finally` block — unconditional, even on throw.
 * This guarantees the model is never left holding freed temporary
 * handles from the closure bridge.
 *
 * @module
 */

import type { FlatEntry, MxArray, ParameterTree } from "@mlxts/core";
import {
  valueAndGrad as coreValueAndGrad,
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

function createCachedTransform(
  model: Module,
  lossFn: (...args: MxArray[]) => MxArray,
  paths: readonly string[][],
): DisposableTransform<(...args: MxArray[]) => [MxArray, MxArray[]]> {
  const numParams = paths.length;
  const argnums = Array.from({ length: numParams }, (_, i) => i);

  const innerFn = (...allArgs: MxArray[]): MxArray => {
    const paramArgs = allArgs.slice(0, numParams);
    const userArgs = allArgs.slice(numParams);
    const entries: FlatEntry[] = [];

    for (let i = 0; i < paths.length; i++) {
      const path = paths[i];
      const arr = paramArgs[i];
      if (path === undefined || arr === undefined) {
        throw new Error(`nn.valueAndGrad: missing parameter at index ${i}`);
      }
      entries.push([path, arr]);
    }

    model.update(treeUnflatten(entries));
    return lossFn(...userArgs);
  };

  return coreValueAndGrad(innerFn, argnums);
}

function assertStableTrainableStructure(
  cachedPaths: readonly string[][],
  currentPaths: readonly string[][],
): void {
  if (samePaths(cachedPaths, currentPaths)) {
    return;
  }

  throw new Error(
    "nn.valueAndGrad: trainable parameter structure changed after the transform was created. " +
      `Expected [${formatPaths(cachedPaths)}], got [${formatPaths(currentPaths)}]. ` +
      "Recreate the transform after freeze/unfreeze or module surgery.",
  );
}

function gradientEntries(paths: readonly string[][], gradArrays: readonly MxArray[]): FlatEntry[] {
  const entries: FlatEntry[] = [];

  for (let index = 0; index < paths.length; index++) {
    const path = paths[index];
    const grad = gradArrays[index];
    if (path === undefined || grad === undefined) {
      throw new Error(`nn.valueAndGrad: missing gradient at index ${index}`);
    }
    entries.push([path, grad]);
  }

  return entries;
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
 * Create a function that computes both the loss and parameter gradients
 * for a neural network module.
 *
 * The returned function takes arbitrary MxArray data inputs and produces
 * `[loss, gradTree]` where `gradTree` has the same structure as
 * `model.trainableParameters()`.
 *
 * @param model - The module whose trainable parameters are differentiated.
 * @param lossFn - A function that takes data inputs and returns a scalar loss.
 *   It should call `model.forward()` internally — the model's parameters are
 *   automatically injected as differentiable variables.
 * @returns A function `(...dataInputs) => [loss, gradTree]`.
 */
export function valueAndGrad(
  model: Module,
  lossFn: (...args: MxArray[]) => MxArray,
): DisposableTransform<(...args: MxArray[]) => [MxArray, ParameterTree]> {
  let cachedPaths: string[][] | null = null;
  let cachedVgFn: DisposableTransform<(...args: MxArray[]) => [MxArray, MxArray[]]> | null = null;
  let disposed = false;

  const transformed = attachDisposable(
    (...args: MxArray[]): [MxArray, ParameterTree] => {
      if (disposed) {
        throw new Error("nn.valueAndGrad: transform has already been disposed");
      }
      // 1. Snapshot trainable parameters
      const flat = treeFlatten(model.trainableParameters());
      const paths = flat.map(([p]) => p);
      const paramArrays = flat.map(([, v]) => v);

      // Edge case: no trainable params (all frozen or empty model)
      if (paramArrays.length === 0) {
        return [lossFn(...args), {}];
      }

      // 2. Save original MxArray references for unconditional restore
      const saved: FlatEntry[] = flat.map(([p, v]) => [p, v]);

      // 3. Build and cache the flat transform once per stable trainable structure.
      if (cachedPaths === null) {
        cachedPaths = paths.map((path) => [...path]);
        cachedVgFn = createCachedTransform(model, lossFn, cachedPaths);
      } else {
        assertStableTrainableStructure(cachedPaths, paths);
      }

      if (cachedVgFn === null) {
        throw new Error("nn.valueAndGrad: cached transform was not initialized");
      }

      try {
        const [loss, gradArrays] = cachedVgFn(...paramArrays, ...args);
        const stablePaths = cachedPaths;
        if (stablePaths === null) {
          throw new Error("nn.valueAndGrad: cached paths were not initialized");
        }
        return [loss, treeUnflatten(gradientEntries(stablePaths, gradArrays))];
      } finally {
        // 6. Unconditional restore — even on throw
        // The closure bridge freed its temporary handles. Restore the
        // model's original live handles so it's never in a bad state.
        model.update(treeUnflatten(saved));
      }
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      cachedPaths = null;
      if (cachedVgFn !== null) {
        cachedVgFn[Symbol.dispose]();
        cachedVgFn = null;
      }
    },
  );

  return transformed;
}
