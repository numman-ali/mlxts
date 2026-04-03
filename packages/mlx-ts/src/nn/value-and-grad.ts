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

import type { MxArray } from "../core/array";
import { valueAndGrad as coreValueAndGrad } from "../core/transforms";
import type { FlatEntry, ParameterTree } from "../utils/tree";
import { treeFlatten, treeUnflatten } from "../utils/tree";
import type { Module } from "./module";

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
): (...args: MxArray[]) => [MxArray, ParameterTree] {
  return (...args: MxArray[]): [MxArray, ParameterTree] => {
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

    // 3. Build inner function: positions [0..N-1] = params, [N..] = user args
    const numParams = paramArrays.length;
    const argnums = Array.from({ length: numParams }, (_, i) => i);

    const innerFn = (...allArgs: MxArray[]): MxArray => {
      const paramArgs = allArgs.slice(0, numParams);
      const userArgs = allArgs.slice(numParams);

      // Unflatten param arrays into a tree and inject into model
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

    // 4. Call Phase 2 valueAndGrad with all param indices
    const vgFn = coreValueAndGrad(innerFn, argnums);

    try {
      const [loss, gradArrays] = vgFn(...paramArrays, ...args);

      // 5. Unflatten gradient arrays back to parameter tree structure
      const gradEntries: FlatEntry[] = [];
      for (let i = 0; i < paths.length; i++) {
        const path = paths[i];
        const grad = gradArrays[i];
        if (path === undefined || grad === undefined) {
          throw new Error(`nn.valueAndGrad: missing gradient at index ${i}`);
        }
        gradEntries.push([path, grad]);
      }

      return [loss, treeUnflatten(gradEntries)];
    } finally {
      // 6. Unconditional restore — even on throw
      // The closure bridge freed its temporary handles. Restore the
      // model's original live handles so it's never in a bad state.
      model.update(treeUnflatten(saved));
    }
  };
}
