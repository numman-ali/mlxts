/**
 * Base optimizer class.
 *
 * Optimizers update model parameters using gradients.
 * State (momentum, variance) is stored in a flat Map keyed
 * by dot-joined parameter paths.
 *
 * @module
 */

import type { FlatEntry, MxArray, ParameterTree } from "@mlxts/core";
import { treeFlatten, treeUnflatten } from "@mlxts/core";
import type { Module } from "@mlxts/nn";

type StateEntry = Record<string, MxArray>;
type ParameterUpdate = {
  parameter: MxArray;
  state?: StateEntry;
};

type StagedUpdates = {
  entries: FlatEntry[];
  nextState: Map<string, StateEntry>;
  newParameters: MxArray[];
  newStateArrays: MxArray[];
};

function toPathKey(path: readonly string[]): string {
  return path.join(".");
}

function buildGradientLookup(
  parameterKeys: ReadonlySet<string>,
  flatGrads: FlatEntry[],
): Map<string, MxArray> {
  const gradByPath = new Map<string, MxArray>();
  for (const [path, grad] of flatGrads) {
    gradByPath.set(toPathKey(path), grad);
  }

  for (const key of parameterKeys) {
    if (!gradByPath.has(key)) {
      throw new Error(`Optimizer: no gradient for parameter "${key}"`);
    }
  }

  for (const key of gradByPath.keys()) {
    if (!parameterKeys.has(key)) {
      throw new Error(`Optimizer: unexpected gradient for parameter "${key}"`);
    }
  }

  return gradByPath;
}

/** Base class for optimizers. Implements Disposable for explicit state cleanup. */
export abstract class Optimizer implements Disposable {
  /** Per-parameter optimizer state (m, v for Adam; velocity for SGD). */
  protected state: Map<string, StateEntry> = new Map();

  protected exportStateSnapshot(): Record<string, StateEntry> {
    const snapshot: Record<string, StateEntry> = {};
    for (const [key, entry] of this.state.entries()) {
      snapshot[key] = entry;
    }
    return snapshot;
  }

  protected replaceStateSnapshot(nextState: Record<string, StateEntry>): void {
    this[Symbol.dispose]();
    this.state = new Map(Object.entries(nextState));
  }

  private preserveInactiveState(parameterKeys: ReadonlySet<string>): Map<string, StateEntry> {
    const nextState = new Map<string, StateEntry>();
    for (const [key, entry] of this.state.entries()) {
      if (!parameterKeys.has(key)) {
        nextState.set(key, entry);
      }
    }
    return nextState;
  }

  private stageUpdates(
    oldParams: FlatEntry[],
    gradByPath: ReadonlyMap<string, MxArray>,
    nextState: Map<string, StateEntry>,
  ): StagedUpdates {
    const entries: FlatEntry[] = [];
    const newParameters: MxArray[] = [];
    const newStateArrays: MxArray[] = [];

    for (const [path, param] of oldParams) {
      const key = toPathKey(path);
      const grad = gradByPath.get(key);
      if (grad === undefined) {
        throw new Error(`Optimizer: no gradient for parameter "${key}"`);
      }

      const result = this.applySingle(key, param, grad, this.state.get(key));
      newParameters.push(result.parameter);
      entries.push([path, result.parameter]);

      if (result.state !== undefined) {
        nextState.set(key, result.state);
        for (const arr of Object.values(result.state)) {
          newStateArrays.push(arr);
        }
      }
    }

    return { entries, nextState, newParameters, newStateArrays };
  }

  private freeStagedUpdates(staged: StagedUpdates | null): void {
    if (staged === null) {
      return;
    }

    for (const arr of staged.newParameters) {
      arr.free();
    }
    for (const arr of staged.newStateArrays) {
      arr.free();
    }
  }

  private freeReplacedState(
    previousState: ReadonlyMap<string, StateEntry>,
    parameterKeys: ReadonlySet<string>,
  ): void {
    for (const [key, entry] of previousState.entries()) {
      if (!parameterKeys.has(key)) {
        continue;
      }

      for (const arr of Object.values(entry)) {
        arr.free();
      }
    }
  }

  /**
   * Apply gradients to model parameters.
   *
   * Uses path-keyed gradient lookup (not positional matching) for safety.
   * Frees old parameter arrays after successful update. Cleans up
   * partially created arrays if applySingle or model.update throws.
   *
   * @param model - The model to update.
   * @param gradients - Gradient tree (same structure as model.trainableParameters()).
   */
  update(model: Module, gradients: ParameterTree): void {
    const oldParams = treeFlatten(model.trainableParameters());
    const flatGrads = treeFlatten(gradients);
    const parameterKeys = new Set(oldParams.map(([path]) => toPathKey(path)));
    const gradByPath = buildGradientLookup(parameterKeys, flatGrads);
    const nextState = this.preserveInactiveState(parameterKeys);
    let staged: StagedUpdates | null = null;

    try {
      staged = this.stageUpdates(oldParams, gradByPath, nextState);
      model.update(treeUnflatten(staged.entries));
    } catch (error) {
      this.freeStagedUpdates(staged);
      throw error;
    }

    const previousState = this.state;
    this.state = nextState;

    // Free OLD parameter arrays that were successfully replaced
    try {
      for (const [, arr] of oldParams) {
        arr.free();
      }
    } finally {
      this.freeReplacedState(previousState, parameterKeys);
    }
  }

  /**
   * Update a single parameter given its gradient.
   * Subclasses implement this with their specific update rule.
   */
  protected abstract applySingle(
    key: string,
    param: MxArray,
    grad: MxArray,
    previousState?: Readonly<StateEntry>,
  ): ParameterUpdate;

  /** Collect all optimizer state arrays (for mxEval). */
  stateArrays(): MxArray[] {
    const arrays: MxArray[] = [];
    for (const entry of this.state.values()) {
      for (const arr of Object.values(entry)) {
        arrays.push(arr);
      }
    }
    return arrays;
  }

  /** Free all state arrays. */
  [Symbol.dispose](): void {
    for (const entry of this.state.values()) {
      for (const arr of Object.values(entry)) {
        arr.free();
      }
    }
    this.state.clear();
  }
}
