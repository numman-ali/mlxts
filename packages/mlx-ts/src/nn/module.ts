/**
 * Base class for neural network modules.
 *
 * Subclasses declare parameters as plain `MxArray` properties and
 * sub-modules as `Module` properties. The base class scans these
 * to build parameter trees for autograd and optimization.
 *
 * Convention: public `MxArray` and `Module` fields are scanned as
 * parameters. Internal state that is not a parameter must use JS
 * `#` private fields or be a non-MxArray type.
 *
 * @module
 */

import { MxArray } from "../core/array";
import type { ParameterTree } from "../utils/tree";
import { isParameterTree } from "../utils/tree";

function formatPath(path: readonly string[]): string {
  return path.length === 0 ? "<root>" : path.join(".");
}

/**
 * Abstract base class for neural network modules.
 *
 * Provides parameter collection, tree-based update, freeze/unfreeze,
 * and training mode propagation. Subclasses implement `forward()`.
 *
 * Phase 3 assumes each public parameter field is a distinct registration
 * site. Shared public aliases are intentionally unsupported until Phase 4
 * defines explicit weight-tying semantics for trees and optimizers.
 */
export abstract class Module implements Disposable {
  #training = true;
  #frozenKeys: Set<string> = new Set();
  #ownKeys: readonly string[] | null = null;
  #ownKeySet: ReadonlySet<string> | null = null;

  /** Whether this module is in training mode. Affects Dropout behavior. */
  get isTraining(): boolean {
    return this.#training;
  }

  /** Forward pass. Subclasses must implement this. */
  abstract forward(...args: MxArray[]): MxArray;

  /**
   * Own enumerable keys are captured on first scan and treated as stable.
   *
   * This keeps the property-scanning design readable while avoiding repeated
   * `Object.keys(this)` work in hot training loops. Public parameter and
   * sub-module fields should therefore be assigned during construction or as
   * class field initializers before the first parameter scan happens.
   */
  private ownKeys(): readonly string[] {
    if (this.#ownKeys === null) {
      const keys = Object.freeze([...Object.keys(this)]);
      this.#ownKeys = keys;
      this.#ownKeySet = new Set(keys);
    }
    return this.#ownKeys;
  }

  private hasOwnKey(key: string): boolean {
    this.ownKeys();
    const ownKeySet = this.#ownKeySet;
    if (ownKeySet === null) {
      throw new Error("Module: own key cache was not initialized");
    }
    return ownKeySet.has(key);
  }

  private slotValue(key: string): unknown {
    return Reflect.get(this, key);
  }

  private collectParameters(trainableOnly: boolean): ParameterTree {
    const tree: ParameterTree = {};
    for (const key of this.ownKeys()) {
      if (trainableOnly && this.#frozenKeys.has(key)) {
        continue;
      }

      const value = this.slotValue(key);
      if (value instanceof MxArray) {
        tree[key] = value;
      } else if (value instanceof Module) {
        const subtree = trainableOnly ? value.trainableParameters() : value.parameters();
        if (!trainableOnly || Object.keys(subtree).length > 0) {
          tree[key] = subtree;
        }
      }
    }
    return tree;
  }

  private assertUpdatableLeaf(current: unknown, path: readonly string[]): void {
    if (current instanceof MxArray) {
      return;
    }
    if (current instanceof Module) {
      throw new Error(
        `Module.update: key "${formatPath(path)}" is a sub-module and expects a nested parameter tree`,
      );
    }
    throw new Error(`Module.update: key "${formatPath(path)}" is not a parameter slot`);
  }

  private assertUpdatableSubtree(current: unknown, path: readonly string[]): Module {
    if (current instanceof Module) {
      return current;
    }
    if (current instanceof MxArray) {
      throw new Error(
        `Module.update: key "${formatPath(path)}" is a parameter leaf and expects an MxArray update`,
      );
    }
    throw new Error(`Module.update: key "${formatPath(path)}" is not a sub-module`);
  }

  private validateUpdateTree(params: ParameterTree, path: readonly string[]): void {
    for (const [key, value] of Object.entries(params)) {
      const nextPath = [...path, key];
      if (!this.hasOwnKey(key)) {
        throw new Error(`Module.update: unknown key "${formatPath(nextPath)}"`);
      }

      const current = this.slotValue(key);
      if (value instanceof MxArray) {
        this.assertUpdatableLeaf(current, nextPath);
      } else if (isParameterTree(value)) {
        const child = this.assertUpdatableSubtree(current, nextPath);
        child.validateUpdateTree(value, nextPath);
      }
    }
  }

  private applyUpdateTree(params: ParameterTree): void {
    for (const [key, value] of Object.entries(params)) {
      if (value instanceof MxArray) {
        Reflect.set(this, key, value);
      } else if (isParameterTree(value)) {
        const current = this.slotValue(key);
        if (current instanceof Module) {
          current.applyUpdateTree(value);
        }
      }
    }
  }

  private assertFreezableKey(key: string, operation: string): void {
    if (!this.hasOwnKey(key)) {
      throw new Error(`${operation}: unknown key "${key}"`);
    }

    const value = this.slotValue(key);
    if (!(value instanceof MxArray) && !(value instanceof Module)) {
      throw new Error(`${operation}: key "${key}" is not a parameter or sub-module`);
    }
  }

  /**
   * Collect all parameters as a nested tree mirroring module hierarchy.
   *
   * Scans own enumerable properties for `MxArray` (leaf parameters) and
   * `Module` (sub-modules to recurse into). Properties that are `null`,
   * numbers, strings, or other non-tensor types are silently skipped.
   */
  parameters(): ParameterTree {
    return this.collectParameters(false);
  }

  /**
   * Collect trainable parameters (excludes frozen keys).
   *
   * Frozen keys are excluded at this module level. Sub-modules manage
   * their own frozen state independently.
   */
  trainableParameters(): ParameterTree {
    return this.collectParameters(true);
  }

  /**
   * Recursive partial merge of parameters into this module.
   *
   * Only touches keys present in `params`. Keys not in the provided tree
   * are left unchanged. This is critical because optimizers and
   * nn.valueAndGrad operate on trainable subtrees, not the full tree.
   */
  update(params: ParameterTree): void {
    this.validateUpdateTree(params, []);
    this.applyUpdateTree(params);
  }

  /**
   * Freeze parameters by key (exclude from `trainableParameters()`).
   *
   * Operates on this module's direct children only. To freeze a
   * sub-module's parameter, call `freeze()` on the sub-module directly.
   *
   * @param keys - Keys to freeze. If omitted, freezes all own parameter keys.
   */
  freeze(keys?: string[]): this {
    if (keys === undefined) {
      for (const key of this.ownKeys()) {
        const value = this.slotValue(key);
        if (value instanceof MxArray || value instanceof Module) {
          this.#frozenKeys.add(key);
        }
      }
    } else {
      for (const key of keys) {
        this.assertFreezableKey(key, "Module.freeze");
      }
      for (const key of keys) {
        this.#frozenKeys.add(key);
      }
    }
    return this;
  }

  /**
   * Unfreeze parameters by key (re-include in `trainableParameters()`).
   *
   * @param keys - Keys to unfreeze. If omitted, unfreezes all keys.
   */
  unfreeze(keys?: string[]): this {
    if (keys === undefined) {
      this.#frozenKeys.clear();
    } else {
      for (const key of keys) {
        this.assertFreezableKey(key, "Module.unfreeze");
      }
      for (const key of keys) {
        this.#frozenKeys.delete(key);
      }
    }
    return this;
  }

  /**
   * Set training mode. Propagates recursively to all sub-modules.
   * Affects behavior of layers like Dropout.
   */
  train(mode?: boolean): this {
    this.#training = mode ?? true;
    for (const key of this.ownKeys()) {
      const val = this.slotValue(key);
      if (val instanceof Module) val.train(mode);
    }
    return this;
  }

  /** Set eval mode (training = false). */
  eval(): this {
    return this.train(false);
  }

  /** Free all owned parameter arrays and dispose sub-modules. */
  [Symbol.dispose](): void {
    for (const key of this.ownKeys()) {
      const val = this.slotValue(key);
      if (val instanceof MxArray) val.free();
      if (val instanceof Module) val[Symbol.dispose]();
    }
  }
}
