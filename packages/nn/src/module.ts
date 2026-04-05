/**
 * Base class for neural network modules.
 * Subclasses declare parameters as plain `MxArray` properties and sub-modules
 * as `Module` properties. The base class scans these to build parameter trees
 * for autograd and optimization.
 * Public `MxArray` and `Module` fields are scanned as parameters. Internal
 * state that is not a parameter must use JS `#` private fields or be a
 * non-MxArray type.
 * @module
 */

import type { ParameterTree } from "@mlxts/core";
import { isParameterTree, MxArray } from "@mlxts/core";

function formatPath(path: readonly string[]): string {
  return path.length === 0 ? "<root>" : path.join(".");
}

type ModuleArrayState =
  | { kind: "module-array"; modules: Module[] }
  | { kind: "mixed-array" }
  | { kind: "other" };

function moduleArrayState(value: unknown): ModuleArrayState {
  if (!Array.isArray(value) || value.length === 0) {
    return { kind: "other" };
  }

  const hasModule = value.some((entry) => entry instanceof Module);
  if (!hasModule) {
    return { kind: "other" };
  }

  if (value.every((entry) => entry instanceof Module)) {
    return { kind: "module-array", modules: value };
  }

  return { kind: "mixed-array" };
}

/**
 * Abstract base class for neural network modules.
 *
 * Provides parameter collection, tree-based update, freeze/unfreeze,
 * and training mode propagation. Subclasses implement `forward()`.
 *
 * Shared public parameter aliases are intentionally unsupported. Weight tying
 * should stay functional (for example via `Embedding.asLinear()`) until the
 * parameter tree and optimizer layers grow explicit alias semantics.
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

  private collectChildParameters(child: Module, trainableOnly: boolean): ParameterTree | null {
    const subtree = trainableOnly ? child.trainableParameters() : child.parameters();
    return trainableOnly && Object.keys(subtree).length === 0 ? null : subtree;
  }

  private collectModuleArrayParameters(
    children: Module[],
    trainableOnly: boolean,
  ): ParameterTree | null {
    const subtree: ParameterTree = {};
    for (let index = 0; index < children.length; index++) {
      const child = children[index];
      if (child === undefined) {
        continue;
      }

      const childParams = this.collectChildParameters(child, trainableOnly);
      if (childParams !== null) {
        subtree[String(index)] = childParams;
      }
    }

    return Object.keys(subtree).length === 0 ? null : subtree;
  }

  private assignParameterValue(
    tree: ParameterTree,
    key: string,
    value: unknown,
    trainableOnly: boolean,
  ): void {
    if (value instanceof MxArray) {
      tree[key] = value;
      return;
    }

    if (value instanceof Module) {
      const subtree = this.collectChildParameters(value, trainableOnly);
      if (subtree !== null) {
        tree[key] = subtree;
      }
      return;
    }

    const arrayState = moduleArrayState(value);
    if (arrayState.kind === "mixed-array") {
      throw new Error(
        `Module.parameters: property "${key}" mixes Module and non-Module values in one array`,
      );
    }

    if (arrayState.kind === "module-array") {
      const subtree = this.collectModuleArrayParameters(arrayState.modules, trainableOnly);
      if (subtree !== null) {
        tree[key] = subtree;
      }
    }
  }

  private collectParameters(trainableOnly: boolean): ParameterTree {
    const tree: ParameterTree = {};
    for (const key of this.ownKeys()) {
      if (trainableOnly && this.#frozenKeys.has(key)) {
        continue;
      }

      this.assignParameterValue(tree, key, this.slotValue(key), trainableOnly);
    }
    return tree;
  }

  private assertUpdatableLeaf(current: unknown, path: readonly string[]): void {
    if (current instanceof MxArray) {
      return;
    }
    const arrayState = moduleArrayState(current);
    if (
      current instanceof Module ||
      arrayState.kind === "module-array" ||
      arrayState.kind === "mixed-array"
    ) {
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

  private validateModuleArrayEntry(
    current: Module[],
    parentPath: readonly string[],
    indexKey: string,
    childValue: MxArray | ParameterTree,
  ): void {
    const childPath = [...parentPath, indexKey];
    const index = Number(indexKey);
    if (!Number.isInteger(index) || index < 0 || index >= current.length) {
      throw new Error(
        `Module.update: invalid index "${formatPath(childPath)}" for module array of length ${current.length}`,
      );
    }

    const child = current[index];
    if (child === undefined) {
      return;
    }

    if (childValue instanceof MxArray) {
      throw new Error(
        `Module.update: key "${formatPath(childPath)}" is a sub-module and expects a nested parameter tree`,
      );
    }

    child.validateUpdateTree(childValue, childPath);
  }

  private validateSubtreeUpdate(
    current: unknown,
    value: ParameterTree,
    nextPath: readonly string[],
  ): void {
    if (current instanceof Module) {
      current.validateUpdateTree(value, nextPath);
      return;
    }

    const arrayState = moduleArrayState(current);
    if (arrayState.kind === "mixed-array") {
      throw new Error(
        `Module.update: key "${formatPath(nextPath)}" mixes Module and non-Module values in one array`,
      );
    }

    if (arrayState.kind === "module-array") {
      for (const [indexKey, childValue] of Object.entries(value)) {
        this.validateModuleArrayEntry(arrayState.modules, nextPath, indexKey, childValue);
      }
      return;
    }

    this.assertUpdatableSubtree(current, nextPath);
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
        continue;
      }

      this.validateSubtreeUpdate(current, value, nextPath);
    }
  }

  private applyModuleArrayUpdate(current: Module[], value: ParameterTree): void {
    for (const [indexKey, childValue] of Object.entries(value)) {
      const child = current[Number(indexKey)];
      if (child !== undefined && isParameterTree(childValue)) {
        child.applyUpdateTree(childValue);
      }
    }
  }

  private applySubtreeUpdate(current: unknown, value: ParameterTree): void {
    if (current instanceof Module) {
      current.applyUpdateTree(value);
      return;
    }

    const arrayState = moduleArrayState(current);
    if (arrayState.kind === "mixed-array") {
      throw new Error("Module.update: mixed Module arrays are unsupported");
    }

    if (arrayState.kind === "module-array") {
      this.applyModuleArrayUpdate(arrayState.modules, value);
    }
  }

  private applyUpdateTree(params: ParameterTree): void {
    for (const [key, value] of Object.entries(params)) {
      if (value instanceof MxArray) {
        Reflect.set(this, key, value);
        continue;
      }

      this.applySubtreeUpdate(this.slotValue(key), value);
    }
  }

  private assertFreezableKey(key: string, operation: string): void {
    if (!this.hasOwnKey(key)) {
      throw new Error(`${operation}: unknown key "${key}"`);
    }

    const value = this.slotValue(key);
    const arrayState = moduleArrayState(value);
    if (arrayState.kind === "mixed-array") {
      throw new Error(`${operation}: key "${key}" mixes Module and non-Module values in one array`);
    }

    if (
      !(value instanceof MxArray) &&
      !(value instanceof Module) &&
      arrayState.kind !== "module-array"
    ) {
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
   * Replace a direct child module and return the previous child.
   *
   * Ownership stays with the caller. The previous child is not disposed.
   */
  replaceChild(key: string, next: Module): Module {
    if (!this.hasOwnKey(key)) {
      throw new Error(`Module.replaceChild: unknown key "${key}"`);
    }

    const current = this.slotValue(key);
    if (!(current instanceof Module)) {
      throw new Error(`Module.replaceChild: key "${key}" is not a direct Module child`);
    }

    next.train(this.#training);
    Reflect.set(this, key, next);
    return current;
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
        const arrayState = moduleArrayState(value);
        if (arrayState.kind === "mixed-array") {
          throw new Error(
            `Module.freeze: key "${key}" mixes Module and non-Module values in one array`,
          );
        }

        if (
          value instanceof MxArray ||
          value instanceof Module ||
          arrayState.kind === "module-array"
        ) {
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
      this.applyTrainingMode(this.slotValue(key), mode);
    }
    return this;
  }

  /** Set eval mode (training = false). */
  eval(): this {
    return this.train(false);
  }

  private applyTrainingMode(value: unknown, mode: boolean | undefined): void {
    if (value instanceof Module) {
      value.train(mode);
      return;
    }

    const arrayState = moduleArrayState(value);
    if (arrayState.kind === "mixed-array") {
      throw new Error("Module.train: mixed Module arrays are unsupported");
    }

    if (arrayState.kind === "module-array") {
      for (const child of arrayState.modules) {
        child.train(mode);
      }
    }
  }

  /** Free all owned parameter arrays and dispose sub-modules. */
  private disposeValue(value: unknown): void {
    if (value instanceof MxArray) {
      value.free();
      return;
    }

    if (value instanceof Module) {
      value[Symbol.dispose]();
      return;
    }

    const arrayState = moduleArrayState(value);
    if (arrayState.kind === "mixed-array") {
      throw new Error("Module.dispose: mixed Module arrays are unsupported");
    }

    if (arrayState.kind === "module-array") {
      for (const child of arrayState.modules) {
        child[Symbol.dispose]();
      }
    }
  }

  /** Free all owned parameter arrays and dispose sub-modules. */
  [Symbol.dispose](): void {
    for (const key of this.ownKeys()) {
      this.disposeValue(this.slotValue(key));
    }
  }
}
