/**
 * Model-weight safetensors interop.
 *
 * This is intentionally separate from the repo's canonical training
 * checkpoint format. Safetensors covers model weights only.
 *
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { loadSafetensors, mxEval, saveSafetensors, treeFlatten, treeUnflatten } from "@mlxts/core";

import type { GPT } from "./model/gpt";

type ModelParameterEntry = {
  path: string[];
  value: MxArray;
};

function parameterKey(path: readonly string[]): string {
  return path.join(".");
}

function sortedParameterEntries(model: GPT): [string[], MxArray][] {
  return [...treeFlatten(model.parameters())].sort(([left], [right]) =>
    parameterKey(left).localeCompare(parameterKey(right)),
  );
}

function currentParameterMap(model: GPT): Map<string, ModelParameterEntry> {
  return new Map(
    sortedParameterEntries(model).map(([pathSegments, value]) => [
      parameterKey(pathSegments),
      { path: pathSegments, value },
    ]),
  );
}

function validateParameterKeys(currentKeys: string[], loadedKeys: string[]): void {
  if (loadedKeys.length !== currentKeys.length) {
    throw new Error(
      `loadModelSafetensors: expected ${currentKeys.length} model parameters, got ${loadedKeys.length}`,
    );
  }

  for (let index = 0; index < currentKeys.length; index++) {
    const expectedKey = currentKeys[index];
    const actualKey = loadedKeys[index];
    if (expectedKey === undefined || actualKey === undefined || expectedKey !== actualKey) {
      throw new Error(
        `loadModelSafetensors: parameter mismatch at index ${index} (${expectedKey ?? "<missing>"} vs ${actualKey ?? "<missing>"})`,
      );
    }
  }
}

function validateTensorCompatibility(
  current: Map<string, ModelParameterEntry>,
  loadedTensors: Record<string, MxArray>,
  orderedKeys: readonly string[],
): void {
  for (const key of orderedKeys) {
    const currentEntry = current.get(key);
    const loadedTensor = loadedTensors[key];
    if (currentEntry === undefined || loadedTensor === undefined) {
      throw new Error(`loadModelSafetensors: missing tensor for "${key}"`);
    }
    if (currentEntry.value.dtype !== loadedTensor.dtype) {
      throw new Error(
        `loadModelSafetensors: dtype mismatch for "${key}" (${loadedTensor.dtype} vs ${currentEntry.value.dtype})`,
      );
    }
    if (currentEntry.value.shape.join(",") !== loadedTensor.shape.join(",")) {
      throw new Error(
        `loadModelSafetensors: shape mismatch for "${key}" (${loadedTensor.shape.join("x")} vs ${currentEntry.value.shape.join("x")})`,
      );
    }
  }
}

function stagedUpdateEntries(
  current: Map<string, ModelParameterEntry>,
  loadedTensors: Record<string, MxArray>,
  orderedKeys: readonly string[],
): [string[], MxArray][] {
  return orderedKeys.map((key) => {
    const entry = current.get(key);
    const value = loadedTensors[key];
    if (entry === undefined || value === undefined) {
      throw new Error(`loadModelSafetensors: missing staged tensor for "${key}"`);
    }
    return [entry.path, value];
  });
}

/**
 * Save model parameters to a safetensors file.
 */
export async function saveModelSafetensors(
  model: GPT,
  path: string,
  metadata: Record<string, string> = {},
): Promise<void> {
  const tensors: Record<string, MxArray> = {};
  for (const [pathSegments, value] of sortedParameterEntries(model)) {
    tensors[parameterKey(pathSegments)] = value;
  }
  await saveSafetensors(tensors, path, metadata);
}

/**
 * Load model parameters from a safetensors file and apply them transactionally.
 */
export async function loadModelSafetensors(
  model: GPT,
  path: string,
): Promise<{ metadata: Record<string, string> }> {
  const loaded = await loadSafetensors(path);
  const current = currentParameterMap(model);
  const loadedKeys = Object.keys(loaded.tensors).sort((left, right) => left.localeCompare(right));
  const currentKeys = [...current.keys()].sort((left, right) => left.localeCompare(right));

  try {
    validateParameterKeys(currentKeys, loadedKeys);
    validateTensorCompatibility(current, loaded.tensors, currentKeys);

    const staged = currentKeys
      .map((key) => loaded.tensors[key])
      .filter((value) => value !== undefined);
    if (staged.length > 0) {
      mxEval(...staged);
    }

    model.update(treeUnflatten(stagedUpdateEntries(current, loaded.tensors, currentKeys)));
    for (const { value } of current.values()) {
      value.free();
    }
    return { metadata: loaded.metadata };
  } catch (error) {
    for (const tensor of Object.values(loaded.tensors)) {
      tensor.free();
    }
    throw error;
  }
}
