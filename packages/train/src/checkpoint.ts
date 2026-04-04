/**
 * Generic model checkpointing for trainable module-like objects.
 *
 * @module
 */

import { type MxArray, mxEval, treeUnflatten } from "@mlxts/core";
import { AdamW } from "@mlxts/optimizers";
import { readFileSync } from "fs";
import { join } from "path";
import { writeCheckpointDirectory } from "./checkpoint-io";
import { readManifest, shiftOptimizerOffsets } from "./checkpoint-manifest";
import {
  createCheckpointArray,
  currentParameterEntries,
  loadOptimizerState,
  readTensorSlice,
  serializeOptimizer,
  serializeParameters,
} from "./checkpoint-serialization";
import {
  type AdamWOptimizerCheckpoint,
  CHECKPOINT_VERSION,
  type CheckpointData,
  type CheckpointTensor,
  type ParameterizedModel,
  type SaveCheckpointOptions,
  TENSOR_DATA_FILENAME,
} from "./checkpoint-types";

export type {
  AdamWOptimizerCheckpoint,
  CheckpointData,
  CheckpointKind,
  CheckpointTensor,
  ParameterizedModel,
  SaveCheckpointOptions,
} from "./checkpoint-types";

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index++) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
}

function currentParameterMap(
  model: ParameterizedModel,
): Map<string, { path: string[]; value: MxArray }> {
  const map = new Map<string, { path: string[]; value: MxArray }>();
  for (const [path, value] of currentParameterEntries(model)) {
    map.set(path.join("."), { path, value });
  }
  return map;
}

function assertCheckpointMatchesModel(
  modelParameters: ReadonlyMap<string, { path: string[]; value: MxArray }>,
  checkpointParameters: Record<string, CheckpointTensor>,
): void {
  const checkpointKeys = new Set(Object.keys(checkpointParameters));

  for (const key of modelParameters.keys()) {
    const checkpointValue = checkpointParameters[key];
    if (checkpointValue === undefined) {
      throw new Error(`applyCheckpoint: missing checkpoint parameter "${key}"`);
    }

    const modelValue = modelParameters.get(key);
    if (modelValue === undefined) {
      throw new Error(`applyCheckpoint: model parameter "${key}" was unexpectedly unavailable`);
    }

    if (modelValue.value.dtype !== checkpointValue.dtype) {
      throw new Error(
        `applyCheckpoint: dtype mismatch for "${key}" (${checkpointValue.dtype} checkpoint vs ${modelValue.value.dtype} model)`,
      );
    }
    if (!sameShape(modelValue.value.shape, checkpointValue.shape)) {
      throw new Error(
        `applyCheckpoint: shape mismatch for "${key}" ([${checkpointValue.shape}] checkpoint vs [${modelValue.value.shape}] model)`,
      );
    }
  }

  for (const key of checkpointKeys) {
    if (!modelParameters.has(key)) {
      throw new Error(`applyCheckpoint: unexpected checkpoint parameter "${key}"`);
    }
  }
}

/** Save a generic checkpoint to a directory. */
export function saveCheckpoint<TMetadata>(options: SaveCheckpointOptions<TMetadata>): void {
  const { model, optimizer, kind, metadata, step, path } = options;
  if (kind === "resume" && optimizer === undefined) {
    throw new Error("saveCheckpoint: resume checkpoints require optimizer state");
  }
  if ((kind === "snapshot" || kind === "best") && optimizer !== undefined) {
    throw new Error("saveCheckpoint: snapshot/best checkpoints must not include optimizer state");
  }
  if (optimizer !== undefined && optimizer.step !== step) {
    throw new Error(
      `saveCheckpoint: optimizer step ${optimizer.step} does not match checkpoint step ${step}`,
    );
  }

  const modelData = serializeParameters(model);
  const optimizerData = optimizer === undefined ? undefined : serializeOptimizer(optimizer);
  const totalBytes = modelData.bytes.byteLength + (optimizerData?.bytes.byteLength ?? 0);
  const bytes = new Uint8Array(totalBytes);
  bytes.set(modelData.bytes, 0);
  if (optimizerData !== undefined) {
    bytes.set(optimizerData.bytes, modelData.bytes.byteLength);
  }

  const manifest = {
    version: CHECKPOINT_VERSION,
    kind,
    metadata,
    step,
    parameters: modelData.parameters,
  } as const;

  writeCheckpointDirectory(
    path,
    optimizerData === undefined
      ? manifest
      : {
          ...manifest,
          optimizer: {
            ...optimizerData.optimizer,
            state: shiftOptimizerOffsets(optimizerData.optimizer.state, modelData.bytes.byteLength),
          },
        },
    bytes,
  );
}

/** Load a generic checkpoint directory into memory. */
export function loadCheckpoint(path: string): CheckpointData {
  const manifest = readManifest(path);
  const tensorBytes = new Uint8Array(readFileSync(join(path, TENSOR_DATA_FILENAME)));
  const parameters: Record<string, CheckpointTensor> = {};

  for (const [key, meta] of Object.entries(manifest.parameters)) {
    parameters[key] = readTensorSlice(key, meta, tensorBytes);
  }

  const checkpoint: CheckpointData = {
    version: CHECKPOINT_VERSION,
    kind: manifest.kind,
    metadata: manifest.metadata,
    step: manifest.step,
    parameters,
  };
  if (manifest.optimizer !== undefined) {
    checkpoint.optimizer = {
      kind: "adamw",
      step: manifest.optimizer.step,
      lr: manifest.optimizer.lr,
      beta1: manifest.optimizer.beta1,
      beta2: manifest.optimizer.beta2,
      eps: manifest.optimizer.eps,
      weightDecay: manifest.optimizer.weightDecay,
      state: loadOptimizerState(manifest.optimizer.state, tensorBytes),
    };
  }
  return checkpoint;
}

/**
 * Apply loaded checkpoint weights to an existing model.
 *
 * Validation happens before mutation. Replacement arrays are staged first,
 * then installed only after the key set, shapes, and dtypes all match.
 */
export function applyCheckpoint<TMetadata>(
  model: ParameterizedModel,
  checkpoint: Pick<CheckpointData<TMetadata>, "parameters">,
): void {
  const current = currentParameterMap(model);
  assertCheckpointMatchesModel(current, checkpoint.parameters);

  const stagedEntries: [string[], MxArray][] = [];
  const stagedArrays: MxArray[] = [];

  try {
    for (const [key, { path }] of current.entries()) {
      const tensor = checkpoint.parameters[key];
      if (tensor === undefined) {
        throw new Error(`applyCheckpoint: missing checkpoint parameter "${key}"`);
      }

      const nextArray = createCheckpointArray(tensor);
      stagedArrays.push(nextArray);
      stagedEntries.push([path, nextArray]);
    }

    mxEval(...stagedArrays);
    model.update(treeUnflatten(stagedEntries));
  } catch (error) {
    for (const array of stagedArrays) {
      array.free();
    }
    throw error;
  }

  for (const { value } of current.values()) {
    value.free();
  }
}

/** Recreate an AdamW optimizer from serialized checkpoint state. */
export function restoreAdamWFromCheckpoint(checkpoint: AdamWOptimizerCheckpoint): AdamW {
  const optimizer = new AdamW({
    learningRate: checkpoint.lr,
    beta1: checkpoint.beta1,
    beta2: checkpoint.beta2,
    eps: checkpoint.eps,
    weightDecay: checkpoint.weightDecay,
  });
  const state: Record<string, Record<string, MxArray>> = {};
  const stagedArrays: MxArray[] = [];

  try {
    for (const [path, slots] of Object.entries(checkpoint.state)) {
      const slotState: Record<string, MxArray> = {};
      for (const [slotName, tensor] of Object.entries(slots)) {
        const array = createCheckpointArray(tensor);
        stagedArrays.push(array);
        slotState[slotName] = array;
      }
      state[path] = slotState;
    }

    mxEval(...stagedArrays);
    optimizer.restore({
      kind: "adamw",
      step: checkpoint.step,
      lr: checkpoint.lr,
      beta1: checkpoint.beta1,
      beta2: checkpoint.beta2,
      eps: checkpoint.eps,
      weightDecay: checkpoint.weightDecay,
      state,
    });
    return optimizer;
  } catch (error) {
    for (const array of stagedArrays) {
      array.free();
    }
    optimizer[Symbol.dispose]();
    throw error;
  }
}
