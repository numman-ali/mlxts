import type { ParameterTree } from "@mlxts/core";
import type { AdamW } from "@mlxts/optimizers";

export const CHECKPOINT_VERSION = 2;
export const MANIFEST_FILENAME = "manifest.json";
export const TENSOR_DATA_FILENAME = "tensors.bin";

export type CheckpointKind = "snapshot" | "resume" | "best";

export type SupportedCheckpointDType =
  | "bool"
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "float32"
  | "float64";

export interface CheckpointTensorMeta {
  shape: number[];
  dtype: SupportedCheckpointDType;
  offset: number;
  byteLength: number;
}

export interface AdamWOptimizerManifest {
  kind: "adamw";
  step: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  state: Record<string, Record<string, CheckpointTensorMeta>>;
}

export interface CheckpointManifest {
  version: 2;
  kind: CheckpointKind;
  metadata: unknown;
  step: number;
  parameters: Record<string, CheckpointTensorMeta>;
  optimizer?: AdamWOptimizerManifest;
}

/** Serialized checkpoint tensor payload. */
export interface CheckpointTensor {
  shape: number[];
  dtype: SupportedCheckpointDType;
  data: Uint8Array;
}

/** Serialized AdamW optimizer payload with binary tensors detached from live arrays. */
export interface AdamWOptimizerCheckpoint {
  kind: "adamw";
  step: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  state: Record<string, Record<string, CheckpointTensor>>;
}

/** In-memory checkpoint representation. */
export interface CheckpointData<TMetadata = unknown> {
  version: 2;
  kind: CheckpointKind;
  metadata: TMetadata;
  step: number;
  parameters: Record<string, CheckpointTensor>;
  optimizer?: AdamWOptimizerCheckpoint;
}

/** Module-like object with a parameter tree and transactional updates. */
export interface ParameterizedModel {
  parameters(): ParameterTree;
  update(tree: ParameterTree): void;
}

export interface SaveCheckpointOptions<TMetadata> {
  model: ParameterizedModel;
  kind: CheckpointKind;
  metadata: TMetadata;
  step: number;
  path: string;
  optimizer?: AdamW | undefined;
}

export type SupportedTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;
