/**
 * GPT-specific checkpoint adapters over @mlxts/train.
 *
 * nanogpt stores GPT config and tokenizer vocab as checkpoint metadata while
 * delegating the generic manifest, tensor, and optimizer state handling to the
 * canonical train package.
 *
 * @module
 */

import type { AdamW } from "@mlxts/optimizers";
import type { CharTokenizer } from "@mlxts/tokenizers";
import {
  applyCheckpoint as applyTrainCheckpoint,
  loadCheckpoint as loadTrainCheckpoint,
  restoreAdamWFromCheckpoint as restoreTrainAdamWFromCheckpoint,
  saveCheckpoint as saveTrainCheckpoint,
  type AdamWOptimizerCheckpoint as TrainAdamWOptimizerCheckpoint,
  type CheckpointData as TrainCheckpointData,
  type CheckpointKind as TrainCheckpointKind,
  type CheckpointTensor as TrainCheckpointTensor,
} from "@mlxts/train";
import type { GPTConfig } from "./config";
import { resolveConfig } from "./config";
import type { GPT } from "./model/gpt";

/** GPT-specific checkpoint metadata persisted via @mlxts/train. */
export interface GPTCheckpointMetadata {
  config: GPTConfig;
  tokenizer: {
    chars: string[];
  };
}

export type CheckpointKind = TrainCheckpointKind;
export type CheckpointTensor = TrainCheckpointTensor;
export type AdamWOptimizerCheckpoint = TrainAdamWOptimizerCheckpoint;
export type CheckpointData = TrainCheckpointData<GPTCheckpointMetadata>;

type SaveCheckpointOptions = {
  model: GPT;
  kind: CheckpointKind;
  config: GPTConfig;
  step: number;
  tokenizer: CharTokenizer;
  path: string;
  optimizer?: AdamW | undefined;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readNumber(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${context}: expected a finite number`);
  }
  return value;
}

function readString(value: unknown, context: string): string {
  if (typeof value !== "string") {
    throw new Error(`${context}: expected a string`);
  }
  return value;
}

function readBoolean(value: unknown, context: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${context}: expected a boolean`);
  }
  return value;
}

function readStringArray(value: unknown, context: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context}: expected a string array`);
  }

  const chars: string[] = [];
  for (let index = 0; index < value.length; index++) {
    chars.push(readString(value[index], `${context}[${index}]`));
  }
  return chars;
}

function readGPTCheckpointMetadata(value: unknown): GPTCheckpointMetadata {
  if (!isRecord(value)) {
    throw new Error("checkpoint metadata: expected an object");
  }
  if (!isRecord(value.config)) {
    throw new Error("checkpoint metadata.config: expected an object");
  }
  if (!isRecord(value.tokenizer)) {
    throw new Error("checkpoint metadata.tokenizer: expected an object");
  }

  const configRecord = value.config;
  const config = resolveConfig(
    {
      nLayer: readNumber(configRecord.nLayer, "checkpoint metadata.config.nLayer"),
      nHead: readNumber(configRecord.nHead, "checkpoint metadata.config.nHead"),
      nEmbd: readNumber(configRecord.nEmbd, "checkpoint metadata.config.nEmbd"),
      blockSize: readNumber(configRecord.blockSize, "checkpoint metadata.config.blockSize"),
      dropout: readNumber(configRecord.dropout, "checkpoint metadata.config.dropout"),
      gradientCheckpointing: readBoolean(
        configRecord.gradientCheckpointing,
        "checkpoint metadata.config.gradientCheckpointing",
      ),
    },
    readNumber(configRecord.vocabSize, "checkpoint metadata.config.vocabSize"),
  );

  return {
    config,
    tokenizer: {
      chars: readStringArray(value.tokenizer.chars, "checkpoint metadata.tokenizer.chars"),
    },
  };
}

/** Save a GPT checkpoint using the canonical @mlxts/train format. */
export function saveCheckpoint(options: SaveCheckpointOptions): void {
  saveTrainCheckpoint({
    model: options.model,
    kind: options.kind,
    metadata: {
      config: options.config,
      tokenizer: {
        chars: options.tokenizer.vocab,
      },
    },
    path: options.path,
    step: options.step,
    optimizer: options.optimizer,
  });
}

/** Load a GPT checkpoint and validate its metadata shape. */
export function loadCheckpoint(path: string): CheckpointData {
  return loadTrainCheckpoint(path, readGPTCheckpointMetadata);
}

/** Apply GPT checkpoint weights transactionally to a model. */
export function applyCheckpoint(model: GPT, checkpoint: Pick<CheckpointData, "parameters">): void {
  applyTrainCheckpoint(model, checkpoint);
}

/** Restore an AdamW optimizer from serialized checkpoint state. */
export function restoreAdamWFromCheckpoint(checkpoint: AdamWOptimizerCheckpoint): AdamW {
  return restoreTrainAdamWFromCheckpoint(checkpoint);
}
