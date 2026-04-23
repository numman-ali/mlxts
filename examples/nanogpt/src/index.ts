/**
 * nanogpt — A GPT implementation in TypeScript, built on @mlxts/*.
 *
 * Educational, self-documenting, and designed to run
 * end-to-end on Apple Silicon.
 *
 * @module nanogpt
 */

// Checkpoint
export type {
  AdamWOptimizerCheckpoint,
  CheckpointData,
  CheckpointKind,
  GPTCheckpointMetadata,
} from "./checkpoint";
export {
  applyCheckpoint,
  loadCheckpoint,
  restoreAdamWFromCheckpoint,
  saveCheckpoint,
} from "./checkpoint";
// Config
export type { GPTConfig, ModelPreset } from "./config";
export { estimateParameterCount, GPT_SMALL, GPT_TINY, resolveConfig } from "./config";
// Generation
export type { GenerateConfig } from "./generate";
export { generate, generateTokens } from "./generate";
// Model
export { GPT } from "./model/gpt";
export { initializeGPT } from "./model/init";
// Interop
export { loadModelSafetensors, saveModelSafetensors } from "./safetensors";
// Training
export type { TrainConfig, TrainEvent, TrainOptions, TrainSummary } from "./train";
export { getLearningRate, train } from "./train";

// The supervised run-manager surface lives under ./run/ intentionally.
// It is an operator-facing process boundary, not part of the library barrel.

export const VERSION = "0.0.1";
