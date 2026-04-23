/**
 * GPT model configuration and presets.
 *
 * Model shape presets define structural parameters only.
 * vocabSize comes from the tokenizer at runtime.
 *
 * @module
 */

/** Structural model parameters — independent of tokenizer. */
export interface ModelPreset {
  nLayer: number;
  nHead: number;
  nEmbd: number;
  blockSize: number;
  dropout: number;
  gradientCheckpointing: boolean;
}

/** Fully resolved GPT configuration including vocab size. */
export interface GPTConfig extends ModelPreset {
  vocabSize: number;
}

/** Total parameter count for a resolved GPT configuration. */
export function estimateParameterCount(config: GPTConfig): number {
  const embeddingParams = config.vocabSize * config.nEmbd + config.blockSize * config.nEmbd;
  const paramsPerBlock = 12 * config.nEmbd * config.nEmbd + 13 * config.nEmbd;
  const finalLayerNormParams = 2 * config.nEmbd;
  return embeddingParams + config.nLayer * paramsPerBlock + finalLayerNormParams;
}

/** Validate and merge a preset with tokenizer-derived vocabSize. */
export function resolveConfig(preset: ModelPreset, vocabSize: number): GPTConfig {
  if (preset.nLayer <= 0) throw new Error(`GPTConfig: nLayer must be > 0, got ${preset.nLayer}`);
  if (preset.nHead <= 0) throw new Error(`GPTConfig: nHead must be > 0, got ${preset.nHead}`);
  if (preset.nEmbd <= 0) throw new Error(`GPTConfig: nEmbd must be > 0, got ${preset.nEmbd}`);
  if (preset.blockSize <= 0)
    throw new Error(`GPTConfig: blockSize must be > 0, got ${preset.blockSize}`);
  if (preset.dropout < 0 || preset.dropout >= 1)
    throw new Error(`GPTConfig: dropout must be in [0, 1), got ${preset.dropout}`);
  if (typeof preset.gradientCheckpointing !== "boolean") {
    throw new Error("GPTConfig: gradientCheckpointing must be a boolean");
  }
  if (preset.nEmbd % preset.nHead !== 0)
    throw new Error(
      `GPTConfig: nEmbd (${preset.nEmbd}) must be divisible by nHead (${preset.nHead})`,
    );
  if (vocabSize <= 0) throw new Error(`GPTConfig: vocabSize must be > 0, got ${vocabSize}`);

  return {
    ...preset,
    vocabSize,
  };
}

/** ~10.8M params with char-level vocab. Trains in minutes on Apple Silicon. */
export const GPT_TINY: ModelPreset = {
  nLayer: 6,
  nHead: 6,
  nEmbd: 384,
  blockSize: 256,
  dropout: 0.2,
  gradientCheckpointing: false,
};

/** GPT-2-small-sized architecture variant. Runtime vocab comes from the tokenizer. */
export const GPT_SMALL: ModelPreset = {
  nLayer: 12,
  nHead: 12,
  nEmbd: 768,
  blockSize: 1024,
  dropout: 0.0,
  gradientCheckpointing: true,
};
