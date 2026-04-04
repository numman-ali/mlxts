/**
 * Autoregressive text generation.
 *
 * Supports greedy decoding (temperature=0) and temperature-scaled
 * categorical sampling. Context is cropped to the last blockSize
 * tokens to stay within the positional embedding range.
 *
 * @module
 */

import {
  argmax,
  array,
  divide,
  type MxArray,
  mxEval,
  random,
  reshape,
  squeeze,
  takeAxis,
} from "@mlxts/core";
import type { GPTConfig } from "./config";
import type { GPT } from "./model/gpt";
import type { CharTokenizer } from "./tokenizer";

/** Generation configuration. */
export interface GenerateConfig {
  maxNewTokens: number;
  temperature: number;
}

function restoreTrainingMode(model: GPT, wasTraining: boolean): void {
  if (wasTraining) {
    model.train();
    return;
  }
  model.eval();
}

function validatePrompt(prompt: MxArray): void {
  if (prompt.dtype !== "int32") {
    throw new Error(`generateTokens: prompt must be int32, got ${prompt.dtype}`);
  }
  if (prompt.size === 0) {
    throw new Error("generateTokens: prompt must contain at least one token");
  }
}

function selectNextToken(logits: MxArray, timeLength: number, temperature: number): number {
  const indexArray = array([timeLength - 1], "int32");

  try {
    using lastLogitsWithAxis = takeAxis(logits, indexArray, 1);
    using lastLogits = squeeze(lastLogitsWithAxis, 1);

    if (temperature === 0) {
      using nextToken = argmax(lastLogits, -1);
      mxEval(nextToken);
      return nextToken.item();
    }

    using scaledLogits = divide(lastLogits, temperature);
    using sampled = random.categorical(scaledLogits, -1);
    mxEval(sampled);
    return sampled.item();
  } finally {
    indexArray.free();
  }
}

/**
 * Generate tokens autoregressively from a prompt.
 *
 * Yields one token ID at a time. The model's prior training mode is restored
 * when generation completes, even if the caller stops iteration early.
 */
export function* generateTokens(
  model: GPT,
  config: GPTConfig,
  prompt: MxArray,
  genConfig: GenerateConfig,
): Generator<number> {
  if (genConfig.maxNewTokens <= 0) {
    throw new Error(`GenerateConfig: maxNewTokens must be > 0, got ${genConfig.maxNewTokens}`);
  }
  if (genConfig.temperature < 0) {
    throw new Error(`GenerateConfig: temperature must be >= 0, got ${genConfig.temperature}`);
  }

  validatePrompt(prompt);
  mxEval(prompt);

  const contextTokens = Array.from(prompt.toTypedArray(), Number);
  const wasTraining = model.isTraining;
  model.eval();

  try {
    for (let index = 0; index < genConfig.maxNewTokens; index++) {
      const windowStart = Math.max(0, contextTokens.length - config.blockSize);
      const windowTokens = contextTokens.slice(windowStart);
      const flatInput = array(windowTokens, "int32");

      try {
        using input = reshape(flatInput, [1, windowTokens.length]);
        using logits = model.forward(input);
        const nextToken = selectNextToken(logits, windowTokens.length, genConfig.temperature);
        contextTokens.push(nextToken);
        yield nextToken;
      } finally {
        flatInput.free();
      }
    }
  } finally {
    restoreTrainingMode(model, wasTraining);
  }
}

/**
 * Generate text from a prompt string.
 *
 * Higher-level wrapper around generateTokens: encodes the prompt,
 * generates tokens, and decodes back to text.
 */
export function generate(
  model: GPT,
  config: GPTConfig,
  tokenizer: CharTokenizer,
  prompt: string,
  genConfig: GenerateConfig,
): string {
  const promptTokens = tokenizer.encode(prompt);
  const promptArray = array(promptTokens, "int32");

  try {
    const generatedTokens: number[] = [];
    for (const token of generateTokens(model, config, promptArray, genConfig)) {
      generatedTokens.push(token);
    }
    return prompt + tokenizer.decode(generatedTokens);
  } finally {
    promptArray.free();
  }
}
