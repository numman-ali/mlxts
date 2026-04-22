import { sftLoss, sftTrain } from "@mlxts/align";
import { mxEval } from "@mlxts/core";
import {
  type ChatMessage,
  collateTokenSupervisionBatch,
  createRandomSource,
  type TokenSupervisionExample,
} from "@mlxts/data";
import { Module } from "@mlxts/nn";
import { Adam } from "@mlxts/optimizers";
import {
  type CausalLM,
  type GenerationOptions,
  generateText,
  type InteractionProfile,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  quantizePretrainedSnapshot,
} from "@mlxts/transformers";
import { existsSync, readdirSync, rmSync } from "fs";

import type { LoadedAssets } from "./types";

export function expectTrainableModule(model: CausalLM): Module {
  if (!(model instanceof Module)) {
    throw new Error("lora-finetune: expected a loaded CausalLM backed by nn.Module.");
  }
  return model;
}

function requireChatTemplate(
  profile: InteractionProfile,
): NonNullable<InteractionProfile["chatTemplate"]> {
  if (profile.chatTemplate === null) {
    throw new Error("lora-finetune: source model must provide a chat template.");
  }
  return profile.chatTemplate;
}

export function readPadTokenId(tokenizer: LoadedAssets["tokenizer"]): number {
  return tokenizer.padTokenId ?? tokenizer.eosTokenIds.at(0) ?? tokenizer.bosTokenId ?? 0;
}

function createGenerationOptions(): GenerationOptions {
  return {
    maxTokens: 32,
    temperature: 0,
    addSpecialTokens: false,
  };
}

function evaluateLoss(loss: ReturnType<typeof sftLoss>): number {
  mxEval(loss);
  const value = loss.item();
  loss.free();
  return value;
}

function freeTokenBatch(batch: ReturnType<typeof collateTokenSupervisionBatch>): void {
  batch.inputIds.free();
  batch.targetIds.free();
  batch.lossMask.free();
}

function evaluateSftBatchLoss(
  model: CausalLM,
  batch: ReturnType<typeof collateTokenSupervisionBatch>,
): number {
  try {
    return evaluateLoss(sftLoss(model, batch));
  } finally {
    freeTokenBatch(batch);
  }
}

function chunkExamples<T>(examples: readonly T[], batchSize: number): T[][] {
  const chunks: T[][] = [];
  for (let index = 0; index < examples.length; index += batchSize) {
    chunks.push(examples.slice(index, index + batchSize));
  }
  return chunks;
}

function createShuffledOrder(length: number, seed: number): number[] {
  const nextRandom = createRandomSource(seed);
  const order = Array.from({ length }, (_, index) => index);
  for (let index = order.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(nextRandom() * (index + 1));
    const current = order[index];
    order[index] = order[swapIndex] ?? current;
    order[swapIndex] = current;
  }
  return order;
}

function createBatchPicker<T>(
  examples: readonly T[],
  batchSize: number,
  seed: number,
): () => readonly T[] {
  const order = createShuffledOrder(examples.length, seed);
  let cursor = 0;
  return () => {
    const batch: T[] = [];
    for (let index = 0; index < batchSize; index += 1) {
      const exampleIndex = order[cursor % order.length];
      const example = examples[exampleIndex];
      if (example === undefined) {
        throw new Error("lora-finetune: selected an undefined batch example.");
      }
      batch.push(example);
      cursor += 1;
    }
    return batch;
  };
}

function generationPrompt(
  profile: InteractionProfile,
  promptMessages: readonly ChatMessage[],
): string {
  const template = requireChatTemplate(profile);
  return template.format(promptMessages, { addGenerationPrompt: true });
}

export function sampleText(
  model: CausalLM,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  promptMessages: readonly ChatMessage[],
): string {
  return generateText(
    model,
    tokenizer,
    generationPrompt(profile, promptMessages),
    createGenerationOptions(),
  );
}

export async function loadAssets(source: string): Promise<LoadedAssets> {
  // Keep snapshot-side artifact discovery serial here so one uncached chat template
  // cannot race itself through the Hugging Face cache on first load.
  const tokenizer = await loadPretrainedTokenizer(source);
  const profile = await loadInteractionProfile(source);
  return { tokenizer, profile };
}

function directoryHasQuantizedSnapshot(outputDir: string): boolean {
  if (!existsSync(outputDir)) {
    return false;
  }

  return readdirSync(outputDir).some(
    (entry) => entry.endsWith(".safetensors") || entry.endsWith(".safetensors.index.json"),
  );
}

export async function ensureQuantizedSnapshot(source: string, outputDir: string): Promise<string> {
  if (directoryHasQuantizedSnapshot(outputDir)) {
    return outputDir;
  }

  if (existsSync(outputDir)) {
    rmSync(outputDir, { recursive: true, force: true });
  }

  await quantizePretrainedSnapshot(source, {
    outputDir,
    overwrite: true,
  });
  return outputDir;
}

export function evaluateDatasetLoss(
  model: CausalLM,
  examples: readonly TokenSupervisionExample[],
  padTokenId: number,
  batchSize: number,
): number {
  let totalLoss = 0;
  let batches = 0;
  for (const chunk of chunkExamples(examples, batchSize)) {
    totalLoss += evaluateSftBatchLoss(model, collateTokenSupervisionBatch(chunk, padTokenId));
    batches += 1;
  }
  return totalLoss / batches;
}

export function runTrainingSteps(
  model: CausalLM,
  examples: readonly TokenSupervisionExample[],
  padTokenId: number,
  batchSize: number,
  steps: number,
  seed: number,
  learningRate: number,
): number {
  const optimizer = new Adam({ learningRate });
  const nextBatch = createBatchPicker(examples, batchSize, seed);
  let totalTrainingLoss = 0;
  for (let step = 0; step < steps; step += 1) {
    const result = sftTrain(model, {
      optimizer,
      batches: [collateTokenSupervisionBatch(nextBatch(), padTokenId)],
      learningRate,
    });
    totalTrainingLoss += result.averageLoss;
  }
  return totalTrainingLoss / steps;
}
