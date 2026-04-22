import { dpoLoss, dpoTrain, preferenceLogProbSums, sftLoss, sftTrain } from "@mlxts/align";
import { mxEval } from "@mlxts/core";
import {
  type ChatMessage,
  collatePreferenceBatch,
  collateTokenSupervisionBatch,
  createRandomSource,
  type PreferenceExample,
  type TokenSupervisionExample,
} from "@mlxts/data";
import { Module } from "@mlxts/nn";
import { Adam } from "@mlxts/optimizers";
import {
  type CausalLM,
  type GenerationOptions,
  generateText,
  type InteractionProfile,
  type LoadSourceOptions,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  quantizePretrainedSnapshot,
} from "@mlxts/transformers";
import { existsSync, readdirSync, rmSync } from "fs";

import type { LoadedAssets, MetricPair } from "./types";

function requireChatTemplate(
  profile: InteractionProfile,
): NonNullable<InteractionProfile["chatTemplate"]> {
  if (profile.chatTemplate === null) {
    throw new Error("Training proof requires an instruct checkpoint with a chat template.");
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

function evaluateLoss(loss: ReturnType<typeof sftLoss> | ReturnType<typeof dpoLoss>): number {
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

function freePreferenceBatch(batch: ReturnType<typeof collatePreferenceBatch>): void {
  freeTokenBatch(batch.chosen);
  freeTokenBatch(batch.rejected);
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

function evaluateDpoBatchLoss(
  policyModel: CausalLM,
  referenceModel: CausalLM,
  batch: ReturnType<typeof collatePreferenceBatch>,
): number {
  try {
    return evaluateLoss(dpoLoss(policyModel, referenceModel, batch));
  } finally {
    freePreferenceBatch(batch);
  }
}

function chunkExamples<T>(examples: readonly T[], batchSize: number): T[][] {
  const chunks: T[][] = [];
  for (let index = 0; index < examples.length; index += batchSize) {
    chunks.push(examples.slice(index, index + batchSize));
  }
  return chunks;
}

export function summarizeMetric(before: number, after: number): MetricPair {
  return {
    before,
    after,
    delta: after - before,
  };
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
        throw new Error("training proof: selected an undefined batch example.");
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

export async function loadAssets(
  source: string,
  options: LoadSourceOptions = {},
): Promise<LoadedAssets> {
  // Keep snapshot-side artifact discovery serial here so one uncached chat template
  // cannot race itself through the Hugging Face cache on first load.
  const tokenizer = await loadPretrainedTokenizer(source, options);
  const profile = await loadInteractionProfile(source, options);
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

export function expectTrainableModule(model: CausalLM): Module {
  if (!(model instanceof Module)) {
    throw new Error("training proof: expected a loaded CausalLM backed by nn.Module.");
  }
  return model;
}

export function evaluateSupervisionDatasetLoss(
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

export function evaluatePreferenceDatasetLoss(
  policyModel: CausalLM,
  referenceModel: CausalLM,
  examples: readonly PreferenceExample[],
  padTokenId: number,
  batchSize: number,
): number {
  let totalLoss = 0;
  let batches = 0;
  for (const chunk of chunkExamples(examples, batchSize)) {
    totalLoss += evaluateDpoBatchLoss(
      policyModel,
      referenceModel,
      collatePreferenceBatch(chunk, padTokenId),
    );
    batches += 1;
  }
  return totalLoss / batches;
}

export function evaluatePreferenceAccuracy(
  policyModel: CausalLM,
  examples: readonly PreferenceExample[],
  padTokenId: number,
  batchSize: number,
): number {
  let wins = 0;
  let total = 0;
  for (const chunk of chunkExamples(examples, batchSize)) {
    const batch = collatePreferenceBatch(chunk, padTokenId);
    try {
      const logProbs = preferenceLogProbSums(policyModel, batch);
      try {
        mxEval(logProbs.chosen, logProbs.rejected);
        const chosenValues = logProbs.chosen.toList() as number[];
        const rejectedValues = logProbs.rejected.toList() as number[];
        for (let index = 0; index < chosenValues.length; index += 1) {
          const chosen = chosenValues[index];
          const rejected = rejectedValues[index];
          if (chosen !== undefined && rejected !== undefined) {
            if (chosen > rejected) {
              wins += 1;
            }
            total += 1;
          }
        }
      } finally {
        logProbs.chosen.free();
        logProbs.rejected.free();
      }
    } finally {
      freePreferenceBatch(batch);
    }
  }
  return total === 0 ? 0 : wins / total;
}

export function runSupervisionTrainingSteps(
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

export function runPreferenceTrainingSteps(
  policyModel: CausalLM,
  referenceModel: CausalLM,
  examples: readonly PreferenceExample[],
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
    const result = dpoTrain(policyModel, {
      referenceModel,
      optimizer,
      batches: [collatePreferenceBatch(nextBatch(), padTokenId)],
      beta: 0.1,
      learningRate,
    });
    totalTrainingLoss += result.averageLoss;
  }
  return totalTrainingLoss / steps;
}
