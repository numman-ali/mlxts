import {
  evaluatePreferenceDatasetLoss as alignEvaluatePreferenceDatasetLoss,
  evaluatePreferenceMetrics as alignEvaluatePreferenceMetrics,
  evaluateSupervisionDatasetLoss as alignEvaluateSupervisionDatasetLoss,
  runPreferenceTrainingSteps as alignRunPreferenceTrainingSteps,
  runSupervisionTrainingSteps as alignRunSupervisionTrainingSteps,
  type PreferenceEvalMetrics,
} from "@mlxts/align";
import type { ChatMessage, PreferenceExample, TokenSupervisionExample } from "@mlxts/data";
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

export const DEFAULT_DPO_BETA = 0.1;

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

export function summarizeMetric(before: number, after: number): MetricPair {
  return {
    before,
    after,
    delta: after - before,
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
  return alignEvaluateSupervisionDatasetLoss(model, {
    examples,
    padTokenId,
    batchSize,
  });
}

export function evaluatePreferenceDatasetLoss(
  policyModel: CausalLM,
  referenceModel: CausalLM,
  examples: readonly PreferenceExample[],
  padTokenId: number,
  batchSize: number,
): number {
  return alignEvaluatePreferenceDatasetLoss(policyModel, {
    referenceModel,
    examples,
    padTokenId,
    batchSize,
  });
}

export function evaluatePreferenceMetrics(
  policyModel: CausalLM,
  referenceModel: CausalLM,
  examples: readonly PreferenceExample[],
  padTokenId: number,
  batchSize: number,
  beta = DEFAULT_DPO_BETA,
): PreferenceEvalMetrics {
  return alignEvaluatePreferenceMetrics(policyModel, {
    referenceModel,
    examples,
    padTokenId,
    batchSize,
    beta,
  });
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
  return alignRunSupervisionTrainingSteps(model, {
    optimizer,
    examples,
    padTokenId,
    batchSize,
    steps,
    seed,
    learningRate,
  }).averageLoss;
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
  beta = DEFAULT_DPO_BETA,
): number {
  const optimizer = new Adam({ learningRate });
  return alignRunPreferenceTrainingSteps(policyModel, {
    referenceModel,
    optimizer,
    examples,
    padTokenId,
    batchSize,
    steps,
    seed,
    beta,
    learningRate,
  }).averageLoss;
}
