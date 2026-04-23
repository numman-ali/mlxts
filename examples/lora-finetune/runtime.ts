import { evaluateSupervisionDatasetLoss, runSupervisionTrainingSteps } from "@mlxts/align";
import type { ChatMessage, TokenSupervisionExample } from "@mlxts/data";
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
  return evaluateSupervisionDatasetLoss(model, {
    examples,
    padTokenId,
    batchSize,
  });
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
  return runSupervisionTrainingSteps(model, {
    optimizer,
    examples,
    padTokenId,
    batchSize,
    steps,
    seed,
    learningRate,
  }).averageLoss;
}
