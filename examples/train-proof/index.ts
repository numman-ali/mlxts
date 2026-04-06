#!/usr/bin/env bun

import {
  buildChatPreferenceExample,
  buildChatSupervisionExample,
  dpoLoss,
  dpoTrain,
  preferenceLogProbSums,
  sftLoss,
  sftTrain,
} from "@mlxts/align";
import { mxEval } from "@mlxts/core";
import {
  type ChatMessage,
  collatePreferenceBatch,
  collateTokenSupervisionBatch,
  createRandomSource,
  type PreferenceExample,
  type TokenSupervisionExample,
} from "@mlxts/data";
import { type ApplyLoRAOptions, applyLoRAToModule, mergeLoRAInModule } from "@mlxts/lora";
import { Module, QuantizedLinear } from "@mlxts/nn";
import { Adam } from "@mlxts/optimizers";
import {
  type CausalLM,
  type GenerationOptions,
  generateText,
  type InteractionProfile,
  type LoadSourceOptions,
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  quantizePretrainedSnapshot,
} from "@mlxts/transformers";
import { existsSync, mkdirSync, readdirSync, rmSync } from "fs";
import { dirname } from "path";

import {
  loadTrainingProofRawDatasets,
  parseTrainingProofArgs,
  type TrainingProofArgs,
  type TrainingProofRawDatasets,
} from "./proof";

type MetricPair = {
  before: number;
  after: number;
  delta: number;
};

type StageReport = {
  stage: string;
  evalLoss?: MetricPair;
  preferenceAccuracy?: MetricPair;
  averageTrainingLoss?: number;
  sampleText?: string;
  notes: string[];
};

type TrainingProofReport = {
  source: string;
  quantizedOutputDir: string;
  datasetSource: TrainingProofArgs["datasetSource"];
  trainLimit: number;
  evalLimit: number;
  batchSize: number;
  steps: number;
  maxSequenceLength: number;
  seed: number;
  dataNotes: string[];
  stages: StageReport[];
};

type LoadedAssets = {
  tokenizer: Awaited<ReturnType<typeof loadPretrainedTokenizer>>;
  profile: InteractionProfile;
};

type PreparedTrainingProofData = {
  supervisionTrain: readonly TokenSupervisionExample[];
  supervisionEval: readonly TokenSupervisionExample[];
  preferenceTrain: readonly PreferenceExample[];
  preferenceEval: readonly PreferenceExample[];
  samplePromptMessages: readonly ChatMessage[];
  notes: string[];
};

function readPadTokenId(tokenizer: LoadedAssets["tokenizer"]): number {
  return tokenizer.padTokenId ?? tokenizer.eosTokenIds.at(0) ?? tokenizer.bosTokenId ?? 0;
}

function requireChatTemplate(
  profile: InteractionProfile,
): NonNullable<InteractionProfile["chatTemplate"]> {
  if (profile.chatTemplate === null) {
    throw new Error("Training proof requires an instruct checkpoint with a chat template.");
  }
  return profile.chatTemplate;
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

function summarizeMetric(before: number, after: number): MetricPair {
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

function sampleText(
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

async function loadAssets(source: string, options: LoadSourceOptions = {}): Promise<LoadedAssets> {
  const [tokenizer, profile] = await Promise.all([
    loadPretrainedTokenizer(source, options),
    loadInteractionProfile(source, options),
  ]);
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

async function ensureQuantizedSnapshot(source: string, outputDir: string): Promise<string> {
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

function makeLoRAOptions(): ApplyLoRAOptions {
  return {
    lastLayers: 2,
    keys: ["q_proj", "v_proj"],
    rank: 8,
    alpha: 16,
    dropout: 0,
  };
}

function buildPreparedSupervisionExamples(
  rawMessages: readonly (readonly ChatMessage[])[],
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  limit: number,
  maxSequenceLength: number,
  label: string,
  notes: string[],
): TokenSupervisionExample[] {
  const template = requireChatTemplate(profile);
  const prepared: TokenSupervisionExample[] = [];
  let skippedMalformed = 0;
  let skippedLong = 0;

  for (const messages of rawMessages) {
    try {
      const example = buildChatSupervisionExample(tokenizer, template, messages);
      if (example.inputIds.length > maxSequenceLength) {
        skippedLong += 1;
        continue;
      }
      prepared.push(example);
      if (prepared.length === limit) {
        break;
      }
    } catch {
      skippedMalformed += 1;
    }
  }

  if (prepared.length < limit) {
    throw new Error(
      `training proof: collected only ${prepared.length} ${label} supervision example(s); expected ${limit}.`,
    );
  }

  notes.push(`${label}_supervision_kept=${prepared.length}`);
  notes.push(`${label}_supervision_skipped_malformed=${skippedMalformed}`);
  notes.push(`${label}_supervision_skipped_long=${skippedLong}`);
  return prepared;
}

function buildPreparedPreferenceExamples(
  rawRows: readonly TrainingProofRawDatasets["preferenceTrainRows"],
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  limit: number,
  maxSequenceLength: number,
  label: string,
  notes: string[],
): PreferenceExample[] {
  const template = requireChatTemplate(profile);
  const prepared: PreferenceExample[] = [];
  let skippedMalformed = 0;
  let skippedLong = 0;

  for (const row of rawRows) {
    try {
      const example = buildChatPreferenceExample(
        tokenizer,
        template,
        row.promptMessages,
        row.chosen,
        row.rejected,
      );
      const chosenLength = example.promptIds.length + example.chosenIds.length;
      const rejectedLength = example.promptIds.length + example.rejectedIds.length;
      if (Math.max(chosenLength, rejectedLength) > maxSequenceLength) {
        skippedLong += 1;
        continue;
      }
      prepared.push(example);
      if (prepared.length === limit) {
        break;
      }
    } catch {
      skippedMalformed += 1;
    }
  }

  if (prepared.length < limit) {
    throw new Error(
      `training proof: collected only ${prepared.length} ${label} preference example(s); expected ${limit}.`,
    );
  }

  notes.push(`${label}_preference_kept=${prepared.length}`);
  notes.push(`${label}_preference_skipped_malformed=${skippedMalformed}`);
  notes.push(`${label}_preference_skipped_long=${skippedLong}`);
  return prepared;
}

async function prepareTrainingProofData(
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  args: TrainingProofArgs,
): Promise<PreparedTrainingProofData> {
  const rawDatasets = await loadTrainingProofRawDatasets(args);
  const notes = [...rawDatasets.notes];

  return {
    supervisionTrain: buildPreparedSupervisionExamples(
      rawDatasets.supervisionTrainMessages,
      tokenizer,
      profile,
      args.trainLimit,
      args.maxSequenceLength,
      "train",
      notes,
    ),
    supervisionEval: buildPreparedSupervisionExamples(
      rawDatasets.supervisionEvalMessages,
      tokenizer,
      profile,
      args.evalLimit,
      args.maxSequenceLength,
      "eval",
      notes,
    ),
    preferenceTrain: buildPreparedPreferenceExamples(
      rawDatasets.preferenceTrainRows,
      tokenizer,
      profile,
      args.trainLimit,
      args.maxSequenceLength,
      "train",
      notes,
    ),
    preferenceEval: buildPreparedPreferenceExamples(
      rawDatasets.preferenceEvalRows,
      tokenizer,
      profile,
      args.evalLimit,
      args.maxSequenceLength,
      "eval",
      notes,
    ),
    samplePromptMessages: rawDatasets.samplePromptMessages,
    notes,
  };
}

function evaluateSupervisionDatasetLoss(
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

function evaluatePreferenceDatasetLoss(
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

function evaluatePreferenceAccuracy(
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

function runSupervisionTrainingSteps(
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

function runPreferenceTrainingSteps(
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

async function runLoRAStage(
  source: string,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<StageReport> {
  using model = await loadCausalLM(source);
  const padTokenId = readPadTokenId(tokenizer);
  const before = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  applyLoRAToModule(model, makeLoRAOptions());
  const averageTrainingLoss = runSupervisionTrainingSteps(
    model,
    data.supervisionTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed,
    1e-4,
  );
  const merged = mergeLoRAInModule(model);
  const after = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );

  return {
    stage: "lora",
    evalLoss: summarizeMetric(before, after),
    averageTrainingLoss,
    sampleText: sampleText(model, tokenizer, profile, data.samplePromptMessages),
    notes: [
      `merged_targets=${merged.targets.length}`,
      `skipped=${merged.skipped.length}`,
      `train_examples=${data.supervisionTrain.length}`,
      `eval_examples=${data.supervisionEval.length}`,
    ],
  };
}

async function runQLoRAStage(
  source: string,
  quantizedOutputDir: string,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<StageReport> {
  const qloraSource = await ensureQuantizedSnapshot(source, quantizedOutputDir);
  using model = await loadCausalLM(qloraSource);
  const padTokenId = readPadTokenId(tokenizer);
  const before = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  applyLoRAToModule(model, makeLoRAOptions());
  const averageTrainingLoss = runSupervisionTrainingSteps(
    model,
    data.supervisionTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed + 1,
    1e-4,
  );
  const merged = mergeLoRAInModule(model);
  const after = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  const lastLayer = model.config.numHiddenLayers - 1;
  const lastProjection = readLastLlamaQProjection(model, lastLayer);
  const preservedQuantizedBase = lastProjection instanceof QuantizedLinear;

  return {
    stage: "qlora",
    evalLoss: summarizeMetric(before, after),
    averageTrainingLoss,
    sampleText: sampleText(model, tokenizer, profile, data.samplePromptMessages),
    notes: [
      `merged_targets=${merged.targets.length}`,
      `quantized_base_preserved=${preservedQuantizedBase}`,
      `train_examples=${data.supervisionTrain.length}`,
      `eval_examples=${data.supervisionEval.length}`,
    ],
  };
}

function readLastLlamaQProjection(model: CausalLM, layerIndex: number): unknown {
  if (!(model instanceof Module)) {
    return null;
  }

  const backbone = Reflect.get(model, "model");
  if (typeof backbone !== "object" || backbone === null) {
    return null;
  }

  const layers = Reflect.get(backbone, "layers");
  if (!Array.isArray(layers)) {
    return null;
  }

  const layer = layers[layerIndex];
  if (typeof layer !== "object" || layer === null) {
    return null;
  }

  const selfAttention = Reflect.get(layer, "selfAttention");
  if (typeof selfAttention !== "object" || selfAttention === null) {
    return null;
  }

  return Reflect.get(selfAttention, "qProjection");
}

async function runSFTStage(
  source: string,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<StageReport> {
  using model = await loadCausalLM(source);
  const padTokenId = readPadTokenId(tokenizer);
  const before = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  const averageTrainingLoss = runSupervisionTrainingSteps(
    model,
    data.supervisionTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed + 2,
    5e-6,
  );
  const after = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );

  return {
    stage: "sft",
    evalLoss: summarizeMetric(before, after),
    averageTrainingLoss,
    sampleText: sampleText(model, tokenizer, profile, data.samplePromptMessages),
    notes: [
      "dense_model=true",
      `train_examples=${data.supervisionTrain.length}`,
      `eval_examples=${data.supervisionEval.length}`,
    ],
  };
}

async function runDPOStage(
  source: string,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<StageReport> {
  using policyModel = await loadCausalLM(source);
  using referenceModel = await loadCausalLM(source);
  const padTokenId = readPadTokenId(tokenizer);
  const beforeLoss = evaluatePreferenceDatasetLoss(
    policyModel,
    referenceModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
  );
  const beforeAccuracy = evaluatePreferenceAccuracy(
    policyModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
  );
  applyLoRAToModule(policyModel, makeLoRAOptions());
  const averageTrainingLoss = runPreferenceTrainingSteps(
    policyModel,
    referenceModel,
    data.preferenceTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed + 3,
    5e-5,
  );
  const merged = mergeLoRAInModule(policyModel);
  const afterLoss = evaluatePreferenceDatasetLoss(
    policyModel,
    referenceModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
  );
  const afterAccuracy = evaluatePreferenceAccuracy(
    policyModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
  );

  return {
    stage: "dpo",
    evalLoss: summarizeMetric(beforeLoss, afterLoss),
    preferenceAccuracy: summarizeMetric(beforeAccuracy, afterAccuracy),
    averageTrainingLoss,
    sampleText: sampleText(policyModel, tokenizer, profile, data.samplePromptMessages),
    notes: [
      "reference_model=frozen_copy",
      `merged_targets=${merged.targets.length}`,
      `train_examples=${data.preferenceTrain.length}`,
      `eval_examples=${data.preferenceEval.length}`,
    ],
  };
}

function printStage(report: StageReport): void {
  const evalLoss =
    report.evalLoss === undefined
      ? ""
      : ` eval_loss_before=${report.evalLoss.before.toFixed(4)} eval_loss_after=${report.evalLoss.after.toFixed(4)} delta=${report.evalLoss.delta.toFixed(4)}`;
  const preferenceAccuracy =
    report.preferenceAccuracy === undefined
      ? ""
      : ` pref_acc_before=${report.preferenceAccuracy.before.toFixed(4)} pref_acc_after=${report.preferenceAccuracy.after.toFixed(4)} delta=${report.preferenceAccuracy.delta.toFixed(4)}`;
  const trainingLoss =
    report.averageTrainingLoss === undefined
      ? ""
      : ` train_loss=${report.averageTrainingLoss.toFixed(4)}`;
  console.log(`[${report.stage}]${evalLoss}${preferenceAccuracy}${trainingLoss}`);
  for (const note of report.notes) {
    console.log(`  - ${note}`);
  }
  if (report.sampleText !== undefined) {
    console.log(`  sample: ${report.sampleText}`);
  }
}

async function main(): Promise<void> {
  const parsed: TrainingProofArgs = parseTrainingProofArgs(Bun.argv.slice(2));
  const sourceOptions: LoadSourceOptions = {};

  console.log(`Training proof source: ${parsed.source}`);
  console.log(`Dataset source: ${parsed.datasetSource}`);
  console.log(`Train limit: ${parsed.trainLimit}`);
  console.log(`Eval limit: ${parsed.evalLimit}`);
  console.log(`Batch size: ${parsed.batchSize}`);
  console.log(`Steps per stage: ${parsed.steps}`);
  console.log(`Max sequence length: ${parsed.maxSequenceLength}`);
  console.log(`Quantized snapshot output: ${parsed.quantizedOutputDir}`);
  console.log(`Report path: ${parsed.reportPath}`);

  const denseAssets = await loadAssets(parsed.source, sourceOptions);
  const preparedData = await prepareTrainingProofData(
    denseAssets.tokenizer,
    denseAssets.profile,
    parsed,
  );
  for (const note of preparedData.notes) {
    console.log(`  data: ${note}`);
  }

  const loraStage = await runLoRAStage(
    parsed.source,
    denseAssets.tokenizer,
    denseAssets.profile,
    preparedData,
    parsed,
  );
  printStage(loraStage);

  const qloraStage = await runQLoRAStage(
    parsed.source,
    parsed.quantizedOutputDir,
    denseAssets.tokenizer,
    denseAssets.profile,
    preparedData,
    parsed,
  );
  printStage(qloraStage);

  const sftStage = await runSFTStage(
    parsed.source,
    denseAssets.tokenizer,
    denseAssets.profile,
    preparedData,
    parsed,
  );
  printStage(sftStage);

  const dpoStage = await runDPOStage(
    parsed.source,
    denseAssets.tokenizer,
    denseAssets.profile,
    preparedData,
    parsed,
  );
  printStage(dpoStage);

  const report: TrainingProofReport = {
    source: parsed.source,
    quantizedOutputDir: parsed.quantizedOutputDir,
    datasetSource: parsed.datasetSource,
    trainLimit: parsed.trainLimit,
    evalLimit: parsed.evalLimit,
    batchSize: parsed.batchSize,
    steps: parsed.steps,
    maxSequenceLength: parsed.maxSequenceLength,
    seed: parsed.seed,
    dataNotes: preparedData.notes,
    stages: [loraStage, qloraStage, sftStage, dpoStage],
  };

  mkdirSync(dirname(parsed.reportPath), { recursive: true });
  await Bun.write(parsed.reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`Report written to ${parsed.reportPath}`);
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
