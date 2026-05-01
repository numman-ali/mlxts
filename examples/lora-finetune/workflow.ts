import { getPeakMemoryBytes, resetPeakMemory, treeFlatten } from "@mlxts/core";
import type { ChatMessage } from "@mlxts/data";
import { applyLoRAToModule, assertQuantizedBasePreserved, mergeLoRAInModule } from "@mlxts/lora";
import type { Module } from "@mlxts/nn";
import {
  expectTrainableModule,
  loadCausalLM,
  loadCausalLMAdapters,
  resolveLoRATargets,
  saveCausalLMAdapters,
} from "@mlxts/transformers";
import { mkdirSync } from "fs";
import { join } from "path";

import type { FinetuneArgs, FinetuneReport } from "./args";
import { loadRawMessages, prepareSupervisionExamples } from "./data";
import {
  ensureQuantizedSnapshot,
  evaluateDatasetLoss,
  loadAssets,
  readPadTokenId,
  runTrainingSteps,
  sampleText,
} from "./runtime";
import { assertFinetuneReport } from "./verification";

function printRunSummary(args: FinetuneArgs, writeLine: (line: string) => void): void {
  writeLine(`LoRA finetune source: ${args.source}`);
  writeLine(`Mode: ${args.mode}`);
  writeLine(`Preset: ${args.preset}`);
  writeLine(`Adapter format: ${args.adapterFormat}`);
  writeLine(`Dataset source: ${args.datasetSource}`);
  writeLine(`Train limit: ${args.trainLimit}`);
  writeLine(`Eval limit: ${args.evalLimit}`);
  writeLine(`Batch size: ${args.batchSize}`);
  writeLine(`Steps: ${args.steps}`);
  writeLine(`Max sequence length: ${args.maxSequenceLength}`);
  writeLine(`Output dir: ${args.outputDir}`);
}

function resolveLoadSource(args: FinetuneArgs): Promise<string> | string {
  if (args.mode === "qlora") {
    return ensureQuantizedSnapshot(args.source, args.quantizedOutputDir);
  }
  return args.source;
}

function buildReport(
  args: FinetuneArgs,
  adapterDirectory: string,
  targetPaths: readonly string[],
  parameterCounts: FinetuneReport["parameterCounts"],
  dataStats: FinetuneReport["dataStats"],
  adapterCheck: FinetuneReport["adapterCheck"],
  evalLossBefore: number,
  evalLossAfter: number,
  training: {
    averageLoss: number;
    stepLosses: FinetuneReport["metrics"]["trainingStepLosses"];
  },
  samplePrompt: readonly ChatMessage[],
  trainedSample: string,
  reloadedSample: string,
  mergedSample: string,
): FinetuneReport {
  return {
    source: args.source,
    mode: args.mode,
    preset: args.preset,
    adapterFormat: args.adapterFormat,
    datasetSource: args.datasetSource,
    trainLimit: args.trainLimit,
    evalLimit: args.evalLimit,
    batchSize: args.batchSize,
    steps: args.steps,
    maxSequenceLength: args.maxSequenceLength,
    outputDir: args.outputDir,
    adapterDir: adapterDirectory,
    metrics: {
      evalLossBefore,
      evalLossAfter,
      averageTrainingLoss: training.averageLoss,
      trainingStepLosses: training.stepLosses,
      targetCount: targetPaths.length,
    },
    targetPaths,
    parameterCounts,
    memory: {
      peakBytes: getPeakMemoryBytes(),
    },
    dataStats,
    adapterCheck,
    samplePrompt,
    sampleText: {
      trained: trainedSample,
      reloaded: reloadedSample,
      merged: mergedSample,
    },
  };
}

async function runReloadAndMergeChecks(
  loadSource: string,
  adapterDirectory: string,
  args: FinetuneArgs,
  assets: Awaited<ReturnType<typeof loadAssets>>,
  samplePrompt: readonly ChatMessage[],
  trainedSample: string,
  targetPaths: readonly string[],
): Promise<{
  reloadedSample: string;
  mergedSample: string;
  adapterCheck: FinetuneReport["adapterCheck"];
}> {
  using reloadedModel = await loadCausalLM(loadSource);
  await loadCausalLMAdapters(reloadedModel, adapterDirectory, {
    format: args.adapterFormat,
  });
  const reloadedSample = sampleText(reloadedModel, assets.tokenizer, assets.profile, samplePrompt);

  const trainableModule = expectTrainableModule(reloadedModel);
  mergeLoRAInModule(trainableModule);
  const qloraQuantizedBasePreserved =
    args.mode === "qlora" ? assertQLoRABasesPreserved(trainableModule, targetPaths) : null;
  const mergedSample = sampleText(reloadedModel, assets.tokenizer, assets.profile, samplePrompt);
  return {
    reloadedSample,
    mergedSample,
    adapterCheck: {
      reloadedMatchesTrained: trainedSample === reloadedSample,
      qloraQuantizedBasePreserved,
    },
  };
}

function assertQLoRABasesPreserved(module: Module, targetPaths: readonly string[]): boolean {
  for (const targetPath of targetPaths) {
    assertQuantizedBasePreserved(module, targetPath);
  }
  return true;
}

function countParameterElements(module: Module): FinetuneReport["parameterCounts"] {
  const total = treeFlatten(module.parameters()).reduce(
    (sum, [, parameter]) => sum + parameter.size,
    0,
  );
  const trainable = treeFlatten(module.trainableParameters()).reduce(
    (sum, [, parameter]) => sum + parameter.size,
    0,
  );
  return { total, trainable };
}

function printCompletion(
  evalLossBefore: number,
  evalLossAfter: number,
  targetCount: number,
  adapterDirectory: string,
  reportPath: string,
  trainedSample: string,
  reloadedSample: string,
  mergedSample: string,
  writeLine: (line: string) => void,
): void {
  writeLine(
    `LoRA finetune complete. eval_loss_before=${evalLossBefore.toFixed(4)} eval_loss_after=${evalLossAfter.toFixed(4)} target_count=${targetCount}`,
  );
  writeLine(`Adapter directory: ${adapterDirectory}`);
  writeLine(`Report: ${reportPath}`);
  writeLine(`Sample (trained): ${trainedSample}`);
  writeLine(`Sample (reloaded): ${reloadedSample}`);
  writeLine(`Sample (merged): ${mergedSample}`);
}

export async function runLoRAFinetune(
  args: FinetuneArgs,
  progress: (line: string) => void = console.error,
): Promise<FinetuneReport> {
  resetPeakMemory();
  mkdirSync(args.outputDir, { recursive: true });
  printRunSummary(args, progress);

  const loadSource = await resolveLoadSource(args);
  const assets = await loadAssets(args.source);
  const rawMessages = await loadRawMessages(args);
  const trainPrepared = prepareSupervisionExamples(
    rawMessages.trainMessages,
    assets.tokenizer,
    assets.profile,
    args.trainLimit,
    args.maxSequenceLength,
  );
  const evalPrepared = prepareSupervisionExamples(
    rawMessages.evalMessages,
    assets.tokenizer,
    assets.profile,
    args.evalLimit,
    args.maxSequenceLength,
  );
  const trainExamples = trainPrepared.examples;
  const evalExamples = evalPrepared.examples;

  const adapterDirectory = join(args.outputDir, "adapter");

  using trainedModel = await loadCausalLM(loadSource);
  const resolvedTargets = resolveLoRATargets(trainedModel, {
    preset: args.preset,
    lastLayers: 2,
  });
  applyLoRAToModule(expectTrainableModule(trainedModel), {
    paths: resolvedTargets.paths,
    rank: 8,
    alpha: 16,
    dropout: 0,
  });
  const parameterCounts = countParameterElements(expectTrainableModule(trainedModel));

  const padTokenId = readPadTokenId(assets.tokenizer);
  const evalLossBefore = evaluateDatasetLoss(
    trainedModel,
    evalExamples,
    padTokenId,
    args.batchSize,
  );
  const training = runTrainingSteps(
    trainedModel,
    trainExamples,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed,
    args.mode === "qlora" ? 1e-4 : 5e-5,
  );
  const evalLossAfter = evaluateDatasetLoss(trainedModel, evalExamples, padTokenId, args.batchSize);
  const trainedSample = sampleText(
    trainedModel,
    assets.tokenizer,
    assets.profile,
    rawMessages.samplePrompt,
  );

  await saveCausalLMAdapters(trainedModel, adapterDirectory, {
    format: args.adapterFormat,
    baseModelNameOrPath: args.source,
  });

  const { reloadedSample, mergedSample, adapterCheck } = await runReloadAndMergeChecks(
    loadSource,
    adapterDirectory,
    args,
    assets,
    rawMessages.samplePrompt,
    trainedSample,
    resolvedTargets.paths,
  );

  const report = buildReport(
    args,
    adapterDirectory,
    resolvedTargets.paths,
    parameterCounts,
    {
      train: trainPrepared.stats,
      eval: evalPrepared.stats,
    },
    adapterCheck,
    evalLossBefore,
    evalLossAfter,
    training,
    rawMessages.samplePrompt,
    trainedSample,
    reloadedSample,
    mergedSample,
  );
  const verification = assertFinetuneReport(report);

  await Bun.write(args.reportPath, `${JSON.stringify(report, null, 2)}\n`);
  progress(`Report verification passed (${verification.checks.length} checks).`);
  printCompletion(
    evalLossBefore,
    evalLossAfter,
    resolvedTargets.paths.length,
    adapterDirectory,
    args.reportPath,
    trainedSample,
    reloadedSample,
    mergedSample,
    progress,
  );
  return report;
}
