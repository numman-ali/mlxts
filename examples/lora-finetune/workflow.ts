import { applyLoRAToModule, mergeLoRAInModule } from "@mlxts/lora";
import {
  expectTrainableModule,
  loadCausalLM,
  loadCausalLMAdapters,
  resolveLoRATargets,
  saveCausalLMAdapters,
} from "@mlxts/transformers";
import { mkdirSync } from "fs";
import { join } from "path";

import { type FinetuneArgs, type FinetuneReport, parseArgs } from "./args";
import { loadRawMessages, prepareSupervisionExamples } from "./data";
import {
  ensureQuantizedSnapshot,
  evaluateDatasetLoss,
  loadAssets,
  readPadTokenId,
  runTrainingSteps,
  sampleText,
} from "./runtime";

function printRunSummary(args: FinetuneArgs): void {
  console.log(`LoRA finetune source: ${args.source}`);
  console.log(`Mode: ${args.mode}`);
  console.log(`Preset: ${args.preset}`);
  console.log(`Adapter format: ${args.adapterFormat}`);
  console.log(`Dataset source: ${args.datasetSource}`);
  console.log(`Train limit: ${args.trainLimit}`);
  console.log(`Eval limit: ${args.evalLimit}`);
  console.log(`Batch size: ${args.batchSize}`);
  console.log(`Steps: ${args.steps}`);
  console.log(`Max sequence length: ${args.maxSequenceLength}`);
  console.log(`Output dir: ${args.outputDir}`);
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
  targetCount: number,
  evalLossBefore: number,
  evalLossAfter: number,
  averageTrainingLoss: number,
  samplePrompt: readonly { role: string; content: string }[],
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
      averageTrainingLoss,
      targetCount,
    },
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
  samplePrompt: readonly { role: string; content: string }[],
): Promise<{ reloadedSample: string; mergedSample: string }> {
  using reloadedModel = await loadCausalLM(loadSource);
  await loadCausalLMAdapters(reloadedModel, adapterDirectory, {
    format: args.adapterFormat,
  });
  const reloadedSample = sampleText(reloadedModel, assets.tokenizer, assets.profile, samplePrompt);

  mergeLoRAInModule(expectTrainableModule(reloadedModel));
  const mergedSample = sampleText(reloadedModel, assets.tokenizer, assets.profile, samplePrompt);
  return { reloadedSample, mergedSample };
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
): void {
  console.log(
    `LoRA finetune complete. eval_loss_before=${evalLossBefore.toFixed(4)} eval_loss_after=${evalLossAfter.toFixed(4)} target_count=${targetCount}`,
  );
  console.log(`Adapter directory: ${adapterDirectory}`);
  console.log(`Report: ${reportPath}`);
  console.log(`Sample (trained): ${trainedSample}`);
  console.log(`Sample (reloaded): ${reloadedSample}`);
  console.log(`Sample (merged): ${mergedSample}`);
}

export async function runLoRAFinetune(argv: readonly string[]): Promise<void> {
  const args = parseArgs(argv);
  mkdirSync(args.outputDir, { recursive: true });
  printRunSummary(args);

  const loadSource = await resolveLoadSource(args);
  const assets = await loadAssets(args.source);
  const rawMessages = await loadRawMessages(args);
  const trainExamples = prepareSupervisionExamples(
    rawMessages.trainMessages,
    assets.tokenizer,
    assets.profile,
    args.trainLimit,
    args.maxSequenceLength,
  );
  const evalExamples = prepareSupervisionExamples(
    rawMessages.evalMessages,
    assets.tokenizer,
    assets.profile,
    args.evalLimit,
    args.maxSequenceLength,
  );

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

  const padTokenId = readPadTokenId(assets.tokenizer);
  const evalLossBefore = evaluateDatasetLoss(
    trainedModel,
    evalExamples,
    padTokenId,
    args.batchSize,
  );
  const averageTrainingLoss = runTrainingSteps(
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

  const { reloadedSample, mergedSample } = await runReloadAndMergeChecks(
    loadSource,
    adapterDirectory,
    args,
    assets,
    rawMessages.samplePrompt,
  );

  const report = buildReport(
    args,
    adapterDirectory,
    resolvedTargets.paths.length,
    evalLossBefore,
    evalLossAfter,
    averageTrainingLoss,
    rawMessages.samplePrompt,
    trainedSample,
    reloadedSample,
    mergedSample,
  );

  await Bun.write(args.reportPath, `${JSON.stringify(report, null, 2)}\n`);
  printCompletion(
    evalLossBefore,
    evalLossAfter,
    resolvedTargets.paths.length,
    adapterDirectory,
    args.reportPath,
    trainedSample,
    reloadedSample,
    mergedSample,
  );
}
