import { prepareData } from "@mlxts/data";
import { CharTokenizer } from "@mlxts/tokenizers";
import { mkdirSync } from "fs";
import { applyCheckpoint, loadCheckpoint, saveCheckpoint } from "../checkpoint";
import { type GenerateConfig, generate } from "../generate";
import { GPT } from "../model/gpt";
import { saveModelSafetensors } from "../safetensors";
import { train } from "../train";
import { printExportHelp, printGenerateHelp, printTrainHelp } from "./help";
import {
  checkpointPath,
  createAutoTrainingPolicy,
  createTrainingSession,
  trainDefaultsForConfig,
} from "./session";
import {
  applyRuntimeLimits,
  emitJson,
  formatNanogptCliError,
  getFlag,
  getNumberFlag,
  RUNTIME_ERROR_EXIT_CODE,
  USAGE_ERROR_EXIT_CODE,
  UserError,
} from "./shared";
import {
  announceTrainingSession,
  type CheckpointPolicy,
  createStopController,
  createTrainEventHandler,
  emitCancelledRun,
  emitStoppedRun,
  emitTrainingSample,
  type SamplingPolicy,
} from "./train-events";

export async function runTrain(flags: Map<string, string>): Promise<void> {
  const checkpointDir = getFlag(flags, "checkpoint-dir", ".nanogpt-checkpoints");
  const runDir = flags.get("run-dir");
  const useJson = flags.has("json");
  const session = await createTrainingSession(flags);
  const defaults = trainDefaultsForConfig(session.config);
  const checkpointPolicy: CheckpointPolicy = {
    checkpointDir,
    runDir,
    snapshotEvery: getNumberFlag(flags, "snapshot-interval", defaults.snapshotInterval),
    resumeEvery: getNumberFlag(flags, "resume-interval", defaults.resumeInterval),
  };
  const samplingPolicy: SamplingPolicy = {
    every: flags.has("sample-interval")
      ? getNumberFlag(flags, "sample-interval", 0)
      : runDir !== undefined
        ? checkpointPolicy.snapshotEvery
        : 0,
    maxNewTokens: getNumberFlag(flags, "sample-tokens", 200),
  };
  const autoTrainingPolicy = createAutoTrainingPolicy(flags, checkpointDir, session.presetName);
  const { trainTokens, valTokens } = prepareData(session.tokenizer.encode(session.text), 0.9);
  const stopController = createStopController(useJson, runDir);
  let autoStopRequested = false;

  mkdirSync(checkpointDir ?? ".nanogpt-checkpoints", { recursive: true });
  applyRuntimeLimits(flags);
  announceTrainingSession(session, checkpointPolicy, samplingPolicy, autoTrainingPolicy, useJson);

  const onEvent = createTrainEventHandler({
    useJson,
    checkpointPolicy,
    samplingPolicy,
    autoTrainingPolicy,
    requestAutoStop(reason) {
      if (autoStopRequested) {
        return;
      }
      autoStopRequested = true;
      autoTrainingPolicy.stopReason = reason;
    },
    session,
  });

  try {
    const summary = train({
      model: session.model,
      config: session.config,
      trainConfig: session.trainConfig,
      trainTokens,
      valTokens,
      optimizer: session.optimizer,
      onEvent,
      shouldStop: () => stopController.shouldStop() || autoStopRequested,
    });

    const controlCommand = stopController.command();
    if (controlCommand === "cancel") {
      emitCancelledRun(summary, useJson);
      return;
    }

    const finalPath = checkpointPath(
      checkpointDir ?? ".nanogpt-checkpoints",
      session.presetName,
      "resume",
      summary.totalSteps,
    );
    saveCheckpoint({
      model: session.model,
      optimizer: session.optimizer,
      kind: "resume",
      config: session.config,
      step: summary.totalSteps,
      tokenizer: session.tokenizer,
      path: finalPath,
    });
    if (useJson) {
      emitJson({ type: "checkpoint", step: summary.totalSteps, kind: "resume", path: finalPath });
    }
    if (summary.totalSteps < session.trainConfig.maxSteps) {
      emitStoppedRun(finalPath, summary, autoTrainingPolicy, useJson);
      return;
    }
    emitTrainingSample(session, finalPath, summary, useJson);
  } finally {
    stopController.cleanup();
    session.optimizer[Symbol.dispose]();
    session.model[Symbol.dispose]();
  }
}

export function runGenerate(flags: Map<string, string>): void {
  const checkpoint = flags.get("checkpoint");
  if (checkpoint === undefined) {
    throw new UserError("Flag --checkpoint is required for generate");
  }

  const data = loadCheckpoint(checkpoint);
  const tokenizer = CharTokenizer.fromVocab(data.metadata.tokenizer.chars);
  const model = new GPT(data.metadata.config);

  try {
    applyCheckpoint(model, data);
    const prompt = getFlag(flags, "prompt", "\n") ?? "\n";
    const config: GenerateConfig = {
      maxNewTokens: getNumberFlag(flags, "max-tokens", 500),
      temperature: getNumberFlag(flags, "temperature", 0.8),
    };
    const text = generate(model, data.metadata.config, tokenizer, prompt, config);

    if (flags.has("json")) {
      emitJson({
        checkpoint,
        prompt,
        text,
        maxNewTokens: config.maxNewTokens,
        temperature: config.temperature,
      });
      return;
    }

    process.stdout.write(text);
    process.stdout.write("\n");
  } finally {
    model[Symbol.dispose]();
  }
}

export async function runExport(flags: Map<string, string>): Promise<void> {
  const checkpointPath = flags.get("checkpoint");
  if (checkpointPath === undefined) {
    throw new UserError("Flag --checkpoint is required for export");
  }

  const outputPath = flags.get("output");
  if (outputPath === undefined) {
    throw new UserError("Flag --output is required for export");
  }

  const checkpoint = loadCheckpoint(checkpointPath);
  const model = new GPT(checkpoint.metadata.config);
  try {
    applyCheckpoint(model, checkpoint);
    await saveModelSafetensors(model, outputPath, {
      checkpoint: checkpointPath,
      step: String(checkpoint.step),
      kind: checkpoint.kind,
    });
  } finally {
    model[Symbol.dispose]();
  }

  process.stdout.write(`${outputPath}\n`);
}

export function handleError(error: unknown): never {
  if (error instanceof UserError) {
    process.stdout.write(`${formatNanogptCliError(error.message, "usage", "nanogpt --help")}\n`);
    process.exit(USAGE_ERROR_EXIT_CODE);
  }

  const message = error instanceof Error ? error.message : String(error);
  process.stdout.write(`${formatNanogptCliError(message, "runtime", "nanogpt --help")}\n`);
  if (error instanceof Error && error.stack !== undefined && error.stack !== error.message) {
    process.stderr.write(`${error.stack}\n`);
  }
  process.exit(RUNTIME_ERROR_EXIT_CODE);
}

export { printExportHelp, printGenerateHelp, printTrainHelp };
