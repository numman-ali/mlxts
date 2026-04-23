import { clearMemoryCache, getMemoryStats } from "@mlxts/core";
import type { CheckpointKind } from "../checkpoint";
import { saveCheckpoint } from "../checkpoint";
import { estimateParameterCount } from "../config";
import { generate } from "../generate";
import { type RunControlCommand, readRunControl } from "../run/files";
import type { TrainEvent, TrainSummary } from "../train";
import { formatEvalEvent, formatStepEvent, trainingTableHeader } from "./help";
import type { AutoTrainingPolicy, TrainingSession } from "./session";
import { checkpointPath } from "./session";
import { emitJson } from "./shared";

export type TelemetrySnapshot = ReturnType<typeof getMemoryStats>;

export type CheckpointPolicy = {
  checkpointDir?: string | undefined;
  runDir?: string | undefined;
  snapshotEvery: number;
  resumeEvery: number;
};

export type SamplingPolicy = {
  every: number;
  maxNewTokens: number;
};

function readTelemetry(): TelemetrySnapshot {
  return getMemoryStats();
}

function samplePrompt(session: TrainingSession): string {
  const vocab = session.tokenizer.vocab;
  return vocab.includes("\n") ? "\n" : (vocab[0] ?? "");
}

function emitControlRequest(
  useJson: boolean,
  command: RunControlCommand,
  requestedAt: string,
): void {
  if (useJson) {
    emitJson({ type: "control", command, requestedAt });
    return;
  }
  if (command === "cancel") {
    process.stderr.write(
      "\nCancellation requested. Work since the latest resume checkpoint may be lost.\n",
    );
    return;
  }
  process.stderr.write("\nGraceful stop requested.\n");
}

export function createStopController(
  useJson: boolean,
  runDir?: string,
): {
  cleanup: () => void;
  command: () => RunControlCommand | undefined;
  shouldStop: () => boolean;
} {
  let requested: RunControlCommand | undefined;
  let requestedAt: string | undefined;

  const rememberRequest = (command: RunControlCommand, at: string): void => {
    if (requested === command && requestedAt === at) {
      return;
    }
    if (requested !== undefined) {
      return;
    }
    requested = command;
    requestedAt = at;
    emitControlRequest(useJson, command, at);
  };

  const onSignal = (signal: NodeJS.Signals): void => {
    rememberRequest("stop", `${new Date().toISOString()} (${signal})`);
  };

  process.on("SIGINT", onSignal);
  process.on("SIGTERM", onSignal);

  return {
    command: () => requested,
    shouldStop: () => {
      if (runDir !== undefined) {
        const control = readRunControl(runDir);
        if (control !== undefined) {
          rememberRequest(control.command, control.requestedAt);
        }
      }
      return requested !== undefined;
    },
    cleanup: () => {
      process.off("SIGINT", onSignal);
      process.off("SIGTERM", onSignal);
    },
  };
}

function maybeClearAllocatorCache(): TelemetrySnapshot {
  const before = readTelemetry();
  if (before.cacheBytes <= Math.max(before.activeBytes * 2, 512 * 1024 * 1024)) {
    return before;
  }
  clearMemoryCache();
  return readTelemetry();
}

function emitCheckpoint(
  useJson: boolean,
  step: number,
  kind: CheckpointKind,
  path: string,
  telemetry: TelemetrySnapshot,
): void {
  if (useJson) {
    emitJson({
      type: "checkpoint",
      step,
      kind,
      path,
      activeMemoryBytes: telemetry.activeBytes,
      cacheMemoryBytes: telemetry.cacheBytes,
      peakMemoryBytes: telemetry.peakBytes,
      memoryLimitBytes: telemetry.limitBytes,
    });
    return;
  }
  process.stderr.write(`  ${kind} checkpoint saved: ${path}\n`);
}

function emitSample(
  session: TrainingSession,
  step: number,
  maxNewTokens: number,
  useJson: boolean,
): void {
  const sample = generate(session.model, session.config, session.tokenizer, samplePrompt(session), {
    maxNewTokens,
    temperature: 0.8,
  });

  if (useJson) {
    emitJson({
      type: "sample",
      step,
      text: sample,
      maxNewTokens,
    });
    return;
  }
  process.stderr.write(`\n--- Sample @ step ${step} ---\n${sample}\n`);
}

function emitBestCheckpoint(
  useJson: boolean,
  step: number,
  valLoss: number,
  path: string | undefined,
): void {
  if (useJson) {
    emitJson({ type: "best-checkpoint", step, valLoss, path });
    return;
  }
  process.stderr.write(
    `  best checkpoint: val ${valLoss.toFixed(4)} at step ${step}${path === undefined ? "" : ` -> ${path}`}\n`,
  );
}

function emitEarlyStop(
  useJson: boolean,
  step: number,
  reason: string,
  policy: AutoTrainingPolicy,
): void {
  if (useJson) {
    emitJson({
      type: "early-stop",
      step,
      reason,
      bestValLoss: policy.bestValLoss,
      bestCheckpointStep: policy.bestCheckpointStep,
      bestCheckpointPath: policy.bestCheckpointPath,
      patience: policy.patience,
      minDelta: policy.minDelta,
      consecutiveBadEvals: policy.consecutiveBadEvals,
    });
    return;
  }
  process.stderr.write(`  early stop: ${reason}\n`);
}

export function announceTrainingSession(
  session: TrainingSession,
  checkpointPolicy: CheckpointPolicy,
  samplingPolicy: SamplingPolicy,
  autoTrainingPolicy: AutoTrainingPolicy,
  useJson: boolean,
): void {
  const telemetry = readTelemetry();
  const parameterCount = estimateParameterCount(session.config);

  if (useJson) {
    emitJson({
      type: "start",
      preset: session.presetName,
      config: session.config,
      params: parameterCount,
      vocabSize: session.tokenizer.vocabSize,
      maxSteps: session.trainConfig.maxSteps,
      batchSize: session.trainConfig.batchSize,
      gradAccumSteps: session.trainConfig.gradAccumSteps,
      maxGradNorm: session.trainConfig.maxGradNorm,
      warmupSteps: session.trainConfig.warmupSteps,
      startStep: session.trainConfig.startStep ?? 0,
      resumeFrom: session.checkpointSource,
      checkpointDir: checkpointPolicy.checkpointDir,
      snapshotEvery: checkpointPolicy.snapshotEvery,
      resumeEvery: checkpointPolicy.resumeEvery,
      sampleEvery: samplingPolicy.every,
      sampleTokens: samplingPolicy.maxNewTokens,
      earlyStopPatience: autoTrainingPolicy.patience,
      earlyStopMinDelta: autoTrainingPolicy.minDelta,
      pid: process.pid,
      activeMemoryBytes: telemetry.activeBytes,
      cacheMemoryBytes: telemetry.cacheBytes,
      peakMemoryBytes: telemetry.peakBytes,
      memoryLimitBytes: telemetry.limitBytes,
    });
    return;
  }

  process.stderr.write(
    `Training ${session.presetName} on ${session.text.length.toLocaleString()} chars (vocab: ${session.tokenizer.vocabSize}, params: ${parameterCount.toLocaleString()})\n\n`,
  );
  process.stderr.write(
    `Checkpoint policy: snapshot every ${checkpointPolicy.snapshotEvery} eval step(s), resume every ${checkpointPolicy.resumeEvery} eval step(s)\n`,
  );
  process.stderr.write(
    `Gradient clipping: ${session.trainConfig.maxGradNorm === null ? "disabled" : session.trainConfig.maxGradNorm}\n`,
  );
  process.stderr.write(
    `Sample output: ${samplingPolicy.every > 0 ? `every ${samplingPolicy.every} step(s), ${samplingPolicy.maxNewTokens} tokens` : "disabled"}\n\n`,
  );
  process.stderr.write(
    `Auto stop: ${
      autoTrainingPolicy.patience === null
        ? "disabled"
        : `patience ${autoTrainingPolicy.patience} eval(s), min delta ${autoTrainingPolicy.minDelta}`
    }\n\n`,
  );
  if (session.checkpointSource !== undefined) {
    const modeLabel = session.resumeStep > 0 ? "Resuming" : "Warm-starting";
    process.stderr.write(
      `${modeLabel} from ${session.checkpointSource} at step ${session.resumeStep}\n\n`,
    );
  }
  process.stderr.write(trainingTableHeader());
}

export function createTrainEventHandler(options: {
  checkpointPolicy: CheckpointPolicy;
  samplingPolicy: SamplingPolicy;
  autoTrainingPolicy: AutoTrainingPolicy;
  requestAutoStop: (reason: string) => void;
  useJson: boolean;
  session: TrainingSession;
}): (event: TrainEvent) => void {
  const {
    checkpointPolicy,
    samplingPolicy,
    autoTrainingPolicy,
    requestAutoStop,
    session,
    useJson,
  } = options;

  function checkpointKindForStep(step: number): CheckpointKind | undefined {
    if (checkpointPolicy.resumeEvery > 0 && step % checkpointPolicy.resumeEvery === 0) {
      return "resume";
    }
    if (checkpointPolicy.snapshotEvery > 0 && step % checkpointPolicy.snapshotEvery === 0) {
      return "snapshot";
    }
    return undefined;
  }

  function emitEventOutput(event: TrainEvent): void {
    const telemetry = readTelemetry();
    if (useJson) {
      emitJson({
        ...event,
        activeMemoryBytes: telemetry.activeBytes,
        cacheMemoryBytes: telemetry.cacheBytes,
        peakMemoryBytes: telemetry.peakBytes,
        memoryLimitBytes: telemetry.limitBytes,
      });
      return;
    }
    if (event.type === "step") {
      process.stderr.write(formatStepEvent(event));
      return;
    }
    if (event.type === "eval") {
      process.stderr.write(formatEvalEvent(event));
      return;
    }
    if (event.type === "done") {
      process.stderr.write(`\nTraining complete. ${event.totalSteps} steps.\n`);
    }
  }

  function maybeSaveBestCheckpoint(event: Extract<TrainEvent, { type: "eval" }>): void {
    const improved =
      autoTrainingPolicy.bestValLoss === null ||
      event.valLoss <= autoTrainingPolicy.bestValLoss - autoTrainingPolicy.minDelta;

    if (!improved) {
      return;
    }

    autoTrainingPolicy.bestValLoss = event.valLoss;
    autoTrainingPolicy.bestCheckpointStep = event.step;
    autoTrainingPolicy.consecutiveBadEvals = 0;
    autoTrainingPolicy.stopReason = undefined;

    if (autoTrainingPolicy.bestCheckpointPath !== undefined) {
      saveCheckpoint({
        model: session.model,
        kind: "best",
        config: session.config,
        step: event.step,
        tokenizer: session.tokenizer,
        path: autoTrainingPolicy.bestCheckpointPath,
      });
    }
    emitBestCheckpoint(useJson, event.step, event.valLoss, autoTrainingPolicy.bestCheckpointPath);
  }

  function maybeRequestEarlyStop(event: Extract<TrainEvent, { type: "eval" }>): void {
    if (
      autoTrainingPolicy.patience === null ||
      autoTrainingPolicy.bestCheckpointStep === event.step
    ) {
      return;
    }

    autoTrainingPolicy.consecutiveBadEvals += 1;
    if (autoTrainingPolicy.consecutiveBadEvals < autoTrainingPolicy.patience) {
      return;
    }

    const bestStep =
      autoTrainingPolicy.bestCheckpointStep === null
        ? "unknown"
        : String(autoTrainingPolicy.bestCheckpointStep);
    const bestVal =
      autoTrainingPolicy.bestValLoss === null
        ? "unknown"
        : autoTrainingPolicy.bestValLoss.toFixed(4);
    const reason =
      `validation loss did not improve by at least ${autoTrainingPolicy.minDelta} for ` +
      `${autoTrainingPolicy.consecutiveBadEvals} eval(s); best ${bestVal} at step ${bestStep}`;
    autoTrainingPolicy.stopReason = reason;
    emitEarlyStop(useJson, event.step, reason, autoTrainingPolicy);
    requestAutoStop(reason);
  }

  function maybeSavePeriodicCheckpoint(event: Extract<TrainEvent, { type: "eval" }>): void {
    const checkpointDir = checkpointPolicy.checkpointDir;
    if (checkpointDir === undefined) {
      return;
    }

    const checkpointKind = checkpointKindForStep(event.step);
    if (checkpointKind === undefined) {
      return;
    }

    const path = checkpointPath(checkpointDir, session.presetName, checkpointKind, event.step);
    saveCheckpoint({
      model: session.model,
      optimizer: checkpointKind === "resume" ? session.optimizer : undefined,
      kind: checkpointKind,
      config: session.config,
      step: event.step,
      tokenizer: session.tokenizer,
      path,
    });
    emitCheckpoint(useJson, event.step, checkpointKind, path, maybeClearAllocatorCache());
  }

  function handleEvalEvent(event: Extract<TrainEvent, { type: "eval" }>): void {
    maybeSaveBestCheckpoint(event);
    maybeRequestEarlyStop(event);
    maybeSavePeriodicCheckpoint(event);
  }

  function handleStepEvent(event: Extract<TrainEvent, { type: "step" }>): void {
    if (samplingPolicy.every <= 0 || event.step % samplingPolicy.every !== 0) {
      return;
    }
    emitSample(session, event.step, samplingPolicy.maxNewTokens, useJson);
  }

  return (event: TrainEvent): void => {
    emitEventOutput(event);
    if (event.type === "eval") {
      handleEvalEvent(event);
    }
    if (event.type === "step") {
      handleStepEvent(event);
    }
  };
}

export function emitTrainingSample(
  session: TrainingSession,
  finalPath: string,
  summary: TrainSummary,
  useJson: boolean,
): void {
  emitSample(session, Number(summary.totalSteps ?? 0), 200, useJson);
  if (useJson) {
    emitJson({
      type: "final-sample",
      preset: session.presetName,
      path: finalPath,
      summary,
    });
  }
}

export function emitStoppedRun(
  finalPath: string,
  summary: TrainSummary,
  policy: AutoTrainingPolicy,
  useJson: boolean,
): void {
  if (useJson) {
    emitJson({
      type: "stopped",
      path: finalPath,
      summary,
      reason: policy.stopReason,
      bestValLoss: policy.bestValLoss,
      bestCheckpointStep: policy.bestCheckpointStep,
      bestCheckpointPath: policy.bestCheckpointPath,
    });
    return;
  }
  process.stderr.write(
    `\nTraining stopped cleanly at step ${summary.totalSteps}. Final checkpoint: ${finalPath}${
      policy.stopReason === undefined ? "" : `\nReason: ${policy.stopReason}`
    }\n`,
  );
}

export function emitCancelledRun(summary: TrainSummary, useJson: boolean): void {
  if (useJson) {
    emitJson({ type: "cancelled", summary });
    return;
  }
  process.stderr.write(`\nTraining cancelled at step ${summary.totalSteps}.\n`);
}
