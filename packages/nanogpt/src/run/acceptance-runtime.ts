import type { CharTokenizer } from "@mlxts/tokenizers";
import { readFileSync } from "fs";
import { resolve } from "path";

import type { RunStatus } from "./files";
import {
  deriveOperatorHealth,
  eventsPath,
  readRunStatus,
  repoRootFromPackageRoot,
  runDir,
} from "./files";

export type StepEventRecord = {
  type: "step";
  step: number;
  tokensPerSec: number;
  activeMemoryBytes?: number | undefined;
};

function packageRoot(): string {
  return resolve(import.meta.dir, "../..");
}

function repoRoot(): string {
  return repoRootFromPackageRoot(packageRoot());
}

function parseJsonLine(line: string): unknown {
  return JSON.parse(line);
}

function readJsonLines(content: string): unknown[] {
  if (content.trim() === "") {
    return [];
  }

  const parsed: unknown[] = [];
  const lines = content.split("\n");
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (line === undefined || line.trim() === "") {
      continue;
    }

    try {
      parsed.push(parseJsonLine(line));
    } catch (error) {
      if (index === lines.length - 1) {
        break;
      }
      throw error;
    }
  }
  return parsed;
}

function isStepEventRecord(value: unknown): value is StepEventRecord {
  return (
    typeof value === "object" &&
    value !== null &&
    "type" in value &&
    value.type === "step" &&
    "step" in value &&
    typeof value.step === "number" &&
    "tokensPerSec" in value &&
    typeof value.tokensPerSec === "number"
  );
}

export function readStepEvents(runId: string): StepEventRecord[] {
  const eventLog = eventsPath(runDir(repoRoot(), runId));
  const stepEvents: StepEventRecord[] = [];
  for (const parsed of readJsonLines(readFileSync(eventLog, "utf-8"))) {
    if (isStepEventRecord(parsed)) {
      stepEvents.push(parsed);
    }
  }
  return stepEvents;
}

function average(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error("Cannot average an empty sequence");
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function assertSoakStabilityForEvents(
  runId: string,
  stepEvents: readonly StepEventRecord[],
  sampleSize: number,
  minRatio: number,
  maxSlopeMbPerEvent: number,
): {
  throughputRatio: number;
  firstAverage: number;
  lastAverage: number;
  slopeMbPerEvent: number;
} {
  if (stepEvents.length < sampleSize * 2) {
    throw new Error(
      `Soak run ${runId} recorded only ${stepEvents.length} step events; need at least ${
        sampleSize * 2
      } to judge throughput drift.`,
    );
  }

  const firstAverage = average(stepEvents.slice(0, sampleSize).map((event) => event.tokensPerSec));
  const lastAverage = average(stepEvents.slice(-sampleSize).map((event) => event.tokensPerSec));
  const ratio = lastAverage / firstAverage;

  if (ratio < minRatio) {
    throw new Error(
      `Soak failed: throughput ratio ${ratio.toFixed(3)} is below the minimum ${minRatio.toFixed(3)} ` +
        `(first ${Math.round(firstAverage).toLocaleString()} tok/s, last ${Math.round(lastAverage).toLocaleString()} tok/s).`,
    );
  }

  const memoryEvents = stepEvents.filter((event) => event.activeMemoryBytes !== undefined);
  if (memoryEvents.length < 2) {
    throw new Error(`Soak run ${runId} did not record enough memory-bearing step events.`);
  }

  const firstMemory = memoryEvents[0];
  const lastMemory = memoryEvents[memoryEvents.length - 1];
  if (firstMemory?.activeMemoryBytes === undefined || lastMemory?.activeMemoryBytes === undefined) {
    throw new Error(`Soak run ${runId} did not record valid active-memory samples.`);
  }

  const slopeMbPerEvent =
    (lastMemory.activeMemoryBytes - firstMemory.activeMemoryBytes) /
    Math.max(memoryEvents.length - 1, 1) /
    (1024 * 1024);
  if (slopeMbPerEvent > maxSlopeMbPerEvent) {
    throw new Error(
      `Soak failed: active memory slope ${slopeMbPerEvent.toFixed(3)} MB/event exceeded ${maxSlopeMbPerEvent.toFixed(3)} MB/event.`,
    );
  }

  return {
    throughputRatio: ratio,
    firstAverage,
    lastAverage,
    slopeMbPerEvent,
  };
}

export function samplePrompt(tokenizer: CharTokenizer): string {
  const vocab = tokenizer.vocab;
  return vocab.includes("\n") ? "\n" : (vocab[0] ?? "");
}

function isTerminalState(state: RunStatus["state"]): boolean {
  return (
    state === "completed" ||
    state === "stopped" ||
    state === "stalled" ||
    state === "failed" ||
    state === "cancelled"
  );
}

export function waitForTerminalState(
  runId: string,
  pollSeconds: number,
): ReturnType<typeof readRunStatus> {
  const directory = runDir(repoRoot(), runId);
  while (true) {
    const status = readRunStatus(directory);
    const health = deriveOperatorHealth(status);
    if (isTerminalState(status.state)) {
      return status;
    }

    if (health.operatorHealth !== "healthy") {
      Bun.sleepSync(250);
      const refreshedStatus = readRunStatus(directory);
      const refreshedHealth = deriveOperatorHealth(refreshedStatus);
      if (isTerminalState(refreshedStatus.state)) {
        return refreshedStatus;
      }
      if (refreshedHealth.operatorHealth !== "healthy") {
        throw new Error(
          `Acceptance run ${runId} became unhealthy (${refreshedHealth.operatorHealth}) before reaching a terminal state`,
        );
      }
    }

    Bun.sleepSync(pollSeconds * 1000);
  }
}

export function assertCompletedStatus(runId: string, status: RunStatus): void {
  if (status.state === "stalled") {
    const reason = status.stallReason === undefined ? "unknown stall reason" : status.stallReason;
    throw new Error(`Acceptance run ${runId} stalled: ${reason}`);
  }
  if (status.state === "stopped" && status.earlyStopReason !== undefined) {
    return;
  }
  if (status.state !== "completed") {
    throw new Error(`Acceptance run ${runId} ended in state ${status.state}`);
  }
}

export function finalLossFromStatus(status: RunStatus): number {
  const validationLosses = [status.bestValLoss, status.lastValLoss].filter(
    (value): value is number => value !== undefined,
  );
  if (validationLosses.length > 0) {
    return Math.min(...validationLosses);
  }

  if (status.lastStepLoss === undefined) {
    throw new Error("Acceptance run completed without a final recorded loss");
  }
  return status.lastStepLoss;
}

export function checkpointPathFromStatus(status: RunStatus): string {
  const checkpointPath =
    status.bestCheckpoint ?? status.latestResumeCheckpoint ?? status.latestCheckpoint;
  if (checkpointPath === undefined) {
    throw new Error("Acceptance run completed without a checkpoint");
  }
  return checkpointPath;
}
