import {
  readRunStatus as readSupervisedRunStatus,
  type RunStatus as SupervisedRunStatus,
  packageRootFromRunDir as supervisedPackageRootFromRunDir,
  runDir as supervisedRunDir,
  runsRoot as supervisedRunsRoot,
  writeRunStatus as writeSupervisedRunStatus,
} from "@mlxts/train/supervised-run";

import type { GPTConfig } from "../config";
import { NANOGPT_RUNS_DIRECTORY } from "./supervised-run-config";

type RunStatusBase = Omit<SupervisedRunStatus, "config">;

export type RunStatus = RunStatusBase & {
  config?: GPTConfig | undefined;
};

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function readBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readConfig(value: SupervisedRunStatus["config"]): GPTConfig | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const nLayer = readNumber(value.nLayer);
  const nHead = readNumber(value.nHead);
  const nEmbd = readNumber(value.nEmbd);
  const blockSize = readNumber(value.blockSize);
  const dropout = readNumber(value.dropout);
  const gradientCheckpointing = readBoolean(value.gradientCheckpointing);
  const vocabSize = readNumber(value.vocabSize);

  if (
    nLayer === undefined ||
    nHead === undefined ||
    nEmbd === undefined ||
    blockSize === undefined ||
    dropout === undefined ||
    gradientCheckpointing === undefined ||
    vocabSize === undefined
  ) {
    return undefined;
  }

  return {
    nLayer,
    nHead,
    nEmbd,
    blockSize,
    dropout,
    gradientCheckpointing,
    vocabSize,
  };
}

export function readRunStatus(runDirectory: string): RunStatus {
  const status = readSupervisedRunStatus(runDirectory);
  return {
    ...status,
    config: readConfig(status.config),
  };
}

export function writeRunStatus(runDirectory: string, status: RunStatus): void {
  writeSupervisedRunStatus(runDirectory, status);
}

export function runsRoot(repoRoot: string): string {
  return supervisedRunsRoot(repoRoot, NANOGPT_RUNS_DIRECTORY);
}

export function runDir(repoRoot: string, runId: string): string {
  return supervisedRunDir(repoRoot, runId, NANOGPT_RUNS_DIRECTORY);
}

export function packageRootFromRunDir(runDirectory: string): string {
  return supervisedPackageRootFromRunDir(runDirectory, "examples/nanogpt");
}

export {
  activePid,
  appendEvent,
  checkpointsDir,
  clearRunControl,
  DEFAULT_STALL_TIMEOUT_SECONDS,
  deriveOperatorHealth,
  ensureRunDir,
  eventsPath,
  type OperatorHealth,
  pidPath,
  type RunControl,
  type RunControlCommand,
  type RunHealth,
  type RunSpec,
  type RunState,
  readLatestCheckpoint,
  readRunControl,
  readRunSpec,
  repoRootFromPackageRoot,
  runControlPath,
  runSpecPath,
  runStatusPath,
  stderrPath,
  writePid,
  writeRunControl,
  writeRunSpec,
} from "@mlxts/train/supervised-run";
