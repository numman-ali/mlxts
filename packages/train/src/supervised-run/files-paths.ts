import { existsSync, mkdirSync, readdirSync } from "fs";
import { join, resolve } from "path";

export const DEFAULT_RUNS_DIRECTORY_NAME = ".supervised-runs";

export function packageRootFromRunDir(
  runDirectory: string,
  packagePathFromRepoRoot: string,
): string {
  return resolve(runDirectory, "..", "..", packagePathFromRepoRoot);
}

export function repoRootFromPackageRoot(packageRoot: string): string {
  return resolve(packageRoot, "..", "..");
}

export function runsRoot(
  repoRoot: string,
  runsDirectoryName = DEFAULT_RUNS_DIRECTORY_NAME,
): string {
  return join(repoRoot, runsDirectoryName);
}

export function runDir(
  repoRoot: string,
  runId: string,
  runsDirectoryName = DEFAULT_RUNS_DIRECTORY_NAME,
): string {
  return join(runsRoot(repoRoot, runsDirectoryName), runId);
}

export function checkpointsDir(runDirectory: string): string {
  return join(runDirectory, "checkpoints");
}

export function runSpecPath(runDirectory: string): string {
  return join(runDirectory, "run.json");
}

export function runStatusPath(runDirectory: string): string {
  return join(runDirectory, "status.json");
}

export function runControlPath(runDirectory: string): string {
  return join(runDirectory, "control.json");
}

export function eventsPath(runDirectory: string): string {
  return join(runDirectory, "events.jsonl");
}

export function stderrPath(runDirectory: string): string {
  return join(runDirectory, "stderr.log");
}

export function pidPath(runDirectory: string): string {
  return join(runDirectory, "pid");
}

export function ensureRunDir(runDirectory: string): void {
  mkdirSync(runDirectory, { recursive: true });
  mkdirSync(checkpointsDir(runDirectory), { recursive: true });
}

function checkpointStep(name: string): number | undefined {
  const match = /-step-(\d+)$/.exec(name);
  if (match === null) {
    return undefined;
  }

  const value = Number(match[1]);
  return Number.isInteger(value) ? value : undefined;
}

export function readLatestCheckpoint(runDirectory: string): string | undefined {
  const checkpointDirectory = checkpointsDir(runDirectory);
  if (!existsSync(checkpointDirectory)) {
    return undefined;
  }

  const entries = readdirSync(checkpointDirectory, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);

  entries.sort((left, right) => {
    const leftStep = checkpointStep(left);
    const rightStep = checkpointStep(right);
    if (leftStep !== undefined && rightStep !== undefined && leftStep !== rightStep) {
      return leftStep - rightStep;
    }
    return left.localeCompare(right);
  });

  const latest = entries.at(-1);
  return latest === undefined ? undefined : join(checkpointDirectory, latest);
}
