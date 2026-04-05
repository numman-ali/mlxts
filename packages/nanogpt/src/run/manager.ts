#!/usr/bin/env bun

import {
  CONTROL_FLAG_ALLOWLIST,
  getFlag,
  hasFlag,
  parseArgs,
  RESUME_FLAG_ALLOWLIST,
  START_FLAG_ALLOWLIST,
  STATUS_FLAG_ALLOWLIST,
  usage,
  validateAllowedFlags,
  WATCH_FLAG_ALLOWLIST,
} from "./manager-args";
import { resumeRun, startRun, writeControl } from "./manager-run";
import { printStatus, watchRun } from "./manager-status";

function handleStart(args: string[]): void {
  validateAllowedFlags(args, START_FLAG_ALLOWLIST, "start");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }
  startRun(args);
}

function handleResume(args: string[]): void {
  validateAllowedFlags(args, RESUME_FLAG_ALLOWLIST, "resume");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }
  resumeRun(args);
}

function handleStatus(args: string[]): void {
  validateAllowedFlags(args, STATUS_FLAG_ALLOWLIST, "status");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }

  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("status requires --name <run-id>");
  }
  printStatus(runId, hasFlag(args, "json"));
}

async function handleWatch(args: string[]): Promise<void> {
  validateAllowedFlags(args, WATCH_FLAG_ALLOWLIST, "watch");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }

  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("watch requires --name <run-id>");
  }

  const interval = Number(getFlag(args, "interval") ?? "10");
  if (!Number.isFinite(interval) || interval <= 0) {
    throw new Error("watch requires --interval to be a positive number");
  }
  await watchRun(runId, interval, hasFlag(args, "json"));
}

function handleStop(args: string[]): void {
  validateAllowedFlags(args, CONTROL_FLAG_ALLOWLIST, "stop");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }

  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("stop requires --name <run-id>");
  }
  writeControl(runId, "stop");
}

function handleCancel(args: string[]): void {
  validateAllowedFlags(args, CONTROL_FLAG_ALLOWLIST, "cancel");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }

  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("cancel requires --name <run-id>");
  }
  writeControl(runId, "cancel");
}

export async function main(argv = process.argv): Promise<void> {
  const { command, args } = parseArgs(argv);
  if (command === "help" || hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }

  const handlers: Record<string, (commandArgs: string[]) => void | Promise<void>> = {
    start: handleStart,
    resume: handleResume,
    status: handleStatus,
    watch: handleWatch,
    stop: handleStop,
    cancel: handleCancel,
  };

  const handler = handlers[command];
  if (handler === undefined) {
    process.stdout.write(usage());
    return;
  }
  await handler(args);
}

await main();
