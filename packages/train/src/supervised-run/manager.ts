import { getFlag, hasFlag, parseArgs, validateAllowedFlags } from "./manager-args";
import {
  resumeRun,
  type SupervisedRunManagerRunOptions,
  startRun,
  writeControl,
} from "./manager-run";
import { printStatus, type SupervisedRunStatusOptions, watchRun } from "./manager-status";

export type SupervisedRunManagerCliOptions = {
  usage: string;
  startFlagAllowlist: ReadonlySet<string>;
  resumeFlagAllowlist: ReadonlySet<string>;
  statusFlagAllowlist: ReadonlySet<string>;
  watchFlagAllowlist: ReadonlySet<string>;
  controlFlagAllowlist: ReadonlySet<string>;
  run: SupervisedRunManagerRunOptions;
  status: SupervisedRunStatusOptions;
};

function handleStart(args: string[], options: SupervisedRunManagerCliOptions): void {
  validateAllowedFlags(args, options.startFlagAllowlist, "start");
  if (hasFlag(args, "help")) {
    process.stdout.write(options.usage);
    return;
  }
  startRun(args, options.run);
}

function handleResume(args: string[], options: SupervisedRunManagerCliOptions): void {
  validateAllowedFlags(args, options.resumeFlagAllowlist, "resume");
  if (hasFlag(args, "help")) {
    process.stdout.write(options.usage);
    return;
  }
  resumeRun(args, options.run);
}

function handleStatus(args: string[], options: SupervisedRunManagerCliOptions): void {
  validateAllowedFlags(args, options.statusFlagAllowlist, "status");
  if (hasFlag(args, "help")) {
    process.stdout.write(options.usage);
    return;
  }

  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("status requires --name <run-id>");
  }
  printStatus(runId, hasFlag(args, "json"), options.status);
}

async function handleWatch(args: string[], options: SupervisedRunManagerCliOptions): Promise<void> {
  validateAllowedFlags(args, options.watchFlagAllowlist, "watch");
  if (hasFlag(args, "help")) {
    process.stdout.write(options.usage);
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
  await watchRun(runId, interval, hasFlag(args, "json"), options.status);
}

function handleStop(args: string[], options: SupervisedRunManagerCliOptions): void {
  validateAllowedFlags(args, options.controlFlagAllowlist, "stop");
  if (hasFlag(args, "help")) {
    process.stdout.write(options.usage);
    return;
  }

  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("stop requires --name <run-id>");
  }
  writeControl(runId, "stop", options.run);
}

function handleCancel(args: string[], options: SupervisedRunManagerCliOptions): void {
  validateAllowedFlags(args, options.controlFlagAllowlist, "cancel");
  if (hasFlag(args, "help")) {
    process.stdout.write(options.usage);
    return;
  }

  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("cancel requires --name <run-id>");
  }
  writeControl(runId, "cancel", options.run);
}

export async function runSupervisedManagerCli(
  options: SupervisedRunManagerCliOptions,
  argv = process.argv,
): Promise<void> {
  const { command, args } = parseArgs(argv);
  if (command === "help" || hasFlag(args, "help")) {
    process.stdout.write(options.usage);
    return;
  }

  const handlers: Record<
    string,
    (commandArgs: string[], options: SupervisedRunManagerCliOptions) => void | Promise<void>
  > = {
    start: handleStart,
    resume: handleResume,
    status: handleStatus,
    watch: handleWatch,
    stop: handleStop,
    cancel: handleCancel,
  };

  const handler = handlers[command];
  if (handler === undefined) {
    process.stdout.write(options.usage);
    return;
  }
  await handler(args, options);
}
