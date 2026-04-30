import {
  getFlag,
  hasFlag,
  parseArgs,
  SupervisedRunManagerUsageError,
  validateAllowedFlags,
} from "./manager-args";
import {
  resumeRun,
  type SupervisedRunManagerRunOptions,
  startRun,
  writeControl,
} from "./manager-run";
import { printStatus, type SupervisedRunStatusOptions, watchRun } from "./manager-status";

export type SupervisedRunManagerCliOptions = {
  usage: string;
  helpCommand?: string | undefined;
  startFlagAllowlist: ReadonlySet<string>;
  resumeFlagAllowlist: ReadonlySet<string>;
  statusFlagAllowlist: ReadonlySet<string>;
  watchFlagAllowlist: ReadonlySet<string>;
  controlFlagAllowlist: ReadonlySet<string>;
  run: SupervisedRunManagerRunOptions;
  status: SupervisedRunStatusOptions;
};

export type SupervisedRunManagerCliRuntime = {
  stdout?: ((text: string) => void) | undefined;
};

const COMMANDS = new Set(["start", "resume", "status", "watch", "stop", "cancel", "help"]);

function writeStdout(stdout: (text: string) => void, text: string): void {
  stdout(text);
}

function usageCommand(options: SupervisedRunManagerCliOptions): string {
  return options.helpCommand ?? "manager help";
}

export function formatSupervisedManagerCliError(
  message: string,
  code: "usage" | "runtime" = "usage",
  helpCommand = "manager help",
): string {
  return [
    "error:",
    `  code: ${JSON.stringify(code)}`,
    `  message: ${JSON.stringify(message)}`,
    "help[1]:",
    `  Run \`${helpCommand}\` for supervised-run manager commands`,
  ].join("\n");
}

function withStdout(
  options: SupervisedRunManagerCliOptions,
  stdout: (text: string) => void,
): SupervisedRunManagerCliOptions {
  return {
    ...options,
    run: {
      ...options.run,
      stdout,
    },
    status: {
      ...options.status,
      stdout,
    },
  };
}

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
    throw new SupervisedRunManagerUsageError("status requires --name <run-id>");
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
    throw new SupervisedRunManagerUsageError("watch requires --name <run-id>");
  }

  const interval = Number(getFlag(args, "interval") ?? "10");
  if (!Number.isFinite(interval) || interval <= 0) {
    throw new SupervisedRunManagerUsageError("watch requires --interval to be a positive number");
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
    throw new SupervisedRunManagerUsageError("stop requires --name <run-id>");
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
    throw new SupervisedRunManagerUsageError("cancel requires --name <run-id>");
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

export async function runSupervisedManagerCliCommand(
  options: SupervisedRunManagerCliOptions,
  argv = process.argv,
  runtime: SupervisedRunManagerCliRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? ((text: string) => process.stdout.write(text));
  const { command, args } = parseArgs(argv);
  if (command === "help" || hasFlag(args, "help")) {
    writeStdout(stdout, options.usage);
    return 0;
  }
  if (!COMMANDS.has(command)) {
    writeStdout(
      stdout,
      `${formatSupervisedManagerCliError(
        `unknown command "${command}"`,
        "usage",
        usageCommand(options),
      )}\n`,
    );
    return 2;
  }

  try {
    await runSupervisedManagerCli(withStdout(options, stdout), argv);
    return 0;
  } catch (error) {
    const isUsage = error instanceof SupervisedRunManagerUsageError;
    writeStdout(
      stdout,
      `${formatSupervisedManagerCliError(
        error instanceof Error ? error.message : String(error),
        isUsage ? "usage" : "runtime",
        usageCommand(options),
      )}\n`,
    );
    return isUsage ? 2 : 1;
  }
}
