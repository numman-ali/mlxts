#!/usr/bin/env bun

/**
 * nanogpt CLI — train GPT models and generate text.
 *
 * Usage:
 *   nanogpt train [options]
 *   nanogpt generate --checkpoint <path> [options]
 *
 * @module
 */

import {
  handleError,
  printExportHelp,
  printGenerateHelp,
  printTrainHelp,
  runExport,
  runGenerate,
  runTrain,
} from "./cli/commands";
import { printHelp } from "./cli/help";
import {
  EXPORT_FLAG_ALLOWLIST,
  EXPORT_VALUE_FLAGS,
  GENERATE_FLAG_ALLOWLIST,
  GENERATE_VALUE_FLAGS,
  parseArgs,
  TRAIN_FLAG_ALLOWLIST,
  TRAIN_VALUE_FLAGS,
  UserError,
  validateFlagValues,
  validateKnownFlags,
} from "./cli/shared";

async function main(): Promise<void> {
  const { command, flags, valuedFlags } = parseArgs(process.argv);

  try {
    switch (command) {
      case "help":
        printHelp();
        return;
      case "train":
        validateKnownFlags(flags, TRAIN_FLAG_ALLOWLIST, "train");
        validateFlagValues(flags, valuedFlags, TRAIN_VALUE_FLAGS, "train");
        if (flags.has("help")) {
          printTrainHelp();
          return;
        }
        await runTrain(flags);
        return;
      case "generate":
        validateKnownFlags(flags, GENERATE_FLAG_ALLOWLIST, "generate");
        validateFlagValues(flags, valuedFlags, GENERATE_VALUE_FLAGS, "generate");
        if (flags.has("help")) {
          printGenerateHelp();
          return;
        }
        runGenerate(flags);
        return;
      case "export":
        validateKnownFlags(flags, EXPORT_FLAG_ALLOWLIST, "export");
        validateFlagValues(flags, valuedFlags, EXPORT_VALUE_FLAGS, "export");
        if (flags.has("help")) {
          printExportHelp();
          return;
        }
        await runExport(flags);
        return;
      default:
        throw new UserError(`Unknown command "${command}"`);
    }
  } catch (error) {
    handleError(error);
  }
}

await main();
