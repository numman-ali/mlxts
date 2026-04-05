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
  GENERATE_FLAG_ALLOWLIST,
  parseArgs,
  TRAIN_FLAG_ALLOWLIST,
  validateKnownFlags,
} from "./cli/shared";

async function main(): Promise<void> {
  const { command, flags } = parseArgs(process.argv);

  try {
    switch (command) {
      case "train":
        validateKnownFlags(flags, TRAIN_FLAG_ALLOWLIST, "train");
        if (flags.has("help")) {
          printTrainHelp();
          return;
        }
        await runTrain(flags);
        return;
      case "generate":
        validateKnownFlags(flags, GENERATE_FLAG_ALLOWLIST, "generate");
        if (flags.has("help")) {
          printGenerateHelp();
          return;
        }
        runGenerate(flags);
        return;
      case "export":
        validateKnownFlags(flags, EXPORT_FLAG_ALLOWLIST, "export");
        if (flags.has("help")) {
          printExportHelp();
          return;
        }
        await runExport(flags);
        return;
      default:
        printHelp();
    }
  } catch (error) {
    handleError(error);
  }
}

await main();
