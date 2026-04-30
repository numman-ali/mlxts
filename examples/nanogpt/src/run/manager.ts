#!/usr/bin/env bun

import { runSupervisedManagerCliCommand } from "@mlxts/train/supervised-run";

import { nanogptManagerCliOptions } from "./supervised-run-config";

export async function main(argv = process.argv): Promise<number> {
  return runSupervisedManagerCliCommand(nanogptManagerCliOptions, argv);
}

const exitCode = await main();
process.exit(exitCode);
