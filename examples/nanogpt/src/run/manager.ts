#!/usr/bin/env bun

import { runSupervisedManagerCli } from "@mlxts/train/supervised-run";

import { nanogptManagerCliOptions } from "./supervised-run-config";

export async function main(argv = process.argv): Promise<void> {
  await runSupervisedManagerCli(nanogptManagerCliOptions, argv);
}

await main();
