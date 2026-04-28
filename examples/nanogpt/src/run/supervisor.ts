#!/usr/bin/env bun

import { runSupervisedSupervisor } from "@mlxts/train/supervised-run";

import { acquireRuntimeCommandLock } from "../../../../scripts/runtime-command-lock";
import { nanogptSupervisorOptions } from "./supervised-run-config";

export async function main(argv = process.argv): Promise<void> {
  using _runtimeLock = acquireRuntimeCommandLock("train:nanogpt-supervisor");
  await runSupervisedSupervisor(nanogptSupervisorOptions, argv);
}

await main();
