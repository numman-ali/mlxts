#!/usr/bin/env bun

import { runTrainingProofCommand } from "./cli";

const exitCode = await runTrainingProofCommand(Bun.argv.slice(2));
process.exit(exitCode);
