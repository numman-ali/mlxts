#!/usr/bin/env bun

import { runLoRAFinetuneCommand } from "./cli";

const exitCode = await runLoRAFinetuneCommand(Bun.argv.slice(2));
process.exit(exitCode);
