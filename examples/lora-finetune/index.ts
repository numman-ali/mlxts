#!/usr/bin/env bun

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { runLoRAFinetune } from "./workflow";

using _runtimeLock = acquireRuntimeCommandLock("example:lora-finetune");
await runLoRAFinetune(Bun.argv.slice(2));
