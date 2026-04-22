#!/usr/bin/env bun

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { runTrainingProof } from "./workflow";

using _runtimeLock = acquireRuntimeCommandLock("proof:training");
await runTrainingProof(Bun.argv.slice(2));
