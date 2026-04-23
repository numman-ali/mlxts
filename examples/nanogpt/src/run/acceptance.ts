#!/usr/bin/env bun

import { CharTokenizer } from "@mlxts/tokenizers";
import { resolve } from "path";
import { acquireRuntimeCommandLock } from "../../../../scripts/runtime-command-lock";
import { applyCheckpoint, loadCheckpoint } from "../checkpoint";
import { generate } from "../generate";
import { GPT } from "../model/gpt";
import {
  buildManagerArgs,
  getNumberFlag,
  parseArgs,
  readMode,
  readPresetName,
  readRunOptions,
} from "./acceptance-options";
import {
  assertCompletedStatus,
  assertSoakStabilityForEvents,
  checkpointPathFromStatus,
  finalLossFromStatus,
  readStepEvents,
  type StepEventRecord,
  samplePrompt,
  waitForTerminalState,
} from "./acceptance-runtime";

function packageRoot(): string {
  return resolve(import.meta.dir, "../..");
}

function decodeOutput(bytes: Uint8Array): string {
  return new TextDecoder().decode(bytes);
}

function assertSoakStability(
  runId: string,
  sampleSize: number,
  minRatio: number,
  maxSlopeMbPerEvent: number,
): void {
  const metrics = assertSoakStabilityForEvents(
    runId,
    readStepEvents(runId),
    sampleSize,
    minRatio,
    maxSlopeMbPerEvent,
  );

  process.stdout.write(
    `throughputRatio=${metrics.throughputRatio.toFixed(3)} firstAvg=${Math.round(metrics.firstAverage).toLocaleString()} lastAvg=${Math.round(metrics.lastAverage).toLocaleString()} activeSlopeMbPerEvent=${metrics.slopeMbPerEvent.toFixed(3)}\n`,
  );
}

export async function main(argv = process.argv): Promise<void> {
  const flags = parseArgs(argv);
  const {
    presetName,
    mode,
    runId,
    pollSeconds,
    lossTarget,
    throughputWindow,
    minThroughputRatio,
    maxSlopeMbPerEvent,
    parameterCount,
    args,
  } = readRunOptions(flags);

  process.stdout.write(
    `Acceptance run: ${presetName} (${parameterCount.toLocaleString()} params)\n`,
  );
  const startResult = Bun.spawnSync(["bun", ...args], {
    cwd: packageRoot(),
    stdout: "pipe",
    stderr: "pipe",
  });
  if (startResult.exitCode !== 0) {
    process.stderr.write(decodeOutput(startResult.stderr));
    process.exit(startResult.exitCode);
  }

  process.stdout.write(decodeOutput(startResult.stdout));
  const status = waitForTerminalState(runId, pollSeconds);
  assertCompletedStatus(runId, status);

  if (mode === "soak") {
    assertSoakStability(runId, throughputWindow, minThroughputRatio, maxSlopeMbPerEvent);
    process.stdout.write(`run=${runId} preset=${presetName} mode=soak status=completed\n`);
    return;
  }

  const finalLoss = finalLossFromStatus(status);
  if (finalLoss >= lossTarget) {
    throw new Error(
      `Acceptance failed: ${presetName} final loss ${finalLoss.toFixed(4)} did not beat ${lossTarget.toFixed(4)}`,
    );
  }

  const checkpointPath = checkpointPathFromStatus(status);

  // The supervised run owns the runtime lock while training. Reacquire only
  // for the post-run checkpoint load and sample generation work.
  using _runtimeLock = acquireRuntimeCommandLock("acceptance:nanogpt-postrun");
  const checkpoint = loadCheckpoint(checkpointPath);
  const tokenizer = CharTokenizer.fromVocab(checkpoint.metadata.tokenizer.chars);
  const model = new GPT(checkpoint.metadata.config);

  try {
    model.eval();
    applyCheckpoint(model, checkpoint);
    const sample = generate(model, checkpoint.metadata.config, tokenizer, samplePrompt(tokenizer), {
      maxNewTokens: 200,
      temperature: 0.8,
    });
    process.stdout.write(`checkpoint=${checkpointPath}\n`);
    process.stdout.write(`finalLoss=${finalLoss.toFixed(4)} target=${lossTarget.toFixed(4)}\n`);
    process.stdout.write(`sample:\n${sample}\n`);
  } finally {
    model[Symbol.dispose]();
  }
}

export {
  assertCompletedStatus,
  assertSoakStabilityForEvents,
  buildManagerArgs,
  checkpointPathFromStatus,
  finalLossFromStatus,
  getNumberFlag,
  parseArgs,
  readMode,
  readPresetName,
  readRunOptions,
  readStepEvents,
  type StepEventRecord,
  samplePrompt,
  waitForTerminalState,
};

if (import.meta.main) {
  await main();
}
