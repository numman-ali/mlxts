#!/usr/bin/env bun

import { array, loadSafetensors, mxEval } from "@mlxts/core";
import { readFileSync } from "fs";
import { join, resolve } from "path";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";

import { loadCausalLM } from "../src/load";

type ParityFixture = {
  model: string;
  promptText?: string;
  tokenIds: number[];
  tolerance?: number;
  logitsTensorName?: string;
};

function usage(): never {
  console.error(
    "Usage: bun run packages/transformers/scripts/verify-llama-parity.ts <model-path-or-repo-id> <fixture-dir>",
  );
  process.exit(1);
}

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") {
    return [value];
  }
  if (!Array.isArray(value)) {
    throw new Error("verify-llama-parity: expected nested numeric arrays in the logits tensor.");
  }
  return value.flatMap((entry) => flattenNumbers(entry));
}

function loadFixture(fixtureDirectory: string): ParityFixture {
  const path = join(fixtureDirectory, "fixture.json");
  const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("verify-llama-parity: fixture.json must contain an object.");
  }

  const record = Object.fromEntries(Object.entries(parsed));
  const tokenIds = record.tokenIds;
  if (!Array.isArray(tokenIds) || tokenIds.some((value) => typeof value !== "number")) {
    throw new Error("verify-llama-parity: fixture.json must include a numeric tokenIds array.");
  }

  return {
    model: typeof record.model === "string" ? record.model : "",
    promptText: typeof record.promptText === "string" ? record.promptText : undefined,
    tokenIds,
    tolerance: typeof record.tolerance === "number" ? record.tolerance : undefined,
    logitsTensorName:
      typeof record.logitsTensorName === "string" ? record.logitsTensorName : undefined,
  };
}

async function main(): Promise<void> {
  using _runtimeLock = acquireRuntimeCommandLock("verify:llama-parity");
  const modelSource = Bun.argv[2];
  const fixtureDirectoryArg = Bun.argv[3];
  if (modelSource === undefined || fixtureDirectoryArg === undefined) {
    usage();
  }

  const fixtureDirectory = resolve(fixtureDirectoryArg);
  const fixture = loadFixture(fixtureDirectory);
  const tolerance = fixture.tolerance ?? 1e-4;
  const logitsTensorName = fixture.logitsTensorName ?? "logits";

  using model = await loadCausalLM(modelSource);
  using inputIds = array([fixture.tokenIds], "int32");
  using actualLogits = model.forward(inputIds);
  const expected = await loadSafetensors(join(fixtureDirectory, "logits.safetensors"));
  const expectedLogits = expected.tensors[logitsTensorName];
  if (expectedLogits === undefined) {
    throw new Error(
      `verify-llama-parity: logits.safetensors does not contain tensor "${logitsTensorName}".`,
    );
  }

  try {
    mxEval(actualLogits, expectedLogits);

    const actualValues = flattenNumbers(actualLogits.toList());
    const expectedValues = flattenNumbers(expectedLogits.toList());
    if (actualValues.length !== expectedValues.length) {
      throw new Error(
        `verify-llama-parity: logits length mismatch (${actualValues.length} vs ${expectedValues.length}).`,
      );
    }

    let maxAbsDiff = 0;
    for (let index = 0; index < actualValues.length; index += 1) {
      const diff = Math.abs((actualValues[index] ?? 0) - (expectedValues[index] ?? 0));
      if (diff > maxAbsDiff) {
        maxAbsDiff = diff;
      }
    }

    console.log(
      JSON.stringify(
        {
          modelSource,
          fixtureModel: fixture.model,
          promptText: fixture.promptText,
          tokenCount: fixture.tokenIds.length,
          logitsTensorName,
          maxAbsDiff,
          tolerance,
          passed: maxAbsDiff <= tolerance,
        },
        null,
        2,
      ),
    );

    if (maxAbsDiff > tolerance) {
      throw new Error(
        `verify-llama-parity: max abs diff ${maxAbsDiff} exceeded tolerance ${tolerance}.`,
      );
    }
  } finally {
    for (const tensor of Object.values(expected.tensors)) {
      tensor.free();
    }
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
