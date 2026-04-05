#!/usr/bin/env bun

import { quantizePretrainedSnapshot } from "../src/quantize";

type CliOptions = {
  source: string;
  outputDir: string;
  revision?: string;
  bits?: number;
  groupSize?: number;
  mode?: "affine" | "mxfp4" | "mxfp8" | "nvfp4";
  localFilesOnly: boolean;
  overwrite: boolean;
};

function usage(): never {
  console.error(
    "Usage: bun run packages/transformers/scripts/quantize-snapshot.ts <source> --out <dir> [--bits <n>] [--group-size <n>] [--mode affine|mxfp4|mxfp8|nvfp4] [--revision <rev>] [--local-files-only] [--overwrite]",
  );
  process.exit(1);
}

function readInteger(flag: string, value: string | undefined): number {
  const parsed = Number.parseInt(value ?? "", 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`quantize-snapshot: ${flag} expects a positive integer.`);
  }
  return parsed;
}

function parseArgs(argv: readonly string[]): CliOptions {
  const source = argv[0];
  if (source === undefined || source.startsWith("--")) {
    usage();
  }

  const options: CliOptions = {
    source,
    outputDir: "",
    localFilesOnly: false,
    overwrite: false,
  };

  for (let index = 1; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }

    switch (arg) {
      case "--out":
        options.outputDir = argv[index + 1] ?? "";
        index += 1;
        break;
      case "--bits":
        options.bits = readInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--group-size":
        options.groupSize = readInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--mode": {
        const mode = argv[index + 1];
        if (mode !== "affine" && mode !== "mxfp4" && mode !== "mxfp8" && mode !== "nvfp4") {
          throw new Error(`quantize-snapshot: unsupported mode "${mode ?? ""}".`);
        }
        options.mode = mode;
        index += 1;
        break;
      }
      case "--revision":
        options.revision = argv[index + 1];
        index += 1;
        break;
      case "--local-files-only":
        options.localFilesOnly = true;
        break;
      case "--overwrite":
        options.overwrite = true;
        break;
      default:
        throw new Error(`quantize-snapshot: unknown flag "${arg}".`);
    }
  }

  if (options.outputDir.trim() === "") {
    throw new Error("quantize-snapshot: --out is required.");
  }

  return options;
}

async function main(): Promise<void> {
  const options = parseArgs(Bun.argv.slice(2));
  console.log(`Quantizing ${options.source} -> ${options.outputDir}`);
  const result = await quantizePretrainedSnapshot(options.source, {
    outputDir: options.outputDir,
    revision: options.revision,
    bits: options.bits,
    groupSize: options.groupSize,
    mode: options.mode,
    localFilesOnly: options.localFilesOnly,
    overwrite: options.overwrite,
  });

  const ratio =
    result.inputBytes === 0
      ? "n/a"
      : `${(result.outputBytes / result.inputBytes).toFixed(3)}x output/input bytes`;
  console.log(
    [
      `shards=${result.shardCount}`,
      `quantized_tensors=${result.quantizedTensorCount}`,
      `copied_tensors=${result.copiedTensorCount}`,
      `input_bytes=${result.inputBytes}`,
      `output_bytes=${result.outputBytes}`,
      ratio,
    ].join(" "),
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
