import type {
  DType,
  QuantizationMode,
  SafetensorTensorInfo,
  SafetensorWriteEntry,
  SupportedSafetensorsDType,
} from "@mlxts/core";
import {
  inspectSafetensors,
  iterateSafetensorByteChunks,
  iterateSafetensors,
  quantize,
  saveSafetensorsStream,
  tensorBytes,
} from "@mlxts/core";
import { type QuantizationParameters, resolveQuantizationParameters } from "@mlxts/quantize";
import { existsSync, mkdirSync, rmSync, statSync } from "fs";
import { dirname, join, relative, resolve } from "path";

import { inspectSnapshot, resolvePretrainedSnapshot } from "./pretrained/snapshot";
import { listSafetensorShardPaths } from "./pretrained/weights";
import { resolveFamily } from "./registry";
import type { BaseModelConfig, FamilyRegistration, LoadSourceOptions } from "./types";

export type QuantizePretrainedSnapshotOptions = LoadSourceOptions & {
  outputDir: string;
  overwrite?: boolean;
  bits?: number;
  groupSize?: number;
  mode?: QuantizationMode;
  quantizeLmHead?: boolean;
};

export type QuantizedSnapshotResult = {
  outputDir: string;
  shardCount: number;
  quantizedTensorCount: number;
  copiedTensorCount: number;
  inputBytes: number;
  outputBytes: number;
  outputConfigPath: string;
  outputIndexPath?: string;
};

function isFloatingDType(dtype: DType): boolean {
  return dtype === "float16" || dtype === "bfloat16" || dtype === "float32" || dtype === "float64";
}

function ensureOutputDirectory(path: string, overwrite: boolean): void {
  if (existsSync(path)) {
    if (!overwrite) {
      throw new Error(`quantizePretrainedSnapshot: output directory "${path}" already exists.`);
    }
    rmSync(path, { recursive: true, force: true });
  }
  mkdirSync(path, { recursive: true });
}

function relativeFileMap(
  files: readonly { localPath: string; relativePath: string }[],
): Map<string, string> {
  return new Map(files.map((file) => [resolve(file.localPath), file.relativePath]));
}

function writeJson(path: string, value: unknown): Promise<number> {
  mkdirSync(dirname(path), { recursive: true });
  return Bun.write(path, `${JSON.stringify(value, null, 2)}\n`);
}

async function copyFile(sourcePath: string, targetPath: string): Promise<number> {
  mkdirSync(dirname(targetPath), { recursive: true });
  return Bun.write(targetPath, Bun.file(sourcePath));
}

function parseConfigForQuantization(inspectionConfig: Record<string, unknown>): {
  registration: FamilyRegistration;
  config: BaseModelConfig;
} {
  const modelType = inspectionConfig.model_type;
  if (typeof modelType !== "string" || modelType === "") {
    throw new Error("quantizePretrainedSnapshot: config.model_type must be a non-empty string.");
  }

  const registration = resolveFamily(modelType);
  return {
    registration,
    config: registration.parseConfig(inspectionConfig),
  };
}

function quantizationConfigRecord(params: QuantizationParameters): Record<string, number | string> {
  return {
    group_size: params.groupSize,
    bits: params.bits,
    mode: params.mode,
  };
}

function rewrittenConfigRecord(
  config: Record<string, unknown>,
  params: QuantizationParameters,
): Record<string, unknown> {
  const {
    quantization: _ignoredQuantization,
    quantization_config: _ignoredQuantizationConfig,
    ...rest
  } = config;
  return {
    ...rest,
    quantization: quantizationConfigRecord(params),
  };
}

function outputIndexRelativePath(
  inspection: ReturnType<typeof inspectSnapshot>,
): string | undefined {
  const indexPath = inspection.model.safetensorsIndexPath;
  if (indexPath === undefined) {
    return undefined;
  }
  return relative(inspection.snapshot.directory, indexPath);
}

function fileSize(path: string): number {
  return statSync(path).size;
}

function quantizedTensorKey(name: string, suffix: ".scales" | ".biases"): string {
  return `${name.slice(0, -".weight".length)}${suffix}`;
}

function excludedSnapshotPaths(
  inspection: ReturnType<typeof inspectSnapshot>,
  shardPaths: readonly string[],
): Set<string> {
  return new Set(
    [inspection.model.configPath, inspection.model.safetensorsIndexPath, ...shardPaths]
      .filter((path) => path !== undefined)
      .map((path) => resolve(path)),
  );
}

async function copyNonModelFiles(
  files: readonly { localPath: string; relativePath: string }[],
  excluded: ReadonlySet<string>,
  outputDir: string,
): Promise<number> {
  let outputBytes = 0;
  for (const file of files) {
    const absolutePath = resolve(file.localPath);
    if (excluded.has(absolutePath)) {
      continue;
    }
    outputBytes += await copyFile(file.localPath, join(outputDir, file.relativePath));
  }
  return outputBytes;
}

type QuantizedShardResult = {
  quantizedTensorCount: number;
  copiedTensorCount: number;
  outputBytes: number;
};

const COPY_CHUNK_BYTES = 8 * 1024 * 1024;

function packedColumnCount(inputDims: number, bits: number): number {
  const packedColumns = (inputDims * bits) / 32;
  if (!Number.isInteger(packedColumns)) {
    throw new Error(
      `quantizePretrainedSnapshot: inputDims ${inputDims} and bits ${bits} do not form a valid packed layout.`,
    );
  }
  return packedColumns;
}

function quantizationGroupCount(inputDims: number, groupSize: number): number {
  if (inputDims % groupSize !== 0) {
    throw new Error(
      `quantizePretrainedSnapshot: inputDims ${inputDims} must be divisible by groupSize ${groupSize}.`,
    );
  }
  return inputDims / groupSize;
}

function isQuantizableMappedWeight(
  mappedPath: string | null,
  tensor: Pick<SafetensorTensorInfo, "shape" | "dtype">,
  params: QuantizationParameters,
  options: Pick<QuantizePretrainedSnapshotOptions, "quantizeLmHead">,
): mappedPath is string {
  if (mappedPath === null || !mappedPath.endsWith(".weight")) {
    return false;
  }
  if (tensor.shape.length !== 2 || !isFloatingDType(tensor.dtype)) {
    return false;
  }
  if (
    mappedPath.endsWith("embedTokens.weight") ||
    mappedPath.endsWith("embedTokensPerLayer.weight")
  ) {
    return false;
  }
  if (options.quantizeLmHead === false && mappedPath === "lmHead.weight") {
    return false;
  }

  const inputDims = tensor.shape[1];
  return inputDims !== undefined && inputDims % params.groupSize === 0;
}

function copyTensorEntry(shardPath: string, tensor: SafetensorTensorInfo): SafetensorWriteEntry {
  return {
    name: tensor.name,
    shape: tensor.shape,
    dtype: tensor.dtype,
    chunks: () =>
      (async function* (): AsyncGenerator<Uint8Array, void, void> {
        for await (const chunk of iterateSafetensorByteChunks(shardPath, tensor.name, {
          maxBytesPerChunk: COPY_CHUNK_BYTES,
        })) {
          yield chunk.bytes;
        }
      })(),
  };
}

type OutputTensorDescriptor = {
  name: string;
  shape: number[];
  dtype: SupportedSafetensorsDType;
};

function quantizedOutputDescriptors(
  tensor: SafetensorTensorInfo,
  params: QuantizationParameters,
): OutputTensorDescriptor[] {
  const [outputDims, inputDims] = tensor.shape;
  if (outputDims === undefined || inputDims === undefined || tensor.shape.length !== 2) {
    throw new Error(
      `quantizePretrainedSnapshot: expected a 2D weight tensor for "${tensor.name}".`,
    );
  }

  const groupCount = quantizationGroupCount(inputDims, params.groupSize);
  const descriptors: OutputTensorDescriptor[] = [
    {
      name: tensor.name,
      shape: [outputDims, packedColumnCount(inputDims, params.bits)],
      dtype: "uint32",
    },
    {
      name: quantizedTensorKey(tensor.name, ".scales"),
      shape: [outputDims, groupCount],
      dtype: "float32",
    },
  ];

  if (params.mode === "affine") {
    descriptors.push({
      name: quantizedTensorKey(tensor.name, ".biases"),
      shape: [outputDims, groupCount],
      dtype: "float32",
    });
  }

  return descriptors;
}

async function loadShardTensorByName(shardPath: string, name: string) {
  for await (const entry of iterateSafetensors(shardPath, {
    include: (candidate) => candidate === name,
  })) {
    return entry.tensor;
  }

  throw new Error(`quantizePretrainedSnapshot: tensor "${name}" was not found in "${shardPath}".`);
}

function quantizedTensorEntries(
  shardPath: string,
  tensor: SafetensorTensorInfo,
  params: QuantizationParameters,
): SafetensorWriteEntry[] {
  const descriptors = quantizedOutputDescriptors(tensor, params);
  let cachedBytes: Record<string, Uint8Array> | null = null;
  let remainingEntries = descriptors.length;

  async function ensureBytes(): Promise<Record<string, Uint8Array>> {
    if (cachedBytes !== null) {
      return cachedBytes;
    }

    using sourceTensor = await loadShardTensorByName(shardPath, tensor.name);
    const quantizedTensor = quantize(sourceTensor, params);
    try {
      cachedBytes = {
        [tensor.name]: tensorBytes(quantizedTensor.weight, "uint32"),
        [quantizedTensorKey(tensor.name, ".scales")]: tensorBytes(
          quantizedTensor.scales,
          "float32",
        ),
        ...(quantizedTensor.biases === undefined
          ? {}
          : {
              [quantizedTensorKey(tensor.name, ".biases")]: tensorBytes(
                quantizedTensor.biases,
                "float32",
              ),
            }),
      };
    } finally {
      quantizedTensor.weight.free();
      quantizedTensor.scales.free();
      quantizedTensor.biases?.free();
    }

    return cachedBytes;
  }

  return descriptors.map((descriptor) => ({
    ...descriptor,
    chunks: () =>
      (async function* (): AsyncGenerator<Uint8Array, void, void> {
        const bytes = (await ensureBytes())[descriptor.name];
        if (bytes === undefined) {
          throw new Error(
            `quantizePretrainedSnapshot: missing quantized bytes for "${descriptor.name}".`,
          );
        }
        yield bytes;
        remainingEntries -= 1;
        if (remainingEntries === 0) {
          cachedBytes = null;
        }
      })(),
  }));
}

async function quantizeShard(
  shardPath: string,
  relativeShardPath: string,
  outputDir: string,
  registration: FamilyRegistration,
  config: BaseModelConfig,
  params: QuantizationParameters,
  options: Pick<QuantizePretrainedSnapshotOptions, "quantizeLmHead">,
  shardWeightMap: Record<string, string>,
): Promise<QuantizedShardResult> {
  const shardInspection = await inspectSafetensors(shardPath);
  const outputEntries: SafetensorWriteEntry[] = [];
  let quantizedTensorCount = 0;
  let copiedTensorCount = 0;

  for (const tensor of shardInspection.tensors) {
    const mappedPath = registration.sanitizeWeight(config, tensor.name);
    if (!isQuantizableMappedWeight(mappedPath, tensor, params, options)) {
      outputEntries.push(copyTensorEntry(shardPath, tensor));
      shardWeightMap[tensor.name] = relativeShardPath;
      copiedTensorCount += 1;
      continue;
    }

    for (const entry of quantizedTensorEntries(shardPath, tensor, params)) {
      outputEntries.push(entry);
      shardWeightMap[entry.name] = relativeShardPath;
    }
    quantizedTensorCount += 1;
  }

  const outputShardPath = join(outputDir, relativeShardPath);
  await saveSafetensorsStream(outputEntries, outputShardPath, shardInspection.metadata);
  return {
    quantizedTensorCount,
    copiedTensorCount,
    outputBytes: fileSize(outputShardPath),
  };
}

function resolvedIndexRelativePath(
  inspection: ReturnType<typeof inspectSnapshot>,
  shardPaths: readonly string[],
): string | undefined {
  return (
    outputIndexRelativePath(inspection) ??
    (shardPaths.length > 1 ? "model.safetensors.index.json" : undefined)
  );
}

/** Quantize a pretrained snapshot shard by shard into an MLX-native 4/8-bit snapshot. */
export async function quantizePretrainedSnapshot(
  source: string,
  options: QuantizePretrainedSnapshotOptions,
): Promise<QuantizedSnapshotResult> {
  const outputDir = resolve(options.outputDir);
  ensureOutputDirectory(outputDir, options.overwrite ?? false);

  const snapshot = await resolvePretrainedSnapshot(source, options);
  const inspection = inspectSnapshot(snapshot);
  const { registration, config } = parseConfigForQuantization(inspection.config);
  const params = resolveQuantizationParameters(options);
  const relativePaths = relativeFileMap(snapshot.files);
  const shardWeightMap: Record<string, string> = {};
  const shardPaths = listSafetensorShardPaths(snapshot);
  const excluded = excludedSnapshotPaths(inspection, shardPaths);

  let quantizedTensorCount = 0;
  let copiedTensorCount = 0;
  let outputBytes = await copyNonModelFiles(snapshot.files, excluded, outputDir);

  for (const shardPath of shardPaths) {
    const relativeShardPath = relativePaths.get(resolve(shardPath));
    if (relativeShardPath === undefined) {
      throw new Error(
        `quantizePretrainedSnapshot: could not resolve relative path for shard "${shardPath}".`,
      );
    }

    const shardResult = await quantizeShard(
      shardPath,
      relativeShardPath,
      outputDir,
      registration,
      config,
      params,
      options,
      shardWeightMap,
    );
    quantizedTensorCount += shardResult.quantizedTensorCount;
    copiedTensorCount += shardResult.copiedTensorCount;
    outputBytes += shardResult.outputBytes;
  }

  const outputConfigPath = join(outputDir, "config.json");
  outputBytes += await writeJson(
    outputConfigPath,
    rewrittenConfigRecord(inspection.config, params),
  );

  const indexRelativePath = resolvedIndexRelativePath(inspection, shardPaths);
  if (indexRelativePath !== undefined) {
    const existingIndex = inspection.safetensorsIndex;
    outputBytes += await writeJson(join(outputDir, indexRelativePath), {
      ...existingIndex,
      weight_map: shardWeightMap,
    });
  }

  return {
    outputDir,
    shardCount: shardPaths.length,
    quantizedTensorCount,
    copiedTensorCount,
    inputBytes: snapshot.totalBytes,
    outputBytes,
    outputConfigPath,
    ...(indexRelativePath === undefined
      ? {}
      : { outputIndexPath: join(outputDir, indexRelativePath) }),
  };
}
