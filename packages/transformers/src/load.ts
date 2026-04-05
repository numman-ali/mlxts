/**
 * Pretrained model and tokenizer loading entry points.
 * @module
 */

import { mxEval, treeFlatten } from "@mlxts/core";
import { Module } from "@mlxts/nn";
import {
  resolveCheckpointQuantizationPlan,
  setupQuantizedModule,
  translateCheckpointQuantizationPlanPaths,
} from "@mlxts/quantize";
import { loadTokenizer, type Tokenizer, type TokenizerFileSet } from "@mlxts/tokenizers";
import { parseGenerationDefaults } from "./infrastructure/generation-defaults";
import { assignWeightPath, listParameterPaths } from "./infrastructure/weight-assignment";
import {
  inspectSnapshot,
  resolvePretrainedSnapshot,
  resolvePretrainedSource,
} from "./pretrained/snapshot";
import { iterateSafetensorWeights, listSafetensorShardPaths } from "./pretrained/weights";
import { resolveFamily } from "./registry";
import {
  type BaseModelConfig,
  type CausalLM,
  ConfigParseError,
  type FamilyRegistration,
  type LoadCausalLMOptions,
  type LoadPretrainedTokenizerOptions,
  MissingWeightsError,
} from "./types";

function sortedUnique(values: readonly string[]): string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}

function expectModelType(config: Record<string, unknown>): string {
  const modelType = config.model_type;
  if (typeof modelType !== "string" || modelType === "") {
    throw new ConfigParseError("loadCausalLM: config.model_type must be a non-empty string.");
  }
  return modelType;
}

function withQuantizedLeafSuffix(path: string, suffix: ".scales" | ".biases"): string | null {
  if (!path.endsWith(".weight")) {
    return null;
  }
  return `${path.slice(0, -".weight".length)}${suffix}`;
}

function sanitizeCheckpointWeightPath<Config extends { family: string }>(
  sanitizeWeight: (config: Config, checkpointName: string) => string | null,
  config: Config,
  checkpointName: string,
): string | null {
  const directPath = sanitizeWeight(config, checkpointName);
  if (directPath !== null) {
    return directPath;
  }

  for (const suffix of [".scales", ".biases"] as const) {
    if (!checkpointName.endsWith(suffix)) {
      continue;
    }

    const weightName = `${checkpointName.slice(0, -suffix.length)}.weight`;
    const weightPath = sanitizeWeight(config, weightName);
    if (weightPath === null) {
      return null;
    }
    return withQuantizedLeafSuffix(weightPath, suffix);
  }

  return null;
}

function translateQuantizedRulePath<Config extends { family: string }>(
  sanitizeWeight: (config: Config, checkpointName: string) => string | null,
  config: Config,
  checkpointPath: string,
): string | null {
  const weightPath = sanitizeCheckpointWeightPath(
    sanitizeWeight,
    config,
    `${checkpointPath}.weight`,
  );
  if (weightPath === null || !weightPath.endsWith(".weight")) {
    return null;
  }
  return weightPath.slice(0, -".weight".length);
}

function tokenizerFileSet(
  snapshotDirectory: string,
  inspection: ReturnType<typeof inspectSnapshot>,
): TokenizerFileSet {
  const fileSet: TokenizerFileSet = {
    directory: snapshotDirectory,
  };
  if (inspection.tokenizer.tokenizerJsonPath !== undefined) {
    fileSet.tokenizerJsonPath = inspection.tokenizer.tokenizerJsonPath;
  }
  if (inspection.tokenizer.tekkenJsonPath !== undefined) {
    fileSet.tekkenJsonPath = inspection.tokenizer.tekkenJsonPath;
  }
  if (inspection.tokenizer.tokenizerModelPath !== undefined) {
    fileSet.tokenizerModelPath = inspection.tokenizer.tokenizerModelPath;
  }
  if (inspection.tokenizer.tokenizerConfigPath !== undefined) {
    fileSet.tokenizerConfigPath = inspection.tokenizer.tokenizerConfigPath;
  }
  if (inspection.tokenizer.specialTokensMapPath !== undefined) {
    fileSet.specialTokensMapPath = inspection.tokenizer.specialTokensMapPath;
  }
  fileSet.tokenizerConfigData = {
    ...inspection.generationConfig,
    ...inspection.tokenizerConfig,
  };
  fileSet.specialTokensMapData = inspection.specialTokensMap;
  return fileSet;
}

type PreparedModel = {
  registration: FamilyRegistration;
  config: BaseModelConfig;
  model: CausalLM & Module;
};

function ensureModuleModel(model: CausalLM, family: string): CausalLM & Module {
  if (!(model instanceof Module)) {
    throw new Error(`loadCausalLM: family "${family}" did not return an nn.Module.`);
  }
  return model;
}

function prepareModel(configRecord: Record<string, unknown>): PreparedModel {
  const modelType = expectModelType(configRecord);
  const registration = resolveFamily(modelType);
  const config = registration.parseConfig(configRecord);
  const model = ensureModuleModel(registration.createModel(config), registration.family);
  return { registration, config, model };
}

function prepareQuantizedCheckpointModel(
  model: Module,
  configRecord: Record<string, unknown>,
  registration: FamilyRegistration,
  config: BaseModelConfig,
): void {
  const checkpointQuantizationPlan = resolveCheckpointQuantizationPlan(configRecord);
  if (checkpointQuantizationPlan === null) {
    return;
  }

  const translatedPlan = translateCheckpointQuantizationPlanPaths(
    checkpointQuantizationPlan,
    (path) => translateQuantizedRulePath(registration.sanitizeWeight, config, path),
  );
  setupQuantizedModule(model, translatedPlan);
}

type WeightLoadState = {
  assignedPaths: Set<string>;
  unexpectedWeights: string[];
};

async function assignCheckpointWeights(
  snapshot: Awaited<ReturnType<typeof resolvePretrainedSnapshot>>,
  registration: FamilyRegistration,
  config: BaseModelConfig,
  model: CausalLM,
): Promise<WeightLoadState> {
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];
  const sanitizedPaths = new Map<string, string>();
  const exceptionalWeights = new Set(registration.exceptionalWeightNames?.(config) ?? []);

  for await (const { name, tensor } of iterateSafetensorWeights(snapshot, {
    include: (name) => {
      if (exceptionalWeights.has(name)) {
        return false;
      }

      const path = sanitizeCheckpointWeightPath(registration.sanitizeWeight, config, name);
      if (path === null) {
        if (!(registration.isIgnoredWeight?.(config, name) ?? false)) {
          unexpectedWeights.push(name);
        }
        return false;
      }

      sanitizedPaths.set(name, path);
      return true;
    },
  })) {
    const path = sanitizedPaths.get(name);
    if (path === undefined) {
      tensor.free();
      throw new Error(`loadCausalLM: missing sanitized path for checkpoint tensor "${name}".`);
    }

    try {
      assignWeightPath(model, path, tensor);
      assignedPaths.add(path);
    } catch (error) {
      tensor.free();
      throw error;
    }
  }

  if (exceptionalWeights.size > 0) {
    if (registration.loadExceptionalWeights === undefined) {
      throw new Error(
        `loadCausalLM: family "${registration.family}" declared exceptional checkpoint weights without a loader.`,
      );
    }

    await registration.loadExceptionalWeights({
      snapshot,
      config,
      model,
      assignWeight: (path, tensor) => {
        try {
          assignWeightPath(model, path, tensor);
          assignedPaths.add(path);
        } catch (error) {
          tensor.free();
          throw error;
        }
      },
    });
  }

  return {
    assignedPaths,
    unexpectedWeights,
  };
}

function finalizeLoadedModel(
  model: CausalLM,
  expectedPaths: ReadonlySet<string>,
  assignedPaths: ReadonlySet<string>,
  unexpectedWeights: readonly string[],
  options: LoadCausalLMOptions,
  shardCount: number,
): CausalLM {
  const missingPaths = [...expectedPaths].filter((path) => !assignedPaths.has(path));
  if (missingPaths.length > 0) {
    throw new MissingWeightsError(missingPaths);
  }

  model.eval();
  const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
  mxEval(...parameters);

  const finalUnexpectedWeights = sortedUnique(unexpectedWeights);
  if (finalUnexpectedWeights.length > 0) {
    const message = `loadCausalLM: checkpoint contained unexpected unmapped weights: ${finalUnexpectedWeights.join(", ")}.`;
    if (options.strictUnexpectedWeights === true) {
      throw new Error(message);
    }
    console.warn(message);
  }

  options.onProgress?.({
    stage: "model",
    status: "weights-complete",
    shardCount,
  });

  return model;
}

/** Load a pretrained decoder model from a local directory or Hugging Face repo. */
export async function loadCausalLM(
  source: string,
  options: LoadCausalLMOptions = {},
): Promise<CausalLM> {
  const snapshot = await resolvePretrainedSnapshot(source, options);
  const inspection = inspectSnapshot(snapshot);
  const { registration, config, model } = prepareModel(inspection.config);
  const generationDefaults = parseGenerationDefaults(inspection.generationConfig);
  if (generationDefaults !== undefined) {
    config.generationDefaults = generationDefaults;
  }
  prepareQuantizedCheckpointModel(model, inspection.config, registration, config);
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const shardCount = listSafetensorShardPaths(snapshot).length;
  options.onProgress?.({
    stage: "model",
    status: "weights-start",
    shardCount,
  });

  try {
    const { assignedPaths, unexpectedWeights } = await assignCheckpointWeights(
      snapshot,
      registration,
      config,
      model,
    );
    return finalizeLoadedModel(
      model,
      expectedPaths,
      assignedPaths,
      unexpectedWeights,
      options,
      shardCount,
    );
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}

/** Load the tokenizer associated with a pretrained model snapshot. */
export async function loadPretrainedTokenizer(
  source: string,
  options: LoadPretrainedTokenizerOptions = {},
): Promise<Tokenizer> {
  const { format, ...snapshotOptions } = options;
  const snapshot = await resolvePretrainedSnapshot(source, snapshotOptions);
  const inspection = inspectSnapshot(snapshot);
  if (
    inspection.tokenizer.tokenizerJsonPath === undefined &&
    inspection.tokenizer.tekkenJsonPath === undefined &&
    inspection.tokenizer.tokenizerModelPath === undefined
  ) {
    throw new Error(
      "loadPretrainedTokenizer: snapshot does not include tokenizer.json, tekken.json, or tokenizer.model.",
    );
  }

  options.onProgress?.({
    stage: "tokenizer",
    status: "start",
    directory: snapshot.directory,
    ...(format === undefined ? {} : { format }),
  });

  const tokenizer = loadTokenizer(
    tokenizerFileSet(snapshot.directory, inspection),
    format === undefined ? {} : { format },
  );
  options.onProgress?.({
    stage: "tokenizer",
    status: "complete",
    directory: snapshot.directory,
    ...(format === undefined ? {} : { format }),
  });
  return tokenizer;
}

/** Resolve a local directory or Hub repo id to a concrete local snapshot path. */
export { resolvePretrainedSource };
