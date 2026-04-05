/**
 * Pretrained model and tokenizer loading entry points.
 * @module
 */

import { mxEval, treeFlatten } from "@mlxts/core";
import { inspectSnapshot, iterateSafetensorWeights, resolveSnapshot } from "@mlxts/hub";
import { loadTokenizer, type Tokenizer, type TokenizerFileSet } from "@mlxts/tokenizers";

import { assignWeightPath, listParameterPaths } from "./infrastructure/weight-assignment";
import { resolveFamily } from "./registry";
import {
  type CausalLM,
  ConfigParseError,
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

/** Load a pretrained decoder model from a local directory or Hugging Face repo. */
export async function loadCausalLM(
  source: string,
  options: LoadCausalLMOptions = {},
): Promise<CausalLM> {
  const snapshot = await resolveSnapshot(source, options);
  const inspection = inspectSnapshot(snapshot);
  const modelType = expectModelType(inspection.config);
  const registration = resolveFamily(modelType);
  const config = registration.parseConfig(inspection.config);
  const model = registration.createModel(config);
  const assignedPaths = new Set<string>();
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const unexpectedWeights: string[] = [];
  const sanitizedPaths = new Map<string, string>();
  const exceptionalWeights = new Set(registration.exceptionalWeightNames?.(config) ?? []);

  try {
    for await (const { name, tensor } of iterateSafetensorWeights(snapshot, {
      include: (name) => {
        if (exceptionalWeights.has(name)) {
          return false;
        }

        const path = registration.sanitizeWeight(config, name);
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

    return model;
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
  const snapshot = await resolveSnapshot(source, snapshotOptions);
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

  return loadTokenizer(
    tokenizerFileSet(snapshot.directory, inspection),
    format === undefined ? {} : { format },
  );
}
