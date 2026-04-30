/**
 * T5 encoder checkpoint weight mapping and loading.
 * @module
 */

import { mxEval, treeFlatten } from "@mlxts/core";

import { assignWeightPath, listParameterPaths } from "../../infrastructure/weight-assignment";
import type { ResolvedSnapshot } from "../../pretrained/types";
import { iterateSafetensorWeights, listSafetensorShardPaths } from "../../pretrained/weights";
import { MissingWeightsError } from "../../types";
import type { T5EncoderModel } from "./model";

export type T5EncoderWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading T5 encoder weights. */
export type T5EncoderWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

const WEIGHT_PATTERNS = [
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.q\.weight$/,
    target: "layers.$1.selfAttention.attention.q.weight",
  },
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.k\.weight$/,
    target: "layers.$1.selfAttention.attention.k.weight",
  },
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.v\.weight$/,
    target: "layers.$1.selfAttention.attention.v.weight",
  },
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.o\.weight$/,
    target: "layers.$1.selfAttention.attention.o.weight",
  },
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.relative_attention_bias\.weight$/,
    target: "layers.$1.selfAttention.attention.relativeAttentionBias.weight",
  },
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.0\.layer_norm\.weight$/,
    target: "layers.$1.selfAttention.layerNorm.weight",
  },
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.wi\.weight$/,
    target: "layers.$1.feedForward.dense.wi.weight",
  },
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.wi_0\.weight$/,
    target: "layers.$1.feedForward.dense.wi0.weight",
  },
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.wi_1\.weight$/,
    target: "layers.$1.feedForward.dense.wi1.weight",
  },
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.wo\.weight$/,
    target: "layers.$1.feedForward.dense.wo.weight",
  },
  {
    pattern: /^encoder\.block\.(\d+)\.layer\.1\.layer_norm\.weight$/,
    target: "layers.$1.feedForward.layerNorm.weight",
  },
  {
    pattern: /^encoder\.final_layer_norm\.weight$/,
    target: "finalLayerNorm.weight",
  },
] as const;

function replacePattern(name: string, pattern: RegExp, target: string): string | null {
  if (!pattern.test(name)) {
    return null;
  }
  return name.replace(pattern, target);
}

function sorted(values: Iterable<string>): string[] {
  return [...values].toSorted((left, right) => left.localeCompare(right));
}

function isIgnoredT5EncoderWeight(checkpointName: string): boolean {
  return checkpointName.startsWith("decoder.") || checkpointName === "lm_head.weight";
}

/** Map a T5 checkpoint tensor name onto the package-owned encoder parameter tree. */
export function t5EncoderWeightPath(checkpointName: string): string | null {
  if (checkpointName === "shared.weight" || checkpointName === "encoder.embed_tokens.weight") {
    return "tokenEmbedding.weight";
  }

  for (const rule of WEIGHT_PATTERNS) {
    const path = replacePattern(checkpointName, rule.pattern, rule.target);
    if (path !== null) {
      return path;
    }
  }
  return null;
}

function throwIfMissingWeights(
  expectedPaths: ReadonlySet<string>,
  assignedPaths: ReadonlySet<string>,
): void {
  const missingPaths = [...expectedPaths].filter((path) => !assignedPaths.has(path));
  if (missingPaths.length > 0) {
    throw new MissingWeightsError(missingPaths);
  }
}

function throwIfUnexpectedWeights(
  unexpectedWeights: readonly string[],
  options: T5EncoderWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadT5EncoderWeights: checkpoint contained unexpected unmapped weights: ${unexpectedWeights.join(", ")}.`,
  );
}

function evalAssignedParameters(model: T5EncoderModel): void {
  const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
  mxEval(...parameters);
}

/** Load T5 encoder weights into an existing model. */
export async function loadT5EncoderWeights(
  model: T5EncoderModel,
  snapshot: ResolvedSnapshot,
  options: T5EncoderWeightLoadOptions = {},
): Promise<T5EncoderWeightLoadResult> {
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];

  for await (const { name, tensor } of iterateSafetensorWeights(snapshot)) {
    const path = t5EncoderWeightPath(name);
    if (path === null || !expectedPaths.has(path)) {
      tensor.free();
      if (!isIgnoredT5EncoderWeight(name)) {
        unexpectedWeights.push(name);
      }
      continue;
    }

    try {
      assignWeightPath(model, path, tensor);
      assignedPaths.add(path);
    } catch (error) {
      tensor.free();
      throw error;
    }
  }

  const finalUnexpectedWeights = sorted(unexpectedWeights);
  throwIfMissingWeights(expectedPaths, assignedPaths);
  throwIfUnexpectedWeights(finalUnexpectedWeights, options);
  model.eval();
  evalAssignedParameters(model);

  return {
    assignedPaths: sorted(assignedPaths),
    unexpectedWeights: finalUnexpectedWeights,
    shardCount: listSafetensorShardPaths(snapshot).length,
  };
}
