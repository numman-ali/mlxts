import { argmax, array, logsumexp, MxArray, reshape, subtract } from "@mlxts/core";

import { prefillPromptCache, takeLastLogits } from "../src/infrastructure/generation/helpers";
import type { loadCausalLM } from "../src/load";
import type { CausalLM, TransformerCache } from "../src/types";

export type BenchmarkModel = Awaited<ReturnType<typeof loadCausalLM>>;

export type GreedyStepResult = {
  token: MxArray;
  logprobs: MxArray;
};

function inputTensor(tokenIds: readonly number[] | MxArray): MxArray {
  return tokenIds instanceof MxArray ? tokenIds : array([tokenIds], "int32");
}

/** Chunk the prompt into the cache and return the remaining tail for first-token decode. */
export function prefillBenchmarkCache(
  model: CausalLM,
  promptTokenIds: readonly number[],
  cache: TransformerCache,
  prefillStepSize: number,
): number[] {
  return prefillPromptCache(model, promptTokenIds, cache, prefillStepSize).tokenIds;
}

/** Greedy next-token prediction for the synthetic throughput benchmark. */
export function predictGreedyToken(
  model: CausalLM,
  tokenIds: readonly number[] | MxArray,
  cache: TransformerCache,
): MxArray {
  const ids = inputTensor(tokenIds);
  const ownsInput = !(tokenIds instanceof MxArray);

  try {
    using logits = model.forward(ids, { cache });
    using lastLogits = takeLastLogits(logits, "benchmark-generation");
    using greedy = argmax(lastLogits, -1);
    return reshape(greedy, [1, 1]);
  } finally {
    if (ownsInput) {
      ids.free();
    }
  }
}

/** Greedy next-token prediction plus logprobs for MLX-LM-style parity timing. */
export function predictGreedyStep(
  model: CausalLM,
  tokenIds: readonly number[] | MxArray,
  cache: TransformerCache,
): GreedyStepResult {
  const ids = inputTensor(tokenIds);
  const ownsInput = !(tokenIds instanceof MxArray);

  try {
    using logits = model.forward(ids, { cache });
    using lastLogits = takeLastLogits(logits, "benchmark-generation-parity");
    using greedy = argmax(lastLogits, -1);
    using normalizer = logsumexp(lastLogits, -1, true);
    return {
      token: reshape(greedy, [1, 1]),
      logprobs: subtract(lastLogits, normalizer),
    };
  } finally {
    if (ownsInput) {
      ids.free();
    }
  }
}
