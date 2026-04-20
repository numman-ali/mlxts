/**
 * Sampling helpers over model logits.
 * @module
 */

import { argmax, array, concatenate, MxArray, random, reshape, retainArray } from "@mlxts/core";

import type { SamplerOptions } from "../../types";
import {
  applySamplingRepetitionPenalty,
  buildFilteredSamplingLogprobs,
  normalizeSamplingLogits,
  sampleCategoricalToken,
  usesProbabilityFiltering,
} from "./runtime";

function sampleGreedyToken(logits: MxArray): MxArray {
  using greedy = argmax(logits, -1);
  return reshape(greedy, [1, 1]);
}

/** Stateful GPU-side sampler data for generation loops. */
export class SamplerState implements Disposable {
  #rngKey: MxArray | null;
  #historyIds: MxArray | null;
  #tracksHistory: boolean;

  constructor(promptTokenIds: readonly number[], options: SamplerOptions) {
    this.#rngKey = options.seed === undefined ? null : random.key(options.seed);
    this.#tracksHistory = (options.repetitionPenalty ?? 1) > 1;
    this.#historyIds =
      this.#tracksHistory && promptTokenIds.length > 0 ? array([promptTokenIds], "int32") : null;
  }

  sampleTokenTensor(logits: MxArray, options: SamplerOptions = {}): MxArray {
    const temperature = options.temperature ?? 1.0;
    using repetitionAdjusted = applySamplingRepetitionPenalty(
      logits,
      this.#historyIds,
      options.repetitionPenalty,
    );

    if (temperature <= 0) {
      return sampleGreedyToken(repetitionAdjusted);
    }

    const rngKey = this.takeRngKey();

    try {
      if (!usesProbabilityFiltering(options)) {
        using logprobs = normalizeSamplingLogits(repetitionAdjusted);
        return sampleCategoricalToken(logprobs, temperature, rngKey);
      }

      using logprobs = buildFilteredSamplingLogprobs(logits, this.#historyIds, options);
      return sampleCategoricalToken(logprobs, temperature, rngKey);
    } finally {
      rngKey?.free();
    }
  }

  appendToken(tokenId: number | MxArray): void {
    if (!this.#tracksHistory) {
      return;
    }

    const tokenTensor =
      tokenId instanceof MxArray ? retainArray(tokenId) : array([[tokenId]], "int32");

    try {
      if (this.#historyIds === null) {
        this.#historyIds = tokenTensor;
        return;
      }

      const nextHistory = concatenate([this.#historyIds, tokenTensor], 1);
      this.#historyIds.free();
      this.#historyIds = nextHistory;
    } finally {
      if (this.#historyIds !== tokenTensor) {
        tokenTensor.free();
      }
    }
  }

  [Symbol.dispose](): void {
    this.#rngKey?.free();
    this.#historyIds?.free();
    this.#rngKey = null;
    this.#historyIds = null;
  }

  private takeRngKey(): MxArray | null {
    if (this.#rngKey === null) {
      return null;
    }

    const [nextState, sampleKey] = random.split(this.#rngKey);
    this.#rngKey.free();
    this.#rngKey = nextState;
    return sampleKey;
  }
}
