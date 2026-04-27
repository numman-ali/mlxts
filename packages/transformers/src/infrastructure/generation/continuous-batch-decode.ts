import type { MxArray } from "@mlxts/core";

import type { CausalLM, SamplerOptions, TransformerBatchCache } from "../../types";
import {
  nextInputTensor,
  sampleNextBatchToken,
  tokenRows,
  tokenTensorToIds,
} from "./continuous-batch-helpers";
import { nextDecodeStepsSincePrefill } from "./continuous-batch-policy";
import type { ScheduledRequest } from "./continuous-batch-types";
import { finishIfEos } from "./runtime";

type DecodeStepOptions = {
  model: CausalLM;
  active: ScheduledRequest[];
  cache: TransformerBatchCache;
  currentToken: MxArray;
  samplerOptions: SamplerOptions;
  eosTokenIds: ReadonlySet<number>;
  hadPendingPrefillWork: boolean;
  decodeStepsSincePrefill: number;
  onFirstToken(request: ScheduledRequest): void;
  finishRequest(request: ScheduledRequest): void;
};

type DecodeStepResult = {
  active: ScheduledRequest[];
  cache: TransformerBatchCache | null;
  currentToken: MxArray | null;
  decodeStepsSincePrefill: number;
};

type ContinuingRows = {
  keepPositions: number[];
  active: ScheduledRequest[];
};

function canScheduleNextTokenBeforeSync(
  active: readonly ScheduledRequest[],
  eosTokenIds: ReadonlySet<number>,
  samplerOptions: SamplerOptions,
): boolean {
  if (eosTokenIds.size > 0 || (samplerOptions.temperature ?? 1.0) > 0) {
    return false;
  }
  if ((samplerOptions.repetitionPenalty ?? 1.0) > 1.0) {
    return false;
  }
  return active.every((request) => request.generated.length + 1 < request.maxTokens);
}

function scheduleNextTokenBeforeSync(
  options: Pick<DecodeStepOptions, "model" | "active" | "cache" | "samplerOptions">,
  emittedToken: MxArray,
): MxArray {
  using nextInput = nextInputTensor(
    emittedToken,
    Array.from({ length: options.active.length }, (_, index) => index),
  );
  return sampleNextBatchToken(
    options.model,
    nextInput,
    options.cache,
    options.active.map((request) => request.samplerState),
    options.samplerOptions,
  );
}

function continuingRowsAfterEmittedToken(
  options: DecodeStepOptions,
  tokenIds: readonly number[],
  hasScheduledNextToken: boolean,
): ContinuingRows {
  const keepPositions: number[] = [];
  const active: ScheduledRequest[] = [];

  for (let position = 0; position < options.active.length; position += 1) {
    const request = options.active[position];
    const tokenId = tokenIds[position];
    if (request === undefined || tokenId === undefined) {
      continue;
    }

    request.generated.push(tokenId);
    request.onToken?.(tokenId, request.generated);
    if (!request.firstTokenEmitted) {
      request.firstTokenEmitted = true;
      options.onFirstToken(request);
    }
    const eosFinishReason = finishIfEos(options.eosTokenIds, tokenId);
    if (eosFinishReason !== null) {
      request.finishReason = eosFinishReason;
      options.finishRequest(request);
      continue;
    }
    if (request.generated.length >= request.maxTokens) {
      options.finishRequest(request);
      continue;
    }

    if (!hasScheduledNextToken) {
      request.samplerState.appendToken(tokenId);
    }
    keepPositions.push(position);
    active.push(request);
  }

  return { keepPositions, active };
}

/** Decode one already-prefilled continuous batch token step. */
export function decodeContinuousBatchStep(options: DecodeStepOptions): DecodeStepResult {
  const emittedToken = options.currentToken;
  let ownsEmittedToken = true;
  let nextToken: MxArray | null = null;

  try {
    if (
      canScheduleNextTokenBeforeSync(options.active, options.eosTokenIds, options.samplerOptions)
    ) {
      nextToken = scheduleNextTokenBeforeSync(options, emittedToken);
    }

    const tokenIds = tokenTensorToIds(emittedToken);
    const hasScheduledNextToken = nextToken !== null;
    const continuing = continuingRowsAfterEmittedToken(options, tokenIds, hasScheduledNextToken);

    if (continuing.keepPositions.length === 0) {
      nextToken?.free();
      nextToken = null;
      emittedToken.free();
      ownsEmittedToken = false;
      options.cache[Symbol.dispose]();
      return {
        active: [],
        cache: null,
        currentToken: null,
        decodeStepsSincePrefill: options.decodeStepsSincePrefill,
      };
    }

    if (continuing.keepPositions.length !== options.active.length) {
      options.cache.filter(continuing.keepPositions);
      if (nextToken !== null) {
        const filteredToken = tokenRows(nextToken, continuing.keepPositions);
        nextToken.free();
        nextToken = filteredToken;
      }
    }

    if (nextToken === null) {
      using nextInput = nextInputTensor(emittedToken, continuing.keepPositions);
      nextToken = sampleNextBatchToken(
        options.model,
        nextInput,
        options.cache,
        continuing.active.map((request) => request.samplerState),
        options.samplerOptions,
      );
    }

    emittedToken.free();
    ownsEmittedToken = false;
    const currentToken = nextToken;
    nextToken = null;
    return {
      active: continuing.active,
      cache: options.cache,
      currentToken,
      decodeStepsSincePrefill: nextDecodeStepsSincePrefill(
        options.decodeStepsSincePrefill,
        options.hadPendingPrefillWork,
      ),
    };
  } catch (error) {
    nextToken?.free();
    if (ownsEmittedToken) {
      emittedToken.free();
    }
    throw error;
  }
}
