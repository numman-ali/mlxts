import {
  argmax,
  array,
  clearMemoryCache,
  concatenate,
  createStream,
  getRecommendedWorkingSetBytes,
  type MxArray,
  mxEval,
  reshape,
  setWiredLimitBytes,
  slice,
  synchronize,
  takeAxis,
  withDefaultStream,
} from "@mlxts/core";

import type { BatchGenerationOptions, CausalLM, GenerationResult } from "../../types";
import { BatchKVCache } from "../cache";
import { GenerationAbortError, throwIfGenerationAborted } from "./cancellation";
import { resolveGenerationOptions } from "./defaults";
import { takeLastLogits } from "./helpers";
import { finishIfEos } from "./runtime";

type RunExclusive = <T>(work: () => Promise<T>) => Promise<T>;

export type ContinuousBatchTokenSchedulerOptions = Omit<
  BatchGenerationOptions,
  "maxTokens" | "abortSignal"
> & {
  /** Maximum active rows admitted into one decode step. */
  maxBatchSize?: number;
  /** Delay before starting a fresh scheduler loop so nearby requests can join. */
  batchWindowMs?: number;
  /** Optional model-lane guard shared with fallback generation. */
  runExclusive?: RunExclusive;
  /** Emitted whenever the scheduler admits rows into the active decode batch. */
  onBatch?: (event: ContinuousBatchEvent) => void;
};

export type ContinuousBatchTokenRequest = {
  id?: string;
  promptTokenIds: readonly number[];
  maxTokens: number;
  abortSignal?: AbortSignal;
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void;
};

export type ContinuousBatchEvent = {
  ids: readonly string[];
  batchSize: number;
  maxTokens: number;
  maxTokensByRequest: readonly number[];
};

type ScheduledRequest = {
  id: string;
  promptTokenIds: readonly number[];
  maxTokens: number;
  generated: number[];
  finishReason: GenerationResult["finishReason"];
  abortSignal?: AbortSignal;
  onAbort?: () => void;
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void;
  resolve(result: GenerationResult): void;
  reject(error: unknown): void;
};

function integerAtLeast(value: number, name: string, minimum: number): number {
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`ContinuousBatchTokenScheduler: ${name} must be >= ${minimum}.`);
  }
  return value;
}

function validateOptions(options: BatchGenerationOptions): void {
  if (options.cache !== undefined) {
    throw new Error("ContinuousBatchTokenScheduler: external caches are not supported.");
  }
  if (options.useCache === false) {
    throw new Error("ContinuousBatchTokenScheduler: cache-enabled decoding is required.");
  }
  if ((options.temperature ?? 1.0) > 0) {
    throw new Error("ContinuousBatchTokenScheduler: continuous batching requires temperature: 0.");
  }
  if ((options.repetitionPenalty ?? 1.0) !== 1.0) {
    throw new Error("ContinuousBatchTokenScheduler: repetition penalty is not supported yet.");
  }
}

function leftPadPrompts(
  prompts: readonly (readonly number[])[],
  padTokenId: number,
): { padded: number[][]; leftPadding: number[] } {
  const maxLength = Math.max(...prompts.map((prompt) => prompt.length));
  const padded: number[][] = [];
  const leftPadding: number[] = [];

  for (const prompt of prompts) {
    const padding = maxLength - prompt.length;
    leftPadding.push(padding);
    padded.push([...Array<number>(padding).fill(padTokenId), ...prompt]);
  }

  return { padded, leftPadding };
}

function sampleGreedyBatchTokenTensor(logits: MxArray, context: string): MxArray {
  const batchSize = logits.shape[0];
  if (batchSize === undefined || logits.shape.length !== 2) {
    throw new Error(
      `${context}: expected rank-2 logits for batch sampling, got [${logits.shape.join(", ")}].`,
    );
  }
  using greedy = argmax(logits, -1);
  return reshape(greedy, [batchSize, 1]);
}

function sampleNextBatchToken(model: CausalLM, inputIds: MxArray, cache: BatchKVCache): MxArray {
  using logits = model.forward(inputIds, { cache });
  using lastLogits = takeLastLogits(logits, "ContinuousBatchTokenScheduler");
  const nextToken = sampleGreedyBatchTokenTensor(lastLogits, "ContinuousBatchTokenScheduler");
  mxEval(nextToken);
  return nextToken;
}

function tokenTensorToIds(tokens: MxArray): number[] {
  const [batchSize, width] = tokens.shape;
  if (batchSize === undefined || width !== 1 || tokens.shape.length !== 2) {
    throw new Error(
      `ContinuousBatchTokenScheduler: expected token tensor shape [batch, 1], got [${tokens.shape.join(", ")}].`,
    );
  }

  const tokenIds: number[] = [];
  for (let index = 0; index < batchSize; index += 1) {
    using token = slice(tokens, [index, 0], [index + 1, 1]);
    tokenIds.push(token.item());
  }
  return tokenIds;
}

function tokenRows(tokens: MxArray, indices: readonly number[]): MxArray {
  using indexTensor = array([...indices], "int32");
  return takeAxis(tokens, indexTensor, 0);
}

function nextInputTensor(tokenIds: readonly number[], keepPositions: readonly number[]): MxArray {
  return array(
    keepPositions.map((position) => [tokenIds[position] ?? 0]),
    "int32",
  );
}

function combineCurrentTokens(left: MxArray | null, right: MxArray): MxArray {
  if (left === null) {
    return tokenRows(
      right,
      Array.from({ length: right.shape[0] ?? 0 }, (_, index) => index),
    );
  }
  using retainedLeft = tokenRows(
    left,
    Array.from({ length: left.shape[0] ?? 0 }, (_, index) => index),
  );
  using retainedRight = tokenRows(
    right,
    Array.from({ length: right.shape[0] ?? 0 }, (_, index) => index),
  );
  const combined = concatenate([retainedLeft, retainedRight], 0);
  mxEval(combined);
  return combined;
}

function yieldToScheduler(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

const defaultRunExclusive: RunExclusive = (work) => work();

/** Scheduler-owned continuous greedy decode loop for full-cache batchable models. */
export class ContinuousBatchTokenScheduler {
  readonly #model: CausalLM;
  readonly #maxBatchSize: number;
  readonly #batchWindowMs: number;
  readonly #runExclusive: RunExclusive;
  readonly #onBatch: ((event: ContinuousBatchEvent) => void) | undefined;
  readonly #eosTokenIds: ReadonlySet<number>;
  readonly #padTokenId: number;
  readonly #waiting: ScheduledRequest[] = [];
  #active: ScheduledRequest[] = [];
  #cache: BatchKVCache | null = null;
  #currentToken: MxArray | null = null;
  #running = false;
  #startScheduled = false;
  #nextId = 0;

  constructor(model: CausalLM, options: ContinuousBatchTokenSchedulerOptions) {
    const resolved = resolveGenerationOptions(model, undefined, {
      ...options,
      maxTokens: 1,
    });
    validateOptions(resolved);
    this.#model = model;
    this.#maxBatchSize = integerAtLeast(options.maxBatchSize ?? 8, "maxBatchSize", 1);
    this.#batchWindowMs = integerAtLeast(options.batchWindowMs ?? 0, "batchWindowMs", 0);
    this.#runExclusive = options.runExclusive ?? defaultRunExclusive;
    this.#onBatch = options.onBatch;
    this.#eosTokenIds = new Set(resolved.eosTokenIds ?? []);
    this.#padTokenId = options.padTokenId ?? 0;
  }

  enqueue(request: ContinuousBatchTokenRequest): Promise<GenerationResult> {
    if (request.promptTokenIds.length === 0) {
      throw new Error(
        "ContinuousBatchTokenScheduler: promptTokenIds must contain at least one token.",
      );
    }
    if (!Number.isInteger(request.maxTokens) || request.maxTokens < 0) {
      throw new Error(
        `ContinuousBatchTokenScheduler: maxTokens must be >= 0, got ${request.maxTokens}.`,
      );
    }
    throwIfGenerationAborted(request.abortSignal, "ContinuousBatchTokenScheduler");
    if (request.maxTokens === 0) {
      return Promise.resolve({ tokenIds: [], finishReason: "length" });
    }

    return new Promise((resolve, reject) => {
      const entry: ScheduledRequest = {
        id: request.id ?? `continuous-${this.#nextId}`,
        promptTokenIds: request.promptTokenIds,
        maxTokens: request.maxTokens,
        generated: [],
        finishReason: "length",
        ...(request.abortSignal === undefined ? {} : { abortSignal: request.abortSignal }),
        ...(request.onToken === undefined ? {} : { onToken: request.onToken }),
        resolve,
        reject,
      };
      this.#nextId += 1;
      this.#attachAbort(entry);
      this.#waiting.push(entry);
      this.#schedule();
    });
  }

  #attachAbort(entry: ScheduledRequest): void {
    if (entry.abortSignal === undefined) {
      return;
    }
    entry.onAbort = () => {
      const index = this.#waiting.indexOf(entry);
      if (index >= 0) {
        this.#waiting.splice(index, 1);
        this.#cleanup(entry);
        entry.reject(
          new GenerationAbortError("ContinuousBatchTokenScheduler: generation was cancelled."),
        );
      }
    };
    entry.abortSignal.addEventListener("abort", entry.onAbort, { once: true });
  }

  #cleanup(entry: ScheduledRequest): void {
    if (entry.onAbort !== undefined) {
      entry.abortSignal?.removeEventListener("abort", entry.onAbort);
      delete entry.onAbort;
    }
  }

  #schedule(): void {
    if (this.#running || this.#startScheduled) {
      return;
    }
    this.#startScheduled = true;
    if (this.#batchWindowMs === 0) {
      queueMicrotask(() => this.#start());
      return;
    }
    setTimeout(() => this.#start(), this.#batchWindowMs);
  }

  #start(): void {
    this.#startScheduled = false;
    if (this.#running || this.#waiting.length === 0) {
      return;
    }
    this.#running = true;
    void this.#runExclusive(() => this.#runLoop())
      .catch((error) => this.#failAll(error))
      .finally(() => {
        this.#running = false;
        if (this.#waiting.length > 0) {
          this.#schedule();
        }
      });
  }

  async #runLoop(): Promise<void> {
    using generationStream = createStream();
    const previousWiredLimit = setWiredLimitBytes(getRecommendedWorkingSetBytes());
    try {
      while (this.#waiting.length > 0 || this.#active.length > 0) {
        withDefaultStream(generationStream, () => {
          this.#dropAbortedActiveRows();
          this.#admitWaitingRows();
          this.#dropAbortedActiveRows();
          this.#decodeOneStep();
        });
        await yieldToScheduler();
      }
    } finally {
      this.#currentToken?.free();
      this.#currentToken = null;
      this.#cache?.[Symbol.dispose]();
      this.#cache = null;
      synchronize(generationStream);
      clearMemoryCache();
      setWiredLimitBytes(previousWiredLimit);
    }
  }

  #dropAbortedActiveRows(): void {
    if (this.#active.length === 0 || this.#currentToken === null || this.#cache === null) {
      return;
    }
    const keepPositions: number[] = [];
    for (let index = 0; index < this.#active.length; index += 1) {
      const request = this.#active[index];
      if (request === undefined) {
        continue;
      }
      if (request.abortSignal?.aborted) {
        this.#cleanup(request);
        request.reject(
          new GenerationAbortError("ContinuousBatchTokenScheduler: generation was cancelled."),
        );
        continue;
      }
      keepPositions.push(index);
    }
    if (keepPositions.length === this.#active.length) {
      return;
    }
    if (keepPositions.length === 0) {
      this.#active = [];
      this.#currentToken.free();
      this.#currentToken = null;
      this.#cache[Symbol.dispose]();
      this.#cache = null;
      return;
    }

    const nextActive = keepPositions
      .map((index) => this.#active[index])
      .filter((entry) => entry !== undefined);
    this.#cache.filter(keepPositions);
    const nextToken = tokenRows(this.#currentToken, keepPositions);
    mxEval(nextToken);
    this.#currentToken.free();
    this.#currentToken = nextToken;
    this.#active = nextActive;
  }

  #admitWaitingRows(): void {
    const capacity = this.#maxBatchSize - this.#active.length;
    if (capacity <= 0 || this.#waiting.length === 0) {
      return;
    }

    const admitted = this.#waiting.splice(0, capacity);
    this.#active.push(...admitted);
    const { padded, leftPadding } = leftPadPrompts(
      admitted.map((request) => request.promptTokenIds),
      this.#padTokenId,
    );
    const nextCache = new BatchKVCache(this.#model.layerCount, leftPadding);
    using promptInput = array(padded, "int32");
    const nextToken = sampleNextBatchToken(this.#model, promptInput, nextCache);

    if (this.#cache === null) {
      this.#cache = nextCache;
      this.#currentToken = nextToken;
    } else {
      this.#cache.extend(nextCache);
      nextCache[Symbol.dispose]();
      const combined = combineCurrentTokens(this.#currentToken, nextToken);
      this.#currentToken?.free();
      this.#currentToken = combined;
      nextToken.free();
    }

    this.#emitBatch();
  }

  #emitBatch(): void {
    if (this.#active.length === 0) {
      return;
    }
    const maxTokensByRequest = this.#active.map((request) => request.maxTokens);
    this.#onBatch?.({
      ids: this.#active.map((request) => request.id),
      batchSize: this.#active.length,
      maxTokens: Math.max(...maxTokensByRequest),
      maxTokensByRequest,
    });
  }

  #decodeOneStep(): void {
    if (this.#currentToken === null || this.#cache === null || this.#active.length === 0) {
      return;
    }

    const emittedToken = this.#currentToken;
    this.#currentToken = null;
    const tokenIds = tokenTensorToIds(emittedToken);
    const keepPositions: number[] = [];
    const nextActive: ScheduledRequest[] = [];

    for (let position = 0; position < this.#active.length; position += 1) {
      const request = this.#active[position];
      const tokenId = tokenIds[position];
      if (request === undefined || tokenId === undefined) {
        continue;
      }

      request.generated.push(tokenId);
      request.onToken?.(tokenId, request.generated);
      const eosFinishReason = finishIfEos(this.#eosTokenIds, tokenId);
      if (eosFinishReason !== null) {
        request.finishReason = eosFinishReason;
        this.#finishRequest(request);
        continue;
      }
      if (request.generated.length >= request.maxTokens) {
        this.#finishRequest(request);
        continue;
      }

      keepPositions.push(position);
      nextActive.push(request);
    }

    if (keepPositions.length === 0) {
      emittedToken.free();
      this.#active = [];
      this.#cache[Symbol.dispose]();
      this.#cache = null;
      this.#admitWaitingRows();
      return;
    }

    if (keepPositions.length !== this.#active.length) {
      this.#cache.filter(keepPositions);
    }
    this.#active = nextActive;
    using nextInput = nextInputTensor(tokenIds, keepPositions);
    const nextToken = sampleNextBatchToken(this.#model, nextInput, this.#cache);
    emittedToken.free();
    this.#currentToken = nextToken;
    this.#admitWaitingRows();
  }

  #finishRequest(request: ScheduledRequest): void {
    this.#cleanup(request);
    request.resolve({
      tokenIds: [...request.generated],
      finishReason: request.finishReason,
    });
  }

  #failAll(error: unknown): void {
    for (const request of [...this.#waiting, ...this.#active]) {
      this.#cleanup(request);
      request.reject(error);
    }
    this.#waiting.length = 0;
    this.#active = [];
    this.#currentToken?.free();
    this.#currentToken = null;
    this.#cache?.[Symbol.dispose]();
    this.#cache = null;
  }
}
