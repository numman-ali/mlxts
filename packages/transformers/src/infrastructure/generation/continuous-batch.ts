import {
  array,
  clearMemoryCache,
  createStream,
  getRecommendedWorkingSetBytes,
  type MxArray,
  mxEval,
  setWiredLimitBytes,
  synchronize,
  withDefaultStream,
} from "@mlxts/core";

import type {
  BatchGenerationOptions,
  CausalLM,
  GenerationResult,
  PrefillProgressEvent,
} from "../../types";
import { BatchKVCache } from "../cache";
import { GenerationAbortError, throwIfGenerationAborted } from "./cancellation";
import {
  combineCurrentTokens,
  integerAtLeast,
  nextInputTensor,
  prefillPromptChunk,
  prefillReadyBatch,
  rejectAbortedPrefillingRows,
  sampleNextBatchToken,
  tokenRows,
  tokenTensorToIds,
  validateContinuousBatchOptions,
  yieldToScheduler,
} from "./continuous-batch-helpers";
import { resolveGenerationOptions } from "./defaults";
import { validatePrefillStepSize } from "./helpers";
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
  onPrefillProgress?: (event: PrefillProgressEvent) => void;
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
  onPrefillProgress?: (event: PrefillProgressEvent) => void;
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void;
  resolve(result: GenerationResult): void;
  reject(error: unknown): void;
};

type PrefillingRequest = {
  request: ScheduledRequest;
  cache: BatchKVCache;
  cursor: number;
};

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
  readonly #prefillStepSize: number;
  readonly #waiting: ScheduledRequest[] = [];
  readonly #prefilling: PrefillingRequest[] = [];
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
    validateContinuousBatchOptions(resolved);
    this.#model = model;
    this.#maxBatchSize = integerAtLeast(options.maxBatchSize ?? 8, "maxBatchSize", 1);
    this.#batchWindowMs = integerAtLeast(options.batchWindowMs ?? 0, "batchWindowMs", 0);
    this.#runExclusive = options.runExclusive ?? defaultRunExclusive;
    this.#onBatch = options.onBatch;
    this.#eosTokenIds = new Set(resolved.eosTokenIds ?? []);
    this.#padTokenId = options.padTokenId ?? 0;
    this.#prefillStepSize = options.prefillStepSize ?? 2048;
    validatePrefillStepSize(this.#prefillStepSize, "ContinuousBatchTokenScheduler");
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
        ...(request.onPrefillProgress === undefined
          ? {}
          : { onPrefillProgress: request.onPrefillProgress }),
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
      const prefillIndex = this.#prefilling.findIndex((prefilling) => prefilling.request === entry);
      if (prefillIndex >= 0) {
        const [prefilling] = this.#prefilling.splice(prefillIndex, 1);
        prefilling?.cache[Symbol.dispose]();
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
    if (this.#running || (this.#waiting.length === 0 && this.#prefilling.length === 0)) {
      return;
    }
    this.#running = true;
    void this.#runExclusive(() => this.#runLoop())
      .catch((error) => this.#failAll(error))
      .finally(() => {
        this.#running = false;
        if (this.#waiting.length > 0 || this.#prefilling.length > 0) {
          this.#schedule();
        }
      });
  }

  async #runLoop(): Promise<void> {
    using generationStream = createStream();
    const previousWiredLimit = setWiredLimitBytes(getRecommendedWorkingSetBytes());
    try {
      while (this.#waiting.length > 0 || this.#prefilling.length > 0 || this.#active.length > 0) {
        withDefaultStream(generationStream, () => {
          rejectAbortedPrefillingRows(this.#prefilling, (request) => this.#cleanup(request));
          this.#dropAbortedActiveRows();
          this.#admitWaitingRows();
          rejectAbortedPrefillingRows(this.#prefilling, (request) => this.#cleanup(request));
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
      for (const prefilling of this.#prefilling.splice(0)) {
        prefilling.cache[Symbol.dispose]();
      }
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
    const capacity = this.#maxBatchSize - this.#active.length - this.#prefilling.length;
    if (capacity <= 0 && this.#prefilling.length === 0) {
      return;
    }
    if (this.#active.length === 0 && this.#prefilling.length === 0) {
      this.#admitFullRows(Math.min(capacity, this.#waiting.length));
      return;
    }
    if (capacity > 0 && this.#prefilling.length === 0 && this.#waiting.length > 0) {
      this.#startPrefillingRow();
    }
    this.#advancePrefillingRow();
  }

  #admitFullRows(count: number): void {
    if (count <= 0) {
      return;
    }
    const admitted = this.#waiting.splice(0, count);
    this.#active.push(...admitted);
    const { cache: nextCache, nextToken } = prefillReadyBatch(
      this.#model,
      admitted.map((request) => request.promptTokenIds),
      this.#padTokenId,
    );

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

  #startPrefillingRow(): void {
    const request = this.#waiting.shift();
    if (request === undefined) {
      return;
    }
    this.#prefilling.push({
      request,
      cache: new BatchKVCache(this.#model.layerCount, [0]),
      cursor: 0,
    });
  }

  #advancePrefillingRow(): void {
    const prefilling = this.#prefilling[0];
    if (prefilling === undefined) {
      return;
    }
    const remainingPrefillTokens = prefilling.request.promptTokenIds.length - prefilling.cursor - 1;
    if (remainingPrefillTokens > 0) {
      this.#prefillOneChunk(prefilling, remainingPrefillTokens);
      return;
    }
    this.#finishPrefillingRow(prefilling);
  }

  #prefillOneChunk(prefilling: PrefillingRequest, remainingPrefillTokens: number): void {
    const progress = prefillPromptChunk(
      this.#model,
      prefilling.request.promptTokenIds,
      prefilling.cache,
      prefilling.cursor,
      this.#prefillStepSize,
      remainingPrefillTokens,
    );
    prefilling.cursor = progress.cursor;
    prefilling.request.onPrefillProgress?.(progress.event);
  }

  #finishPrefillingRow(prefilling: PrefillingRequest): void {
    const tail = prefilling.request.promptTokenIds.slice(prefilling.cursor);
    using tailInput = array([tail], "int32");
    const nextToken = sampleNextBatchToken(this.#model, tailInput, prefilling.cache);
    const [ready] = this.#prefilling.splice(0, 1);
    if (ready === undefined) {
      nextToken.free();
      return;
    }
    this.#mergeReadyRow(ready.request, ready.cache, nextToken);
  }

  #mergeReadyRow(request: ScheduledRequest, nextCache: BatchKVCache, nextToken: MxArray): void {
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

    this.#active.push(request);
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
  }

  #finishRequest(request: ScheduledRequest): void {
    this.#cleanup(request);
    request.resolve({
      tokenIds: [...request.generated],
      finishReason: request.finishReason,
    });
  }

  #failAll(error: unknown): void {
    for (const request of [
      ...this.#waiting,
      ...this.#prefilling.map((prefilling) => prefilling.request),
      ...this.#active,
    ]) {
      this.#cleanup(request);
      request.reject(error);
    }
    this.#waiting.length = 0;
    for (const prefilling of this.#prefilling.splice(0)) {
      prefilling.cache[Symbol.dispose]();
    }
    this.#active = [];
    this.#currentToken?.free();
    this.#currentToken = null;
    this.#cache?.[Symbol.dispose]();
    this.#cache = null;
  }
}
