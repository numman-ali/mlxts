import {
  array,
  clearMemoryCache,
  createStream,
  getRecommendedWorkingSetBytes,
  type MxArray,
  setWiredLimitBytes,
  synchronize,
  withDefaultStream,
} from "@mlxts/core";

import type {
  CausalLM,
  GenerationResult,
  SamplerOptions,
  TransformerBatchCache,
} from "../../types";
import { createBatchCacheForModel } from "./batch-cache-factory";
import { GenerationAbortError, throwIfGenerationAborted } from "./cancellation";
import {
  continuousBatchQueueSnapshot,
  takeAdmittableWaitingRows,
} from "./continuous-batch-admission";
import type { ContinuousBatchQueueSnapshot } from "./continuous-batch-events";
import {
  combineCurrentTokens,
  createScheduledRequest,
  disposePrefillingCaches,
  dropAbortedActiveRows,
  integerAtLeast,
  nextInputTensor,
  prefillPromptChunk,
  prefillReadyBatch,
  rejectAbortedPrefillingRows,
  remainingPrefillTokens,
  sampleNextBatchToken,
  samplerOptionsFrom,
  shouldChunkInitialRows,
  tokenTensorToIds,
  validateContinuousBatchOptions,
  yieldToScheduler,
} from "./continuous-batch-helpers";
import { attachScheduledRequestAbort } from "./continuous-batch-lifecycle";
import { ContinuousBatchTelemetry } from "./continuous-batch-telemetry";
import {
  type ContinuousBatchAdmissionController,
  type ContinuousBatchTokenRequest,
  type ContinuousBatchTokenSchedulerOptions,
  defaultRunExclusive,
  type PrefillingRequest,
  type RunExclusive,
  type ScheduledRequest,
} from "./continuous-batch-types";
import { AdmissionReleaseWakeup } from "./continuous-batch-wakeup";
import { resolveGenerationOptions } from "./defaults";
import { validatePrefillStepSize } from "./helpers";
import { finishIfEos } from "./runtime";

/** Scheduler-owned continuous decode loop for batch-cache-capable models. */
export class ContinuousBatchTokenScheduler {
  readonly #model: CausalLM;
  readonly #maxBatchSize: number;
  readonly #batchWindowMs: number;
  readonly #runExclusive: RunExclusive;
  readonly #admissionController: ContinuousBatchAdmissionController | undefined;
  readonly #telemetry: ContinuousBatchTelemetry;
  readonly #eosTokenIds: ReadonlySet<number>;
  readonly #samplerOptions: SamplerOptions;
  readonly #padTokenId: number;
  readonly #prefillStepSize: number;
  readonly #waiting: ScheduledRequest[] = [];
  readonly #prefilling: PrefillingRequest[] = [];
  #active: ScheduledRequest[] = [];
  #cache: TransformerBatchCache | null = null;
  #currentToken: MxArray | null = null;
  #running = false;
  #startScheduled = false;
  readonly #admissionWakeup: AdmissionReleaseWakeup;
  #nextId = 0;

  constructor(model: CausalLM, options: ContinuousBatchTokenSchedulerOptions) {
    const resolved = resolveGenerationOptions(model, undefined, { ...options, maxTokens: 1 });
    validateContinuousBatchOptions(resolved);
    this.#model = model;
    this.#maxBatchSize = integerAtLeast(options.maxBatchSize ?? 8, "maxBatchSize", 1);
    this.#batchWindowMs = integerAtLeast(options.batchWindowMs ?? 0, "batchWindowMs", 0);
    this.#runExclusive = options.runExclusive ?? defaultRunExclusive;
    this.#admissionController = options.admissionController;
    this.#admissionWakeup = new AdmissionReleaseWakeup(this.#admissionController);
    this.#telemetry = new ContinuousBatchTelemetry(options.onBatch, options.onSchedulerEvent);
    this.#eosTokenIds = new Set(resolved.eosTokenIds ?? []);
    this.#samplerOptions = samplerOptionsFrom(resolved);
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
      const entry = createScheduledRequest(
        request.id ?? `continuous-${this.#nextId}`,
        request,
        this.#samplerOptions,
        resolve,
        reject,
      );
      this.#nextId += 1;
      attachScheduledRequestAbort(entry, {
        waiting: this.#waiting,
        prefilling: this.#prefilling,
        telemetry: this.#telemetry,
        snapshot: () => this.#snapshot(),
        cleanup: (request) => this.#cleanup(request),
      });
      const queuedAhead = this.#waiting.length;
      this.#waiting.push(entry);
      this.#telemetry.queued(entry, queuedAhead, this.#snapshot());
      this.#schedule();
    });
  }

  #cleanup(entry: ScheduledRequest): void {
    if (entry.onAbort !== undefined) {
      entry.abortSignal?.removeEventListener("abort", entry.onAbort);
      delete entry.onAbort;
    }
    entry.admissionReservation?.[Symbol.dispose]();
    delete entry.admissionReservation;
    entry.samplerState[Symbol.dispose]();
  }

  #schedule(): void {
    if (this.#running || this.#startScheduled || this.#admissionWakeup.waiting) {
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
    this.#admissionWakeup.cancel();
    this.#startScheduled = false;
    if (this.#running || (this.#waiting.length === 0 && this.#prefilling.length === 0)) {
      return;
    }
    this.#running = true;
    void this.#runExclusive(() => this.#runLoop())
      .catch((error) => this.#failAll(error))
      .finally(() => {
        this.#running = false;
        if (
          !this.#admissionWakeup.waiting &&
          (this.#waiting.length > 0 || this.#prefilling.length > 0)
        ) {
          this.#schedule();
        }
      });
  }

  async #runLoop(): Promise<void> {
    using generationStream = createStream();
    const previousWiredLimit = setWiredLimitBytes(getRecommendedWorkingSetBytes());
    let failed = false;
    try {
      while (this.#waiting.length > 0 || this.#prefilling.length > 0 || this.#active.length > 0) {
        let didWork = false;
        withDefaultStream(generationStream, () => {
          rejectAbortedPrefillingRows(this.#prefilling, (request) => this.#cleanup(request));
          this.#dropAbortedActiveRows();
          didWork = this.#admitWaitingRows() || didWork;
          rejectAbortedPrefillingRows(this.#prefilling, (request) => this.#cleanup(request));
          this.#dropAbortedActiveRows();
          didWork = this.#decodeOneStep() || didWork;
        });
        if (!didWork && this.#waiting.length > 0 && this.#prefilling.length === 0) {
          this.#admissionWakeup.pause(() => this.#schedule());
          break;
        }
        await yieldToScheduler();
      }
    } catch (error) {
      failed = true;
      this.#failAll(error);
    } finally {
      this.#currentToken?.free();
      this.#currentToken = null;
      this.#cache?.[Symbol.dispose]();
      this.#cache = null;
      if (!failed) {
        disposePrefillingCaches(this.#prefilling.splice(0));
      }
      synchronize(generationStream);
      clearMemoryCache();
      setWiredLimitBytes(previousWiredLimit);
    }
  }

  #dropAbortedActiveRows(): void {
    const next = dropAbortedActiveRows(this.#active, this.#currentToken, this.#cache, (request) => {
      this.#cleanup(request);
      this.#telemetry.cancelled(request, this.#snapshot());
      request.reject(
        new GenerationAbortError("ContinuousBatchTokenScheduler: generation was cancelled."),
      );
    });
    this.#active = next.active;
    this.#currentToken = next.currentToken;
    this.#cache = next.cache;
  }

  #takeAdmittableWaitingRows(limit: number): ScheduledRequest[] {
    return takeAdmittableWaitingRows(
      this.#waiting,
      limit,
      this.#admissionController,
      () => this.#snapshot(),
      this.#telemetry,
      (request) => this.#cleanup(request),
    );
  }

  #admitWaitingRows(): boolean {
    const capacity = this.#maxBatchSize - this.#active.length - this.#prefilling.length;
    if (capacity <= 0 && this.#prefilling.length === 0) {
      return false;
    }
    if (this.#active.length === 0 && this.#prefilling.length === 0) {
      const count = Math.min(capacity, this.#waiting.length);
      if (shouldChunkInitialRows(this.#waiting, count, this.#prefillStepSize)) {
        let started = false;
        for (let index = 0; index < count; index += 1) {
          started = this.#startPrefillingRow() || started;
        }
        return this.#advancePrefillingRow() || started;
      }
      return this.#admitFullRows(count);
    }
    let didWork = false;
    if (capacity > 0 && this.#waiting.length > 0) {
      didWork = this.#startPrefillingRow();
    }
    return this.#advancePrefillingRow() || didWork;
  }

  #admitFullRows(count: number): boolean {
    if (count <= 0) {
      return false;
    }
    const admitted = this.#takeAdmittableWaitingRows(count);
    if (admitted.length === 0) {
      return false;
    }
    this.#active.push(...admitted);
    const { cache: nextCache, nextToken } = prefillReadyBatch(
      this.#model,
      admitted,
      this.#padTokenId,
      this.#samplerOptions,
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

    this.#telemetry.admitted(this.#active, this.#snapshot());
    return true;
  }

  #startPrefillingRow(): boolean {
    const [request] = this.#takeAdmittableWaitingRows(1);
    if (request === undefined) {
      return false;
    }
    this.#prefilling.push({
      request,
      cache: createBatchCacheForModel(this.#model, [0], "ContinuousBatchTokenScheduler"),
      cursor: 0,
    });
    this.#telemetry.prefillStart(request, this.#snapshot());
    return true;
  }

  #advancePrefillingRow(): boolean {
    const readyIndex = this.#prefilling.findIndex(
      (prefilling) => remainingPrefillTokens(prefilling) <= 0,
    );
    if (readyIndex >= 0) {
      return this.#finishPrefillingRow(readyIndex);
    }
    const prefilling = this.#prefilling[0];
    if (prefilling === undefined) {
      return false;
    }
    this.#prefillOneChunk(prefilling, remainingPrefillTokens(prefilling));
    if (this.#prefilling.length > 1 && remainingPrefillTokens(prefilling) > 0) {
      const [rotated] = this.#prefilling.splice(0, 1);
      if (rotated !== undefined) {
        this.#prefilling.push(rotated);
      }
    }
    return true;
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

  #finishPrefillingRow(index: number): boolean {
    const prefilling = this.#prefilling[index];
    if (prefilling === undefined) {
      return false;
    }
    const tail = prefilling.request.promptTokenIds.slice(prefilling.cursor);
    using tailInput = array([tail], "int32");
    const nextToken = sampleNextBatchToken(
      this.#model,
      tailInput,
      prefilling.cache,
      [prefilling.request.samplerState],
      prefilling.request.samplerOptions,
    );
    const [ready] = this.#prefilling.splice(index, 1);
    if (ready === undefined) {
      nextToken.free();
      return false;
    }
    this.#mergeReadyRow(ready.request, ready.cache, nextToken);
    return true;
  }

  #mergeReadyRow(
    request: ScheduledRequest,
    nextCache: TransformerBatchCache,
    nextToken: MxArray,
  ): void {
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
    this.#telemetry.admitted(this.#active, this.#snapshot());
  }

  #decodeOneStep(): boolean {
    if (this.#currentToken === null || this.#cache === null || this.#active.length === 0) {
      return false;
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
      if (!request.firstTokenEmitted) {
        request.firstTokenEmitted = true;
        this.#telemetry.firstToken(request, this.#snapshot());
      }
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

      request.samplerState.appendToken(tokenId);
      keepPositions.push(position);
      nextActive.push(request);
    }

    if (keepPositions.length === 0) {
      emittedToken.free();
      this.#active = [];
      this.#cache[Symbol.dispose]();
      this.#cache = null;
      return true;
    }

    if (keepPositions.length !== this.#active.length) {
      this.#cache.filter(keepPositions);
    }
    this.#active = nextActive;
    using nextInput = nextInputTensor(tokenIds, keepPositions);
    const nextToken = sampleNextBatchToken(
      this.#model,
      nextInput,
      this.#cache,
      this.#active.map((request) => request.samplerState),
      this.#samplerOptions,
    );
    emittedToken.free();
    this.#currentToken = nextToken;
    return true;
  }

  #finishRequest(request: ScheduledRequest): void {
    this.#cleanup(request);
    this.#telemetry.finished(request, this.#snapshot());
    request.resolve({ tokenIds: [...request.generated], finishReason: request.finishReason });
  }

  #snapshot(): ContinuousBatchQueueSnapshot {
    return continuousBatchQueueSnapshot(
      this.#waiting,
      this.#prefilling,
      this.#active,
      this.#maxBatchSize,
      this.#admissionController,
    );
  }

  #failAll(error: unknown): void {
    this.#admissionWakeup.cancel();
    for (const request of [
      ...this.#waiting,
      ...this.#prefilling.map((prefilling) => prefilling.request),
      ...this.#active,
    ]) {
      this.#cleanup(request);
      request.reject(error);
    }
    this.#waiting.length = 0;
    disposePrefillingCaches(this.#prefilling.splice(0));
    this.#active = [];
    this.#currentToken?.free();
    this.#currentToken = null;
    this.#cache?.[Symbol.dispose]();
    this.#cache = null;
  }
}
