import { describe, expect, test } from "bun:test";
import type { MxArray, ParameterTree } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import type {
  BaseModelConfig,
  CausalLM,
  ForwardOptions,
  InteractionProfile,
  TransformerCache,
} from "@mlxts/transformers";
import { type ServeModelsRuntime, serveModelsWithRuntime } from "./sources";

class FakeModel implements CausalLM {
  readonly family = "llama";
  readonly layerCount = 0;
  readonly config: BaseModelConfig = {
    family: "llama",
    modelType: "fake",
    rawConfig: {},
    vocabSize: 8,
    hiddenSize: 4,
    numHiddenLayers: 0,
  };
  disposeCount = 0;

  forward(_inputIds: MxArray, _options?: ForwardOptions): MxArray {
    throw new Error("FakeModel.forward should not be called by source-loading tests.");
  }

  createCache(): TransformerCache {
    throw new Error("FakeModel.createCache should not be called by source-loading tests.");
  }

  parameters(): ParameterTree {
    return {};
  }

  trainableParameters(): ParameterTree {
    return {};
  }

  update(_params: ParameterTree): void {}

  freeze(): this {
    return this;
  }

  unfreeze(): this {
    return this;
  }

  eval(): this {
    return this;
  }

  train(): this {
    return this;
  }

  [Symbol.dispose](): void {
    this.disposeCount += 1;
  }
}

class FailingAdmissionModel extends FakeModel {
  override readonly config: BaseModelConfig = new Proxy(
    {
      family: "llama",
      modelType: "fake",
      rawConfig: {},
      vocabSize: 8,
      hiddenSize: 4,
      numHiddenLayers: 0,
    },
    {
      get(target, property, receiver) {
        if (property === "rawConfig") {
          throw new Error("admission metadata failed");
        }
        return Reflect.get(target, property, receiver);
      },
    },
  );
}

class FakeTokenizer implements Tokenizer {
  readonly vocabSize = 8;
  readonly bosTokenId = undefined;
  readonly eosTokenIds: number[] = [];
  readonly padTokenId = undefined;

  encode(text: string): number[] {
    return [...text].map((char) => char.charCodeAt(0));
  }

  encodeWithOffsets(text: string) {
    return { ids: this.encode(text) };
  }

  encodeBatch(texts: string[]) {
    return texts.map((text) => this.encodeWithOffsets(text));
  }

  decode(tokenIds: number[]): string {
    return tokenIds.map((tokenId) => String.fromCharCode(tokenId)).join("");
  }

  decodeBatch(batch: number[][]): string[] {
    return batch.map((tokenIds) => this.decode(tokenIds));
  }
}

const fakeInteractionProfile: InteractionProfile = {
  kind: "completion",
  chatTemplate: null,
  compileTextPrompt(_tokenizer, prompt) {
    return { text: prompt, tokenIds: [] };
  },
  compileMessages() {
    throw new Error("not used");
  },
};

function fakeRunningServer(modelIds: readonly string[]) {
  const server = Bun.serve({
    port: 0,
    fetch() {
      return new Response("ok");
    },
  });
  const [modelId] = modelIds;
  if (modelId === undefined) {
    throw new Error("expected at least one model id");
  }
  return {
    endpoint: "http://127.0.0.1:8000",
    modelId,
    modelIds,
    server,
    stop() {
      server.stop(true);
    },
    [Symbol.dispose]() {
      server.stop(true);
    },
  };
}

function testRuntime(options: {
  calls: string[];
  models?: FakeModel[];
  failOn?: string;
  loadQwenConditional?: boolean;
  recordContentAdapters?: boolean;
}): ServeModelsRuntime {
  const models = options.models ?? [new FakeModel(), new FakeModel()];
  return {
    async resolvePretrainedSource(source, loadOptions) {
      options.calls.push(
        `resolve:${source}:${loadOptions?.revision}:${loadOptions?.localFilesOnly}`,
      );
      loadOptions?.onProgress?.({ stage: "resolve", status: "start", source });
      if (options.failOn === `resolve:${source}`) {
        throw new Error(`resolve failed:${source}`);
      }
      return source.startsWith("/") ? source : `/snapshots/${source}`;
    },
    async loadCausalLM(source) {
      options.calls.push(`model:${source}`);
      if (options.failOn === `model:${source}`) {
        throw new Error(`model failed:${source}`);
      }
      const model = models.shift();
      if (model === undefined) {
        throw new Error("missing fake model");
      }
      return model;
    },
    async loadPretrainedTokenizer(source) {
      options.calls.push(`tokenizer:${source}`);
      if (options.failOn === `tokenizer:${source}`) {
        throw new Error(`tokenizer failed:${source}`);
      }
      return new FakeTokenizer();
    },
    async loadInteractionProfile(source) {
      options.calls.push(`profile:${source}`);
      if (options.failOn === `profile:${source}`) {
        throw new Error(`profile failed:${source}`);
      }
      return fakeInteractionProfile;
    },
    ...(options.loadQwenConditional === true
      ? {
          async loadQwen3_5ForConditionalGeneration(source) {
            options.calls.push(`qwen-conditional-model:${source}`);
            const model = models.shift();
            if (model === undefined) {
              throw new Error("missing fake model");
            }
            return model;
          },
          async loadQwen3_5VisionPreprocessor(source) {
            options.calls.push(`qwen-preprocessor:${source}`);
            return {
              size: { shortestEdge: 3136, longestEdge: 50176 },
              patchSize: 16,
              temporalPatchSize: 2,
              mergeSize: 2,
              imageMean: [0.5, 0.5, 0.5],
              imageStd: [0.5, 0.5, 0.5],
              processorClass: "Qwen3VLProcessor",
              imageProcessorType: "Qwen2VLImageProcessorFast",
            };
          },
        }
      : {}),
    serveLoadedModels(serveOptions) {
      const modelIds = serveOptions.models.map((model) => model.modelId);
      if (options.recordContentAdapters === true) {
        options.calls.push(
          `content-adapters:${serveOptions.models
            .map((model) => (model.contentAdapter === undefined ? "no" : "yes"))
            .join(",")}`,
        );
      }
      const remoteImageHosts = serveOptions.remoteImageHosts ?? [];
      if (remoteImageHosts.length > 0) {
        options.calls.push(`remote-image-hosts:${remoteImageHosts.join(",")}`);
      }
      const localImageRoots = serveOptions.localImageRoots ?? [];
      if (localImageRoots.length > 0) {
        options.calls.push(`local-image-roots:${localImageRoots.join(",")}`);
      }
      options.calls.push(
        `serve:${modelIds.join(",")}:${serveOptions.apiKey}:${serveOptions.maxBatchSize}:${serveOptions.disposeModelsOnStop}`,
      );
      if (options.failOn === "serve") {
        throw new Error("serve failed");
      }
      return fakeRunningServer(modelIds);
    },
  };
}

function createQwenConditionalDirectory(): string {
  const directory = `${Bun.env.TMPDIR ?? "/tmp"}/mlxts-qwen-conditional-${crypto.randomUUID()}`;
  const result = Bun.spawnSync(["mkdir", "-p", directory]);
  if (result.exitCode !== 0) {
    throw new Error("failed to create temporary Qwen checkpoint directory");
  }
  return directory;
}

async function createSafetensorDirectory(bytes: number): Promise<string> {
  const directory = createQwenConditionalDirectory();
  await Bun.write(`${directory}/model.safetensors`, new Uint8Array(bytes));
  return directory;
}

function removeDirectory(path: string): void {
  Bun.spawnSync(["rm", "-rf", path]);
}

function deferred<T>() {
  let resolveValue: (value: T) => void = () => {};
  const promise = new Promise<T>((resolve) => {
    resolveValue = resolve;
  });
  return { promise, resolve: resolveValue };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function completionRequest(model: string): RequestInit {
  return {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, prompt: "hello", max_tokens: 1 }),
  };
}

describe("serveModels", () => {
  test("loads multiple sources sequentially and serves them as owned models", async () => {
    const calls: string[] = [];
    const progress: string[] = [];
    const runtime = testRuntime({ calls });
    const localImageRoot = createQwenConditionalDirectory();

    try {
      const running = await serveModelsWithRuntime(
        {
          models: [
            { source: "repo/gemma", modelId: "gemma-local" },
            { source: "repo/qwen", revision: "qwen-rev", localFilesOnly: true },
          ],
          revision: "main",
          apiKey: "secret",
          maxBatchSize: 16,
          localImageRoots: [localImageRoot],
          remoteImageHosts: ["example.com", "cdn.example.com"],
          onProgress(event, context) {
            progress.push(`${context.index}:${context.modelId}:${event.stage}:${context.source}`);
          },
        },
        runtime,
      );

      expect(running.modelIds).toEqual(["gemma-local", "repo/qwen"]);
      expect(calls).toEqual([
        "resolve:repo/gemma:main:false",
        "model:/snapshots/repo/gemma",
        "tokenizer:/snapshots/repo/gemma",
        "profile:/snapshots/repo/gemma",
        "resolve:repo/qwen:qwen-rev:true",
        "model:/snapshots/repo/qwen",
        "tokenizer:/snapshots/repo/qwen",
        "profile:/snapshots/repo/qwen",
        "remote-image-hosts:example.com,cdn.example.com",
        `local-image-roots:${localImageRoot}`,
        "serve:gemma-local,repo/qwen:secret:16:true",
      ]);
      expect(progress).toEqual([
        "0:gemma-local:resolve:repo/gemma",
        "1:repo/qwen:resolve:repo/qwen",
      ]);
      running.stop();
    } finally {
      removeDirectory(localImageRoot);
    }
  });

  test("validates source entries before resolving anything", async () => {
    const calls: string[] = [];
    const runtime = testRuntime({ calls });

    await expect(serveModelsWithRuntime({ models: [] }, runtime)).rejects.toThrow(
      "models must contain at least one model source",
    );
    await expect(serveModelsWithRuntime({ models: [{ source: "" }] }, runtime)).rejects.toThrow(
      "source must be a non-empty string",
    );
    await expect(
      serveModelsWithRuntime(
        { models: [{ source: "repo/one" }, { source: "repo/two", modelId: "repo/one" }] },
        runtime,
      ),
    ).rejects.toThrow('modelId "repo/one" is duplicated.');
    await expect(
      serveModelsWithRuntime({ models: [{ source: "repo/one", pinned: true }] }, runtime),
    ).rejects.toThrow('pinned source models require modelLoadPolicy="lazy"');
    await expect(
      serveModelsWithRuntime({ models: [{ source: "repo/one" }], modelIdleTtlMs: 1000 }, runtime),
    ).rejects.toThrow('modelIdleTtlMs requires modelLoadPolicy="lazy"');
    await expect(
      serveModelsWithRuntime(
        { models: [{ source: "repo/one" }], modelPressurePolicy: "shed_non_pinned" },
        runtime,
      ),
    ).rejects.toThrow('modelPressurePolicy requires modelLoadPolicy="lazy"');
    await expect(
      serveModelsWithRuntime(
        { models: [{ source: "repo/one" }], modelPressureReleaseTimeoutMs: 1000 },
        runtime,
      ),
    ).rejects.toThrow('modelPressureReleaseTimeoutMs requires modelLoadPolicy="lazy"');
    await expect(
      serveModelsWithRuntime(
        {
          models: [{ source: "repo/one" }],
          modelLoadPolicy: "lazy",
          modelPressureReleaseTimeoutMs: 0,
        },
        runtime,
      ),
    ).rejects.toThrow("modelPressureReleaseTimeoutMs must be a positive integer");
    await expect(
      serveModelsWithRuntime(
        {
          models: [{ source: "repo/one" }],
          modelLoadPolicy: "lazy",
          pinnedModels: ["missing"],
        },
        runtime,
      ),
    ).rejects.toThrow('pinned model "missing" is not part of this serveModels() call');
    expect(calls).toEqual([]);
  });

  test("starts a lazy source pool without resolving models at startup", async () => {
    const calls: string[] = [];
    const runtime = testRuntime({ calls });

    const running = await serveModelsWithRuntime(
      {
        models: [
          { source: "repo/gemma", modelId: "gemma" },
          { source: "repo/qwen", modelId: "qwen", pinned: true },
        ],
        modelLoadPolicy: "lazy",
        modelIdleTtlMs: 1000,
        port: 0,
      },
      runtime,
    );

    try {
      expect(calls).toEqual([]);
      expect(running.modelIds).toEqual(["gemma", "qwen"]);
      const response = await fetch(`${running.endpoint}/v1/models`);
      expect(response.status).toBe(200);
      expect(await response.json()).toMatchObject({
        data: [{ id: "gemma" }, { id: "qwen" }],
      });
      const infoResponse = await fetch(`${running.endpoint}/info`);
      expect(infoResponse.status).toBe(200);
      expect(await infoResponse.json()).toMatchObject({
        model_pool: {
          load_policy: "lazy",
          pressure_policy: "reject",
          pressure_release_timeout_ms: null,
          idle_ttl_ms: 1000,
          models: [
            {
              id: "gemma",
              pinned: false,
              state: "not_loaded",
              active_requests: 0,
              pressure_aborted_requests: 0,
            },
            {
              id: "qwen",
              pinned: true,
              state: "not_loaded",
              active_requests: 0,
              pressure_aborted_requests: 0,
            },
          ],
        },
      });
      expect(calls).toEqual([]);
    } finally {
      running.stop();
    }
  });

  test("disposes lazy-loaded models when setup fails before engine publication", async () => {
    const calls: string[] = [];
    const model = new FailingAdmissionModel();
    const runtime = testRuntime({ calls, models: [model] });
    const running = await serveModelsWithRuntime(
      {
        models: [{ source: "repo/gemma", modelId: "gemma" }],
        modelLoadPolicy: "lazy",
        port: 0,
      },
      runtime,
    );

    try {
      const response = await fetch(
        `${running.endpoint}/v1/completions`,
        completionRequest("gemma"),
      );
      expect(response.status).toBe(500);
      expect(model.disposeCount).toBe(1);
    } finally {
      running.stop();
    }
  });

  test("disposes previously loaded models when later loading fails", async () => {
    const calls: string[] = [];
    const first = new FakeModel();
    const second = new FakeModel();
    const runtime = testRuntime({
      calls,
      models: [first, second],
      failOn: "tokenizer:/snapshots/repo/qwen",
    });

    await expect(
      serveModelsWithRuntime(
        { models: [{ source: "repo/gemma" }, { source: "repo/qwen" }] },
        runtime,
      ),
    ).rejects.toThrow("tokenizer failed:/snapshots/repo/qwen");

    expect(first.disposeCount).toBe(1);
    expect(second.disposeCount).toBe(1);
  });

  test("disposes loaded models when final serving fails", async () => {
    const calls: string[] = [];
    const first = new FakeModel();
    const second = new FakeModel();
    const runtime = testRuntime({ calls, models: [first, second], failOn: "serve" });

    await expect(
      serveModelsWithRuntime(
        { models: [{ source: "repo/gemma" }, { source: "repo/qwen" }] },
        runtime,
      ),
    ).rejects.toThrow("serve failed");

    expect(first.disposeCount).toBe(1);
    expect(second.disposeCount).toBe(1);
  });

  test("rejects later model loads that exceed the active memory preflight", async () => {
    const calls: string[] = [];
    const firstDirectory = await createSafetensorDirectory(100);
    const secondDirectory = await createSafetensorDirectory(100);
    const first = new FakeModel();
    const second = new FakeModel();
    let memoryReads = 0;
    const runtime: ServeModelsRuntime = {
      ...testRuntime({ calls, models: [first, second] }),
      readGenerationMemoryUsage() {
        memoryReads += 1;
        return {
          activeBytes: memoryReads === 1 ? 0 : 850,
          cacheBytes: 0,
          peakBytes: 0,
          limitBytes: 1000,
        };
      },
    };

    try {
      await expect(
        serveModelsWithRuntime(
          {
            models: [
              { source: firstDirectory, modelId: "first" },
              { source: secondDirectory, modelId: "second" },
            ],
            gpuMemoryUtilization: 0.9,
          },
          runtime,
        ),
      ).rejects.toThrow('Model "second" is estimated to need');

      expect(calls).toEqual([
        `resolve:${firstDirectory}:undefined:false`,
        `model:${firstDirectory}`,
        `tokenizer:${firstDirectory}`,
        `profile:${firstDirectory}`,
        `resolve:${secondDirectory}:undefined:false`,
      ]);
      expect(first.disposeCount).toBe(1);
      expect(second.disposeCount).toBe(0);
    } finally {
      removeDirectory(firstDirectory);
      removeDirectory(secondDirectory);
    }
  });

  test("serializes lazy first loads so memory preflight sees earlier loads", async () => {
    const calls: string[] = [];
    const firstDirectory = await createSafetensorDirectory(100);
    const secondDirectory = await createSafetensorDirectory(100);
    const first = new FakeModel();
    const second = new FakeModel();
    const firstLoadStarted = deferred<void>();
    const releaseFirstLoad = deferred<FakeModel>();
    let memoryReads = 0;
    const runtime: ServeModelsRuntime = {
      ...testRuntime({ calls, models: [second] }),
      readGenerationMemoryUsage() {
        memoryReads += 1;
        return {
          activeBytes: memoryReads === 1 ? 0 : 850,
          cacheBytes: 0,
          peakBytes: 0,
          limitBytes: 1000,
        };
      },
      async loadCausalLM(source) {
        calls.push(`model:${source}`);
        if (source === firstDirectory) {
          firstLoadStarted.resolve();
          return releaseFirstLoad.promise;
        }
        return second;
      },
    };

    try {
      const running = await serveModelsWithRuntime(
        {
          models: [
            { source: firstDirectory, modelId: "first" },
            { source: secondDirectory, modelId: "second" },
          ],
          modelLoadPolicy: "lazy",
          gpuMemoryUtilization: 0.9,
          port: 0,
        },
        runtime,
      );
      try {
        const firstResponse = fetch(
          `${running.endpoint}/v1/completions`,
          completionRequest("first"),
        );
        await firstLoadStarted.promise;
        const secondResponse = fetch(
          `${running.endpoint}/v1/completions`,
          completionRequest("second"),
        );
        await sleep(0);
        releaseFirstLoad.resolve(first);

        const [, secondResult] = await Promise.all([firstResponse, secondResponse]);
        expect(secondResult.status).toBe(503);
        expect(memoryReads).toBe(2);
        expect(calls).toEqual([
          `resolve:${firstDirectory}:undefined:false`,
          `model:${firstDirectory}`,
          `tokenizer:${firstDirectory}`,
          `profile:${firstDirectory}`,
          `resolve:${secondDirectory}:undefined:false`,
        ]);
        expect(first.disposeCount).toBe(0);
        expect(second.disposeCount).toBe(0);
      } finally {
        running.stop();
      }
    } finally {
      removeDirectory(firstDirectory);
      removeDirectory(secondDirectory);
    }
  });

  test("reruns lazy memory preflight after HTTP idle eviction", async () => {
    const calls: string[] = [];
    const directory = await createSafetensorDirectory(100);
    const first = new FakeModel();
    const second = new FakeModel();
    let memoryReads = 0;
    const runtime: ServeModelsRuntime = {
      ...testRuntime({ calls, models: [first, second] }),
      readGenerationMemoryUsage() {
        memoryReads += 1;
        return {
          activeBytes: 0,
          cacheBytes: 0,
          peakBytes: 0,
          limitBytes: 1000,
        };
      },
    };

    try {
      const running = await serveModelsWithRuntime(
        {
          models: [{ source: directory, modelId: "first" }],
          modelLoadPolicy: "lazy",
          modelIdleTtlMs: 1,
          gpuMemoryUtilization: 0.9,
          port: 0,
        },
        runtime,
      );
      try {
        const firstResponse = await fetch(
          `${running.endpoint}/v1/completions`,
          completionRequest("first"),
        );
        expect(firstResponse.status).toBe(500);
        await sleep(10);
        expect(first.disposeCount).toBe(1);

        const secondResponse = await fetch(
          `${running.endpoint}/v1/completions`,
          completionRequest("first"),
        );
        expect(secondResponse.status).toBe(500);
        await sleep(10);

        expect(second.disposeCount).toBe(1);
        expect(memoryReads).toBe(2);
        expect(calls).toEqual([
          `resolve:${directory}:undefined:false`,
          `model:${directory}`,
          `tokenizer:${directory}`,
          `profile:${directory}`,
          `resolve:${directory}:undefined:false`,
          `model:${directory}`,
          `tokenizer:${directory}`,
          `profile:${directory}`,
        ]);
      } finally {
        running.stop();
      }
    } finally {
      removeDirectory(directory);
    }
  });

  test("loads Qwen conditional checkpoints with an image content adapter", async () => {
    const calls: string[] = [];
    const directory = createQwenConditionalDirectory();
    await Bun.write(
      `${directory}/config.json`,
      JSON.stringify({
        model_type: "qwen3_5",
        architectures: ["Qwen3_5ForConditionalGeneration"],
        vision_config: { hidden_size: 8 },
      }),
    );
    const runtime = testRuntime({
      calls,
      loadQwenConditional: true,
      recordContentAdapters: true,
    });

    try {
      const running = await serveModelsWithRuntime(
        {
          models: [{ source: directory, modelId: "mlx-community/Qwen3.6-27B-4bit" }],
          localFilesOnly: true,
        },
        runtime,
      );

      expect(running.modelIds).toEqual(["mlx-community/Qwen3.6-27B-4bit"]);
      expect(calls).toEqual([
        `resolve:${directory}:undefined:true`,
        `qwen-conditional-model:${directory}`,
        `tokenizer:${directory}`,
        `profile:${directory}`,
        `qwen-preprocessor:${directory}`,
        "content-adapters:yes",
        "serve:mlx-community/Qwen3.6-27B-4bit:undefined:undefined:true",
      ]);
      running.stop();
    } finally {
      removeDirectory(directory);
    }
  });

  test("loads Qwen MoE conditional checkpoints with an image content adapter", async () => {
    const calls: string[] = [];
    const directory = createQwenConditionalDirectory();
    await Bun.write(
      `${directory}/config.json`,
      JSON.stringify({
        model_type: "qwen3_5_moe",
        architectures: ["Qwen3_5MoeForConditionalGeneration"],
        vision_config: { hidden_size: 8 },
      }),
    );
    const runtime = testRuntime({
      calls,
      loadQwenConditional: true,
      recordContentAdapters: true,
    });

    try {
      const running = await serveModelsWithRuntime(
        {
          models: [{ source: directory, modelId: "Qwen/Qwen3.6-35B-A3B" }],
          localFilesOnly: true,
        },
        runtime,
      );

      expect(running.modelIds).toEqual(["Qwen/Qwen3.6-35B-A3B"]);
      expect(calls).toEqual([
        `resolve:${directory}:undefined:true`,
        `qwen-conditional-model:${directory}`,
        `tokenizer:${directory}`,
        `profile:${directory}`,
        `qwen-preprocessor:${directory}`,
        "content-adapters:yes",
        "serve:Qwen/Qwen3.6-35B-A3B:undefined:undefined:true",
      ]);
      running.stop();
    } finally {
      removeDirectory(directory);
    }
  });
});
