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
import { type ServeModelsRuntime, serveModelsWithRuntime } from "./model-sources";

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

function removeDirectory(path: string): void {
  Bun.spawnSync(["rm", "-rf", path]);
}

describe("serveModels", () => {
  test("loads multiple sources sequentially and serves them as owned models", async () => {
    const calls: string[] = [];
    const progress: string[] = [];
    const runtime = testRuntime({ calls });

    const running = await serveModelsWithRuntime(
      {
        models: [
          { source: "repo/gemma", modelId: "gemma-local" },
          { source: "repo/qwen", revision: "qwen-rev", localFilesOnly: true },
        ],
        revision: "main",
        apiKey: "secret",
        maxBatchSize: 16,
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
      "serve:gemma-local,repo/qwen:secret:16:true",
    ]);
    expect(progress).toEqual(["0:gemma-local:resolve:repo/gemma", "1:repo/qwen:resolve:repo/qwen"]);
    running.stop();
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
    expect(calls).toEqual([]);
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
});
