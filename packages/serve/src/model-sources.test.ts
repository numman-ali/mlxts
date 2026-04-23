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
      return `/snapshots/${source}`;
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
    serveLoadedModels(serveOptions) {
      const modelIds = serveOptions.models.map((model) => model.modelId);
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
});
