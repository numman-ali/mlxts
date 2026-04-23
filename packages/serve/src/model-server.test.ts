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

import {
  type ServeModelRuntime,
  serveLoadedModel,
  serveModel,
  serveModelWithRuntime,
} from "./model-server";

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

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray {
    void inputIds;
    void options;
    throw new Error("FakeModel.forward should not be called by model metadata requests.");
  }

  createCache(): TransformerCache {
    throw new Error("FakeModel.createCache should not be called by model metadata requests.");
  }

  parameters(): ParameterTree {
    return {};
  }

  trainableParameters(): ParameterTree {
    return {};
  }

  update(params: ParameterTree): void {
    void params;
  }

  freeze(keys?: string[]): this {
    void keys;
    return this;
  }

  unfreeze(keys?: string[]): this {
    void keys;
    return this;
  }

  eval(): this {
    return this;
  }

  train(mode?: boolean): this {
    void mode;
    return this;
  }

  [Symbol.dispose](): void {
    this.disposeCount += 1;
  }
}

class FakeTokenizer implements Tokenizer {
  readonly vocabSize = 8;
  readonly bosTokenId = undefined;
  readonly eosTokenIds = [];
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

describe("serveLoadedModel", () => {
  test("serves model metadata and disposes owned models on stop", async () => {
    const model = new FakeModel();
    const running = serveLoadedModel({
      model,
      tokenizer: new FakeTokenizer(),
      modelId: "tiny",
      port: 0,
      disposeModelOnStop: true,
    });

    try {
      const response = await fetch(`${running.endpoint}/v1/models`);
      expect(response.status).toBe(200);
      expect(await response.json()).toMatchObject({
        object: "list",
        data: [{ id: "tiny", object: "model" }],
      });
    } finally {
      running.stop();
    }

    running.stop();
    expect(model.disposeCount).toBe(1);
  });

  test("does not dispose borrowed models by default", () => {
    const model = new FakeModel();
    const running = serveLoadedModel({
      model,
      tokenizer: new FakeTokenizer(),
      modelId: "borrowed",
      port: 0,
    });

    running.stop();
    expect(model.disposeCount).toBe(0);
  });

  test("supports explicit disposal through the returned server handle", () => {
    const model = new FakeModel();
    const running = serveLoadedModel({
      model,
      tokenizer: new FakeTokenizer(),
      modelId: "owned",
      port: 0,
      disposeModelOnStop: true,
    });

    running[Symbol.dispose]();
    expect(model.disposeCount).toBe(1);
  });

  test("validates operator-facing server options before binding", () => {
    const model = new FakeModel();
    const tokenizer = new FakeTokenizer();

    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "",
      }),
    ).toThrow("modelId must be a non-empty string.");
    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "bad-limit",
        maxGeneratedTokens: 0,
      }),
    ).toThrow("maxGeneratedTokens must be a positive integer.");
    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "bad-total-limit",
        maxTotalTokens: 0,
      }),
    ).toThrow("maxTotalTokens must be a positive integer.");
    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "bad-port",
        port: -1,
      }),
    ).toThrow("port must be a non-negative integer.");
  });

  test("validates public serveModel options before resolving sources", async () => {
    await expect(serveModel({ source: "" })).rejects.toThrow("source must be a non-empty string.");
  });

  test("loads, serves, and disposes model-owned servers through injectable runtime", async () => {
    const model = new FakeModel();
    const calls: string[] = [];
    const runtime: ServeModelRuntime = {
      async resolvePretrainedSource(source, options) {
        calls.push(`resolve:${source}:${options?.revision}:${options?.localFilesOnly}`);
        return "/snapshots/qwen";
      },
      async loadCausalLM(source) {
        calls.push(`model:${source}`);
        return model;
      },
      async loadPretrainedTokenizer(source) {
        calls.push(`tokenizer:${source}`);
        return new FakeTokenizer();
      },
      async loadInteractionProfile(source) {
        calls.push(`profile:${source}`);
        return fakeInteractionProfile;
      },
      serveLoadedModel(options) {
        calls.push(
          `serve:${options.modelId}:${options.apiKey}:${options.maxTotalTokens}:${options.disposeModelOnStop}`,
        );
        return fakeRunningServer(options.modelId);
      },
    };

    const running = await serveModelWithRuntime(
      {
        source: "repo/model",
        modelId: "qwen-local",
        revision: "main",
        apiKey: "secret",
        localFilesOnly: true,
      },
      runtime,
    );

    expect(running.modelId).toBe("qwen-local");
    expect(calls).toEqual([
      "resolve:repo/model:main:true",
      "model:/snapshots/qwen",
      "tokenizer:/snapshots/qwen",
      "profile:/snapshots/qwen",
      "serve:qwen-local:secret:4096:true",
    ]);
    expect(model.disposeCount).toBe(0);
    running.stop();
  });

  test("disposes loaded models when tokenizer loading fails", async () => {
    const model = new FakeModel();
    const runtime: ServeModelRuntime = {
      async resolvePretrainedSource() {
        return "/snapshots/qwen";
      },
      async loadCausalLM() {
        return model;
      },
      async loadPretrainedTokenizer() {
        throw new Error("tokenizer failed");
      },
      async loadInteractionProfile() {
        return fakeInteractionProfile;
      },
      serveLoadedModel() {
        throw new Error("serve should not start");
      },
    };

    await expect(serveModelWithRuntime({ source: "repo/model" }, runtime)).rejects.toThrow(
      "tokenizer failed",
    );
    expect(model.disposeCount).toBe(1);
  });
});

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

function fakeRunningServer(modelId: string) {
  const server = Bun.serve({
    port: 0,
    fetch() {
      return new Response("ok");
    },
  });
  return {
    endpoint: "http://127.0.0.1:8000",
    modelId,
    server,
    stop() {
      server.stop(true);
    },
    [Symbol.dispose]() {
      server.stop(true);
    },
  };
}
