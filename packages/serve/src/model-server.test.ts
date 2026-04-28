import { describe, expect, test } from "bun:test";
import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import type {
  BaseModelConfig,
  CausalLM,
  ForwardOptions,
  InteractionProfile,
  Qwen3_5VisionPreprocessorConfig,
  TransformerCache,
} from "@mlxts/transformers";
import { KVCache } from "@mlxts/transformers";

import {
  type ServeModelRuntime,
  serveLoadedModel,
  serveLoadedModels,
  serveModel,
  serveModelWithRuntime,
} from "./model-server";
import type { ServeEvent } from "./types";

const ROUTE_STRATEGY = {
  schedulerMode: "auto",
  cacheBackend: "managed",
  attentionBackend: "auto",
  decodingBackend: "model",
} as const;

class FakeModel implements CausalLM {
  readonly family = "llama";
  readonly layerCount: number = 0;
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
  readonly vocabSize: number = 8;
  readonly bosTokenId: number | undefined = undefined;
  readonly eosTokenIds: number[] = [];
  readonly padTokenId: number | undefined = undefined;

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

class GeneratingModel extends FakeModel {
  readonly #tokenId: number;
  override readonly config: BaseModelConfig = {
    family: "llama",
    modelType: "llama",
    rawConfig: {},
    vocabSize: 4,
    hiddenSize: 1,
    numHiddenLayers: 1,
  };
  override readonly layerCount: number = 1;
  readonly forwardBatchSizes: number[] = [];

  constructor(tokenId = 2) {
    super();
    this.#tokenId = tokenId;
  }

  get batchForwardCount(): number {
    return this.forwardBatchSizes.filter((batchSize) => batchSize > 1).length;
  }

  override forward(inputIds: MxArray, options?: ForwardOptions): MxArray {
    const [batchSize, sequenceLength] = inputIds.shape;
    if (batchSize === undefined || sequenceLength === undefined) {
      throw new Error("GeneratingModel.forward expected rank-2 token ids.");
    }
    this.forwardBatchSizes.push(batchSize);
    options?.cache?.advance(sequenceLength);
    return array(
      Array.from({ length: batchSize }, () =>
        Array.from({ length: sequenceLength }, () =>
          Array.from({ length: this.config.vocabSize }, (_, tokenId) =>
            tokenId === this.#tokenId ? 1.0 : 0.0,
          ),
        ),
      ),
      "float32",
    );
  }

  override createCache(): TransformerCache {
    return new KVCache(this.layerCount);
  }
}

class PreparedGeneratingModel extends GeneratingModel {
  readonly forwardedInputEmbeddingShapes: number[][] = [];

  override forward(inputIds: MxArray, options?: ForwardOptions): MxArray {
    if (options?.inputEmbeddings !== undefined) {
      this.forwardedInputEmbeddingShapes.push([...options.inputEmbeddings.shape]);
    }
    return super.forward(inputIds, options);
  }
}

class GeneratingTokenizer extends FakeTokenizer {
  override readonly vocabSize: number = 4;

  override decode(tokenIds: number[]): string {
    return tokenIds.map((tokenId) => String.fromCharCode(97 + tokenId)).join("");
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
      maxGeneratedTokens: 128,
      maxPromptTokens: 192,
      maxTotalTokens: 256,
      maxBatchSize: 4,
      batchWindowMs: 2,
      prefillStepSize: 1024,
      activePrefillStepSize: 64,
      activeDecodeStepsPerPrefillChunk: 7,
      maxConcurrentRequests: 1,
      gpuMemoryUtilization: 0.75,
      disposeModelOnStop: true,
    });

    try {
      const response = await fetch(`${running.endpoint}/v1/models`);
      const infoResponse = await fetch(`${running.endpoint}/info`);
      expect(response.status).toBe(200);
      expect(infoResponse.status).toBe(200);
      expect(await response.json()).toMatchObject({
        object: "list",
        data: [{ id: "tiny", object: "model" }],
      });
      expect(await infoResponse.json()).toMatchObject({
        model_id: "tiny",
        model_ids: ["tiny"],
        limits: {
          max_generated_tokens: 128,
          max_prompt_tokens: 192,
          max_total_tokens: 256,
          max_client_batch_size: 4,
          batch_window_ms: 2,
          prefill_step_size: 1024,
          active_prefill_step_size: 64,
          active_decode_steps_per_prefill_chunk: 7,
          stream_decode_interval: 1,
          max_concurrent_requests: 1,
          gpu_memory_utilization: 0.75,
        },
        runtime_strategy: {
          scheduler: {
            mode: "auto",
            max_batch_size: 4,
            batch_window_ms: 2,
            prefill_step_size: 1024,
            active_prefill_step_size: 64,
            active_decode_steps_per_prefill_chunk: 7,
            max_concurrent_requests: 1,
          },
          cache: {
            backend: "managed",
            precision: "model",
          },
          attention: {
            backend: "auto",
          },
          decoding: {
            backend: "model",
          },
          streaming: {
            stream_decode_interval: 1,
          },
          memory: {
            policy: "admit_only",
            gpu_memory_utilization: 0.75,
          },
        },
        models: [
          {
            id: "tiny",
            context_window: null,
            max_prompt_tokens: 192,
            max_total_tokens: 256,
            effective_total_tokens: 256,
          },
        ],
      });
      expect(running.modelIds).toEqual(["tiny"]);
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

  test("schedules concurrent non-streaming completions through continuous batching", async () => {
    const model = new GeneratingModel();
    const events: ServeEvent[] = [];
    const running = serveLoadedModel({
      model,
      tokenizer: new GeneratingTokenizer(),
      modelId: "tiny",
      port: 0,
      maxBatchSize: 2,
      batchWindowMs: 5,
      maxConcurrentRequests: 1,
      onEvent: (event) => events.push(event),
    });

    try {
      const body = JSON.stringify({
        model: "tiny",
        prompt: "hi",
        max_tokens: 16,
        temperature: 0,
      });
      const [first, second] = await Promise.all([
        fetch(`${running.endpoint}/v1/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body,
        }),
        fetch(`${running.endpoint}/v1/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body,
        }),
      ]);

      expect(first.status).toBe(200);
      expect(second.status).toBe(200);
      expect(await first.json()).toMatchObject({ choices: [{ text: "c".repeat(16) }] });
      expect(await second.json()).toMatchObject({ choices: [{ text: "c".repeat(16) }] });
      expect(model.batchForwardCount).toBeGreaterThan(0);
      expect(
        events.find(
          (event) =>
            event.type === "generation_scheduler_phase" &&
            event.phase === "admitted" &&
            event.batchSize === 2,
        ),
      ).toMatchObject({
        mode: "continuous",
        model: "tiny",
        batchSize: 2,
      });
      expect(events.filter((event) => event.type === "generation_route_decision")).toContainEqual({
        type: "generation_route_decision",
        id: expect.any(String),
        protocol: "openai.completions",
        model: "tiny",
        route: "continuous",
        eligible: true,
        reason: "eligible",
        modelType: "llama",
        maxBatchSize: 2,
        ...ROUTE_STRATEGY,
        stream: false,
      });
      const metricsResponse = await fetch(`${running.endpoint}/metrics`);
      const metrics = await metricsResponse.text();
      expect(metrics).toContain(
        'mlxts_serve_generation_route_decisions_total{model="tiny",protocol="openai.completions",route="continuous",eligible="true",reason="eligible",model_type="llama",scheduler="auto",cache="managed",attention="auto",decoding="model",stream="false"} 2',
      );
      expect(metrics).toContain(
        'mlxts_serve_scheduler_phases_total{model="tiny",mode="continuous",phase="admitted"}',
      );
    } finally {
      running.stop();
    }
  });

  test("schedules concurrent completions with mixed max_tokens", async () => {
    const model = new GeneratingModel();
    const running = serveLoadedModel({
      model,
      tokenizer: new GeneratingTokenizer(),
      modelId: "tiny",
      port: 0,
      maxBatchSize: 2,
      batchWindowMs: 5,
      maxConcurrentRequests: 1,
    });

    try {
      const request = (maxTokens: number) =>
        fetch(`${running.endpoint}/v1/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "tiny",
            prompt: "hi",
            max_tokens: maxTokens,
            temperature: 0,
          }),
        });
      const [short, long] = await Promise.all([request(8), request(16)]);

      expect(short.status).toBe(200);
      expect(long.status).toBe(200);
      expect(await short.json()).toMatchObject({ choices: [{ text: "c".repeat(8) }] });
      expect(await long.json()).toMatchObject({ choices: [{ text: "c".repeat(16) }] });
      expect(model.batchForwardCount).toBeGreaterThan(0);
    } finally {
      running.stop();
    }
  });

  test("serves multiple loaded models behind one endpoint", async () => {
    const gemma = new GeneratingModel(1);
    const qwen = new GeneratingModel(2);
    const running = serveLoadedModels({
      models: [
        { model: gemma, tokenizer: new GeneratingTokenizer(), modelId: "gemma-local" },
        {
          model: qwen,
          tokenizer: new GeneratingTokenizer(),
          modelId: "mlx-community/Qwen3.6-27B-4bit",
        },
      ],
      port: 0,
      disposeModelsOnStop: true,
    });

    try {
      const modelsResponse = await fetch(`${running.endpoint}/v1/models`);
      const infoResponse = await fetch(`${running.endpoint}/info`);
      expect(modelsResponse.status).toBe(200);
      expect(infoResponse.status).toBe(200);
      expect(await modelsResponse.json()).toMatchObject({
        object: "list",
        data: [
          { id: "gemma-local", object: "model" },
          { id: "mlx-community/Qwen3.6-27B-4bit", object: "model" },
        ],
      });
      expect(await infoResponse.json()).toMatchObject({
        model_id: "gemma-local",
        model_ids: ["gemma-local", "mlx-community/Qwen3.6-27B-4bit"],
        model_count: 2,
      });
      expect(running.modelId).toBe("gemma-local");
      expect(running.modelIds).toEqual(["gemma-local", "mlx-community/Qwen3.6-27B-4bit"]);

      const gemmaResponse = await fetch(`${running.endpoint}/v1/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "gemma-local",
          prompt: "hi",
          max_tokens: 2,
          temperature: 0,
        }),
      });
      const qwenResponse = await fetch(`${running.endpoint}/v1/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "mlx-community/Qwen3.6-27B-4bit",
          prompt: "hi",
          max_tokens: 2,
          temperature: 0,
        }),
      });
      const missingResponse = await fetch(`${running.endpoint}/v1/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "missing-local",
          prompt: "hi",
          max_tokens: 1,
          temperature: 0,
        }),
      });

      expect(gemmaResponse.status).toBe(200);
      expect(qwenResponse.status).toBe(200);
      expect(missingResponse.status).toBe(404);
      expect(await gemmaResponse.json()).toMatchObject({ choices: [{ text: "bb" }] });
      expect(await qwenResponse.json()).toMatchObject({ choices: [{ text: "cc" }] });
    } finally {
      running.stop();
    }

    expect(gemma.disposeCount).toBe(1);
    expect(qwen.disposeCount).toBe(1);
  });

  test("serves OpenAI chat image content through a loaded content adapter", async () => {
    const model = new PreparedGeneratingModel();
    const running = serveLoadedModel({
      model,
      tokenizer: new GeneratingTokenizer(),
      modelId: "mlx-community/Qwen3.6-27B-4bit",
      port: 0,
      contentAdapter: {
        async load() {
          return {
            prompt: { text: "user:<image>", tokenIds: [0, 1] },
            preparePrompt() {
              return {
                tokenIds: [0, 1],
                inputEmbeddings: array([[[0], [1]]], "float32"),
              };
            },
          };
        },
      },
    });

    try {
      const response = await fetch(`${running.endpoint}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "mlx-community/Qwen3.6-27B-4bit",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "Describe this." },
                { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } },
              ],
            },
          ],
          max_tokens: 2,
          temperature: 0,
        }),
      });

      expect(response.status).toBe(200);
      expect(await response.json()).toMatchObject({
        choices: [{ message: { role: "assistant", content: "cc" } }],
      });
      expect(model.forwardedInputEmbeddingShapes).toContainEqual([1, 1, 1]);
    } finally {
      running.stop();
    }
  });

  test("serves OpenAI responses image content through a loaded content adapter", async () => {
    const model = new PreparedGeneratingModel();
    const running = serveLoadedModel({
      model,
      tokenizer: new GeneratingTokenizer(),
      modelId: "mlx-community/Qwen3.6-27B-4bit",
      port: 0,
      contentAdapter: {
        async load() {
          return {
            prompt: { text: "user:<image>", tokenIds: [0, 1] },
            preparePrompt() {
              return {
                tokenIds: [0, 1],
                inputEmbeddings: array([[[0], [1]]], "float32"),
              };
            },
          };
        },
      },
    });

    try {
      const response = await fetch(`${running.endpoint}/v1/responses`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "mlx-community/Qwen3.6-27B-4bit",
          input: [
            {
              role: "user",
              content: [
                { type: "input_text", text: "Describe this." },
                { type: "input_image", image_url: "data:image/png;base64,AA==" },
              ],
            },
          ],
          max_output_tokens: 2,
          temperature: 0,
        }),
      });

      expect(response.status).toBe(200);
      expect(await response.json()).toMatchObject({
        object: "response",
        output_text: "cc",
      });
      expect(model.forwardedInputEmbeddingShapes).toContainEqual([1, 1, 1]);
    } finally {
      running.stop();
    }
  });

  test("validates operator-facing server options before binding", () => {
    const model = new FakeModel();
    const tokenizer = new FakeTokenizer();

    expect(() =>
      serveLoadedModels({
        models: [],
      }),
    ).toThrow("models must contain at least one loaded model.");
    expect(() =>
      serveLoadedModels({
        models: [
          { model, tokenizer, modelId: "duplicate" },
          { model, tokenizer, modelId: "duplicate" },
        ],
      }),
    ).toThrow('modelId "duplicate" is duplicated.');
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
        modelId: "bad-prompt-limit",
        maxPromptTokens: 0,
      }),
    ).toThrow("maxPromptTokens must be a positive integer.");
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
    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "bad-batch-size",
        maxBatchSize: 0,
      }),
    ).toThrow("maxBatchSize must be a positive integer.");
    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "bad-batch-window",
        batchWindowMs: -1,
      }),
    ).toThrow("batchWindowMs must be a non-negative integer.");
    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "bad-active-prefill-step",
        activePrefillStepSize: 0,
      }),
    ).toThrow("activePrefillStepSize must be a positive integer.");
    expect(() =>
      serveLoadedModel({
        model: new FakeModel(),
        tokenizer: new FakeTokenizer(),
        modelId: "bad-prefill-step",
        prefillStepSize: 0,
      }),
    ).toThrow("prefillStepSize must be a positive integer.");
    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "bad-active-decode-quantum",
        activeDecodeStepsPerPrefillChunk: 0,
      }),
    ).toThrow("activeDecodeStepsPerPrefillChunk must be a positive integer.");
    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "bad-stream-decode",
        streamDecodeInterval: 0,
      }),
    ).toThrow("streamDecodeInterval must be a positive integer.");
    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "bad-concurrency",
        maxConcurrentRequests: 0,
      }),
    ).toThrow("maxConcurrentRequests must be a positive integer.");
    expect(() =>
      serveLoadedModel({
        model,
        tokenizer,
        modelId: "bad-memory-budget",
        gpuMemoryUtilization: 1.5,
      }),
    ).toThrow("gpuMemoryUtilization must be a number greater than 0");
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
          `serve:${options.modelId}:${options.apiKey}:${options.maxPromptTokens}:${options.maxTotalTokens}:${options.maxBatchSize}:${options.batchWindowMs}:${options.prefillStepSize}:${options.activePrefillStepSize}:${options.activeDecodeStepsPerPrefillChunk}:${options.streamDecodeInterval}:${options.maxConcurrentRequests}:${options.gpuMemoryUtilization}:${options.disposeModelOnStop}`,
        );
        return fakeRunningServer(options.modelId);
      },
    };

    const running = await serveModelWithRuntime(
      {
        source: "repo/model",
        modelId: "mlx-community/Qwen3.6-27B-4bit",
        revision: "main",
        apiKey: "secret",
        localFilesOnly: true,
        maxBatchSize: 16,
        batchWindowMs: 3,
        prefillStepSize: 1024,
        activePrefillStepSize: 64,
        activeDecodeStepsPerPrefillChunk: 7,
        streamDecodeInterval: 2,
        maxConcurrentRequests: 2,
        gpuMemoryUtilization: 0.8,
      },
      runtime,
    );

    expect(running.modelId).toBe("mlx-community/Qwen3.6-27B-4bit");
    expect(calls).toEqual([
      "resolve:repo/model:main:true",
      "model:/snapshots/qwen",
      "tokenizer:/snapshots/qwen",
      "profile:/snapshots/qwen",
      "serve:mlx-community/Qwen3.6-27B-4bit:secret:4096:4096:16:3:1024:64:7:2:2:0.8:true",
    ]);
    expect(model.disposeCount).toBe(0);
    running.stop();
  });

  test("loads single Qwen conditional checkpoints with an image content adapter", async () => {
    const model = new FakeModel();
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
    const runtime: ServeModelRuntime = {
      async resolvePretrainedSource(source, options) {
        calls.push(`resolve:${source}:${options?.localFilesOnly}`);
        return directory;
      },
      async loadCausalLM() {
        throw new Error("single Qwen conditional checkpoints should not use loadCausalLM");
      },
      async loadQwen3_5ForConditionalGeneration(source) {
        calls.push(`qwen-conditional-model:${source}`);
        return model;
      },
      async loadQwen3_5VisionPreprocessor(source) {
        calls.push(`qwen-preprocessor:${source}`);
        return qwenPreprocessorConfig;
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
          `serve:${options.modelId}:${options.contentAdapter === undefined ? "no" : "yes"}`,
        );
        return fakeRunningServer(options.modelId);
      },
    };

    try {
      const running = await serveModelWithRuntime(
        {
          source: "repo/qwen",
          modelId: "mlx-community/Qwen3.6-27B-4bit",
          localFilesOnly: true,
        },
        runtime,
      );

      expect(running.modelId).toBe("mlx-community/Qwen3.6-27B-4bit");
      expect(calls).toEqual([
        "resolve:repo/qwen:true",
        `qwen-conditional-model:${directory}`,
        `tokenizer:${directory}`,
        `profile:${directory}`,
        `qwen-preprocessor:${directory}`,
        "serve:mlx-community/Qwen3.6-27B-4bit:yes",
      ]);
      running.stop();
    } finally {
      removeDirectory(directory);
    }
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
    modelIds: [modelId],
    server,
    stop() {
      server.stop(true);
    },
    [Symbol.dispose]() {
      server.stop(true);
    },
  };
}

function createQwenConditionalDirectory(): string {
  const directory = `${Bun.env.TMPDIR ?? "/tmp"}/mlxts-single-qwen-${crypto.randomUUID()}`;
  const result = Bun.spawnSync(["mkdir", "-p", directory]);
  if (result.exitCode !== 0) {
    throw new Error("failed to create temporary Qwen checkpoint directory");
  }
  return directory;
}

function removeDirectory(path: string): void {
  Bun.spawnSync(["rm", "-rf", path]);
}

const qwenPreprocessorConfig: Qwen3_5VisionPreprocessorConfig = {
  size: {
    shortestEdge: 56,
    longestEdge: 256,
  },
  patchSize: 14,
  temporalPatchSize: 2,
  mergeSize: 2,
  imageMean: [0, 0, 0],
  imageStd: [1, 1, 1],
  processorClass: null,
  imageProcessorType: null,
};
