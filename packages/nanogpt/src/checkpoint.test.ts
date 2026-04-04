import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from "fs";
import { AdamW, full, treeFlatten, treeUnflatten } from "mlx-ts";
import { tmpdir } from "os";
import { join } from "path";
import {
  applyCheckpoint,
  loadCheckpoint,
  restoreAdamWFromCheckpoint,
  saveCheckpoint,
} from "./checkpoint";
import { GPT_TINY, resolveConfig } from "./config";
import { prepareData } from "./data";
import { GPT } from "./model/gpt";
import { initializeGPT } from "./model/init";
import { createDefaultAdamW } from "./optimizer-defaults";
import { CharTokenizer } from "./tokenizer";
import { train } from "./train";

function createCheckpointFixture() {
  const tokenizer = CharTokenizer.fromText("to be or not to be");
  const config = resolveConfig(
    { ...GPT_TINY, nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 16, dropout: 0 },
    tokenizer.vocabSize,
  );
  const model = new GPT(config);
  initializeGPT(model, config);
  return { tokenizer, config, model };
}

function createCheckpointDirectory(name: string): string {
  return mkdtempSync(join(tmpdir(), `${name}-`));
}

describe("checkpoint", () => {
  test("saveCheckpoint writes a canonical checkpoint directory and loadCheckpoint restores it", () => {
    const { tokenizer, config, model } = createCheckpointFixture();
    const directory = createCheckpointDirectory("nanogpt-checkpoint");
    const checkpointPath = join(directory, "step-5");

    try {
      saveCheckpoint({
        model,
        kind: "snapshot",
        config,
        step: 5,
        tokenizer,
        path: checkpointPath,
      });
      expect(existsSync(join(checkpointPath, "manifest.json"))).toBe(true);
      expect(existsSync(join(checkpointPath, "tensors.bin"))).toBe(true);

      const loaded = loadCheckpoint(checkpointPath);
      expect(loaded.version).toBe(2);
      expect(loaded.step).toBe(5);
      expect(loaded.config).toEqual(config);
      expect(loaded.tokenizer.chars).toEqual(tokenizer.vocab);

      const parameterKeys = Object.keys(loaded.parameters);
      expect(parameterKeys.length).toBeGreaterThan(0);
      const firstTensor = loaded.parameters[parameterKeys[0] ?? ""];
      expect(firstTensor?.data.byteLength).toBeGreaterThan(0);
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("saveCheckpoint supports best checkpoints without optimizer state", () => {
    const { tokenizer, config, model } = createCheckpointFixture();
    const directory = createCheckpointDirectory("nanogpt-best-checkpoint");
    const checkpointPath = join(directory, "best");

    try {
      saveCheckpoint({
        model,
        kind: "best",
        config,
        step: 7,
        tokenizer,
        path: checkpointPath,
      });

      const loaded = loadCheckpoint(checkpointPath);
      expect(loaded.kind).toBe("best");
      expect(loaded.step).toBe(7);
      expect(loaded.optimizer).toBeUndefined();
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("saveCheckpoint can replace an existing checkpoint path without leaving it unreadable", () => {
    const { tokenizer, config, model } = createCheckpointFixture();
    const directory = createCheckpointDirectory("nanogpt-checkpoint-replace");
    const checkpointPath = join(directory, "step-5");

    try {
      saveCheckpoint({
        model,
        kind: "snapshot",
        config,
        step: 5,
        tokenizer,
        path: checkpointPath,
      });

      const mutatedEntry = treeFlatten(model.parameters())[0];
      if (mutatedEntry === undefined) {
        throw new Error("expected at least one parameter to mutate");
      }
      const [path, mutatedWeight] = mutatedEntry;
      const replacement = full([...mutatedWeight.shape], 9, mutatedWeight.dtype);
      model.update(treeUnflatten([[path, replacement]]));

      saveCheckpoint({
        model,
        kind: "snapshot",
        config,
        step: 6,
        tokenizer,
        path: checkpointPath,
      });

      const loaded = loadCheckpoint(checkpointPath);
      expect(loaded.step).toBe(6);
      expect(existsSync(join(checkpointPath, "manifest.json"))).toBe(true);
      expect(existsSync(join(checkpointPath, "tensors.bin"))).toBe(true);
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("applyCheckpoint restores parameter values and frees replaced arrays", () => {
    const source = createCheckpointFixture();
    const target = createCheckpointFixture();
    const directory = createCheckpointDirectory("nanogpt-apply");
    const checkpointPath = join(directory, "step-7");

    try {
      saveCheckpoint({
        model: source.model,
        kind: "snapshot",
        config: source.config,
        step: 7,
        tokenizer: source.tokenizer,
        path: checkpointPath,
      });

      const before = treeFlatten(target.model.parameters());
      const loaded = loadCheckpoint(checkpointPath);
      applyCheckpoint(target.model, loaded);
      const after = treeFlatten(target.model.parameters());

      expect(before).toHaveLength(after.length);
      for (let index = 0; index < before.length; index++) {
        expect(before[index]?.[1].isDisposed).toBe(true);
        expect(before[index]?.[0]).toEqual(after[index]?.[0]);
      }

      const expected = treeFlatten(source.model.parameters());
      for (let index = 0; index < expected.length; index++) {
        expect(expected[index]?.[1].toList()).toEqual(after[index]?.[1].toList());
      }
    } finally {
      source.model[Symbol.dispose]();
      target.model[Symbol.dispose]();
    }
  });

  test("applyCheckpoint rejects dtype mismatches", () => {
    const { tokenizer, config, model } = createCheckpointFixture();
    const target = createCheckpointFixture();
    const directory = createCheckpointDirectory("nanogpt-checkpoint-dtype");
    const checkpointPath = join(directory, "step-1");

    try {
      saveCheckpoint({
        model,
        kind: "snapshot",
        config,
        step: 1,
        tokenizer,
        path: checkpointPath,
      });
      const loaded = loadCheckpoint(checkpointPath);
      const firstKey = Object.keys(loaded.parameters)[0];
      if (firstKey === undefined) {
        throw new Error("checkpoint test fixture produced no parameters");
      }
      const original = loaded.parameters[firstKey];
      if (original === undefined) {
        throw new Error(`checkpoint parameter "${firstKey}" was unexpectedly unavailable`);
      }
      const mutatedParameters: typeof loaded.parameters = {
        ...loaded.parameters,
        [firstKey]: {
          shape: [...original.shape],
          dtype: "float64",
          data: Uint8Array.from(original.data),
        },
      };

      const mutated = {
        ...loaded,
        parameters: mutatedParameters,
      };

      expect(() => applyCheckpoint(target.model, mutated)).toThrow("dtype mismatch");
    } finally {
      model[Symbol.dispose]();
      target.model[Symbol.dispose]();
    }
  });

  test("applyCheckpoint rejects shape and key mismatches", () => {
    const { tokenizer, config, model } = createCheckpointFixture();
    const target = createCheckpointFixture();
    const directory = createCheckpointDirectory("nanogpt-checkpoint-shape");
    const checkpointPath = join(directory, "step-2");

    try {
      saveCheckpoint({
        model,
        kind: "snapshot",
        config,
        step: 2,
        tokenizer,
        path: checkpointPath,
      });
      const loaded = loadCheckpoint(checkpointPath);
      const firstKey = Object.keys(loaded.parameters)[0];
      if (firstKey === undefined) {
        throw new Error("checkpoint test fixture produced no parameters");
      }
      const original = loaded.parameters[firstKey];
      if (original === undefined) {
        throw new Error(`checkpoint parameter "${firstKey}" was unexpectedly unavailable`);
      }

      expect(() =>
        applyCheckpoint(target.model, {
          ...loaded,
          parameters: {
            ...loaded.parameters,
            [firstKey]: {
              shape: [...original.shape, 1],
              dtype: original.dtype,
              data: Uint8Array.from(original.data),
            },
          },
        }),
      ).toThrow("shape mismatch");

      expect(() =>
        applyCheckpoint(target.model, {
          ...loaded,
          parameters: {
            ...loaded.parameters,
            rogue: {
              shape: [1],
              dtype: "float32",
              data: new Uint8Array(4),
            },
          },
        }),
      ).toThrow("unexpected checkpoint parameter");
    } finally {
      model[Symbol.dispose]();
      target.model[Symbol.dispose]();
    }
  });

  test("loadCheckpoint rejects unsupported manifest versions", () => {
    const directory = createCheckpointDirectory("nanogpt-bad-checkpoint");
    const checkpointPath = join(directory, "bad");
    mkdirSync(checkpointPath, { recursive: true });
    const manifestPath = join(checkpointPath, "manifest.json");
    const tensorPath = join(checkpointPath, "tensors.bin");

    writeFileSync(
      manifestPath,
      JSON.stringify({
        version: 1,
        step: 0,
        config: {
          nLayer: 1,
          nHead: 1,
          nEmbd: 1,
          blockSize: 1,
          dropout: 0,
          gradientCheckpointing: false,
          vocabSize: 1,
        },
        tokenizer: { chars: ["a"] },
        parameters: {},
      }),
      "utf-8",
    );
    writeFileSync(tensorPath, new Uint8Array(0));

    expect(() => loadCheckpoint(checkpointPath)).toThrow("unsupported");
  });

  test("loadCheckpoint rejects unsupported kinds and malformed manifest sections", () => {
    const directory = createCheckpointDirectory("nanogpt-bad-manifest");
    const checkpointPath = join(directory, "bad");
    mkdirSync(checkpointPath, { recursive: true });

    const baseManifest = {
      version: 2,
      kind: "snapshot",
      step: 0,
      config: {
        nLayer: 1,
        nHead: 1,
        nEmbd: 1,
        blockSize: 1,
        dropout: 0,
        gradientCheckpointing: false,
        vocabSize: 1,
      },
      tokenizer: { chars: ["a"] },
      parameters: {},
    };

    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({ ...baseManifest, kind: "legacy" }),
      "utf-8",
    );
    writeFileSync(join(checkpointPath, "tensors.bin"), new Uint8Array(0));
    expect(() => loadCheckpoint(checkpointPath)).toThrow('kind "legacy" is unsupported');

    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({ ...baseManifest, tokenizer: [] }),
      "utf-8",
    );
    expect(() => loadCheckpoint(checkpointPath)).toThrow("tokenizer: expected an object");

    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({ ...baseManifest, config: "bad-config" }),
      "utf-8",
    );
    expect(() => loadCheckpoint(checkpointPath)).toThrow("config: expected an object");
  });

  test("loadCheckpoint rejects tensor metadata that exceeds tensors.bin", () => {
    const directory = createCheckpointDirectory("nanogpt-short-tensor");
    const checkpointPath = join(directory, "bad");
    mkdirSync(checkpointPath, { recursive: true });
    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({
        version: 2,
        kind: "snapshot",
        step: 0,
        config: {
          nLayer: 1,
          nHead: 1,
          nEmbd: 1,
          blockSize: 1,
          dropout: 0,
          gradientCheckpointing: false,
          vocabSize: 1,
        },
        tokenizer: { chars: ["a"] },
        parameters: {
          weight: { shape: [1], dtype: "float32", offset: 0, byteLength: 4 },
        },
      }),
      "utf-8",
    );
    writeFileSync(join(checkpointPath, "tensors.bin"), new Uint8Array(2));

    expect(() => loadCheckpoint(checkpointPath)).toThrow("exceeds tensors.bin size");
  });

  test("loadCheckpoint enforces canonical snapshot and resume manifest rules", () => {
    const directory = createCheckpointDirectory("nanogpt-manifest-rules");
    const checkpointPath = join(directory, "bad");
    mkdirSync(checkpointPath, { recursive: true });

    const baseManifest = {
      version: 2,
      kind: "resume",
      step: 3,
      config: {
        nLayer: 1,
        nHead: 1,
        nEmbd: 1,
        blockSize: 1,
        dropout: 0,
        gradientCheckpointing: false,
        vocabSize: 1,
      },
      tokenizer: { chars: ["a"] },
      parameters: {},
    };

    writeFileSync(join(checkpointPath, "tensors.bin"), new Uint8Array(0));

    writeFileSync(join(checkpointPath, "manifest.json"), JSON.stringify(baseManifest), "utf-8");
    expect(() => loadCheckpoint(checkpointPath)).toThrow("resume checkpoints require optimizer");

    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({
        ...baseManifest,
        kind: "snapshot",
        optimizer: {
          kind: "adamw",
          step: 3,
          lr: 1e-3,
          beta1: 0.9,
          beta2: 0.95,
          eps: 1e-8,
          weightDecay: 0.1,
          state: {},
        },
      }),
      "utf-8",
    );
    expect(() => loadCheckpoint(checkpointPath)).toThrow(
      "snapshot/best checkpoints must not include optimizer",
    );

    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({
        ...baseManifest,
        optimizer: {
          kind: "adamw",
          step: 2,
          lr: 1e-3,
          beta1: 0.9,
          beta2: 0.95,
          eps: 1e-8,
          weightDecay: 0.1,
          state: {},
        },
      }),
      "utf-8",
    );
    expect(() => loadCheckpoint(checkpointPath)).toThrow(
      "optimizer.step 2 does not match checkpoint step 3",
    );

    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({
        ...baseManifest,
        optimizer: {
          kind: "sgd",
          step: 3,
          lr: 1e-3,
          beta1: 0.9,
          beta2: 0.95,
          eps: 1e-8,
          weightDecay: 0.1,
          state: {},
        },
      }),
      "utf-8",
    );
    expect(() => loadCheckpoint(checkpointPath)).toThrow('optimizer.kind: expected "adamw"');
  });

  test("saveCheckpoint enforces snapshot/resume optimizer contracts", () => {
    const { tokenizer, config, model } = createCheckpointFixture();
    const directory = createCheckpointDirectory("nanogpt-save-rules");
    const checkpointPath = join(directory, "step-1");
    const optimizer = new AdamW();

    try {
      expect(() =>
        saveCheckpoint({
          model,
          kind: "resume",
          config,
          step: 1,
          tokenizer,
          path: checkpointPath,
        }),
      ).toThrow("resume checkpoints require optimizer state");

      expect(() =>
        saveCheckpoint({
          model,
          optimizer,
          kind: "snapshot",
          config,
          step: 1,
          tokenizer,
          path: checkpointPath,
        }),
      ).toThrow("snapshot/best checkpoints must not include optimizer state");

      optimizer.restore({
        kind: "adamw",
        step: 3,
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weightDecay: 0.01,
        state: {},
      });
      expect(() =>
        saveCheckpoint({
          model,
          optimizer,
          kind: "resume",
          config,
          step: 1,
          tokenizer,
          path: checkpointPath,
        }),
      ).toThrow("optimizer step 3 does not match checkpoint step 1");
    } finally {
      model[Symbol.dispose]();
      optimizer[Symbol.dispose]();
    }
  });

  test("optimizer state round-trips through the canonical checkpoint format", () => {
    const text = "abcdefghijklmnopqrstuvwxyz ".repeat(120);
    const tokenizer = CharTokenizer.fromText(text);
    const config = resolveConfig(
      { ...GPT_TINY, nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 16, dropout: 0 },
      tokenizer.vocabSize,
    );
    const model = new GPT(config);
    initializeGPT(model, config);
    const directory = createCheckpointDirectory("nanogpt-checkpoint-optimizer");
    const checkpointPath = join(directory, "step-1");
    const optimizer = createDefaultAdamW(1e-3, 0.1);

    try {
      const { trainTokens, valTokens } = prepareData(tokenizer.encode(text), 0.9);
      train({
        model,
        optimizer,
        config,
        trainTokens,
        valTokens,
        trainConfig: {
          maxSteps: 1,
          batchSize: 1,
          learningRate: 1e-3,
          weightDecay: 0.1,
          warmupSteps: 0,
          minLearningRate: 1e-4,
          gradAccumSteps: 1,
          evalInterval: 1,
          evalSteps: 1,
          logInterval: 1,
          maxGradNorm: 1,
          seed: 42,
        },
      });

      saveCheckpoint({
        model,
        optimizer,
        kind: "resume",
        config,
        step: 1,
        tokenizer,
        path: checkpointPath,
      });

      const loaded = loadCheckpoint(checkpointPath);
      expect(loaded.optimizer?.kind).toBe("adamw");
      expect(loaded.optimizer?.step).toBe(1);
      const optimizerState = loaded.optimizer;
      if (optimizerState === undefined) {
        throw new Error("expected checkpoint optimizer state");
      }

      const restored = restoreAdamWFromCheckpoint(optimizerState);
      try {
        expect(restored.step).toBe(1);
        expect(restored.stateArrays().length).toBeGreaterThan(0);
      } finally {
        restored[Symbol.dispose]();
      }
    } finally {
      optimizer[Symbol.dispose]();
      model[Symbol.dispose]();
    }
  });

  test("saveCheckpoint enforces snapshot vs resume checkpoint kinds", () => {
    const { tokenizer, config, model } = createCheckpointFixture();
    const optimizer = createDefaultAdamW(1e-3, 0.1);
    const directory = createCheckpointDirectory("nanogpt-checkpoint-kinds");

    try {
      expect(() =>
        saveCheckpoint({
          model,
          optimizer,
          kind: "snapshot",
          config,
          step: 1,
          tokenizer,
          path: join(directory, "snapshot-with-optimizer"),
        }),
      ).toThrow("snapshot/best checkpoints must not");

      expect(() =>
        saveCheckpoint({
          model,
          kind: "resume",
          config,
          step: 1,
          tokenizer,
          path: join(directory, "resume-without-optimizer"),
        }),
      ).toThrow("resume checkpoints require");
    } finally {
      optimizer[Symbol.dispose]();
      model[Symbol.dispose]();
    }
  });

  test("loadCheckpoint rejects invalid checkpoint kind and inconsistent optimizer metadata", () => {
    const directory = createCheckpointDirectory("nanogpt-invalid-manifest");
    const checkpointPath = join(directory, "bad");
    mkdirSync(checkpointPath, { recursive: true });
    writeFileSync(join(checkpointPath, "tensors.bin"), new Uint8Array(0));

    const baseManifest = {
      version: 2,
      step: 0,
      config: {
        nLayer: 1,
        nHead: 1,
        nEmbd: 1,
        blockSize: 1,
        dropout: 0,
        gradientCheckpointing: false,
        vocabSize: 1,
      },
      tokenizer: { chars: ["a"] },
      parameters: {},
    };

    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({ ...baseManifest, kind: "mystery" }),
      "utf-8",
    );
    expect(() => loadCheckpoint(checkpointPath)).toThrow("kind");

    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({ ...baseManifest, kind: "resume" }),
      "utf-8",
    );
    expect(() => loadCheckpoint(checkpointPath)).toThrow("require optimizer metadata");

    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({
        ...baseManifest,
        kind: "snapshot",
        optimizer: {
          kind: "adamw",
          step: 0,
          lr: 1e-3,
          beta1: 0.9,
          beta2: 0.999,
          eps: 1e-8,
          weightDecay: 0.1,
          state: {},
        },
      }),
      "utf-8",
    );
    expect(() => loadCheckpoint(checkpointPath)).toThrow("must not include optimizer");

    writeFileSync(
      join(checkpointPath, "manifest.json"),
      JSON.stringify({
        ...baseManifest,
        kind: "resume",
        step: 2,
        optimizer: {
          kind: "adamw",
          step: 1,
          lr: 1e-3,
          beta1: 0.9,
          beta2: 0.999,
          eps: 1e-8,
          weightDecay: 0.1,
          state: {},
        },
      }),
      "utf-8",
    );
    expect(() => loadCheckpoint(checkpointPath)).toThrow("does not match checkpoint step");
  });
});
