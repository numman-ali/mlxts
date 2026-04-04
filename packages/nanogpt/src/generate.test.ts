import { describe, expect, test } from "bun:test";
import { array, random, reshape } from "mlx-ts";
import { GPT_TINY, resolveConfig } from "./config";
import { generate, generateTokens } from "./generate";
import { GPT } from "./model/gpt";
import { initializeGPT } from "./model/init";
import { CharTokenizer } from "./tokenizer";

const TEXT = "abcdefghijklmnopqrstuvwxyz ";
const TOKENIZER = CharTokenizer.fromText(TEXT);
const TEST_CONFIG = resolveConfig(
  { ...GPT_TINY, nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 16, dropout: 0 },
  TOKENIZER.vocabSize,
);

function createModel(): GPT {
  random.seed(42);
  const model = new GPT(TEST_CONFIG);
  initializeGPT(model, TEST_CONFIG);
  return model;
}

describe("generate", () => {
  test("generates valid text with temperature > 0", () => {
    const model = createModel();
    model.eval();

    try {
      const result = generate(model, TEST_CONFIG, TOKENIZER, "abc", {
        maxNewTokens: 10,
        temperature: 0.8,
      });

      expect(result.startsWith("abc")).toBe(true);
      expect(result.length).toBe(13);
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("greedy generation is deterministic", () => {
    const model = createModel();
    model.eval();

    try {
      const first = generate(model, TEST_CONFIG, TOKENIZER, "abc", {
        maxNewTokens: 5,
        temperature: 0,
      });
      const second = generate(model, TEST_CONFIG, TOKENIZER, "abc", {
        maxNewTokens: 5,
        temperature: 0,
      });
      expect(first).toBe(second);
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("restores the caller's training mode after token generation", () => {
    const model = createModel();
    model.train();
    const prompt = array([[0, 1, 2]], "int32");

    try {
      const tokens = Array.from(
        generateTokens(model, TEST_CONFIG, prompt, { maxNewTokens: 3, temperature: 0 }),
      );
      expect(tokens).toHaveLength(3);
      expect(model.isTraining).toBe(true);
    } finally {
      prompt.free();
      model[Symbol.dispose]();
    }
  });

  test("crops prompt context to the last blockSize tokens", () => {
    const model = createModel();
    model.eval();
    const longPrompt = TEXT.repeat(2);

    try {
      const generated = generate(model, TEST_CONFIG, TOKENIZER, longPrompt, {
        maxNewTokens: 2,
        temperature: 0,
      });
      expect(generated.startsWith(longPrompt)).toBe(true);
      expect(generated.length).toBe(longPrompt.length + 2);
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("rejects non-int32 prompts", () => {
    const model = createModel();
    const flatPrompt = array([0, 1, 2]);

    try {
      using prompt = reshape(flatPrompt, [1, 3]);
      expect(() =>
        Array.from(generateTokens(model, TEST_CONFIG, prompt, { maxNewTokens: 1, temperature: 0 })),
      ).toThrow("prompt must be int32");
    } finally {
      flatPrompt.free();
      model[Symbol.dispose]();
    }
  });
});
