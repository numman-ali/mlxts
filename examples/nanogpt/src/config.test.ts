import { describe, expect, test } from "bun:test";
import { estimateParameterCount, GPT_SMALL, GPT_TINY, resolveConfig } from "./config";

describe("GPTConfig", () => {
  test("GPT_TINY preset has valid values", () => {
    expect(GPT_TINY.nLayer).toBeGreaterThan(0);
    expect(GPT_TINY.nHead).toBeGreaterThan(0);
    expect(GPT_TINY.nEmbd).toBeGreaterThan(0);
    expect(GPT_TINY.nEmbd % GPT_TINY.nHead).toBe(0);
    expect(GPT_TINY.blockSize).toBeGreaterThan(0);
    expect(GPT_TINY.dropout).toBeGreaterThanOrEqual(0);
    expect(GPT_TINY.dropout).toBeLessThan(1);
    expect(GPT_TINY.gradientCheckpointing).toBe(false);
  });

  test("GPT_SMALL preset has valid values", () => {
    expect(GPT_SMALL.nEmbd % GPT_SMALL.nHead).toBe(0);
    expect(GPT_SMALL.gradientCheckpointing).toBe(true);
  });

  test("estimateParameterCount matches the tiny formula", () => {
    const config = resolveConfig(GPT_TINY, 65);
    expect(estimateParameterCount(config)).toBe(10_770_816);
  });

  test("GPT_SMALL has more parameters than GPT_TINY for the same vocab", () => {
    const tiny = resolveConfig(GPT_TINY, 65);
    const small = resolveConfig(GPT_SMALL, 65);
    expect(estimateParameterCount(small)).toBeGreaterThan(estimateParameterCount(tiny));
  });

  test("resolveConfig merges vocabSize", () => {
    const config = resolveConfig(GPT_TINY, 65);
    expect(config.vocabSize).toBe(65);
    expect(config.nLayer).toBe(GPT_TINY.nLayer);
    expect(config.gradientCheckpointing).toBe(false);
  });

  test("resolveConfig rejects nEmbd not divisible by nHead", () => {
    expect(() => resolveConfig({ ...GPT_TINY, nEmbd: 385 }, 65)).toThrow("divisible");
  });

  test("resolveConfig rejects invalid vocabSize", () => {
    expect(() => resolveConfig(GPT_TINY, 0)).toThrow("vocabSize");
  });

  test("resolveConfig rejects negative nLayer", () => {
    expect(() => resolveConfig({ ...GPT_TINY, nLayer: 0 }, 65)).toThrow("nLayer");
  });

  test("resolveConfig rejects dropout >= 1", () => {
    expect(() => resolveConfig({ ...GPT_TINY, dropout: 1.0 }, 65)).toThrow("dropout");
  });

  test("resolveConfig rejects non-boolean gradientCheckpointing", () => {
    const badPreset = { ...GPT_TINY };
    Reflect.set(badPreset, "gradientCheckpointing", undefined);
    expect(() => resolveConfig(badPreset, 65)).toThrow("gradientCheckpointing");
  });
});
