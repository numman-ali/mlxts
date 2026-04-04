import { describe, expect, test } from "bun:test";
import {
  applyCheckpoint,
  CharTokenizer,
  estimateParameterCount,
  GPT,
  GPT_SMALL,
  GPT_TINY,
  loadCheckpoint,
  loadModelSafetensors,
  resolveConfig,
  saveCheckpoint,
  saveModelSafetensors,
  VERSION,
} from "./index";

describe("nanogpt public API", () => {
  test("VERSION is defined", () => {
    expect(VERSION).toBe("0.0.1");
  });

  test("GPT class is exported", () => {
    expect(GPT).toBeDefined();
  });

  test("presets are exported", () => {
    expect(GPT_TINY).toBeDefined();
    expect(GPT_SMALL).toBeDefined();
  });

  test("CharTokenizer is exported", () => {
    expect(CharTokenizer).toBeDefined();
  });

  test("resolveConfig is exported", () => {
    const config = resolveConfig(GPT_TINY, 65);
    expect(config.vocabSize).toBe(65);
  });

  test("estimateParameterCount is exported", () => {
    const config = resolveConfig(GPT_TINY, 65);
    expect(estimateParameterCount(config)).toBeGreaterThan(0);
  });

  test("checkpoint helpers are exported", () => {
    expect(saveCheckpoint).toBeDefined();
    expect(loadCheckpoint).toBeDefined();
    expect(applyCheckpoint).toBeDefined();
  });

  test("safetensors helpers are exported", () => {
    expect(saveModelSafetensors).toBeDefined();
    expect(loadModelSafetensors).toBeDefined();
  });
});
