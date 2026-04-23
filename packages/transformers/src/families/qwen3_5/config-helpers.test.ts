import { describe, expect, test } from "bun:test";

import {
  expectPositiveInteger,
  optionalIntegerOrNull,
  parseEosTokenId,
  parseIntegerArrayField,
  parseIntegerOrIntegerArray,
  parseLayerTypes,
  parseTextRopeParameters,
} from "./config-helpers";

describe("Qwen 3.5 config helpers", () => {
  test("validate integer helper fields and optional token ids", () => {
    expect(expectPositiveInteger(4, "field")).toBe(4);
    expect(() => expectPositiveInteger(0, "field")).toThrow("field must be positive");

    expect(
      parseIntegerArrayField(
        { deepstack_visual_indexes: [1, 3] },
        "deepstack_visual_indexes",
        "config",
      ),
    ).toEqual([1, 3]);
    expect(
      parseIntegerArrayField(
        { deepstack_visual_indexes: [] },
        "deepstack_visual_indexes",
        "config",
        true,
      ),
    ).toEqual([]);
    expect(parseIntegerArrayField({}, "deepstack_visual_indexes", "config")).toBeUndefined();
    expect(() =>
      parseIntegerArrayField(
        { deepstack_visual_indexes: ["x"] },
        "deepstack_visual_indexes",
        "config",
      ),
    ).toThrow("must be an integer");

    expect(parseIntegerOrIntegerArray({ patch_size: 2 }, "patch_size", "config")).toBe(2);
    expect(parseIntegerOrIntegerArray({ patch_size: [2, 4] }, "patch_size", "config")).toEqual([
      2, 4,
    ]);
    expect(() => parseIntegerOrIntegerArray({ patch_size: 0 }, "patch_size", "config")).toThrow(
      "must be positive",
    );

    expect(optionalIntegerOrNull({ bos_token_id: 1 }, "bos_token_id", "config")).toBe(1);
    expect(optionalIntegerOrNull({ bos_token_id: null }, "bos_token_id", "config")).toBeNull();
    expect(() => optionalIntegerOrNull({ bos_token_id: "x" }, "bos_token_id", "config")).toThrow(
      "must be an integer when present",
    );
  });

  test("parse eos ids, layer patterns, and rope defaults", () => {
    expect(parseEosTokenId({ eos_token_id: 2 }, "config")).toBe(2);
    expect(parseEosTokenId({ eos_token_id: [2, 3] }, "config")).toEqual([2, 3]);
    expect(parseEosTokenId({ eos_token_id: null }, "config")).toBeNull();
    expect(() => parseEosTokenId({ eos_token_id: "x" }, "config")).toThrow(
      "must be an integer or integer array",
    );

    expect(parseLayerTypes({}, 4, "config")).toEqual({
      layerTypes: ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
      fullAttentionInterval: 4,
    });
    expect(
      parseLayerTypes(
        {
          full_attention_interval: 2,
          layer_types: ["linear_attention", "full_attention", "linear_attention", "full_attention"],
        },
        4,
        "config",
      ),
    ).toEqual({
      layerTypes: ["linear_attention", "full_attention", "linear_attention", "full_attention"],
      fullAttentionInterval: 2,
    });
    expect(() =>
      parseLayerTypes(
        {
          full_attention_interval: 2,
          layer_types: ["full_attention", "linear_attention", "linear_attention", "full_attention"],
        },
        4,
        "config",
      ),
    ).toThrow("must match full_attention_interval=2");
    expect(() => parseLayerTypes({ layer_types: ["weird"] }, 1, "config")).toThrow(
      'must be "linear_attention" or "full_attention"',
    );

    expect(parseTextRopeParameters({}, "config")).toEqual({
      ropeType: "default",
      ropeTheta: 10_000_000,
      partialRotaryFactor: 0.25,
      mropeSection: [11, 11, 10],
      mropeInterleaved: true,
    });
    expect(
      parseTextRopeParameters(
        {
          rope_theta: 5000,
          partial_rotary_factor: 0.5,
          rope_parameters: {
            rope_type: "default",
            mrope_section: [3, 3, 2],
            mrope_interleaved: false,
          },
        },
        "config",
      ),
    ).toEqual({
      ropeType: "default",
      ropeTheta: 5000,
      partialRotaryFactor: 0.5,
      mropeSection: [3, 3, 2],
      mropeInterleaved: false,
    });
  });
});
