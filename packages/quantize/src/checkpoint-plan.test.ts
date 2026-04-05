import { describe, expect, test } from "bun:test";

import {
  registerQuantizedCheckpointProvider,
  resolveCheckpointQuantizationPlan,
  translateCheckpointQuantizationPlanPaths,
} from "./checkpoint-plan";
import { resolveQuantizationParameters } from "./parameters";

describe("resolveCheckpointQuantizationPlan", () => {
  test("parses root quantization defaults", () => {
    const plan = resolveCheckpointQuantizationPlan({
      quantization: { group_size: 64, bits: 4, mode: "affine" },
    });

    expect(plan).not.toBeNull();
    expect(plan?.provider).toBe("mlx");
    expect(plan?.defaults).toEqual({
      groupSize: 64,
      bits: 4,
      mode: "affine",
    });
    expect(plan?.explicitOnly).toBe(false);
  });

  test("parses explicit per-path rules", () => {
    const plan = resolveCheckpointQuantizationPlan({
      quantization: {
        bits: 4,
        group_size: 64,
        "model.layers.0.self_attn.q_proj": { bits: 8 },
        "model.layers.0.self_attn.k_proj": false,
      },
    });

    expect(plan?.explicitOnly).toBe(true);
    expect(plan?.rules).toEqual([
      {
        path: "model.layers.0.self_attn.q_proj",
        enabled: true,
        params: { bits: 8 },
      },
      {
        path: "model.layers.0.self_attn.k_proj",
        enabled: false,
      },
    ]);
  });

  test("reads text_config fallback metadata", () => {
    const plan = resolveCheckpointQuantizationPlan({
      text_config: {
        quantization_config: {
          quant_method: "mxfp4",
        },
      },
    });

    expect(plan?.provider).toBe("mlx");
    expect(plan?.defaults).toEqual({
      groupSize: 32,
      bits: 4,
      mode: "mxfp4",
    });
  });

  test("keeps unsupported legacy providers explicit", () => {
    const plan = resolveCheckpointQuantizationPlan({
      quantization_config: {
        quant_method: "awq",
        group_size: 128,
        bits: 4,
      },
    });

    expect(plan?.provider).toBe("awq");
    expect(plan?.defaults.groupSize).toBe(128);
  });

  test("parses compressed-tensors legacy configs as mlx affine defaults", () => {
    const plan = resolveCheckpointQuantizationPlan({
      quantization_config: {
        quant_method: "compressed-tensors",
      },
    });

    expect(plan?.provider).toBe("mlx");
    expect(plan?.defaults).toEqual({
      groupSize: 32,
      bits: 4,
      mode: "affine",
    });
  });

  test("rejects invalid quantization modes", () => {
    expect(() =>
      resolveCheckpointQuantizationPlan({
        quantization: {
          mode: "bad-mode",
        },
      }),
    ).toThrow("must be one of affine");
  });

  test("returns null when no quantization metadata is present", () => {
    expect(resolveCheckpointQuantizationPlan({})).toBeNull();
    expect(resolveCheckpointQuantizationPlan({ quantization: [] })).toBeNull();
  });

  test("reads text_config quantization metadata", () => {
    const plan = resolveCheckpointQuantizationPlan({
      text_config: {
        quantization: {
          bits: 8,
          mode: "mxfp8",
        },
      },
    });

    expect(plan?.defaults).toEqual({
      groupSize: 32,
      bits: 8,
      mode: "mxfp8",
    });
  });

  test("keeps legacy gptq providers explicit", () => {
    const plan = resolveCheckpointQuantizationPlan({
      quantization_config: {
        quant_method: "gptq",
        group_size: 128,
        bits: 4,
      },
    });

    expect(plan?.provider).toBe("gptq");
  });

  test("allows custom providers to participate in resolution", () => {
    registerQuantizedCheckpointProvider({
      name: "test-provider",
      resolve(config) {
        if (config.custom_quant === true) {
          return {
            provider: "custom",
            sourceKey: "quantization",
            defaults: { groupSize: 64, bits: 4, mode: "affine" },
            explicitOnly: false,
            rules: [],
          };
        }
        return null;
      },
    });

    const plan = resolveCheckpointQuantizationPlan({ custom_quant: true });
    expect(plan?.provider).toBe("custom");
  });
});

describe("translateCheckpointQuantizationPlanPaths", () => {
  test("translates explicit paths", () => {
    const plan = resolveCheckpointQuantizationPlan({
      quantization: {
        bits: 4,
        group_size: 64,
        "model.layers.0.self_attn.q_proj": true,
      },
    });
    if (plan === null) {
      throw new Error("expected plan");
    }

    const translated = translateCheckpointQuantizationPlanPaths(plan, (path) =>
      path === "model.layers.0.self_attn.q_proj"
        ? "model.layers.0.selfAttention.qProjection"
        : null,
    );
    expect(translated.rules).toEqual([
      {
        path: "model.layers.0.selfAttention.qProjection",
        enabled: true,
      },
    ]);
  });

  test("throws when an enabled path cannot be translated", () => {
    const plan = resolveCheckpointQuantizationPlan({
      quantization: {
        bits: 4,
        group_size: 64,
        "model.layers.0.self_attn.q_proj": true,
      },
    });
    if (plan === null) {
      throw new Error("expected plan");
    }

    expect(() => translateCheckpointQuantizationPlanPaths(plan, () => null)).toThrow(
      "could not translate",
    );
  });

  test("throws when translated rule paths collide", () => {
    const plan = resolveCheckpointQuantizationPlan({
      quantization: {
        bits: 4,
        group_size: 64,
        "model.layers.0.self_attn.q_proj": true,
        "model.layers.0.self_attn.k_proj": true,
      },
    });
    if (plan === null) {
      throw new Error("expected plan");
    }

    expect(() =>
      translateCheckpointQuantizationPlanPaths(plan, () => "model.layers.0.sharedProjection"),
    ).toThrow("duplicate quantization rule");
  });

  test("drops disabled rules that do not translate", () => {
    const plan = resolveCheckpointQuantizationPlan({
      quantization: {
        bits: 4,
        group_size: 64,
        "model.layers.0.self_attn.q_proj": false,
      },
    });
    if (plan === null) {
      throw new Error("expected plan");
    }

    const translated = translateCheckpointQuantizationPlanPaths(plan, () => null);
    expect(translated.rules).toEqual([]);
  });
});

describe("resolveQuantizationParameters", () => {
  test("uses mode-specific defaults", () => {
    expect(resolveQuantizationParameters({ mode: "mxfp8" })).toEqual({
      groupSize: 32,
      bits: 8,
      mode: "mxfp8",
    });
    expect(resolveQuantizationParameters({ mode: "nvfp4" })).toEqual({
      groupSize: 16,
      bits: 4,
      mode: "nvfp4",
    });
  });
});
