import { describe, expect, test } from "bun:test";

import { FAMILY_REGISTRY, resolveFamily } from "./registry";
import { UnsupportedModelError } from "./types";

describe("family registry", () => {
  test("resolves every supported model_type through the explicit registry", () => {
    const supportedModelTypes = [...FAMILY_REGISTRY.keys()].sort();

    expect(resolveFamily("llama").family).toBe("llama");
    expect(resolveFamily("mistral").family).toBe("mistral");
    expect(resolveFamily("gemma").family).toBe("gemma");
    expect(supportedModelTypes).toEqual([
      "gemma",
      "gemma3_text",
      "gemma4",
      "gemma4_text",
      "llama",
      "mistral",
      "mistral3",
      "phi3",
      "qwen3_5",
      "qwen3_5_text",
    ]);
  });

  test("throws a typed error for unsupported model types", () => {
    const supportedModelTypes = [...FAMILY_REGISTRY.keys()].sort();

    expect(() => resolveFamily("mixtral")).toThrow(UnsupportedModelError);
    expect(() => resolveFamily("mixtral")).toThrow(
      `Supported model types: ${supportedModelTypes.join(", ")}`,
    );
  });
});
