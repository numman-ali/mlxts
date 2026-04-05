import { describe, expect, test } from "bun:test";

import {
  ConfigParseError,
  MissingWeightsError,
  UnsupportedModelError,
  WeightMismatchError,
} from "./types";

describe("transformers errors", () => {
  test("format typed loader errors with stable names and messages", () => {
    const unsupported = new UnsupportedModelError("mixtral", ["llama", "gemma"]);
    const config = new ConfigParseError("broken config");
    const mismatch = new WeightMismatchError("model.weight", [2, 3], [3, 2]);
    const missing = new MissingWeightsError(["b", "a"]);

    expect(unsupported.name).toBe("UnsupportedModelError");
    expect(unsupported.message).toContain('unsupported model_type "mixtral"');
    expect(config.name).toBe("ConfigParseError");
    expect(config.message).toBe("broken config");
    expect(mismatch.name).toBe("WeightMismatchError");
    expect(mismatch.message).toContain('"model.weight"');
    expect(mismatch.message).toContain("expected [2, 3]");
    expect(missing.name).toBe("MissingWeightsError");
    expect(missing.message).toContain("a, b");
  });
});
