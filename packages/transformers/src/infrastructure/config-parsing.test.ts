import { describe, expect, test } from "bun:test";

import { ConfigParseError } from "../types";
import {
  expectConfigRecord,
  expectInteger,
  expectString,
  optionalBoolean,
  optionalInteger,
  optionalNumber,
  optionalString,
} from "./config-parsing";

describe("config parsing helpers", () => {
  test("parse required and optional fields from a config record", () => {
    const record = expectConfigRecord(
      {
        model_type: "llama",
        hidden_size: 8,
        rope_theta: 10000,
        attention_bias: false,
        hidden_act: "gelu_pytorch_tanh",
      },
      "test config",
    );

    expect(expectString(record, "model_type", "test config")).toBe("llama");
    expect(expectInteger(record, "hidden_size", "test config")).toBe(8);
    expect(optionalInteger(record, "num_layers", "test config")).toBeUndefined();
    expect(optionalNumber(record, "rope_theta", "test config")).toBe(10000);
    expect(optionalBoolean(record, "attention_bias", "test config")).toBe(false);
    expect(optionalString(record, "hidden_act", "test config")).toBe("gelu_pytorch_tanh");
  });

  test("throw ConfigParseError for malformed required and optional values", () => {
    expect(() => expectConfigRecord([], "broken config")).toThrow(ConfigParseError);

    const record = { model_type: "", hidden_size: 3.5, rope_theta: "bad", attention_bias: 1 };

    expect(() => expectString(record, "model_type", "broken config")).toThrow(
      "broken config.model_type must be a non-empty string",
    );
    expect(() => expectInteger(record, "hidden_size", "broken config")).toThrow(
      "broken config.hidden_size must be an integer",
    );
    expect(() => optionalInteger(record, "hidden_size", "broken config")).toThrow(
      "broken config.hidden_size must be an integer when present",
    );
    expect(() => optionalNumber(record, "rope_theta", "broken config")).toThrow(
      "broken config.rope_theta must be a number when present",
    );
    expect(() => optionalBoolean(record, "attention_bias", "broken config")).toThrow(
      "broken config.attention_bias must be a boolean when present",
    );
    expect(() => optionalString({ hidden_act: "" }, "hidden_act", "broken config")).toThrow(
      "broken config.hidden_act must be a non-empty string when present",
    );
  });
});
