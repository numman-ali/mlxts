import { describe, expect, test } from "bun:test";
import { VERSION } from "./index";

describe("nanogpt", () => {
  test("exports a version string", () => {
    expect(VERSION).toBe("0.0.1");
  });
});
