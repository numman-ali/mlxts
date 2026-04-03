import { describe, expect, test } from "bun:test";
import { VERSION } from "./index";

describe("mlx-ts", () => {
  test("exports a version string", () => {
    expect(VERSION).toBe("0.0.1");
  });
});
