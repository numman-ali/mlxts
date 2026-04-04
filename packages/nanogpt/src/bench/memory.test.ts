import { describe, expect, test } from "bun:test";
import { spawnSync } from "child_process";
import { join } from "path";

function packageRoot(): string {
  return join(import.meta.dir, "..", "..");
}

describe("memory benchmark", () => {
  test("reshape-transpose scenario emits structured output", () => {
    const result = spawnSync(
      "bun",
      [
        "run",
        "src/bench/memory.ts",
        "--scenario",
        "reshape-transpose",
        "--warmup",
        "1",
        "--iterations",
        "2",
        "--max-end-growth-mb",
        "512",
        "--json",
      ],
      {
        cwd: packageRoot(),
        encoding: "utf-8",
      },
    );

    expect(result.status).toBe(0);
    const payload: unknown = JSON.parse(result.stdout.trim());
    if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
      throw new Error("Expected JSON object payload from memory benchmark");
    }
    expect(payload).toHaveProperty("scenario", "reshape-transpose");
    expect(payload).toHaveProperty("measuredIterations", 2);
  });
});
