import { describe, expect, test } from "bun:test";
import { spawnSync } from "child_process";
import { mkdtempSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

function packageRoot(): string {
  return join(import.meta.dir, "..", "..");
}

function lockedEnv(): NodeJS.ProcessEnv {
  const directory = mkdtempSync(join(tmpdir(), "nanogpt-memory-lock-"));
  const lockPath = join(directory, "runtime-lock.json");
  writeFileSync(
    lockPath,
    JSON.stringify({
      token: "busy",
      pid: process.pid,
      parentPid: process.ppid,
      command: "other-heavy-command",
      cwd: packageRoot(),
      startedAt: new Date().toISOString(),
    }),
    "utf-8",
  );
  return {
    ...process.env,
    MLXTS_RUNTIME_LOCK_PATH: lockPath,
  };
}

function runMemoryBench(args: string[], env?: NodeJS.ProcessEnv) {
  return spawnSync("bun", ["run", "bench:memory", ...args], {
    cwd: packageRoot(),
    encoding: "utf-8",
    env,
  });
}

describe("memory benchmark", () => {
  test("help is compact AXI stdout and does not acquire the runtime lock", () => {
    const result = runMemoryBench(["--help"], lockedEnv());

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("description:");
    expect(result.stdout).toContain("options[8]");
    expect(result.stdout).toContain("exit_codes[3]");
    expect(result.stdout).not.toContain("runtime lock");
  });

  test("usage errors use structured stdout before the runtime lock", () => {
    const result = runMemoryBench(["--scenario"], lockedEnv());

    expect(result.status).toBe(2);
    expect(result.stdout).toContain("error:");
    expect(result.stdout).toContain('"usage"');
    expect(result.stdout).toContain("Flag --scenario requires a value");
    expect(result.stdout).not.toContain("runtime lock");
    expect(result.stderr).not.toContain("Flag --scenario");
  });

  test("runtime lock conflicts use structured runtime errors", () => {
    const result = runMemoryBench(
      [
        "--scenario",
        "reshape-transpose",
        "--warmup",
        "1",
        "--iterations",
        "1",
        "--max-end-growth-mb",
        "512",
      ],
      lockedEnv(),
    );

    expect(result.status).toBe(1);
    expect(result.stdout).toContain("error:");
    expect(result.stdout).toContain('"runtime"');
    expect(result.stdout).toContain("runtime lock");
  });

  test("reshape-transpose scenario emits structured output", () => {
    const result = runMemoryBench([
      "--scenario",
      "reshape-transpose",
      "--warmup",
      "1",
      "--iterations",
      "2",
      "--max-end-growth-mb",
      "512",
      "--json",
    ]);

    expect(result.status).toBe(0);
    const payload: unknown = JSON.parse(result.stdout.trim());
    if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
      throw new Error("Expected JSON object payload from memory benchmark");
    }
    expect(payload).toHaveProperty("scenario", "reshape-transpose");
    expect(payload).toHaveProperty("measuredIterations", 2);
  });
});
