import { afterEach, describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { join } from "path";

import { acquireRuntimeCommandLock } from "./runtime-command-lock";

const TEMP_DIRECTORY = process.env.TMPDIR ?? "/tmp";

function withLockPath(path: string, body: () => void): void {
  const previous = process.env.MLXTS_RUNTIME_LOCK_PATH;
  process.env.MLXTS_RUNTIME_LOCK_PATH = path;
  try {
    body();
  } finally {
    if (previous === undefined) {
      delete process.env.MLXTS_RUNTIME_LOCK_PATH;
    } else {
      process.env.MLXTS_RUNTIME_LOCK_PATH = previous;
    }
  }
}

function clearInheritedLockToken(): void {
  delete process.env.MLXTS_RUNTIME_LOCK_TOKEN;
}

describe("runtime command lock", () => {
  const tempRoots: string[] = [];

  afterEach(() => {
    clearInheritedLockToken();
    for (const root of tempRoots.splice(0, tempRoots.length)) {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("acquires and releases the lock file", () => {
    const root = mkdtempSync(join(TEMP_DIRECTORY, "mlxts-runtime-lock-"));
    tempRoots.push(root);
    const lockPath = join(root, "runtime-command.lock.json");

    withLockPath(lockPath, () => {
      using lock = acquireRuntimeCommandLock("bench:generation");
      expect(existsSync(lockPath)).toBe(true);

      const record = JSON.parse(readFileSync(lockPath, "utf8")) as { command: string; pid: number };
      expect(record.command).toBe("bench:generation");
      expect(record.pid).toBe(process.pid);
      lock[Symbol.dispose]();
    });

    expect(existsSync(lockPath)).toBe(false);
  });

  test("blocks an independent second owner while the first lock is live", () => {
    const root = mkdtempSync(join(TEMP_DIRECTORY, "mlxts-runtime-lock-"));
    tempRoots.push(root);
    const lockPath = join(root, "runtime-command.lock.json");

    withLockPath(lockPath, () => {
      using firstLock = acquireRuntimeCommandLock("bench:generation");
      clearInheritedLockToken();
      expect(() => acquireRuntimeCommandLock("bench:generation:parity")).toThrow(
        "already holding the MLX runtime lock",
      );
      firstLock[Symbol.dispose]();
    });
  });

  test("allows inherited nested execution under the same lock token", () => {
    const root = mkdtempSync(join(TEMP_DIRECTORY, "mlxts-runtime-lock-"));
    tempRoots.push(root);
    const lockPath = join(root, "runtime-command.lock.json");

    withLockPath(lockPath, () => {
      using parentLock = acquireRuntimeCommandLock("acceptance:gpt-small");
      using nestedLock = acquireRuntimeCommandLock("run:nanogpt supervisor");
      expect(existsSync(lockPath)).toBe(true);
      nestedLock[Symbol.dispose]();
      expect(existsSync(lockPath)).toBe(true);
      parentLock[Symbol.dispose]();
    });

    expect(existsSync(lockPath)).toBe(false);
  });

  test("recovers from a stale lock owned by a dead pid", () => {
    const root = mkdtempSync(join(TEMP_DIRECTORY, "mlxts-runtime-lock-"));
    tempRoots.push(root);
    const lockPath = join(root, "runtime-command.lock.json");
    mkdirSync(root, { recursive: true });
    writeFileSync(
      lockPath,
      JSON.stringify(
        {
          token: "stale-token",
          pid: 999_999,
          parentPid: 1,
          command: "stale-run",
          cwd: root,
          startedAt: new Date(0).toISOString(),
        },
        null,
        2,
      ),
    );

    withLockPath(lockPath, () => {
      using lock = acquireRuntimeCommandLock("bench:generation");
      const record = JSON.parse(readFileSync(lockPath, "utf8")) as { command: string; pid: number };
      expect(record.command).toBe("bench:generation");
      expect(record.pid).toBe(process.pid);
      lock[Symbol.dispose]();
    });
  });
});
