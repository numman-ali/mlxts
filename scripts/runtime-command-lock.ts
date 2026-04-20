import { closeSync, mkdirSync, openSync, readFileSync, rmSync, writeFileSync } from "fs";
import { dirname, resolve } from "path";

type RuntimeCommandLockRecord = {
  token: string;
  pid: number;
  parentPid: number;
  command: string;
  cwd: string;
  startedAt: string;
};

const RUNTIME_LOCK_TOKEN_ENV = "MLXTS_RUNTIME_LOCK_TOKEN";
const DEFAULT_LOCK_PATH = resolve(import.meta.dir, "..", ".tmp", "runtime-command.lock.json");

function runtimeLockPath(): string {
  return process.env.MLXTS_RUNTIME_LOCK_PATH ?? DEFAULT_LOCK_PATH;
}

function readLockRecord(lockPath: string): RuntimeCommandLockRecord | null {
  try {
    const parsed: unknown = JSON.parse(readFileSync(lockPath, "utf8"));
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return null;
    }

    const record = parsed as Partial<RuntimeCommandLockRecord>;
    if (
      typeof record.token !== "string" ||
      typeof record.pid !== "number" ||
      typeof record.parentPid !== "number" ||
      typeof record.command !== "string" ||
      typeof record.cwd !== "string" ||
      typeof record.startedAt !== "string"
    ) {
      return null;
    }

    return {
      token: record.token,
      pid: record.pid,
      parentPid: record.parentPid,
      command: record.command,
      cwd: record.cwd,
      startedAt: record.startedAt,
    };
  } catch {
    return null;
  }
}

function isLiveProcess(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false;
  }

  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (typeof error === "object" && error !== null && "code" in error && error.code === "ESRCH") {
      return false;
    }
    return true;
  }
}

function staleOrMissingRecord(lockPath: string): boolean {
  const record = readLockRecord(lockPath);
  return record === null || !isLiveProcess(record.pid);
}

function acquireFreshLockRecord(
  lockPath: string,
  record: RuntimeCommandLockRecord,
  allowRetry: boolean,
): void {
  mkdirSync(dirname(lockPath), { recursive: true });

  try {
    const descriptor = openSync(lockPath, "wx");
    try {
      writeFileSync(descriptor, JSON.stringify(record, null, 2));
    } finally {
      closeSync(descriptor);
    }
    return;
  } catch (error) {
    if (
      allowRetry &&
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      error.code === "EEXIST" &&
      staleOrMissingRecord(lockPath)
    ) {
      rmSync(lockPath, { force: true });
      acquireFreshLockRecord(lockPath, record, false);
      return;
    }

    throw error;
  }
}

function formatLockConflict(lockPath: string, requestedCommand: string): string {
  const record = readLockRecord(lockPath);
  if (record === null) {
    return (
      `runtime lock: ${requestedCommand} cannot start because another heavy MLX command is already running.\n` +
      `  lock: ${lockPath}`
    );
  }

  return (
    `runtime lock: ${requestedCommand} cannot start because ${record.command} is already holding the MLX runtime lock.\n` +
    `  pid: ${record.pid}\n` +
    `  cwd: ${record.cwd}\n` +
    `  started: ${record.startedAt}\n` +
    `  lock: ${lockPath}`
  );
}

class RuntimeCommandLock implements Disposable {
  #lockPath: string;
  #token: string | null;
  #ownsLock: boolean;
  #released = false;
  #previousToken: string | undefined;
  #exitHandler: (() => void) | null;

  constructor(lockPath: string, token: string | null, ownsLock: boolean) {
    this.#lockPath = lockPath;
    this.#token = token;
    this.#ownsLock = ownsLock;
    this.#previousToken = process.env[RUNTIME_LOCK_TOKEN_ENV];
    this.#exitHandler = ownsLock ? () => this.release() : null;

    if (token !== null) {
      process.env[RUNTIME_LOCK_TOKEN_ENV] = token;
    }

    if (this.#exitHandler !== null) {
      process.on("exit", this.#exitHandler);
    }
  }

  release(): void {
    if (this.#released) {
      return;
    }
    this.#released = true;

    if (this.#exitHandler !== null) {
      process.removeListener("exit", this.#exitHandler);
      this.#exitHandler = null;
    }

    if (this.#token !== null) {
      if (this.#previousToken === undefined) {
        delete process.env[RUNTIME_LOCK_TOKEN_ENV];
      } else {
        process.env[RUNTIME_LOCK_TOKEN_ENV] = this.#previousToken;
      }
    }

    if (!this.#ownsLock || this.#token === null) {
      return;
    }

    const record = readLockRecord(this.#lockPath);
    if (record?.token === this.#token) {
      rmSync(this.#lockPath, { force: true });
    }
  }

  [Symbol.dispose](): void {
    this.release();
  }
}

/**
 * Acquire the shared MLX runtime lock for a heavy command.
 *
 * Benchmark, soak, acceptance, and long-running training entrypoints should use
 * this so the repo cannot accidentally run multiple heavy MLX programs at the
 * same time on one machine.
 */
export function acquireRuntimeCommandLock(command: string): Disposable {
  const lockPath = runtimeLockPath();
  const inheritedToken = process.env[RUNTIME_LOCK_TOKEN_ENV];
  if (inheritedToken !== undefined) {
    const record = readLockRecord(lockPath);
    if (record?.token === inheritedToken) {
      return new RuntimeCommandLock(lockPath, inheritedToken, false);
    }
  }

  const record: RuntimeCommandLockRecord = {
    token: crypto.randomUUID(),
    pid: process.pid,
    parentPid: process.ppid,
    command,
    cwd: process.cwd(),
    startedAt: new Date().toISOString(),
  };

  try {
    acquireFreshLockRecord(lockPath, record, true);
  } catch (_error) {
    throw new Error(formatLockConflict(lockPath, command));
  }

  return new RuntimeCommandLock(lockPath, record.token, true);
}
