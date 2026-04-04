import { existsSync, mkdirSync, renameSync, rmSync, writeFileSync } from "fs";
import { basename, dirname, join } from "path";

import {
  type CheckpointManifest,
  MANIFEST_FILENAME,
  TENSOR_DATA_FILENAME,
} from "./checkpoint-types";

function checkpointTempPath(path: string): string {
  return join(
    dirname(path),
    `.${basename(path) || "checkpoint"}.${process.pid}.${crypto.randomUUID()}.tmp`,
  );
}

function checkpointBackupPath(path: string): string {
  return join(
    dirname(path),
    `.${basename(path) || "checkpoint"}.${process.pid}.${crypto.randomUUID()}.bak`,
  );
}

export function writeCheckpointDirectory(
  path: string,
  manifest: CheckpointManifest,
  bytes: Uint8Array,
): void {
  const tempPath = checkpointTempPath(path);
  const backupPath = checkpointBackupPath(path);
  let backupCreated = false;
  let renamedIntoPlace = false;

  rmSync(tempPath, { recursive: true, force: true });
  mkdirSync(tempPath, { recursive: true });

  try {
    writeFileSync(
      join(tempPath, MANIFEST_FILENAME),
      `${JSON.stringify(manifest, null, 2)}\n`,
      "utf-8",
    );
    writeFileSync(join(tempPath, TENSOR_DATA_FILENAME), bytes);

    if (existsSync(path)) {
      rmSync(backupPath, { recursive: true, force: true });
      renameSync(path, backupPath);
      backupCreated = true;
    }

    renameSync(tempPath, path);
    renamedIntoPlace = true;
  } catch (error) {
    if (backupCreated && !renamedIntoPlace && !existsSync(path)) {
      renameSync(backupPath, path);
      backupCreated = false;
    }
    rmSync(tempPath, { recursive: true, force: true });
    throw error;
  } finally {
    if (backupCreated) {
      rmSync(backupPath, { recursive: true, force: true });
    }
  }
}
