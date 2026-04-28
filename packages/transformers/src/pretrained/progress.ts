import type { PretrainedLoadProgressEvent } from "./types";

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) {
    return `${(bytes / 1e9).toFixed(1)} GB`;
  }
  if (bytes >= 1e6) {
    return `${(bytes / 1e6).toFixed(1)} MB`;
  }
  if (bytes >= 1e3) {
    return `${(bytes / 1e3).toFixed(1)} KB`;
  }
  return `${bytes} B`;
}

function logResolveEvent(
  writeLine: (line: string) => void,
  event: Extract<PretrainedLoadProgressEvent, { stage: "resolve" }>,
): void {
  if (event.status === "start") {
    writeLine(`[resolve] resolving ${event.source}`);
    return;
  }

  const revision =
    event.resolvedRevision === undefined ? "" : ` @ ${event.resolvedRevision.slice(0, 12)}`;
  const repo = event.repoId === undefined ? event.directory : `${event.repoId}${revision}`;
  writeLine(
    `[resolve] ${repo} -> ${event.directory} (${event.fileCount} files, ${formatBytes(event.totalBytes)})`,
  );
}

function logDownloadEvent(
  writeLine: (line: string) => void,
  event: Extract<PretrainedLoadProgressEvent, { stage: "download" }>,
): void {
  writeLine(
    `[download] ${event.index}/${event.totalFiles} ${event.status} ${event.relativePath} (${formatBytes(event.size)}) ${formatBytes(event.completedBytes)} / ${formatBytes(event.totalBytes)}`,
  );
}

function logModelEvent(
  writeLine: (line: string) => void,
  event: Extract<PretrainedLoadProgressEvent, { stage: "model" }>,
): void {
  writeLine(
    event.status === "weights-start"
      ? `[model] loading ${event.shardCount} safetensor shard(s)`
      : `[model] finished loading ${event.shardCount} safetensor shard(s)`,
  );
}

function logTokenizerEvent(
  writeLine: (line: string) => void,
  event: Extract<PretrainedLoadProgressEvent, { stage: "tokenizer" }>,
): void {
  writeLine(
    event.status === "start" ? `[tokenizer] loading from ${event.directory}` : "[tokenizer] ready",
  );
}

/** Create a human-readable pretrained loader progress reporter. */
export function createProgressReporter(
  writeLine: (line: string) => void = (line) => console.log(line),
): (event: PretrainedLoadProgressEvent) => void {
  return (event) => {
    switch (event.stage) {
      case "resolve":
        logResolveEvent(writeLine, event);
        return;
      case "download":
        logDownloadEvent(writeLine, event);
        return;
      case "model":
        logModelEvent(writeLine, event);
        return;
      case "tokenizer":
        logTokenizerEvent(writeLine, event);
        return;
    }
  };
}
