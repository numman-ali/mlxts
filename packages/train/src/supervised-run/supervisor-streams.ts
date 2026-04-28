import { createWriteStream } from "fs";

import { stderrPath } from "./files";
import { appendSupervisorEvent, managerEvent, readEvent } from "./supervisor-events";

function appendTrainerEventLine(
  runDirectory: string,
  line: string,
  onEvent: (event: Record<string, unknown>) => void,
): void {
  const event = readEvent(line);
  if (event === null) {
    appendSupervisorEvent(runDirectory, managerEvent("trainer-nonjson", { line }));
    return;
  }

  appendSupervisorEvent(runDirectory, event);
  onEvent(event);
}

function drainTrainerBuffer(
  runDirectory: string,
  buffer: string,
  onEvent: (event: Record<string, unknown>) => void,
): string {
  let newlineIndex = buffer.indexOf("\n");
  while (newlineIndex >= 0) {
    const line = buffer.slice(0, newlineIndex).replace(/\r$/, "");
    buffer = buffer.slice(newlineIndex + 1);
    if (line.length > 0) {
      appendTrainerEventLine(runDirectory, line, onEvent);
    }
    newlineIndex = buffer.indexOf("\n");
  }
  return buffer;
}

export async function pipeTextStream(
  stream: ReadableStream<Uint8Array>,
  sink: (chunk: string) => void,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        const tail = decoder.decode();
        if (tail.length > 0) {
          sink(tail);
        }
        return;
      }
      if (value.byteLength > 0) {
        sink(decoder.decode(value, { stream: true }));
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function pumpTrainerStdout(
  stream: ReadableStream<Uint8Array>,
  runDirectory: string,
  onEvent: (event: Record<string, unknown>) => void,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        break;
      }
      if (value.byteLength > 0) {
        buffer += decoder.decode(value, { stream: true });
      }
      buffer = drainTrainerBuffer(runDirectory, buffer, onEvent);
    }

    const tail = buffer.trim();
    if (tail.length > 0) {
      appendTrainerEventLine(runDirectory, tail, onEvent);
    }
  } finally {
    reader.releaseLock();
  }
}

export function createStderrStream(runDirectory: string) {
  return createWriteStream(stderrPath(runDirectory), { flags: "a" });
}
