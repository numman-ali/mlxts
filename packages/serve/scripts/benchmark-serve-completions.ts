import type {
  SamplingMode,
  ServeBenchmarkOptions,
  ServeBenchmarkRung,
  TransportMode,
} from "./benchmark-serve-options";

type CompletionUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

type CompletionChoice = {
  text?: string;
  finish_reason?: string | null;
};

type CompletionResponseBody = {
  choices?: CompletionChoice[];
  usage?: CompletionUsage | null;
};

export type RequestMetrics = {
  durationMs: number;
  ttftMs: number | null;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  finishReason: string;
  streamChunks: number;
  streamBytes: number;
};

function completionRequestBody(
  modelId: string,
  promptTokenIds: readonly number[],
  generationTokens: number,
  samplingMode: SamplingMode,
  transportMode: TransportMode,
  ignoreEos: boolean,
) {
  return {
    model: modelId,
    prompt: [...promptTokenIds],
    max_tokens: generationTokens,
    ...(transportMode === "streaming"
      ? { stream: true, stream_options: { include_usage: true } }
      : {}),
    ...(ignoreEos ? { ignore_eos: true } : {}),
    ...(samplingMode === "greedy" ? { temperature: 0 } : {}),
  };
}

function completionFinishReason(body: CompletionResponseBody): string {
  const choice = body.choices?.[0];
  return choice?.finish_reason ?? "unknown";
}

function completionUsage(body: CompletionResponseBody): Required<CompletionUsage> {
  const usage = body.usage ?? {};
  return {
    prompt_tokens: usage.prompt_tokens ?? 0,
    completion_tokens: usage.completion_tokens ?? 0,
    total_tokens: usage.total_tokens ?? 0,
  };
}

async function runBufferedCompletionRequest(
  endpoint: string,
  modelId: string,
  promptTokenIds: readonly number[],
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
): Promise<RequestMetrics> {
  const started = performance.now();
  const response = await fetch(`${endpoint}/v1/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(
      completionRequestBody(
        modelId,
        promptTokenIds,
        rung.generationTokens,
        options.samplingMode,
        options.transportMode,
        options.ignoreEos,
      ),
    ),
  });
  const durationMs = performance.now() - started;
  const body = (await response.json()) as CompletionResponseBody;
  if (!response.ok) {
    throw new Error(
      `benchmark-serve: request failed (${response.status}): ${JSON.stringify(body)}`,
    );
  }
  const usage = completionUsage(body);
  return {
    durationMs,
    ttftMs: null,
    promptTokens: usage.prompt_tokens,
    completionTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens,
    finishReason: completionFinishReason(body),
    streamChunks: 0,
    streamBytes: 0,
  };
}

function sseDataPayloads(frame: string): string[] {
  return frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart());
}

function consumeSseFrames(buffer: string, onFrame: (frame: string) => void): { remainder: string } {
  let cursor = 0;
  while (true) {
    const nextFrameEnd = buffer.indexOf("\n\n", cursor);
    if (nextFrameEnd === -1) {
      return { remainder: buffer.slice(cursor) };
    }
    onFrame(buffer.slice(cursor, nextFrameEnd));
    cursor = nextFrameEnd + 2;
  }
}

type StreamingCompletionMetrics = {
  ttftMs: number | null;
  usage: Required<CompletionUsage>;
  finishReason: string;
  streamChunks: number;
};

function initialStreamingCompletionMetrics(): StreamingCompletionMetrics {
  return {
    ttftMs: null,
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    finishReason: "unknown",
    streamChunks: 0,
  };
}

function updateStreamingCompletionMetrics(
  metrics: StreamingCompletionMetrics,
  body: CompletionResponseBody,
  started: number,
): void {
  if (body.usage !== undefined && body.usage !== null) {
    metrics.usage = completionUsage(body);
  }

  const choice = body.choices?.[0];
  if (choice === undefined) {
    return;
  }
  if (choice.finish_reason !== undefined && choice.finish_reason !== null) {
    metrics.finishReason = choice.finish_reason;
  }
  if (choice.text !== undefined && choice.text !== "") {
    metrics.streamChunks += 1;
    metrics.ttftMs ??= performance.now() - started;
  }
}

function handleCompletionSsePayload(
  payload: string,
  metrics: StreamingCompletionMetrics,
  started: number,
): void {
  if (payload === "" || payload === "[DONE]") {
    return;
  }
  updateStreamingCompletionMetrics(metrics, JSON.parse(payload) as CompletionResponseBody, started);
}

async function runStreamingCompletionRequest(
  endpoint: string,
  modelId: string,
  promptTokenIds: readonly number[],
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
): Promise<RequestMetrics> {
  const started = performance.now();
  const response = await fetch(`${endpoint}/v1/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(
      completionRequestBody(
        modelId,
        promptTokenIds,
        rung.generationTokens,
        options.samplingMode,
        options.transportMode,
        options.ignoreEos,
      ),
    ),
  });
  if (!response.ok) {
    const body = (await response.json()) as CompletionResponseBody;
    throw new Error(
      `benchmark-serve: request failed (${response.status}): ${JSON.stringify(body)}`,
    );
  }
  if (response.body === null) {
    throw new Error("benchmark-serve: streaming response had no body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const metrics = initialStreamingCompletionMetrics();
  let buffer = "";
  let streamBytes = 0;

  while (true) {
    const read = await reader.read();
    if (read.done) {
      break;
    }
    streamBytes += read.value.byteLength;
    buffer += decoder.decode(read.value, { stream: true });
    const consumed = consumeSseFrames(buffer, (frame) => {
      for (const payload of sseDataPayloads(frame)) {
        handleCompletionSsePayload(payload, metrics, started);
      }
    });
    buffer = consumed.remainder;
  }

  buffer += decoder.decode();
  consumeSseFrames(`${buffer}\n\n`, (frame) => {
    for (const payload of sseDataPayloads(frame)) {
      handleCompletionSsePayload(payload, metrics, started);
    }
  });

  const durationMs = performance.now() - started;
  return {
    durationMs,
    ttftMs: metrics.ttftMs,
    promptTokens: metrics.usage.prompt_tokens,
    completionTokens: metrics.usage.completion_tokens,
    totalTokens: metrics.usage.total_tokens,
    finishReason: metrics.finishReason,
    streamChunks: metrics.streamChunks,
    streamBytes,
  };
}

export async function runCompletionRequest(
  endpoint: string,
  modelId: string,
  promptTokenIds: readonly number[],
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
): Promise<RequestMetrics> {
  return options.transportMode === "streaming"
    ? await runStreamingCompletionRequest(endpoint, modelId, promptTokenIds, rung, options)
    : await runBufferedCompletionRequest(endpoint, modelId, promptTokenIds, rung, options);
}
