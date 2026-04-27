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
  id?: string;
  choices?: CompletionChoice[];
  usage?: CompletionUsage | null;
};

type ChatResponseBody = {
  id?: string;
  choices?: Array<{
    message?: { content?: string | null; reasoning_content?: string };
    delta?: { content?: string; reasoning_content?: string };
    finish_reason?: string | null;
  }>;
  usage?: CompletionUsage | null;
};

type ResponseApiUsage = {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
};

type ResponseApiBody = {
  id?: string;
  status?: string;
  incomplete_details?: { reason?: string } | null;
  usage?: ResponseApiUsage | null;
  output_text?: string;
  response?: ResponseApiBody;
};

type AnthropicUsage = {
  input_tokens?: number;
  output_tokens?: number;
};

type AnthropicMessageBody = {
  id?: string;
  type?: string;
  content?: Array<
    | { type?: "text"; text?: string }
    | { type?: "thinking"; thinking?: string }
    | Record<string, unknown>
  >;
  stop_reason?: string | null;
  usage?: AnthropicUsage | null;
  message?: AnthropicMessageBody;
  delta?: { stop_reason?: string | null };
};

export type BenchmarkPrompt = {
  tokenIds: readonly number[];
  text: string;
};

export type RequestMetrics = {
  id?: string;
  durationMs: number;
  ttftMs: number | null;
  promptToFirstTokenTps: number | null;
  postTtftCompletionTps: number | null;
  meanStreamChunkGapMs: number | null;
  maxStreamChunkGapMs: number | null;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  finishReason: string;
  streamChunks: number;
  streamBytes: number;
};

function completionRequestBody(
  modelId: string,
  prompt: BenchmarkPrompt,
  generationTokens: number,
  samplingMode: SamplingMode,
  transportMode: TransportMode,
  ignoreEos: boolean,
) {
  return {
    model: modelId,
    prompt: [...prompt.tokenIds],
    max_tokens: generationTokens,
    ...(transportMode === "streaming"
      ? { stream: true, stream_options: { include_usage: true } }
      : {}),
    ...(ignoreEos ? { ignore_eos: true } : {}),
    ...(samplingMode === "greedy" ? { temperature: 0 } : {}),
  };
}

function chatRequestBody(
  modelId: string,
  prompt: BenchmarkPrompt,
  generationTokens: number,
  samplingMode: SamplingMode,
  transportMode: TransportMode,
  ignoreEos: boolean,
) {
  return {
    model: modelId,
    messages: [{ role: "user", content: prompt.text }],
    max_tokens: generationTokens,
    ...(transportMode === "streaming"
      ? { stream: true, stream_options: { include_usage: true } }
      : {}),
    ...(ignoreEos ? { ignore_eos: true } : {}),
    ...(samplingMode === "greedy" ? { temperature: 0 } : {}),
  };
}

function responsesRequestBody(
  modelId: string,
  prompt: BenchmarkPrompt,
  generationTokens: number,
  samplingMode: SamplingMode,
  transportMode: TransportMode,
) {
  return {
    model: modelId,
    input: [{ role: "user", content: prompt.text }],
    max_output_tokens: generationTokens,
    ...(transportMode === "streaming"
      ? { stream: true, stream_options: { include_obfuscation: false } }
      : {}),
    ...(samplingMode === "greedy" ? { temperature: 0 } : {}),
  };
}

function anthropicRequestBody(
  modelId: string,
  prompt: BenchmarkPrompt,
  generationTokens: number,
  samplingMode: SamplingMode,
  transportMode: TransportMode,
) {
  return {
    model: modelId,
    messages: [{ role: "user", content: prompt.text }],
    max_tokens: generationTokens,
    ...(transportMode === "streaming" ? { stream: true } : {}),
    ...(samplingMode === "greedy" ? { temperature: 0 } : {}),
  };
}

function requestPath(options: ServeBenchmarkOptions): string {
  switch (options.protocolMode) {
    case "completions":
      return "/v1/completions";
    case "chat":
      return "/v1/chat/completions";
    case "responses":
      return "/v1/responses";
    case "anthropic":
      return "/v1/messages";
  }
}

function requestBody(
  modelId: string,
  prompt: BenchmarkPrompt,
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
) {
  switch (options.protocolMode) {
    case "completions":
      return completionRequestBody(
        modelId,
        prompt,
        rung.generationTokens,
        options.samplingMode,
        options.transportMode,
        options.ignoreEos,
      );
    case "chat":
      return chatRequestBody(
        modelId,
        prompt,
        rung.generationTokens,
        options.samplingMode,
        options.transportMode,
        options.ignoreEos,
      );
    case "responses":
      return responsesRequestBody(
        modelId,
        prompt,
        rung.generationTokens,
        options.samplingMode,
        options.transportMode,
      );
    case "anthropic":
      return anthropicRequestBody(
        modelId,
        prompt,
        rung.generationTokens,
        options.samplingMode,
        options.transportMode,
      );
  }
}

function requestSignal(options: ServeBenchmarkOptions): AbortSignal {
  return AbortSignal.timeout(options.requestTimeoutMs);
}

function completionFinishReason(body: {
  choices?: Array<{ finish_reason?: string | null }>;
}): string {
  const choice = body.choices?.[0];
  return choice?.finish_reason ?? "unknown";
}

function completionUsage(body: { usage?: CompletionUsage | null }): Required<CompletionUsage> {
  const usage = body.usage ?? {};
  return {
    prompt_tokens: usage.prompt_tokens ?? 0,
    completion_tokens: usage.completion_tokens ?? 0,
    total_tokens: usage.total_tokens ?? 0,
  };
}

function responsesUsage(body: ResponseApiBody): Required<CompletionUsage> {
  const usage = body.usage ?? {};
  return {
    prompt_tokens: usage.input_tokens ?? 0,
    completion_tokens: usage.output_tokens ?? 0,
    total_tokens: usage.total_tokens ?? 0,
  };
}

function anthropicUsage(body: AnthropicMessageBody): Required<CompletionUsage> {
  const usage = body.usage ?? {};
  const promptTokens = usage.input_tokens ?? 0;
  const completionTokens = usage.output_tokens ?? 0;
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  };
}

function isResponseApiBody(
  body: CompletionResponseBody | ChatResponseBody | ResponseApiBody | AnthropicMessageBody,
): body is ResponseApiBody {
  return "output_text" in body || "response" in body || "status" in body;
}

function isAnthropicMessageBody(
  body: CompletionResponseBody | ChatResponseBody | ResponseApiBody | AnthropicMessageBody,
): body is AnthropicMessageBody {
  return ("type" in body && body.type === "message") || "stop_reason" in body || "message" in body;
}

function responseApiFinishReason(body: ResponseApiBody): string {
  if (body.status === "incomplete" && body.incomplete_details?.reason === "max_output_tokens") {
    return "length";
  }
  if (body.status === "completed") {
    return "stop";
  }
  return body.status ?? "unknown";
}

function responseUsage(
  body: CompletionResponseBody | ChatResponseBody | ResponseApiBody | AnthropicMessageBody,
) {
  if (isResponseApiBody(body)) {
    return responsesUsage(body);
  }
  if (isAnthropicMessageBody(body)) {
    return anthropicUsage(body);
  }
  return completionUsage(body);
}

function responseFinishReason(
  body: CompletionResponseBody | ChatResponseBody | ResponseApiBody | AnthropicMessageBody,
) {
  if (isResponseApiBody(body)) {
    return responseApiFinishReason(body);
  }
  if (isAnthropicMessageBody(body)) {
    return body.stop_reason ?? "unknown";
  }
  return completionFinishReason(body);
}

function responseId(
  body: CompletionResponseBody | ChatResponseBody | ResponseApiBody | AnthropicMessageBody,
): string | undefined {
  if (typeof body.id === "string") {
    return body.id;
  }
  if (isResponseApiBody(body) && typeof body.response?.id === "string") {
    return body.response.id;
  }
  if (isAnthropicMessageBody(body) && typeof body.message?.id === "string") {
    return body.message.id;
  }
  return undefined;
}

async function runBufferedCompletionRequest(
  endpoint: string,
  modelId: string,
  prompt: BenchmarkPrompt,
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
): Promise<RequestMetrics> {
  const started = performance.now();
  const response = await fetch(`${endpoint}${requestPath(options)}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal: requestSignal(options),
    body: JSON.stringify(requestBody(modelId, prompt, rung, options)),
  });
  const body = (await response.json()) as
    | CompletionResponseBody
    | ChatResponseBody
    | ResponseApiBody
    | AnthropicMessageBody;
  const durationMs = performance.now() - started;
  if (!response.ok) {
    throw new Error(
      `benchmark-serve: request failed (${response.status}): ${JSON.stringify(body)}`,
    );
  }
  const usage = responseUsage(body);
  const id = responseId(body);
  return {
    ...(id === undefined ? {} : { id }),
    durationMs,
    ttftMs: null,
    promptToFirstTokenTps: null,
    postTtftCompletionTps: null,
    meanStreamChunkGapMs: null,
    maxStreamChunkGapMs: null,
    promptTokens: usage.prompt_tokens,
    completionTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens,
    finishReason: responseFinishReason(body),
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
  streamChunkTimesMs: number[];
  id?: string;
};

function initialStreamingCompletionMetrics(): StreamingCompletionMetrics {
  return {
    ttftMs: null,
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    finishReason: "unknown",
    streamChunks: 0,
    streamChunkTimesMs: [],
  };
}

function recordTextChunk(metrics: StreamingCompletionMetrics, started: number): void {
  const elapsedMs = performance.now() - started;
  metrics.streamChunks += 1;
  metrics.streamChunkTimesMs.push(elapsedMs);
  metrics.ttftMs ??= elapsedMs;
}

function retainResponseId(
  metrics: StreamingCompletionMetrics,
  body: CompletionResponseBody | ChatResponseBody | ResponseApiBody | AnthropicMessageBody,
): void {
  const id = responseId(body);
  if (metrics.id === undefined && id !== undefined) {
    metrics.id = id;
  }
}

function mergeAnthropicUsage(
  previous: Required<CompletionUsage>,
  usage: AnthropicUsage | undefined | null,
): Required<CompletionUsage> {
  const promptTokens = usage?.input_tokens ?? previous.prompt_tokens;
  const completionTokens = usage?.output_tokens ?? previous.completion_tokens;
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  };
}

function updateStreamingCompletionMetrics(
  metrics: StreamingCompletionMetrics,
  body: CompletionResponseBody | ChatResponseBody,
  started: number,
): void {
  if (body.usage !== undefined && body.usage !== null) {
    metrics.usage = completionUsage(body);
  }
  retainResponseId(metrics, body);

  const choice = body.choices?.[0];
  if (choice === undefined) {
    return;
  }
  if (choice.finish_reason !== undefined && choice.finish_reason !== null) {
    metrics.finishReason = choice.finish_reason;
  }
  let text: string | undefined;
  if ("text" in choice) {
    text = choice.text;
  } else if ("delta" in choice) {
    text = choice.delta?.content ?? choice.delta?.reasoning_content;
  }
  if (text !== undefined && text !== "") {
    recordTextChunk(metrics, started);
  }
}

function updateStreamingResponseMetrics(
  metrics: StreamingCompletionMetrics,
  body: ResponseApiBody,
  event: string | null,
  started: number,
): void {
  const response = body.response ?? body;
  if (response.usage !== undefined && response.usage !== null) {
    metrics.usage = responsesUsage(response);
  }
  retainResponseId(metrics, body);
  if (event === "response.completed" || event === "response.incomplete") {
    metrics.finishReason = responseApiFinishReason(response);
  }
  const delta = body.output_text ?? (body as { delta?: unknown }).delta;
  if (
    (event === "response.output_text.delta" || event === "response.reasoning_text.delta") &&
    typeof delta === "string" &&
    delta !== ""
  ) {
    recordTextChunk(metrics, started);
  }
}

function updateStreamingAnthropicMetrics(
  metrics: StreamingCompletionMetrics,
  body: AnthropicMessageBody,
  event: string | null,
  started: number,
): void {
  retainResponseId(metrics, body);
  if (event === "message_start" && body.message !== undefined) {
    retainResponseId(metrics, body.message);
    metrics.usage = mergeAnthropicUsage(metrics.usage, body.message.usage);
  }
  if (event === "message_delta") {
    metrics.usage = mergeAnthropicUsage(metrics.usage, body.usage);
    metrics.finishReason = body.delta?.stop_reason ?? metrics.finishReason;
  }
  if (event === "content_block_delta") {
    const delta = (body as { delta?: { text?: string; thinking?: string } }).delta;
    const text = delta?.text ?? delta?.thinking;
    if (typeof text === "string" && text !== "") {
      recordTextChunk(metrics, started);
    }
  }
}

function sseEventName(frame: string): string | null {
  const eventLine = frame.split(/\r?\n/).find((line) => line.startsWith("event:"));
  return eventLine === undefined ? null : eventLine.slice("event:".length).trimStart();
}

function validateStreamingMetrics(metrics: StreamingCompletionMetrics): void {
  if (metrics.usage.total_tokens <= 0) {
    throw new Error("benchmark-serve: streaming request ended without usage.");
  }
  if (metrics.finishReason === "unknown") {
    throw new Error("benchmark-serve: streaming request ended without a finish reason.");
  }
}

function streamChunkGapsMs(timesMs: readonly number[]): number[] {
  const gaps: number[] = [];
  for (let index = 1; index < timesMs.length; index += 1) {
    const previous = timesMs[index - 1];
    const current = timesMs[index];
    if (previous !== undefined && current !== undefined) {
      gaps.push(Math.max(0, current - previous));
    }
  }
  return gaps;
}

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / Math.max(values.length, 1);
}

function meanOrNull(values: readonly number[]): number | null {
  return values.length === 0 ? null : mean(values);
}

function maxOrNull(values: readonly number[]): number | null {
  return values.length === 0 ? null : Math.max(...values);
}

function handleSsePayload(
  payload: string,
  metrics: StreamingCompletionMetrics,
  started: number,
  options: ServeBenchmarkOptions,
  event: string | null,
): void {
  if (payload === "" || payload === "[DONE]") {
    return;
  }
  if (options.protocolMode === "responses") {
    updateStreamingResponseMetrics(metrics, JSON.parse(payload) as ResponseApiBody, event, started);
    return;
  }
  if (options.protocolMode === "anthropic") {
    updateStreamingAnthropicMetrics(
      metrics,
      JSON.parse(payload) as AnthropicMessageBody,
      event,
      started,
    );
    return;
  }
  updateStreamingCompletionMetrics(
    metrics,
    JSON.parse(payload) as CompletionResponseBody | ChatResponseBody,
    started,
  );
}

async function runStreamingCompletionRequest(
  endpoint: string,
  modelId: string,
  prompt: BenchmarkPrompt,
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
): Promise<RequestMetrics> {
  const started = performance.now();
  const response = await fetch(`${endpoint}${requestPath(options)}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal: requestSignal(options),
    body: JSON.stringify(requestBody(modelId, prompt, rung, options)),
  });
  if (!response.ok) {
    const body = (await response.json()) as
      | CompletionResponseBody
      | ChatResponseBody
      | ResponseApiBody
      | AnthropicMessageBody;
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
      const event = sseEventName(frame);
      for (const payload of sseDataPayloads(frame)) {
        handleSsePayload(payload, metrics, started, options, event);
      }
    });
    buffer = consumed.remainder;
  }

  buffer += decoder.decode();
  consumeSseFrames(`${buffer}\n\n`, (frame) => {
    const event = sseEventName(frame);
    for (const payload of sseDataPayloads(frame)) {
      handleSsePayload(payload, metrics, started, options, event);
    }
  });

  const durationMs = performance.now() - started;
  const ttftMs = metrics.ttftMs;
  const streamChunkGaps = streamChunkGapsMs(metrics.streamChunkTimesMs);
  const promptToFirstTokenTps =
    ttftMs === null || ttftMs <= 0 ? null : metrics.usage.prompt_tokens / (ttftMs / 1000);
  const postTtftCompletionTps =
    ttftMs === null || durationMs <= ttftMs || metrics.usage.completion_tokens <= 1
      ? null
      : (metrics.usage.completion_tokens - 1) / ((durationMs - ttftMs) / 1000);

  validateStreamingMetrics(metrics);

  return {
    ...(metrics.id === undefined ? {} : { id: metrics.id }),
    durationMs,
    ttftMs,
    promptToFirstTokenTps,
    postTtftCompletionTps,
    meanStreamChunkGapMs: meanOrNull(streamChunkGaps),
    maxStreamChunkGapMs: maxOrNull(streamChunkGaps),
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
  prompt: BenchmarkPrompt,
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
): Promise<RequestMetrics> {
  return options.transportMode === "streaming"
    ? await runStreamingCompletionRequest(endpoint, modelId, prompt, rung, options)
    : await runBufferedCompletionRequest(endpoint, modelId, prompt, rung, options);
}
