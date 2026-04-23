import {
  createMicroBatchingGenerationEngine,
  createModelRouterGenerationEngine,
  type GenerationEngine,
  type GenerationInput,
  type NormalizedGenerationRequest,
  type NormalizedGenerationResult,
  startServeServer,
} from "@mlxts/serve";

type CompletionChoice = {
  text: string;
  finish_reason: string | null;
};

type CompletionResponse = {
  id: string;
  object: "text_completion";
  model: string;
  choices: CompletionChoice[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  } | null;
};

type RecordingEngine = {
  engine: GenerationEngine;
  batches: string[][];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function inputText(input: GenerationInput): string {
  if (input.kind === "text") {
    return input.text;
  }
  if (input.kind === "tokens") {
    return `[tokens:${input.tokenIds.join(",")}]`;
  }
  return input.messages.map((message) => `${message.role}:${message.content}`).join("\n");
}

function completionFor(
  label: string,
  request: NormalizedGenerationRequest,
): NormalizedGenerationResult {
  const prompt = inputText(request.input);
  const text = `${label} saw "${prompt}"`;
  return {
    text,
    finishReason: "stop",
    usage: {
      promptTokens: prompt.length,
      completionTokens: text.length,
      totalTokens: prompt.length + text.length,
    },
  };
}

function createRecordingEngine(label: string): RecordingEngine {
  const batches: string[][] = [];
  return {
    batches,
    engine: {
      generate(request) {
        return completionFor(label, request);
      },
      generateBatch(requests) {
        batches.push(requests.map((request) => request.id));
        return requests.map((request) => completionFor(label, request));
      },
    },
  };
}

function parseCompletionResponse(value: unknown): CompletionResponse {
  assert(isRecord(value), "completion response must be an object");
  assert(value.object === "text_completion", "completion response object must be text_completion");
  assert(typeof value.id === "string", "completion response id must be a string");
  assert(typeof value.model === "string", "completion response model must be a string");
  assert(Array.isArray(value.choices), "completion response choices must be an array");
  for (const choice of value.choices) {
    assert(isRecord(choice), "completion choice must be an object");
    assert(typeof choice.text === "string", "completion choice text must be a string");
  }

  const usage = parseCompletionUsage(value.usage);
  return {
    id: value.id,
    object: value.object,
    model: value.model,
    choices: value.choices.map((choice) => ({
      text: isRecord(choice) && typeof choice.text === "string" ? choice.text : "",
      finish_reason:
        isRecord(choice) && typeof choice.finish_reason === "string" ? choice.finish_reason : null,
    })),
    ...(usage === undefined ? {} : { usage }),
  };
}

function optionalUsageNumber(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function parseCompletionUsage(value: unknown): CompletionResponse["usage"] | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value === null) {
    return null;
  }
  assert(isRecord(value), "completion usage must be an object or null");
  const promptTokens = optionalUsageNumber(value, "prompt_tokens");
  const completionTokens = optionalUsageNumber(value, "completion_tokens");
  const totalTokens = optionalUsageNumber(value, "total_tokens");
  return {
    ...(promptTokens === undefined ? {} : { prompt_tokens: promptTokens }),
    ...(completionTokens === undefined ? {} : { completion_tokens: completionTokens }),
    ...(totalTokens === undefined ? {} : { total_tokens: totalTokens }),
  };
}

async function completion(
  endpoint: string,
  body: Record<string, unknown>,
): Promise<CompletionResponse> {
  const response = await fetch(`${endpoint}/v1/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(JSON.stringify(payload));
  }
  return parseCompletionResponse(payload);
}

async function main(): Promise<void> {
  const alpha = createRecordingEngine("alpha");
  const beta = createRecordingEngine("beta");
  const engine = createMicroBatchingGenerationEngine({
    engine: createModelRouterGenerationEngine({
      engines: {
        alpha: alpha.engine,
        beta: beta.engine,
      },
    }),
    batchWindowMs: 25,
    maxBatchSize: 8,
  });

  const server = startServeServer({
    hostname: "127.0.0.1",
    port: 0,
    engine,
  });

  try {
    const endpoint = `http://${server.hostname}:${server.port}`;
    const agentCalls = [
      completion(endpoint, { model: "alpha", prompt: "agent one", max_tokens: 8 }),
      completion(endpoint, { model: "beta", prompt: "agent two", max_tokens: 8 }),
      completion(endpoint, { model: "alpha", prompt: "agent three", max_tokens: 8 }),
      completion(endpoint, { model: "beta", prompt: "agent four", max_tokens: 8 }),
    ];
    const responses = await Promise.all(agentCalls);

    assert(alpha.batches.length === 1, "alpha requests should share one model batch");
    assert(beta.batches.length === 1, "beta requests should share one model batch");
    assert(alpha.batches[0]?.length === 2, "alpha batch should contain two agent requests");
    assert(beta.batches[0]?.length === 2, "beta batch should contain two agent requests");

    console.log(
      JSON.stringify(
        {
          endpoint,
          responses: responses.map((response) => ({
            id: response.id,
            model: response.model,
            text: response.choices[0]?.text,
            usage: response.usage,
          })),
          batches: {
            alpha: alpha.batches,
            beta: beta.batches,
          },
        },
        null,
        2,
      ),
    );
  } finally {
    server.stop(true);
  }
}

await main();
