/**
 * OpenAI-compatible chat completions adapter for agent loops.
 * @module
 */

import { aggregateAgentModelStream, streamOpenAIChatCompletionEvents } from "./chat-streaming";
import { formatToolInstructions } from "./tool-calls";
import type { AgentMessage, AgentModel, AgentTool, AgentToolCall } from "./types";

export type ChatAgentModelOptions = {
  endpoint: string;
  model: string;
  apiKey?: string;
  maxTokens?: number;
  temperature?: number;
  enableThinking?: boolean;
  stream?: boolean;
  verbose?: boolean;
  fetch?: (input: string | URL | Request, init?: ChatFetchInit) => Promise<Response>;
};

export const DEFAULT_AGENT_MAX_TOKENS = 512;

type OpenAIChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  reasoning_content?: string;
  name?: string;
  tool_call_id?: string;
};

type ChatFetchInit = RequestInit & {
  verbose?: boolean;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function chatUrl(endpoint: string): string {
  const trimmed = endpoint.replace(/\/+$/, "");
  if (trimmed.endsWith("/v1/chat/completions")) {
    return trimmed;
  }
  if (trimmed.endsWith("/v1")) {
    return `${trimmed}/chat/completions`;
  }
  return `${trimmed}/v1/chat/completions`;
}

function toOpenAIMessage(message: AgentMessage): OpenAIChatMessage {
  return {
    role: message.role,
    content: message.content,
    ...(message.reasoningContent === undefined
      ? {}
      : { reasoning_content: message.reasoningContent }),
    ...(message.name === undefined ? {} : { name: message.name }),
    ...(message.toolCallId === undefined ? {} : { tool_call_id: message.toolCallId }),
  };
}

function toOpenAITool(tool: AgentTool): Record<string, unknown> {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters ?? { type: "object", properties: {} },
    },
  };
}

function messagesWithToolInstructions(
  messages: readonly AgentMessage[],
  tools: readonly AgentTool[],
): OpenAIChatMessage[] {
  const formattedMessages = messages.map(toOpenAIMessage);
  const instructions = formatToolInstructions(tools);
  if (instructions === "") {
    return formattedMessages;
  }
  return [{ role: "system", content: instructions }, ...formattedMessages];
}

function requestBody(
  options: ChatAgentModelOptions,
  messages: readonly AgentMessage[],
  tools: readonly AgentTool[],
  stream: boolean,
) {
  return {
    model: options.model,
    messages: messagesWithToolInstructions(messages, tools),
    ...(tools.length === 0 ? {} : { tools: tools.map(toOpenAITool), tool_choice: "auto" }),
    max_tokens: options.maxTokens ?? DEFAULT_AGENT_MAX_TOKENS,
    ...(stream ? { stream: true } : {}),
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
    ...(options.enableThinking === undefined
      ? {}
      : { chat_template_kwargs: { enable_thinking: options.enableThinking } }),
  };
}

const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";

function cleanReasoningFromText(text: string): { content: string; reasoningContent?: string } {
  const openIndex = text.indexOf(THINK_OPEN);
  const closeIndex = text.indexOf(THINK_CLOSE);
  if (closeIndex < 0 && openIndex < 0) {
    return { content: text.trim() };
  }

  if (closeIndex >= 0 && (openIndex < 0 || openIndex < closeIndex)) {
    const reasoningStart = openIndex < 0 ? 0 : openIndex + THINK_OPEN.length;
    const reasoning = text.slice(reasoningStart, closeIndex).trim();
    const content = text.slice(closeIndex + THINK_CLOSE.length).trim();
    return reasoning === "" ? { content } : { content, reasoningContent: reasoning };
  }

  const content = text.slice(0, openIndex).trim();
  const reasoning = text.slice(openIndex + THINK_OPEN.length).trim();
  return reasoning === "" ? { content } : { content, reasoningContent: reasoning };
}

function parseArguments(value: unknown): Record<string, unknown> {
  if (typeof value !== "string" || value.trim() === "") {
    return {};
  }
  try {
    const parsed: unknown = JSON.parse(value);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function toolCall(value: unknown, index: number): AgentToolCall | null {
  if (!isRecord(value) || !isRecord(value.function)) {
    return null;
  }
  const name = value.function.name;
  if (typeof name !== "string" || name.trim() === "") {
    return null;
  }
  return {
    id: typeof value.id === "string" && value.id.trim() !== "" ? value.id : `tool-${index + 1}`,
    name,
    arguments: parseArguments(value.function.arguments),
  };
}

function messageToolCalls(message: Record<string, unknown>): AgentToolCall[] | undefined {
  const value = message.tool_calls;
  if (!Array.isArray(value)) {
    return undefined;
  }
  const calls = value
    .map((entry, index) => toolCall(entry, index))
    .filter((entry): entry is AgentToolCall => entry !== null);
  return calls.length === 0 ? undefined : calls;
}

function responseMessage(value: unknown): {
  content: string;
  reasoningContent?: string;
  toolCalls?: AgentToolCall[];
} {
  if (!isRecord(value) || !Array.isArray(value.choices)) {
    throw new Error("Chat completion response did not include choices.");
  }
  const firstChoice = value.choices[0];
  if (!isRecord(firstChoice) || !isRecord(firstChoice.message)) {
    throw new Error("Chat completion response did not include a message choice.");
  }
  const content = firstChoice.message.content;
  const parsedContent = cleanReasoningFromText(typeof content === "string" ? content : "");
  const reasoningContent =
    typeof firstChoice.message.reasoning_content === "string"
      ? firstChoice.message.reasoning_content.trim()
      : parsedContent.reasoningContent;
  const toolCalls = messageToolCalls(firstChoice.message);
  return toolCalls === undefined
    ? {
        content: parsedContent.content,
        ...(reasoningContent === undefined || reasoningContent === "" ? {} : { reasoningContent }),
      }
    : {
        content: parsedContent.content,
        ...(reasoningContent === undefined || reasoningContent === "" ? {} : { reasoningContent }),
        toolCalls,
      };
}

async function errorText(response: Response): Promise<string> {
  try {
    return await response.text();
  } catch {
    return response.statusText;
  }
}

/** Create an agent model backed by an OpenAI-compatible `/v1/chat/completions` endpoint. */
export function createOpenAIChatAgentModel(options: ChatAgentModelOptions): AgentModel {
  const fetchImpl = options.fetch ?? fetch;
  const url = chatUrl(options.endpoint);

  const model: AgentModel = {
    async complete(request) {
      const headers = new Headers({ "content-type": "application/json" });
      if (options.apiKey !== undefined) {
        headers.set("authorization", `Bearer ${options.apiKey}`);
      }

      const init: ChatFetchInit = {
        method: "POST",
        headers,
        body: JSON.stringify(
          requestBody(options, request.messages, request.tools, options.stream === true),
        ),
      };
      if (options.verbose === true) {
        init.verbose = true;
      }

      const response = await fetchImpl(url, init);
      if (!response.ok) {
        throw new Error(
          `Chat completion request failed (${response.status}): ${await errorText(response)}`,
        );
      }

      if (options.stream === true) {
        return await aggregateAgentModelStream(streamOpenAIChatCompletionEvents(response));
      }

      return responseMessage(await response.json());
    },
  };

  if (options.stream === false) {
    return model;
  }

  model.stream = async (request) => {
    const headers = new Headers({ "content-type": "application/json" });
    if (options.apiKey !== undefined) {
      headers.set("authorization", `Bearer ${options.apiKey}`);
    }

    const init: ChatFetchInit = {
      method: "POST",
      headers,
      body: JSON.stringify(requestBody(options, request.messages, request.tools, true)),
    };
    if (options.verbose === true) {
      init.verbose = true;
    }

    const response = await fetchImpl(url, init);
    if (!response.ok) {
      throw new Error(
        `Chat completion request failed (${response.status}): ${await errorText(response)}`,
      );
    }

    return streamOpenAIChatCompletionEvents(response);
  };

  return model;
}
