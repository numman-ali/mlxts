/**
 * OpenResponses function-tool request parsing.
 * @module
 */

import type { ChatTool } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";

export type OpenAIResponseFunctionTool = {
  type: "function";
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  strict?: boolean;
};

export type ParsedOpenAIResponseTools = {
  responseTools: readonly OpenAIResponseFunctionTool[];
  chatTools?: readonly ChatTool[];
};

function nonEmptyFunctionName(value: unknown): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new ServeError("OpenAI responses: function tool name must be non-empty.", {
      param: "tools",
    });
  }
  return value;
}

function optionalDescription(value: unknown): { description?: string } {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "string") {
    throw new ServeError("OpenAI responses: function tool description must be a string.", {
      param: "tools",
    });
  }
  return { description: value };
}

function optionalParameters(value: unknown): { parameters?: Record<string, unknown> } {
  if (value === undefined || value === null) {
    return {};
  }
  if (!isRecord(value)) {
    throw new ServeError("OpenAI responses: function tool parameters must be an object.", {
      param: "tools",
    });
  }
  return { parameters: value };
}

function optionalStrict(value: unknown): { strict?: boolean } {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "boolean") {
    throw new ServeError("OpenAI responses: function tool strict must be a boolean.", {
      param: "tools",
    });
  }
  return { strict: value };
}

function functionToolRecord(value: Record<string, unknown>): Record<string, unknown> {
  return isRecord(value.function) ? value.function : value;
}

function responseFunctionTool(value: unknown): OpenAIResponseFunctionTool {
  if (!isRecord(value) || value.type !== "function") {
    throw new ServeError('OpenAI responses: "tools" entries must be function tools.', {
      param: "tools",
    });
  }
  const source = functionToolRecord(value);
  return {
    type: "function",
    name: nonEmptyFunctionName(source.name),
    ...optionalDescription(source.description),
    ...optionalParameters(source.parameters),
    ...optionalStrict(source.strict),
  };
}

function chatTool(tool: OpenAIResponseFunctionTool): ChatTool {
  return {
    type: "function",
    function: {
      name: tool.name,
      ...(tool.description === undefined ? {} : { description: tool.description }),
      ...(tool.parameters === undefined ? {} : { parameters: tool.parameters }),
    },
  };
}

/** Parse OpenResponses function tools into wire-echo and chat-template shapes. */
export function parseOpenAIResponseTools(
  record: Record<string, unknown>,
): ParsedOpenAIResponseTools {
  const value = record.tools;
  if (value === undefined || value === null) {
    return { responseTools: [] };
  }
  if (!Array.isArray(value)) {
    throw new ServeError('OpenAI responses: "tools" must be an array or null.', {
      param: "tools",
    });
  }
  const responseTools = value.map(responseFunctionTool);
  return {
    responseTools,
    ...(responseTools.length === 0 ? {} : { chatTools: responseTools.map(chatTool) }),
  };
}
