/**
 * OpenAI Responses text-input parsing.
 * @module
 */

import type { ChatMessage } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";

type ResponseInputRole = "assistant" | "developer" | "system" | "user";

function responseInputRole(value: unknown): ResponseInputRole {
  switch (value) {
    case "assistant":
    case "developer":
    case "system":
    case "user":
      return value;
    default:
      throw new ServeError(
        'OpenAI responses: input message role must be "system", "developer", "user", or "assistant".',
        { param: "input" },
      );
  }
}

function textContentPart(value: unknown): string {
  if (!isRecord(value)) {
    throw new ServeError("OpenAI responses: input content parts must be objects.", {
      param: "input",
    });
  }

  switch (value.type) {
    case "input_text":
    case "output_text":
    case "text":
      if (typeof value.text !== "string") {
        throw new ServeError(
          'OpenAI responses: text input content parts require a string "text" field.',
          { param: "input" },
        );
      }
      return value.text;
    case "input_image":
    case "image_url":
      throw new ServeError(
        "OpenAI responses: image input parts are not supported by this endpoint yet.",
        { param: "input" },
      );
    case "input_file":
      throw new ServeError(
        "OpenAI responses: file input parts are not supported by this endpoint yet.",
        { param: "input" },
      );
    case "input_audio":
      throw new ServeError(
        "OpenAI responses: audio input parts are not supported by this endpoint yet.",
        { param: "input" },
      );
    default:
      throw new ServeError(
        'OpenAI responses: only text content parts are supported in "input" today.',
        { param: "input" },
      );
  }
}

function inputContent(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(textContentPart).join("");
  }
  throw new ServeError(
    'OpenAI responses: input message "content" must be a string or an array of text parts.',
    { param: "input" },
  );
}

function inputMessage(value: unknown): ChatMessage {
  if (!isRecord(value)) {
    throw new ServeError("OpenAI responses: input array entries must be objects.", {
      param: "input",
    });
  }
  if (value.type !== undefined && value.type !== null && value.type !== "message") {
    if (value.type === "function_call_output") {
      throw new ServeError(
        "OpenAI responses: function-call output input is not supported until tools are implemented.",
        { param: "input" },
      );
    }
    if (value.type === "reasoning") {
      throw new ServeError(
        "OpenAI responses: reasoning input items are not supported by this endpoint yet.",
        { param: "input" },
      );
    }
    throw new ServeError("OpenAI responses: only text message input items are supported today.", {
      param: "input",
    });
  }

  const role = responseInputRole(value.role);
  const content = inputContent(value.content);
  switch (role) {
    case "assistant":
      return { role: "assistant", content };
    case "developer":
    case "system":
      return { role: "system", content };
    case "user":
      return { role: "user", content };
  }
}

/** Parse OpenAI Responses string or text-only item-array input into chat messages. */
export function parseOpenAIResponseInputMessages(
  record: Record<string, unknown>,
  instructions: string | null,
): ChatMessage[] {
  const input = record.input;
  const messages: ChatMessage[] = [];
  if (instructions !== null) {
    messages.push({ role: "system", content: instructions });
  }

  if (typeof input === "string") {
    if (input.trim() === "") {
      throw new ServeError('OpenAI responses: "input" must be a non-empty string.', {
        param: "input",
      });
    }
    messages.push({ role: "user", content: input });
    return messages;
  }

  if (!Array.isArray(input) || input.length === 0) {
    throw new ServeError(
      'OpenAI responses: "input" must be a non-empty string or text message array.',
      { param: "input" },
    );
  }

  messages.push(...input.map(inputMessage));
  return messages;
}
