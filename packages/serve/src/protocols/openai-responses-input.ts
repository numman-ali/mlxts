/**
 * OpenAI Responses text-input parsing.
 * @module
 */

import type { ChatMessage } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";
import type { GenerationContentMessage, GenerationContentPart, GenerationInput } from "../types";
import {
  hasMediaContent,
  mediaSourceFromDataUrl,
  mediaSourceFromFileId,
  mediaSourceFromUrl,
  textContentPart,
} from "./media-content";

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

type ParsedResponseContent = {
  text: string;
  parts: readonly GenerationContentPart[];
  hasMedia: boolean;
};

type ParsedResponseMessage = {
  chat: ChatMessage;
  content: GenerationContentMessage;
  hasMedia: boolean;
};

function responseImagePart(value: Record<string, unknown>): GenerationContentPart {
  if (value.image_url !== undefined && value.image_url !== null) {
    return {
      kind: "image",
      source: mediaSourceFromUrl(value.image_url, "OpenAI responses: image input parts", "input"),
    };
  }
  if (value.file_id !== undefined && value.file_id !== null) {
    return {
      kind: "image",
      source: mediaSourceFromFileId(value.file_id, "OpenAI responses: image input parts", "input"),
    };
  }
  throw new ServeError('OpenAI responses: image input parts require "image_url" or "file_id".', {
    param: "input",
  });
}

function responseFilePart(value: Record<string, unknown>): GenerationContentPart {
  const source =
    value.file_id !== undefined && value.file_id !== null
      ? mediaSourceFromFileId(value.file_id, "OpenAI responses: file input parts", "input")
      : value.file_data !== undefined && value.file_data !== null
        ? mediaSourceFromDataUrl(value.file_data, "OpenAI responses: file input parts", "input")
        : null;
  if (source === null) {
    throw new ServeError(
      'OpenAI responses: file input parts require "file_id" or data-url "file_data".',
      { param: "input" },
    );
  }
  return {
    kind: "file",
    source,
    ...(typeof value.filename === "string" ? { filename: value.filename } : {}),
  };
}

function responseAudioPart(value: Record<string, unknown>): GenerationContentPart {
  if (!isRecord(value.input_audio)) {
    throw new ServeError('OpenAI responses: audio input parts require an "input_audio" object.', {
      param: "input",
    });
  }
  const data = value.input_audio.data;
  const format = value.input_audio.format;
  if (typeof data !== "string" || data.trim() === "") {
    throw new ServeError('OpenAI responses: audio input requires non-empty base64 "data".', {
      param: "input",
    });
  }
  if (typeof format !== "string" || format.trim() === "") {
    throw new ServeError('OpenAI responses: audio input requires a non-empty "format".', {
      param: "input",
    });
  }
  return {
    kind: "audio",
    source: {
      kind: "data",
      mediaType: `audio/${format}`,
      data,
    },
    format,
  };
}

function responseContentPart(value: unknown): GenerationContentPart {
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
      return textContentPart(value.text);
    case "input_image":
    case "image_url":
      return responseImagePart(value);
    case "input_file":
      return responseFilePart(value);
    case "input_audio":
      return responseAudioPart(value);
    default:
      throw new ServeError(
        'OpenAI responses: only text, image, file, and audio content parts are supported in "input" today.',
        { param: "input" },
      );
  }
}

function joinTextParts(parts: readonly GenerationContentPart[]): string {
  let text = "";
  for (const part of parts) {
    if (part.kind === "text") {
      text += part.text;
    }
  }
  return text;
}

function parsedTextContent(text: string): ParsedResponseContent {
  return {
    text,
    parts: text === "" ? [] : [textContentPart(text)],
    hasMedia: false,
  };
}

function inputContent(value: unknown): ParsedResponseContent {
  if (value === undefined || value === null) {
    return parsedTextContent("");
  }
  if (typeof value === "string") {
    return parsedTextContent(value);
  }
  if (Array.isArray(value)) {
    const parts = value.map(responseContentPart);
    return {
      text: joinTextParts(parts),
      parts,
      hasMedia: hasMediaContent(parts),
    };
  }
  throw new ServeError(
    'OpenAI responses: input message "content" must be a string or an array of content parts.',
    { param: "input" },
  );
}

function inputMessage(value: unknown): ParsedResponseMessage {
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
      return {
        chat: { role: "assistant", content: content.text },
        content: { role: "assistant", content: content.parts },
        hasMedia: content.hasMedia,
      };
    case "developer":
    case "system":
      return {
        chat: { role: "system", content: content.text },
        content: { role: "system", content: content.parts },
        hasMedia: content.hasMedia,
      };
    case "user":
      return {
        chat: { role: "user", content: content.text },
        content: { role: "user", content: content.parts },
        hasMedia: content.hasMedia,
      };
  }
}

/** Parse OpenAI Responses input into text-only messages or ordered media content. */
export function parseOpenAIResponseInput(
  record: Record<string, unknown>,
  instructions: string | null,
): GenerationInput {
  const input = record.input;
  const messages: ChatMessage[] = [];
  const contentMessages: GenerationContentMessage[] = [];
  if (instructions !== null) {
    messages.push({ role: "system", content: instructions });
    contentMessages.push({ role: "system", content: [textContentPart(instructions)] });
  }

  if (typeof input === "string") {
    if (input.trim() === "") {
      throw new ServeError('OpenAI responses: "input" must be a non-empty string.', {
        param: "input",
      });
    }
    messages.push({ role: "user", content: input });
    return {
      kind: "messages",
      messages,
    };
  }

  if (!Array.isArray(input) || input.length === 0) {
    throw new ServeError(
      'OpenAI responses: "input" must be a non-empty string or text message array.',
      { param: "input" },
    );
  }

  const parsedMessages = input.map(inputMessage);
  messages.push(...parsedMessages.map((message) => message.chat));
  contentMessages.push(...parsedMessages.map((message) => message.content));
  if (!parsedMessages.some((message) => message.hasMedia)) {
    return {
      kind: "messages",
      messages,
    };
  }
  return {
    kind: "content",
    messages: contentMessages,
  };
}
